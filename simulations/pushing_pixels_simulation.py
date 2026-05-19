"""Simulation runner for the IBC BlockPush env in RGB-observation mode.

Sibling of `pushing_simulation.py` for the IBC paper's "Block Pushing —
Single target, Images" task. Mirrors `PushingSimulation`'s overall shape
(same env id namespace, same per-seed eval, same denormalized actions) but
with two important differences:

  1. Observations are images (H, W, 3) uint8 — frame-stacking is channel-wise
     (stacks two frames into (H, W, 3*frame_stack)) rather than the
     flat-concatenation `BaseSimulation` does for vector obs.
  2. `self.obs_normalizer` is never invoked. The conv encoder
     (`utils.models.ConvMaxpoolEncoder`) does its own preprocessing
     (uint8→float / 255, bilinear resize to 180×240). The base-class
     normalizer is left in place only because BaseSimulation.__init__
     constructs it unconditionally; a stub entry in
     `observation_bounds.json` keeps the construction from crashing.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import torch

from .base_simulation import BaseSimulation
from .pushing_pixels_env import PushingPixelsEnv


class PushingPixelsSimulation(BaseSimulation):
    """Evaluator for trained Q3CIBC pixel policies on PushingPixels-v0."""

    def __init__(
        self,
        control_point_generator: torch.nn.Module,
        q_estimator: torch.nn.Module,
        device: str = "cpu",
        max_episode_steps: int = 100,
        render_mode: Optional[str] = None,
        frame_stack: int = 1,
        norm_stats: Optional[dict] = None,
        goal_dist_tolerance: float = 0.02,
    ) -> None:
        super().__init__(
            env_id="PushingPixels-v0",
            control_point_generator=control_point_generator,
            q_estimator=q_estimator,
            device=device,
            max_episode_steps=max_episode_steps,
            frame_stack=frame_stack,
        )
        self.render_mode = render_mode
        self.goal_dist_tolerance = goal_dist_tolerance
        self.norm_stats = norm_stats

        # See PushingSimulation: these `_act_*_t` are reserved for the legacy
        # ibc_with_cps action remapping. Set to None to short-circuit the
        # LangevinRefinedParticleSimulation wrapper in hyperparam_search.
        self._act_min_t = None
        self._act_rng_t = None
        if norm_stats is not None:
            self._raw_act_min = np.asarray(norm_stats["act_min"], dtype=np.float32)
            self._raw_act_max = np.asarray(norm_stats["act_max"], dtype=np.float32)
            lo_hi = norm_stats.get("action_norm_range", (-1.0, 1.0))
            self._act_lo = float(lo_hi[0])
            self._act_hi = float(lo_hi[1])
        else:
            self._raw_act_min = None
            self._raw_act_max = None
            self._act_lo = None
            self._act_hi = None

    def create_env(self) -> PushingPixelsEnv:
        return PushingPixelsEnv(
            n_steps=self.max_episode_steps,
            render_mode=self.render_mode,
            goal_dist_tolerance=self.goal_dist_tolerance,
        )

    def _denormalize_action(self, action_normalized: np.ndarray) -> np.ndarray:
        if self._raw_act_min is None:
            return action_normalized
        scale = (self._raw_act_max - self._raw_act_min) / (self._act_hi - self._act_lo)
        return (
            self._raw_act_min + (np.asarray(action_normalized, dtype=np.float32) - self._act_lo) * scale
        ).astype(np.float32)

    # ── Frame-stacking overrides (channel-wise for images) ──────────────────
    # BaseSimulation's defaults concat 1D vectors. Images need channel-wise
    # stack: two (H, W, 3) frames → one (H, W, 6) frame.

    def _get_stacked_obs(self) -> np.ndarray:
        if self.frame_stack <= 1:
            return self._frame_buffer[-1]
        return np.concatenate(list(self._frame_buffer), axis=-1)  # (H, W, 3*fs)

    def _obs_to_tensor(self, stacked_obs_hwc: np.ndarray) -> torch.Tensor:
        """(H, W, 3*fs) uint8 → (1, 3*fs, H, W) uint8 tensor on device."""
        # Channels-last → channels-first for the conv encoder.
        chw = np.transpose(stacked_obs_hwc, (2, 0, 1))  # (3*fs, H, W)
        return torch.from_numpy(chw).unsqueeze(0).to(self.device)  # (1, 3*fs, H, W)

    def select_action(self, observation: np.ndarray, return_q_range: bool = False):
        """Pick best CP under Q. Encoder runs ONCE per state (late fusion).

        Args:
            observation: (H, W, 3*frame_stack) uint8 image stack as returned
                by `_get_stacked_obs`.
        """
        obs_tensor = self._obs_to_tensor(observation)  # (1, C, H, W) uint8

        with torch.no_grad():
            control_points = self.control_point_generator(obs_tensor)  # (1, N, action_dim)
            # Late fusion: encode once, broadcast features over the N CPs.
            features = self.q_estimator.encode(obs_tensor)  # (1, F)
            q_values = self.q_estimator.score(features, control_points).squeeze(-1)  # (1, N)
            best_idx = q_values.argmax(dim=1)
            action_normalized = control_points[0, best_idx[0], :].cpu().numpy()
            q_range = (q_values.min().item(), q_values.max().item())

        action = self._denormalize_action(action_normalized)
        if return_q_range:
            return action, q_range
        return action

    def run_episode(self, seed: int | None = None) -> dict:
        if self.env is None:
            self.env = self.create_env()

        obs, _ = self.env.reset(seed=seed)
        stacked_obs = self._reset_frame_buffer(obs)

        total_reward = 0.0
        episode_length = 0
        min_dist = float("inf")
        done = False
        info: dict = {}

        while not done:
            action = self.select_action(stacked_obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            stacked_obs = self._update_frame_buffer(obs)
            total_reward += float(reward)
            episode_length += 1
            min_dist = min(min_dist, float(info.get("block_to_target_distance", np.inf)))
            done = terminated or truncated

        return {
            "episode_length": episode_length,
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "success": bool(info.get("success", False)),
            "min_dist_to_target": min_dist,
            "final_dist_to_target": float(info.get("block_to_target_distance", np.inf)),
        }

    def get_summary(self) -> dict[str, float]:
        summary = super().get_summary()
        if not self.results:
            return summary
        successes = [r.get("success", False) for r in self.results]
        summary["success_rate"] = float(np.mean(successes))
        summary["success_rate_std"] = float(np.std(successes))
        dists = [r.get("min_dist_to_target", np.inf) for r in self.results]
        finite_dists = [d for d in dists if np.isfinite(d)]
        summary["avg_min_dist_to_target"] = float(np.mean(finite_dists)) if finite_dists else float("inf")
        summary["std_min_dist_to_target"] = float(np.std(finite_dists)) if finite_dists else float("inf")
        return summary
