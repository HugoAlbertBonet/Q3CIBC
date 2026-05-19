"""Simulation runner for the IBC BlockPushMultimodal environment.

Sibling of `pushing_simulation.py` for the IBC paper's "Block Pushing —
Multimodal, States" task (2 blocks, 2 targets). Mirrors `PushingSimulation`
so `hyperparam_search.evaluate_q3c` can swap it in by `active_env`.

Paper-faithful normalization (matches `get_normalizers.py` / `ibc/train/stats.py`):
  - Observations: standardize (x - mean) / std using stats saved at training
    time in `norm_stats.pt`.
  - Actions: model output lives in [-1, 1]; we linearly map to the env's
    native effector-delta range before `env.step`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from utils.normalizations import ObservationNormalizer

from .base_simulation import BaseSimulation
from .pushing_multi_env import PushingMultiEnv


class PushingMultiSimulation(BaseSimulation):
    """Evaluator for trained Q3CIBC policies on PushingMulti-v0.

    `succeeded` comes from the IBC env's internal success flag (both blocks
    are inside distinct targets within `goal_dist_tolerance`). We report
    mean and per-block block→closest-target distances so trial logs surface
    partial progress when only one block ends up in a target.
    """

    def __init__(
        self,
        control_point_generator: torch.nn.Module,
        q_estimator: torch.nn.Module,
        device: str = "cpu",
        # IBC paper's BlockPushMultimodal-v0 registers with max_episode_steps=200.
        max_episode_steps: int = 200,
        render_mode: Optional[str] = None,
        frame_stack: int = 1,
        norm_stats: Optional[dict] = None,
        # Class default matches IBC (0.04, looser than single-target).
        goal_dist_tolerance: float = 0.04,
    ) -> None:
        super().__init__(
            env_id="PushingMulti-v0",
            control_point_generator=control_point_generator,
            q_estimator=q_estimator,
            device=device,
            max_episode_steps=max_episode_steps,
            frame_stack=frame_stack,
        )
        self.render_mode = render_mode
        self.goal_dist_tolerance = goal_dist_tolerance
        self.norm_stats = norm_stats

        if norm_stats is not None and "obs_mean" in norm_stats and "obs_std" in norm_stats:
            self.obs_normalizer = ObservationNormalizer(
                env_id="PushingMulti-v0",
                device=device,
                frame_stack=frame_stack,
                obs_mean=np.asarray(norm_stats["obs_mean"], dtype=np.float32),
                obs_std=np.asarray(norm_stats["obs_std"], dtype=np.float32),
            )

        # See PushingSimulation for why these are set to None — short-circuits
        # the legacy ibc_with_cps action remapping inside hyperparam_search's
        # LangevinRefinedParticleSimulation wrapper.
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

    def create_env(self) -> PushingMultiEnv:
        return PushingMultiEnv(
            n_steps=self.max_episode_steps,
            render_mode=self.render_mode,
            goal_dist_tolerance=self.goal_dist_tolerance,
        )

    def _denormalize_action(self, action_normalized: np.ndarray) -> np.ndarray:
        if self._raw_act_min is None:
            return action_normalized
        scale = (self._raw_act_max - self._raw_act_min) / (self._act_hi - self._act_lo)
        return (self._raw_act_min + (np.asarray(action_normalized, dtype=np.float32) - self._act_lo) * scale).astype(np.float32)

    def select_action(self, observation: np.ndarray, return_q_range: bool = False):
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        obs_tensor = self.obs_normalizer.normalize(obs_tensor)

        with torch.no_grad():
            control_points = self.control_point_generator(obs_tensor)
            obs_expanded = obs_tensor.unsqueeze(1).expand(-1, control_points.shape[1], -1)
            q_values = self.q_estimator(obs_expanded, control_points).squeeze(-1)
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
        min_mean_dist = float("inf")
        min_b0_dist = float("inf")
        min_b1_dist = float("inf")
        done = False
        info: dict = {}

        while not done:
            action = self.select_action(stacked_obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            stacked_obs = self._update_frame_buffer(obs)
            total_reward += float(reward)
            episode_length += 1
            min_mean_dist = min(min_mean_dist, float(info.get("block_to_target_distance", np.inf)))
            min_b0_dist = min(min_b0_dist, float(info.get("block0_to_closest_target_distance", np.inf)))
            min_b1_dist = min(min_b1_dist, float(info.get("block1_to_closest_target_distance", np.inf)))
            done = terminated or truncated

        return {
            "episode_length": episode_length,
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "success": bool(info.get("success", False)),
            "min_mean_dist_to_target": min_mean_dist,
            "min_block0_dist_to_target": min_b0_dist,
            "min_block1_dist_to_target": min_b1_dist,
            "final_mean_dist_to_target": float(info.get("block_to_target_distance", np.inf)),
        }

    def get_summary(self) -> dict[str, float]:
        summary = super().get_summary()
        if not self.results:
            return summary
        successes = [r.get("success", False) for r in self.results]
        summary["success_rate"] = float(np.mean(successes))
        summary["success_rate_std"] = float(np.std(successes))
        dists = [r.get("min_mean_dist_to_target", np.inf) for r in self.results]
        finite_dists = [d for d in dists if np.isfinite(d)]
        summary["avg_min_mean_dist_to_target"] = float(np.mean(finite_dists)) if finite_dists else float("inf")
        summary["std_min_mean_dist_to_target"] = float(np.std(finite_dists)) if finite_dists else float("inf")
        return summary
