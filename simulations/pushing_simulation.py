"""Simulation runner for the IBC BlockPush(PUSH) environment.

Mirrors `ParticleSimulation` so the rest of the Q3CIBC pipeline
(`hyperparam_search.evaluate_q3c`, multi-seed evaluation) can swap in
this env by env-id.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from .base_simulation import BaseSimulation
from .pushing_env import PushingEnv


class PushingSimulation(BaseSimulation):
    """Evaluator for trained Q3CIBC policies on Pushing-v0.

    `succeeded` comes from the IBC env's internal success flag (block within
    its goal tolerance of the target after enough settling time). We report
    final and minimum block→target distances so trial logs surface partial
    progress even when the strict success bit doesn't flip.
    """

    def __init__(
        self,
        control_point_generator: torch.nn.Module,
        q_estimator: torch.nn.Module,
        device: str = "cpu",
        max_episode_steps: int = 100,
        render_mode: Optional[str] = None,
        frame_stack: int = 1,
        norm_stats: Optional[dict] = None,
    ) -> None:
        super().__init__(
            env_id="Pushing-v0",
            control_point_generator=control_point_generator,
            q_estimator=q_estimator,
            device=device,
            max_episode_steps=max_episode_steps,
            frame_stack=frame_stack,
        )
        self.render_mode = render_mode
        if norm_stats is not None:
            act_min = np.asarray(norm_stats["act_min"], dtype=np.float32)
            act_max = np.asarray(norm_stats["act_max"], dtype=np.float32)
            rng = act_max - act_min
            rng = np.where(rng == 0, np.ones_like(rng), rng)
            self._act_min_t = torch.from_numpy(act_min).to(device)
            self._act_rng_t = torch.from_numpy(rng).to(device)
        else:
            self._act_min_t = None
            self._act_rng_t = None

    def create_env(self) -> PushingEnv:
        return PushingEnv(
            n_steps=self.max_episode_steps,
            render_mode=self.render_mode,
        )

    def select_action(self, observation: np.ndarray, return_q_range: bool = False):
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        obs_tensor = self.obs_normalizer.normalize(obs_tensor)

        with torch.no_grad():
            control_points = self.control_point_generator(obs_tensor)  # (1, N, A)
            obs_expanded = obs_tensor.unsqueeze(1).expand(-1, control_points.shape[1], -1)

            if self._act_min_t is not None:
                cp_for_q = (control_points - self._act_min_t) / self._act_rng_t
            else:
                cp_for_q = control_points

            q_values = self.q_estimator(obs_expanded, cp_for_q).squeeze(-1)
            best_idx = q_values.argmax(dim=1)
            action = control_points[0, best_idx[0], :].cpu().numpy()
            q_range = (q_values.min().item(), q_values.max().item())

        # PushingEnv clips to its action box internally; no extra clipping here.
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
        dists = [r.get("min_dist_to_target", np.inf) for r in self.results]
        summary["avg_min_dist_to_target"] = float(np.mean([d for d in dists if np.isfinite(d)]))
        return summary
