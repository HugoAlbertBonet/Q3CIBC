"""Simulation class for the AdroitHandPen-v1 (D4RL/pen/human-v2) environment."""

import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
import mujoco

from .base_simulation import BaseSimulation


class PenHumanV2Simulation(BaseSimulation):
    """Evaluator for trained Q3CIBC policies on AdroitHandPen-v1 (D4RL pen-human).

    The IBC paper reports per-episode return as the primary metric on D4RL pen
    (no strict success bit in their Table 2). gymnasium-robotics' Adroit envs
    do emit `info["success"]` at terminal states, so we surface both:
      - `total_reward` (paper-comparable)
      - `success` (binary, from env info)
    """

    def __init__(
        self,
        control_point_generator,
        q_estimator,
        device: str = "cpu",
        max_episode_steps: int = 200,
        render_mode: str | None = None,
        frame_stack: int = 1,
        norm_stats: dict | None = None,  # accepted for API parity, not used (no action norm)
    ) -> None:
        super().__init__(
            env_id="AdroitHandPen-v1",
            control_point_generator=control_point_generator,
            q_estimator=q_estimator,
            device=device,
            max_episode_steps=max_episode_steps,
            frame_stack=frame_stack,
        )
        self.render_mode = render_mode
        # The Langevin/DFO refinement wrappers in hyperparam_search.evaluate_q3c
        # check `_act_min_t` to decide whether to normalize CPs before scoring.
        # Pen's CP generator emits actions in env space ([-1, 1]) and the Q
        # estimator was trained on the same range, so no transform is needed.
        self._act_min_t = None
        self._act_rng_t = None

    def create_env(self) -> gym.Env:
        gym.register_envs(gymnasium_robotics)
        env = gym.make(
            self.env_id,
            reward_type="dense",
            max_episode_steps=self.max_episode_steps,
            render_mode=self.render_mode,
        )
        return env

    def _render_callback(self, reward: float, total_reward: float) -> None:
        if self.render_mode == "human" and hasattr(self.env.unwrapped, "mujoco_renderer"):
            renderer = self.env.unwrapped.mujoco_renderer
            if hasattr(renderer, "viewer") and renderer.viewer:
                renderer.viewer.add_overlay(
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "Reward",
                    f"{reward:.4f}",
                )
                renderer.viewer.add_overlay(
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "Total Reward",
                    f"{total_reward:.4f}",
                )

    def select_action(self, observation: np.ndarray, return_q_range: bool = False):
        obs_tensor = (
            torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        obs_tensor = self.obs_normalizer.normalize(obs_tensor)

        with torch.no_grad():
            control_points = self.control_point_generator(obs_tensor)
            obs_expanded = obs_tensor.unsqueeze(1).expand(-1, control_points.shape[1], -1)
            q_values = self.q_estimator(obs_expanded, control_points).squeeze(-1)
            best_idx = q_values.argmax(dim=1)
            action = control_points[0, best_idx[0], :].cpu().numpy()
            q_range = (q_values.min().item(), q_values.max().item())

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
        done = False
        info: dict = {}
        success = False

        while not done:
            action = self.select_action(stacked_obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            stacked_obs = self._update_frame_buffer(obs)
            total_reward += float(reward)
            episode_length += 1
            # AdroitHand envs (gymnasium-robotics ≥1.2) emit "success" each step
            # while the goal-pose tolerance is met. We treat the episode as a
            # success if the goal was reached at any point — matches Adroit's
            # standard evaluation protocol.
            step_success = bool(info.get("success", info.get("is_success", False)))
            if step_success:
                success = True
            self._render_callback(reward, total_reward)
            done = terminated or truncated

        return {
            "episode_length": episode_length,
            "total_reward": total_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": success,
        }

    def get_summary(self) -> dict[str, float]:
        summary = super().get_summary()
        if not self.results:
            return summary
        successes = [bool(r.get("success", False)) for r in self.results]
        summary["success_rate"] = float(np.mean(successes))
        summary["success_rate_std"] = float(np.std(successes))
        return summary
