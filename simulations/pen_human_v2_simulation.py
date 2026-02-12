"""Simulation class for the AdroitHandPen-v1 (D4RL/pen/human-v2) environment."""

import numpy as np
import gymnasium as gym
import gymnasium_robotics
import mujoco

from .base_simulation import BaseSimulation


class PenHumanV2Simulation(BaseSimulation):
    """Simulation for testing trained policies on the AdroitHandPen environment.
    
    This class evaluates the trained control point generator by running
    episodes in the actual gymnasium environment and collecting metrics.
    """

    def __init__(
        self,
        control_point_generator,
        q_estimator,
        smoothing_param,
        device: str = "cpu",
        max_episode_steps: int = 200,
        render_mode: str | None = None,
        frame_stack: int = 1,
    ) -> None:
        """Initialize the PenHumanV2 simulation.
        
        Args:
            control_point_generator: The trained policy model.
            q_estimator: The trained Q-value estimator.
            smoothing_param: The smoothing parameter for wire fitting normalization.
            device: The device to run computations on.
            max_episode_steps: Maximum steps per episode.
            render_mode: Gymnasium render mode (None, 'human', 'rgb_array').
        """
        super().__init__(
            env_id="AdroitHandPen-v1",
            control_point_generator=control_point_generator,
            q_estimator=q_estimator,
            smoothing_param=smoothing_param,
            device=device,
            max_episode_steps=max_episode_steps,
            frame_stack=frame_stack,
        )
        self.render_mode = render_mode

    def create_env(self) -> gym.Env:
        """Create the AdroitHandPen gymnasium environment."""
        # Register gymnasium-robotics environments
        gym.register_envs(gymnasium_robotics)
        
        env = gym.make(
            self.env_id,
            reward_type="dense",
            max_episode_steps=self.max_episode_steps,
            render_mode=self.render_mode,
        )
        return env

    def _render_callback(self, reward: float, total_reward: float) -> None:
        """Render reward overlay."""
        if self.render_mode == "human" and hasattr(self.env.unwrapped, "mujoco_renderer"):
            renderer = self.env.unwrapped.mujoco_renderer
            if hasattr(renderer, "viewer") and renderer.viewer:
                renderer.viewer.add_overlay(
                    mujoco.mjtGridPos.mjGRID_TOPLEFT, 
                    "Reward", 
                    f"{reward:.4f}"
                )
                renderer.viewer.add_overlay(
                    mujoco.mjtGridPos.mjGRID_TOPLEFT, 
                    "Total Reward", 
                    f"{total_reward:.4f}"
                )

    def get_summary(self) -> dict[str, float]:
        """Get summary statistics including success rate if available.
        
        Returns:
            Dictionary with summary statistics.
        """
        summary = super().get_summary()
        
        if not self.results:
            return summary
        
        # Check for success info (some environments report this)
        successes = [r.get("success", False) for r in self.results if "success" in r]
        if successes:
            summary["success_rate"] = np.mean(successes)
        
        return summary
