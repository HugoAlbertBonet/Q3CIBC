"""Base simulation class for testing trained policies on gym environments."""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Any
import gymnasium as gym

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from normalizations import ObservationNormalizer


class BaseSimulation(ABC):
    """Abstract base class for running simulations with trained policies.
    
    This class provides the foundation for testing trained control point generators
    on gymnasium environments. Subclasses should implement environment-specific
    setup and action selection logic.
    """

    def __init__(
        self,
        env_id: str,
        control_point_generator: torch.nn.Module,
        q_estimator: torch.nn.Module,
        smoothing_param: torch.Tensor,
        device: str = "cpu",
        max_episode_steps: int = 400,
    ) -> None:
        """Initialize the simulation.
        
        Args:
            env_id: The gymnasium environment ID (e.g., 'AdroitHandPen-v1').
            control_point_generator: The trained policy model that generates control points.
            q_estimator: The trained Q-value estimator.
            smoothing_param: The smoothing parameter for wire fitting normalization.
            device: The device to run computations on ('cpu' or 'cuda').
            max_episode_steps: Maximum steps per episode.
        """
        self.env_id = env_id
        self.control_point_generator = control_point_generator
        self.q_estimator = q_estimator
        self.smoothing_param = smoothing_param
        self.device = device
        self.max_episode_steps = max_episode_steps
        self.env = None
        self.results: list[dict[str, Any]] = []
        
        # Observation normalizer (uses official bounds from JSON file)
        self.obs_normalizer = ObservationNormalizer(env_id=env_id, device=device)

    @abstractmethod
    def create_env(self) -> gym.Env:
        """Create and return the gymnasium environment."""
        pass

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        """Select action from control points based on Q-values.
        
        Uses the Q-estimator to evaluate each control point and selects
        the one with the maximum Q-value.
        
        Args:
            observation: The current observation from the environment.
            
        Returns:
            The selected action as a numpy array.
        """
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        obs_tensor = self.obs_normalizer.normalize(obs_tensor)  # Normalize to [0, 1]
        with torch.no_grad():
            control_points = self.control_point_generator(obs_tensor)  # (1, N, action_dim)
            
            # Get Q-values for all control points
            q_values = self.q_estimator(control_points).squeeze(-1)  # (1, N)
            
            # Select control point with maximum Q-value
            best_idx = q_values.argmax(dim=1)  # (1,)
            action = control_points[0, best_idx[0], :].cpu().numpy()
        return action

    def run_episode(self, seed: int | None = None) -> dict[str, Any]:
        """Run a single episode and return metrics.
        
        Args:
            seed: Optional random seed for this episode.
            
        Returns:
            A dictionary containing episode metrics.
        """
        if self.env is None:
            self.env = self.create_env()
        
        obs, info = self.env.reset(seed=seed)
        
        total_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            action = self.select_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        return {
            "episode_length": episode_length,
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
        }

    def run_simulation(
        self, 
        num_episodes: int = 100, 
        seed: int | None = None
    ) -> list[dict[str, Any]]:
        """Run the simulation for a specified number of episodes.
        
        Args:
            num_episodes: Number of episodes to run.
            seed: Base random seed. Each episode uses seed + episode_idx.
            
        Returns:
            A list of dictionaries containing per-episode metrics.
        """
        self.results = []
        self.control_point_generator.eval()
        self.q_estimator.eval()

        for idx in range(num_episodes):
            episode_seed = (seed + idx) if seed is not None else None
            result = self.run_episode(seed=episode_seed)
            result["episode_index"] = idx
            self.results.append(result)
                
        return self.results

    def get_summary(self) -> dict[str, float]:
        """Get summary statistics from the simulation results.
        
        Returns:
            A dictionary with summary statistics.
        """
        if not self.results:
            return {}
        
        episode_lengths = [r.get("episode_length", 0) for r in self.results]
        total_rewards = [r.get("total_reward", 0.0) for r in self.results]
        
        return {
            "num_episodes": len(self.results),
            "reward_mean": np.mean(total_rewards),
            "reward_std": np.std(total_rewards),
            "episode_length_mean": np.mean(episode_lengths),
            "episode_length_std": np.std(episode_lengths),
        }

    def close(self) -> None:
        """Close the environment."""
        if self.env is not None:
            self.env.close()
            self.env = None
