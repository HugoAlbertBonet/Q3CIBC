
"""2D Grid Navigation Environment for diagnostic testing.

The agent moves on a [-1, 1]² grid towards a randomly placed goal.
Action is a single scalar in [-1, 1] mapped to angle θ = a × π.
The agent moves step_size in that direction each timestep.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DummyEnv(gym.Env):
    """
    2D Grid Navigation: Agent seeks a goal on [-1, 1]².

    State: [goal_x, goal_y, agent_x, agent_y]
    Action: Scalar a ∈ [-1, 1], mapped to angle θ = a × π.
    Dynamics: agent_pos += step_size × [cos(θ), sin(θ)], clamped to [-1, 1]².
    Reward: -distance(agent, goal).
    Termination: distance < goal_radius.
    Truncation: step >= max_steps.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, step_size=0.1, goal_radius=0.05, max_steps=200, render_mode=None):
        self.render_mode = render_mode
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.max_steps = max_steps

        # Action: single scalar in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation: [goal_x, goal_y, agent_x, agent_y]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.goal = np.zeros(2, dtype=np.float32)
        self.agent_pos = np.zeros(2, dtype=np.float32)
        self.current_step = 0

    def _get_obs(self):
        return np.concatenate([self.goal, self.agent_pos]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Random positions within [-0.9, 0.9] to avoid spawning at edges
        self.goal = self.np_random.uniform(-0.9, 0.9, size=2).astype(np.float32)
        self.agent_pos = self.np_random.uniform(-0.9, 0.9, size=2).astype(np.float32)
        # Ensure agent doesn't spawn on top of goal
        while np.linalg.norm(self.agent_pos - self.goal) < self.goal_radius * 3:
            self.agent_pos = self.np_random.uniform(-0.9, 0.9, size=2).astype(np.float32)
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        # Map action [-1, 1] -> angle [-π, π]
        angle = float(action[0]) * np.pi

        # Move agent
        dx = self.step_size * np.cos(angle)
        dy = self.step_size * np.sin(angle)
        self.agent_pos = np.clip(
            self.agent_pos + np.array([dx, dy], dtype=np.float32),
            -1.0, 1.0
        )

        # Compute reward and termination
        dist = np.linalg.norm(self.agent_pos - self.goal)
        reward = -dist
        terminated = dist < self.goal_radius
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), float(reward), terminated, truncated, {"distance": dist}

    def render(self):
        pass

    def close(self):
        pass
