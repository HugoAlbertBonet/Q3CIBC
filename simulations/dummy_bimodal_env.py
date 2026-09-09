"""2D Grid Navigation with a fixed circular obstacle — bimodal variant of DummyEnv.

Same task as `dummy_env.py` (agent seeks a goal on [-1, 1]^2, action is a
heading angle) plus one addition: a FIXED circular obstacle at the origin
that the agent cannot pass through. Whenever the straight line from agent to
goal would cross the obstacle, there are two equally valid detours (clockwise
or counterclockwise) — this is the deliberate multimodality hook. State stays
[goal_x, goal_y, agent_x, agent_y] (4D, same as Dummy-v0): the obstacle is a
constant world feature, not part of the observation, so the policy must infer
"there's a wall in the middle" purely from geometry, exactly like the expert
demonstrations do.

Collision is a hard block: a step that would move the agent inside the
obstacle radius is rejected (agent stays put) rather than clipped to the
boundary, so a policy that averages the two valid headings (as MSE-BC does)
visibly stalls against the obstacle instead of best-effort sliding around it.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DummyBimodalEnv(gym.Env):
    """2D Grid Navigation with a fixed circular obstacle at the origin.

    State: [goal_x, goal_y, agent_x, agent_y]
    Action: Scalar a in [-1, 1], mapped to angle theta = a * pi.
    Dynamics: agent_pos += step_size * [cos(theta), sin(theta)], clamped to
        [-1, 1]^2; the move is rejected (agent stays put) if it would land
        inside the obstacle disk.
    Reward: -distance(agent, goal).
    Termination: distance < goal_radius.
    Truncation: step >= max_steps.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        step_size=0.1,
        goal_radius=0.05,
        max_steps=200,
        obstacle_radius=0.25,
        obstacle_center=(0.0, 0.0),
        render_mode=None,
    ):
        self.render_mode = render_mode
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.obstacle_radius = float(obstacle_radius)
        self.obstacle_center = np.array(obstacle_center, dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.goal = np.zeros(2, dtype=np.float32)
        self.agent_pos = np.zeros(2, dtype=np.float32)
        self.current_step = 0

    def _get_obs(self):
        return np.concatenate([self.goal, self.agent_pos]).astype(np.float32)

    def _clear_of_obstacle(self, pos, margin=0.0):
        return np.linalg.norm(pos - self.obstacle_center) >= self.obstacle_radius + margin

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.goal = self._sample_point_outside_obstacle()
        self.agent_pos = self._sample_point_outside_obstacle()
        while np.linalg.norm(self.agent_pos - self.goal) < self.goal_radius * 3:
            self.agent_pos = self._sample_point_outside_obstacle()

        return self._get_obs(), {}

    def _sample_point_outside_obstacle(self):
        pos = self.np_random.uniform(-0.9, 0.9, size=2).astype(np.float32)
        while not self._clear_of_obstacle(pos, margin=0.05):
            pos = self.np_random.uniform(-0.9, 0.9, size=2).astype(np.float32)
        return pos

    def step(self, action):
        self.current_step += 1

        angle = float(action[0]) * np.pi
        dx = self.step_size * np.cos(angle)
        dy = self.step_size * np.sin(angle)
        candidate = np.clip(
            self.agent_pos + np.array([dx, dy], dtype=np.float32),
            -1.0, 1.0,
        )

        blocked = not self._clear_of_obstacle(candidate)
        if not blocked:
            self.agent_pos = candidate

        dist = np.linalg.norm(self.agent_pos - self.goal)
        reward = -dist
        terminated = bool(dist < self.goal_radius)
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), float(reward), terminated, truncated, {
            "distance": dist,
            "blocked": blocked,
        }

    def render(self):
        pass

    def close(self):
        pass


gym.register(
    id="DummyBimodal-v0",
    entry_point="simulations.dummy_bimodal_env:DummyBimodalEnv",
    max_episode_steps=200,
)
