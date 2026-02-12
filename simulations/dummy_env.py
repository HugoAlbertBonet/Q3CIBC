
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class DummyEnv(gym.Env):
    """
    Dummy Diagnostic Environment: 2D Directional Task
    
    Objective: Agent is at origin, Goal is at fixed relative position (1, 0).
    Action: Angle in [-pi, pi].
    Reward: Cosine similarity to the goal direction (angle=0).
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        
        # Action: Continuous Angle [-pi, pi]
        self.action_space = spaces.Box(
            low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
        )
        
        # Observation: Fixed relative goal position (x, y)
        # It's fixed to [1.0, 0.0] for simplicity in diagnostics
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(2,), dtype=np.float32
        )
        
        self.state = np.array([1.0, 0.0], dtype=np.float32)
        self.max_steps = 200 # Dummy, as state doesn't change
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        
        # Action is an angle in radians
        angle = action[0]
        
        # Goal angle is 0.0 (since goal is at [1, 0])
        # Reward = (1 + cos(angle - goal_angle)) / 2  -> [0, 1]
        reward = (1.0 + np.cos(angle)) / 2.0
        
        # Done if max steps reached
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        return self.state, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass
