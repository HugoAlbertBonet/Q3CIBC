"""TwoChoice — the simplest possible multimodality toy.

Literature-standard shape (IBC Fig. 2 / Diffusion Policy's toy example): a
single-step regression problem where the state directly SPECIFIES two valid
target actions, and the expert commits to one of them with a fresh 50/50
coin flip every sample. No physics, no trajectory, no obstacle geometry — no
"hunting for the interesting state," because every state IS the interesting
state.

State: [mode_a, mode_b] — two target headings in [-1, 1] (angle / pi),
    always separated by at least `min_separation` (circular distance) so the
    two target clusters are never visually ambiguous with each other.
Action: scalar in [-1, 1] (angle / pi), same convention as dummy_bimodal —
    reuses the same polar-plot machinery almost unchanged.
Episode: exactly one step. reset() draws a fresh (mode_a, mode_b) pair;
    step(action) scores it against whichever of the two is closer and
    terminates immediately.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def circular_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest distance between two angles in [-1, 1] (i.e. angle/pi units),
    accounting for wraparound at +-1."""
    diff = np.abs(a - b)
    return np.minimum(diff, 2.0 - diff)


class TwoChoiceEnv(gym.Env):
    """Single-step bimodal target-selection toy.

    State: [mode_a, mode_b], both in [-1, 1], circularly separated by at
        least `min_separation`.
    Action: scalar in [-1, 1].
    Reward: -circular_dist(action, closer_mode). Success if within
        `success_tolerance` of EITHER mode.
    """

    metadata = {"render_modes": [], "render_fps": 1}

    def __init__(
        self,
        min_separation: float = 0.3,
        success_tolerance: float = 0.05,
        render_mode=None,
    ):
        super().__init__()
        self.min_separation = min_separation
        self.success_tolerance = success_tolerance
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.mode_a = np.zeros(1, dtype=np.float32)
        self.mode_b = np.zeros(1, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mode_a = self.np_random.uniform(-1.0, 1.0)
        mode_b = self.np_random.uniform(-1.0, 1.0)
        while circular_dist(np.array(mode_a), np.array(mode_b)) < self.min_separation:
            mode_b = self.np_random.uniform(-1.0, 1.0)
        self.mode_a = np.float32(mode_a)
        self.mode_b = np.float32(mode_b)
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.mode_a, self.mode_b], dtype=np.float32)

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        d_a = float(circular_dist(np.array(a), np.array(self.mode_a)))
        d_b = float(circular_dist(np.array(a), np.array(self.mode_b)))
        dist = min(d_a, d_b)
        reward = -dist
        success = dist < self.success_tolerance
        terminated = True
        truncated = False
        return self._get_obs(), float(reward), terminated, truncated, {
            "success": success,
            "dist_to_mode_a": d_a,
            "dist_to_mode_b": d_b,
        }

    def render(self):
        return None

    def close(self):
        pass


gym.register(
    id="TwoChoice-v0",
    entry_point="simulations.two_choice_env:TwoChoiceEnv",
    max_episode_steps=1,
)
