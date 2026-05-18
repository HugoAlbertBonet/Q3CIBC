"""Gymnasium wrapper around the vendored IBC BlockPushMultimodal env.

Sibling of `pushing_env.py` for the IBC paper's "Block Pushing — Multimodal,
States" task (Florence et al. 2021, Table 3): 2 blocks (green/red), 2 targets
(green/red), succeed when each block ends up in a distinct target.

State layout (16D, before frame-stacking) — MUST stay aligned with
`utils.datasets.PushingMultiDataset._FEATURE_KEYS`:
    [block_translation (2), block_orientation (1),
     block2_translation (2), block2_orientation (1),
     effector_translation (2), effector_target_translation (2),
     target_translation (2), target_orientation (1),
     target2_translation (2), target2_orientation (1)]

Action: 2D effector planar delta — same scale as the single-target task; we
reuse `ACTION_MIN`/`ACTION_MAX` from the vendored `block_pushing` module.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import simulations.ibc_block_pushing  # noqa: F401  — registers gin stub
from simulations.ibc_block_pushing.block_pushing import (
    ACTION_MAX,
    ACTION_MIN,
    BlockTaskVariant,
    WORKSPACE_BOUNDS,
)
from simulations.ibc_block_pushing.block_pushing_multimodal import (
    BlockPushMultimodal,
)


# Canonical, stable feature ordering — sync with PushingMultiDataset.
OBS_KEYS_AND_DIMS = (
    ("block_translation", 2),
    ("block_orientation", 1),
    ("block2_translation", 2),
    ("block2_orientation", 1),
    ("effector_translation", 2),
    ("effector_target_translation", 2),
    ("target_translation", 2),
    ("target_orientation", 1),
    ("target2_translation", 2),
    ("target2_orientation", 1),
)
OBS_DIM = sum(d for _, d in OBS_KEYS_AND_DIMS)  # 16


def flatten_pushing_multi_obs(obs: dict) -> np.ndarray:
    return np.concatenate([
        np.atleast_1d(np.asarray(obs[k], dtype=np.float32)).flatten()
        for k, _ in OBS_KEYS_AND_DIMS
    ]).astype(np.float32)


class PushingMultiEnv(gym.Env):
    """Gymnasium-style 2-block / 2-target pushing env, backed by IBC's BlockPushMultimodal.

    Reference: Florence et al. 2021 (Implicit BC), §5 / Table 3
    ("Block Pushing — Multimodal, States").
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        n_steps: int = 200,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        # IBC's `pushing_multimodal_states/mlp_ebm_langevin.gin` keeps the
        # class default (0.04) — looser than the single-target task's 0.02
        # because both blocks must satisfy the criterion simultaneously.
        goal_dist_tolerance: float = 0.04,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.render_mode = render_mode

        self._env = BlockPushMultimodal(
            task=BlockTaskVariant.PUSH,
            goal_dist_tolerance=goal_dist_tolerance,
        )
        if seed is not None:
            self._env.seed(seed)

        self.action_space = spaces.Box(
            low=ACTION_MIN.astype(np.float32),
            high=ACTION_MAX.astype(np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        ws_min = WORKSPACE_BOUNDS[0]
        ws_max = WORKSPACE_BOUNDS[1]
        lows: list[float] = []
        highs: list[float] = []
        for k, d in OBS_KEYS_AND_DIMS:
            if k in ("block_translation", "block2_translation",
                     "target_translation", "target2_translation"):
                lows.extend([-2.0] * d)
                highs.extend([2.0] * d)
            elif k in ("effector_translation", "effector_target_translation"):
                lows.extend([float(ws_min[0]), float(ws_min[1])])
                highs.extend([float(ws_max[0]), float(ws_max[1])])
            elif k.endswith("_orientation"):
                lows.extend([-np.pi] * d)
                highs.extend([np.pi] * d)
            else:
                lows.extend([-np.inf] * d)
                highs.extend([np.inf] * d)
        self.observation_space = spaces.Box(
            low=np.array(lows, dtype=np.float32),
            high=np.array(highs, dtype=np.float32),
            dtype=np.float32,
        )

        self.steps = 0
        self._last_obs_dict: Optional[dict] = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._env.seed(seed)
        obs_dict = self._env.reset()
        self.steps = 0
        self._last_obs_dict = obs_dict
        return flatten_pushing_multi_obs(obs_dict), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs_dict, reward, ibc_done, _ = self._env.step(action)
        self._last_obs_dict = obs_dict

        self.steps += 1
        terminated = bool(ibc_done) or self.steps >= self.n_steps

        b0 = np.asarray(obs_dict["block_translation"], dtype=np.float32)
        b1 = np.asarray(obs_dict["block2_translation"], dtype=np.float32)
        t0 = np.asarray(obs_dict["target_translation"], dtype=np.float32)
        t1 = np.asarray(obs_dict["target2_translation"], dtype=np.float32)

        # Best (min) block→target distance per block, mirroring IBC's
        # _get_reward "closest target" assignment.
        b0_dist = float(min(np.linalg.norm(b0 - t0), np.linalg.norm(b0 - t1)))
        b1_dist = float(min(np.linalg.norm(b1 - t0), np.linalg.norm(b1 - t1)))
        mean_dist = 0.5 * (b0_dist + b1_dist)
        success = bool(getattr(self._env, "succeeded", False))

        info = {
            "block_to_target_distance": mean_dist,
            "block0_to_closest_target_distance": b0_dist,
            "block1_to_closest_target_distance": b1_dist,
            "success": success,
            "block_translation": b0,
            "block2_translation": b1,
            "target_translation": t0,
            "target2_translation": t1,
        }
        return flatten_pushing_multi_obs(obs_dict), float(reward), terminated, False, info

    def render(self):
        return None

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


gym.register(
    id="PushingMulti-v0",
    entry_point="simulations.pushing_multi_env:PushingMultiEnv",
    max_episode_steps=200,
)


if __name__ == "__main__":
    print("Smoke-testing vendored IBC BlockPushMultimodal wrapper ...")
    env = PushingMultiEnv()
    obs, _ = env.reset(seed=0)
    print(f"  obs shape: {obs.shape}  dtype: {obs.dtype}")
    print(f"  action_space: {env.action_space}")
    total = 0.0
    info: dict = {}
    for step in range(env.n_steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    print(f"  episode ended at step {step}, success={info.get('success', False)}, "
          f"mean block-target dist={info.get('block_to_target_distance', float('nan')):.3f}")
    env.close()
