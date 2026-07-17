"""Dataset classes for loading training data from various sources.

Supports frame stacking: concatenating N consecutive observations into a single
state vector to give the model temporal context.
"""

import os
import glob
import pickle
import re
import zipfile
from typing import Optional
import numpy as np
from torch.utils.data import Dataset
import minari

try:
    # TF preallocates the whole GPU at first device touch by default, which
    # starves PyTorch. Setting allow-growth before import keeps TF on the GPU
    # for fast tf.data pipeline ops while only reserving what it actually uses
    # (typically <1 GB for TFRecord parsing).
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import tensorflow as tf
    # Force TF to CPU. We only use TF for:
    #   - TFRecord parsing in __init__ (CPU op)
    #   - tf.io.decode_image in PushingPixelsDataset.__getitem__ (CPU op)
    # If TF grabs CUDA, DataLoader workers forked after PyTorch's CUDA init
    # crash with `CUDA_ERROR_NOT_INITIALIZED` when they touch the (post-fork
    # broken) CUDA context. set_visible_devices([], "GPU") prevents that
    # without affecting PyTorch's GPU access. The try/except handles the
    # case where TF has already initialized GPUs (then this is a no-op).
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def stack_frames(observations: np.ndarray, episode_starts: np.ndarray, frame_stack: int) -> np.ndarray:
    """Stack consecutive frames into a single observation vector.
    
    For each timestep t, the stacked observation is:
        [obs[t - (frame_stack-1)], obs[t - (frame_stack-2)], ..., obs[t]]
    
    At episode boundaries, earlier frames are filled by repeating the first
    observation of the episode (zero-padding alternative would lose position info).
    
    Args:
        observations: Array of shape (N, obs_dim) with all observations.
        episode_starts: Array of shape (N,) with True at the start of each episode.
        frame_stack: Number of frames to stack.
        
    Returns:
        Stacked observations of shape (N, obs_dim * frame_stack).
    """
    if frame_stack <= 1:
        return observations
    
    n_samples, obs_dim = observations.shape
    stacked = np.zeros((n_samples, obs_dim * frame_stack), dtype=observations.dtype)
    
    for i in range(n_samples):
        frames = []
        for k in range(frame_stack - 1, -1, -1):  # oldest to newest
            idx = i - k
            # Check if we crossed an episode boundary
            if idx < 0 or np.any(episode_starts[idx + 1:i + 1]) if idx < i else False:
                # Pad with the earliest available frame in this episode
                # Find episode start
                ep_start = i
                while ep_start > 0 and not episode_starts[ep_start]:
                    ep_start -= 1
                idx = max(idx, ep_start)
            elif idx < 0:
                idx = 0
            frames.append(observations[idx])
        stacked[i] = np.concatenate(frames)
    
    return stacked


def build_chunked_actions(raw_actions: np.ndarray, episode_starts: np.ndarray, K: int) -> np.ndarray:
    """Turn per-step actions into K-step chunk targets (action chunking).

    Sample t's target becomes [a_t, ..., a_{t+K-1}] flattened to (N, K*A).
    Windows never cross an episode boundary: indices past the episode's last
    step repeat that final action (same padding policy as the LIBERO pixel
    dataset, where chunking was first validated). Call BEFORE computing action
    stats / normalization so act_min/max cover the chunked vector.
    """
    if K <= 1:
        return raw_actions
    n, a = raw_actions.shape
    episode_id = np.cumsum(episode_starts) - 1
    # For each step, the absolute index of its episode's LAST step.
    last_of_ep = np.empty(n, dtype=np.int64)
    ep_last: dict[int, int] = {}
    for i in range(n - 1, -1, -1):
        e = int(episode_id[i])
        if e not in ep_last:
            ep_last[e] = i
        last_of_ep[i] = ep_last[e]
    chunks = np.empty((n, K, a), dtype=np.float32)
    for k in range(K):
        idx = np.minimum(np.arange(n) + k, last_of_ep)
        chunks[:, k] = raw_actions[idx]
    return chunks.reshape(n, K * a)


class D4RLDataset(Dataset):
    """Minari D4RL dataset wrapper with IBC-paper-faithful normalization.

    IBC paper (Florence et al. 2021, App. B.1 / B.3) normalizes:
      - observations: per-dim zero-mean unit-variance (standardize), and
      - actions:      per-dim min-max to `action_norm_range` (default [-1, 1]).

    Stats are computed from the UNSTACKED observations / raw actions so they
    apply to one frame at a time; ObservationNormalizer repeats them
    `frame_stack` times when consuming stacked obs. `act_min`/`act_max` are
    exposed for the eval-time simulation to invert via
    `unnormalize_action()` before stepping the env (mirrors PushingDataset).
    """

    def __init__(
        self,
        root: str,
        download: bool = True,
        frame_stack: int = 1,
        normalize_actions: bool = True,
        action_norm_range: tuple[float, float] = (-1.0, 1.0),
        obs_indices: list[int] | None = None,
        action_chunk: int = 1,
    ):
        self.dataset_name = root
        self.action_chunk = max(1, int(action_chunk))
        self.dataset = self._load_dataset(root, download=download)
        self.frame_stack = frame_stack
        self.normalize_actions = normalize_actions
        self.action_norm_range = action_norm_range
        # Optional column selection on the raw observation vector, applied
        # BEFORE stats/stacking. Used to reproduce the IBC paper's kitchen
        # input: legacy d4rl kitchen obs = robot qpos(9)+obj qpos(21)+goal(30,
        # constant for -complete). The gymnasium-robotics port instead emits
        # qpos+QVEL (59-D); selecting [0:9]+[18:39] recovers the paper's
        # informative content (velocities add 29 noisy dims on 4.2k samples).
        self.obs_indices = list(obs_indices) if obs_indices is not None else None

        all_observations = []
        all_actions = []
        episode_starts = []

        for ep in self.dataset.iterate_episodes():
            # FrankaKitchen episodes carry a Dict observation
            # {observation, achieved_goal, desired_goal}; the policy trains on
            # the flat 'observation' vector (goal is fixed for -complete).
            ep_obs = ep.observations
            if isinstance(ep_obs, dict):
                ep_obs = np.asarray(ep_obs["observation"])
            if self.obs_indices is not None:
                ep_obs = ep_obs[:, self.obs_indices]
            obs = ep_obs[:-1]  # exclude terminal observation
            acts = ep.actions
            starts = np.zeros(len(obs), dtype=bool)
            starts[0] = True
            all_observations.append(obs)
            all_actions.append(acts)
            episode_starts.append(starts)

        self.observations = np.concatenate(all_observations).astype(np.float32)
        raw_actions = np.concatenate(all_actions).astype(np.float32)
        self._episode_starts = np.concatenate(episode_starts)

        # Action chunking: replace per-step targets with K-step windows BEFORE
        # stats so normalization covers the full (K*A) chunk vector.
        if self.action_chunk > 1:
            raw_actions = build_chunked_actions(
                raw_actions, self._episode_starts, self.action_chunk
            )

        # ─── Dataset statistics (paper-faithful, from raw obs/actions) ──────
        # Obs stats are computed on UNSTACKED obs — ObservationNormalizer tiles
        # them frame_stack times. Small std floor avoids div-by-zero on any
        # degenerate dim (Adroit obs dims are healthy in practice; defensive).
        self.obs_mean = self.observations.mean(axis=0).astype(np.float32)
        self.obs_std = (self.observations.std(axis=0) + 1e-6).astype(np.float32)
        self.act_min = raw_actions.min(axis=0).astype(np.float32)
        self.act_max = raw_actions.max(axis=0).astype(np.float32)

        if normalize_actions:
            lo, hi = float(action_norm_range[0]), float(action_norm_range[1])
            denom = self.act_max - self.act_min
            denom = np.where(denom == 0, np.ones_like(denom), denom)
            self.actions = (
                lo + (raw_actions - self.act_min) * (hi - lo) / denom
            ).astype(np.float32)
        else:
            self.actions = raw_actions

        if frame_stack > 1:
            self.observations = stack_frames(
                self.observations, self._episode_starts, frame_stack
            )

        self.state_shape = self.observations.shape[1]  # obs_dim * frame_stack
        self.action_shape = self.actions.shape[1]

    @staticmethod
    def _load_dataset(root: str, download: bool):
        """Load from Minari cache, downloading into it when the dataset is missing."""
        try:
            return minari.load_dataset(root, download=False)
        except Exception:
            if not download:
                raise
            print(f"Minari dataset {root!r} not found locally; downloading...")
            return minari.load_dataset(root, download=True)

    def unnormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """Inverse of the action normalization applied in __init__.

        Use this at env.step time to convert the model's output (in
        `action_norm_range`) back to the env's native action box.
        """
        if not self.normalize_actions:
            return np.asarray(normalized_action, dtype=np.float32)
        lo, hi = float(self.action_norm_range[0]), float(self.action_norm_range[1])
        scale = (self.act_max - self.act_min) / (hi - lo)
        return (
            self.act_min + (np.asarray(normalized_action, dtype=np.float32) - lo) * scale
        ).astype(np.float32)

    def __getitem__(self, index):
        return {'state': self.observations[index], 'action': self.actions[index]}

    def __len__(self):
        return len(self.observations)


class ParticleDataset(Dataset):
    """Dataset for loading particle environment demonstrations from TFRecord files.
    
    The particle environment observation consists of:
    - pos_agent (n_dim): Agent position
    - vel_agent (n_dim): Agent velocity  
    - pos_first_goal (n_dim): First goal position
    - pos_second_goal (n_dim): Second goal position
    
    Total observation dim: 4 * n_dim (before stacking)
    After stacking: 4 * n_dim * frame_stack
    Action dim: n_dim (position setpoint)
    """
    
    def __init__(self, data_dir: str, n_dim: int = 2, frame_stack: int = 1):
        """Initialize the particle dataset.
        
        Args:
            data_dir: Directory containing TFRecord files.
            n_dim: Dimensionality of the particle environment (1, 2, 3, ..., 32).
            frame_stack: Number of consecutive frames to stack into one observation.
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required to load particle TFRecord files. "
                "Install with: pip install tensorflow"
            )
        
        self.data_dir = data_dir
        self.n_dim = n_dim
        self.frame_stack = frame_stack
        self._base_obs_dim = 4 * n_dim  # single-frame observation dimension
        self.action_shape = n_dim       # position setpoint
        
        # Find all matching TFRecord files
        pattern = os.path.join(data_dir, f"{n_dim}d_oracle_particle_*.tfrecord")
        self.tfrecord_files = sorted(glob.glob(pattern))
        
        if not self.tfrecord_files:
            raise FileNotFoundError(
                f"No TFRecord files found matching pattern: {pattern}\n"
                f"Available files in {data_dir}: {os.listdir(data_dir)[:10]}..."
            )
        
        # Load all data into memory
        self.observations, self.actions, self._episode_starts = self._load_all_data()
        
        # Apply frame stacking
        if frame_stack > 1:
            self.observations = stack_frames(self.observations, self._episode_starts, frame_stack)
        
        self.state_shape = self.observations.shape[1]  # obs_dim * frame_stack
        
    def _parse_tfrecord(self, serialized_example):
        """Parse a single TFRecord example."""
        feature_description = {
            'observation/pos_agent': tf.io.FixedLenFeature([self.n_dim], tf.float32),
            'observation/vel_agent': tf.io.FixedLenFeature([self.n_dim], tf.float32),
            'observation/pos_first_goal': tf.io.FixedLenFeature([self.n_dim], tf.float32),
            'observation/pos_second_goal': tf.io.FixedLenFeature([self.n_dim], tf.float32),
            'action': tf.io.FixedLenFeature([self.n_dim], tf.float32),
        }
        
        try:
            example = tf.io.parse_single_example(serialized_example, feature_description)
            return example
        except tf.errors.InvalidArgumentError:
            return None

    @staticmethod
    def _decode_step_type(feature) -> int | None:
        """Decode TF-Agents step_type from a TF Example feature.

        In these particle TFRecords, step_type may be stored either as an
        int64_list scalar or as raw bytes (little-endian integer).
        Returns None when the value cannot be decoded.
        """
        try:
            if feature.int64_list.value:
                return int(feature.int64_list.value[0])
            if feature.bytes_list.value:
                raw = feature.bytes_list.value[0]
                # TF-Agents often serializes small scalar ints into raw bytes.
                return int.from_bytes(raw, byteorder="little", signed=False)
        except Exception:
            return None
        return None
    
    def _load_all_data(self):
        """Load all data from TFRecord files into numpy arrays.
        
        Returns:
            observations, actions, episode_starts arrays.
        """
        all_observations = []
        all_actions = []
        episode_starts = []
        
        for tfrecord_file in self.tfrecord_files:
            raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
            is_first_in_episode = True
            
            for raw_record in raw_dataset:
                try:
                    example = tf.train.Example()
                    example.ParseFromString(raw_record.numpy())
                    features = example.features.feature
                    
                    pos_agent = np.array(features['observation/pos_agent'].float_list.value, dtype=np.float32)
                    vel_agent = np.array(features['observation/vel_agent'].float_list.value, dtype=np.float32)
                    pos_first_goal = np.array(features['observation/pos_first_goal'].float_list.value, dtype=np.float32)
                    pos_second_goal = np.array(features['observation/pos_second_goal'].float_list.value, dtype=np.float32)
                    
                    observation = np.concatenate([pos_agent, vel_agent, pos_first_goal, pos_second_goal])
                    action = np.array(features['action'].float_list.value, dtype=np.float32)
                    
                    all_observations.append(observation)
                    all_actions.append(action)
                    
                    # Detect episode boundaries via step_type (0=FIRST) or file boundaries.
                    # Important: step_type is byte-encoded in these TFRecords.
                    is_start = is_first_in_episode
                    try:
                        if 'step_type' in features:
                            step_type = self._decode_step_type(features['step_type'])
                            if step_type is not None:
                                is_start = (step_type == 0)
                    except Exception:
                        pass
                    episode_starts.append(is_start)
                    is_first_in_episode = False
                    
                except Exception:
                    continue
        
        if not all_observations:
            raise ValueError(
                f"No valid records found in TFRecord files. "
                f"Files checked: {self.tfrecord_files}"
            )
        
        return (
            np.array(all_observations),
            np.array(all_actions),
            np.array(episode_starts, dtype=bool)
        )
    
    def __getitem__(self, index):
        return {
            'state': self.observations[index],
            'action': self.actions[index]
        }
    
    def __len__(self):
        return len(self.observations)


if __name__ == "__main__":
    # Test D4RL dataset
    print("Testing D4RLDataset...")
    dataset = D4RLDataset('D4RL/pen/human-v2', download=True)
    print(f"Dataset length: {len(dataset), len(dataset.observations), len(dataset.actions)}")
    sample = dataset[0]
    print(f"Sample state shape: {sample['state'].shape}, action shape: {sample['action'].shape}")

    # Test D4RL with frame stacking
    print("\nTesting D4RLDataset with frame_stack=3...")
    dataset_stacked = D4RLDataset('D4RL/pen/human-v2', download=True, frame_stack=3)
    print(f"Stacked state shape: {dataset_stacked.state_shape}")

    # Test Particle dataset
    print("\nTesting ParticleDataset...")
    particle_ds = ParticleDataset("datasets/particle", n_dim=2)
    print(f"Dataset length: {len(particle_ds)}")
    sample = particle_ds[0]
    print(f"Sample state shape: {sample['state'].shape}, action shape: {sample['action'].shape}")

    # Test Particle with frame stacking
    print("\nTesting ParticleDataset with frame_stack=3...")
    particle_stacked = ParticleDataset("datasets/particle", n_dim=2, frame_stack=3)
    print(f"Stacked state shape: {particle_stacked.state_shape}")
    sample = particle_stacked[0]
    print(f"Sample state shape: {sample['state'].shape}")
class DummyDataset(Dataset):
    """Synthetic dataset for 2D Grid Navigation task.

    Generates expert trajectories where an agent navigates towards a goal
    on a [-1, 1]² grid. The expert uses atan2 to compute the optimal angle
    towards the goal, with small Gaussian noise for diversity.

    State: [goal_x, goal_y, agent_x, agent_y] (before frame stacking)
    Action: Scalar in [-1, 1], representing angle / π.
    """

    def __init__(
        self,
        size: int = 10000,
        step_size: float = 0.1,
        goal_radius: float = 0.05,
        max_steps_per_episode: int = 200,
        expert_noise_std: float = 0.05,
        n_dim: int = 2,
        frame_stack: int = 1,
    ):
        self.frame_stack = frame_stack
        self.step_size = step_size
        self.goal_radius = goal_radius

        all_observations = []
        all_actions = []
        episode_starts = []

        total_samples = 0
        rng = np.random.default_rng(seed=42)

        while total_samples < size:
            # Random goal and start
            goal = rng.uniform(-0.9, 0.9, size=2).astype(np.float32)
            agent_pos = rng.uniform(-0.9, 0.9, size=2).astype(np.float32)
            # Avoid spawning on top of goal
            while np.linalg.norm(agent_pos - goal) < goal_radius * 3:
                agent_pos = rng.uniform(-0.9, 0.9, size=2).astype(np.float32)

            ep_obs = []
            ep_acts = []

            for step_i in range(max_steps_per_episode):
                # Current observation
                obs = np.concatenate([goal, agent_pos]).astype(np.float32)

                # Expert action: angle towards goal + noise
                diff = goal - agent_pos
                optimal_angle = np.arctan2(diff[1], diff[0])
                # Map to [-1, 1] (action space)
                optimal_action = optimal_angle / np.pi
                noise = rng.normal(0, expert_noise_std)
                action = np.clip(optimal_action + noise, -1.0, 1.0).astype(np.float32)

                ep_obs.append(obs)
                ep_acts.append(np.array([action], dtype=np.float32))

                # Move agent
                angle = action * np.pi
                dx = step_size * np.cos(angle)
                dy = step_size * np.sin(angle)
                agent_pos = np.clip(
                    agent_pos + np.array([dx, dy], dtype=np.float32),
                    -1.0, 1.0
                )

                # Check termination
                if np.linalg.norm(agent_pos - goal) < goal_radius:
                    break

            ep_starts = np.zeros(len(ep_obs), dtype=bool)
            ep_starts[0] = True

            all_observations.append(np.array(ep_obs))
            all_actions.append(np.array(ep_acts))
            episode_starts.append(ep_starts)
            total_samples += len(ep_obs)

        self.observations = np.concatenate(all_observations)[:size]
        self.actions = np.concatenate(all_actions)[:size]
        self._episode_starts = np.concatenate(episode_starts)[:size]

        # Apply frame stacking
        if frame_stack > 1:
            self.observations = stack_frames(
                self.observations, self._episode_starts, frame_stack
            )

        self.state_shape = self.observations.shape[1]
        self.action_shape = self.actions.shape[1]

    def __getitem__(self, index):
        return {'state': self.observations[index], 'action': self.actions[index]}

    def __len__(self):
        return len(self.observations)


class PushingDataset(Dataset):
    """Dataset for the IBC paper's Simulated Pushing task (single target).

    Loads the official `block_push_states_location` TFRecord oracle dataset
    published with Florence et al. 2021 (Implicit Behavioral Cloning) —
    download instructions in the IBC README:
        https://storage.googleapis.com/brain-reach-public/ibc_data/block_push_states_location.zip

    State layout (10D before frame-stacking) — MUST stay aligned with the
    canonical ordering used by `simulations.pushing_env.OBS_KEYS_AND_DIMS`:
        [block_translation (2), block_orientation (1),
         effector_translation (2), effector_target_translation (2),
         target_translation (2), target_orientation (1)]
    Action (2D): xArm planar position delta (data-driven range
        [-0.0255, -0.0209] → [0.0287, 0.0427]).
    """

    # Canonical key order. Sync with simulations.pushing_env.OBS_KEYS_AND_DIMS.
    _FEATURE_KEYS = (
        ("observation/block_translation", 2),
        ("observation/block_orientation", 1),
        ("observation/effector_translation", 2),
        ("observation/effector_target_translation", 2),
        ("observation/target_translation", 2),
        ("observation/target_orientation", 1),
    )

    # Glob pattern for the IBC TFRecord shards. Subclasses (multimodal) override.
    _TFRECORD_GLOB = "oracle_push_*.tfrecord"
    # Short human-readable name of the IBC zip to point users at when the glob
    # finds no files. Subclasses override.
    _DATASET_ZIP_NAME = "block_push_states_location.zip"

    def __init__(
        self,
        data_dir: str = "datasets/block_push/block_push_states_location",
        frame_stack: int = 1,
        max_samples: Optional[int] = None,
        normalize_actions: bool = True,
        action_norm_range: tuple[float, float] = (-1.0, 1.0),
    ):
        """Load the IBC block_push oracle dataset.

        Args:
            data_dir: Directory of `oracle_push_*.tfrecord` files.
            frame_stack: Concatenate the previous (frame_stack - 1) obs into
                the current observation. IBC paper uses 2.
            max_samples: Optional cap (default: load all 75k transitions).
            normalize_actions: When True, return actions linearly mapped to
                `action_norm_range`. This matches the IBC pipeline
                (`compute_dataset_statistics.min_max_actions=True` in
                `pushing_states/mlp_ebm_langevin.gin`) — the network operates
                in normalized action space, denormalized back to raw effector
                deltas only at env.step time. Stats persist in attributes
                `act_min` / `act_max` so callers can denormalize.
            action_norm_range: Linear target range for action normalization.
                Default `(-1, 1)` matches IBC; pass `(0, 1)` for the
                ibc_with_cps convention.
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required to load IBC block_push TFRecord files. "
                "Install with: uv add tensorflow"
            )

        self.frame_stack = frame_stack
        self.data_dir = data_dir
        self.normalize_actions = normalize_actions
        self.action_norm_range = action_norm_range
        self._base_obs_dim = sum(d for _, d in self._FEATURE_KEYS)  # 10

        pattern = os.path.join(data_dir, self._TFRECORD_GLOB)
        self.tfrecord_files = sorted(glob.glob(pattern))
        if not self.tfrecord_files:
            raise FileNotFoundError(
                f"No TFRecord files match {pattern}. Did you download "
                f"{self._DATASET_ZIP_NAME}?"
            )

        self.observations, raw_actions, self._episode_starts = self._load_all_data(
            max_samples=max_samples
        )

        # ─── Dataset statistics (paper-faithful: from raw obs/actions) ──────
        # Computed on UNSTACKED obs so they apply to one frame at a time.
        # The ObservationNormalizer will repeat them frame_stack times.
        self.obs_mean = self.observations.mean(axis=0).astype(np.float32)
        # Small floor on std avoids divide-by-zero for any degenerate dim
        # (block_orientation in particular has near-uniform coverage so std
        # is healthy; this is defensive).
        self.obs_std = (self.observations.std(axis=0) + 1e-6).astype(np.float32)
        self.act_min = raw_actions.min(axis=0).astype(np.float32)
        self.act_max = raw_actions.max(axis=0).astype(np.float32)

        # ─── Action normalization (paper-faithful) ───────────────────────────
        # Linearly map per-dim from [act_min, act_max] → action_norm_range.
        # The reverse map is `_unnormalize_action` for use at env.step time.
        if normalize_actions:
            lo, hi = float(action_norm_range[0]), float(action_norm_range[1])
            denom = (self.act_max - self.act_min)
            # Guard near-degenerate dims (shouldn't happen for pushing but
            # cheap insurance for future datasets).
            denom = np.where(denom == 0, np.ones_like(denom), denom)
            self.actions = (lo + (raw_actions - self.act_min) * (hi - lo) / denom).astype(np.float32)
        else:
            self.actions = raw_actions

        if frame_stack > 1:
            self.observations = stack_frames(
                self.observations, self._episode_starts, frame_stack
            )

        self.state_shape = self.observations.shape[1]
        self.action_shape = self.actions.shape[1]

    def unnormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """Inverse of the action normalization applied in __init__.

        Use this at env.step time to convert the model's output (in
        `action_norm_range`) back to a raw effector delta in the env's
        native action box.
        """
        if not self.normalize_actions:
            return np.asarray(normalized_action, dtype=np.float32)
        lo, hi = float(self.action_norm_range[0]), float(self.action_norm_range[1])
        scale = (self.act_max - self.act_min) / (hi - lo)
        return (self.act_min + (np.asarray(normalized_action, dtype=np.float32) - lo) * scale).astype(np.float32)

    @staticmethod
    def _decode_step_type(feature) -> Optional[int]:
        """Decode tf-agents step_type, which is stored as 1-byte raw bytes."""
        try:
            if feature.int64_list.value:
                return int(feature.int64_list.value[0])
            if feature.bytes_list.value:
                raw = feature.bytes_list.value[0]
                return int.from_bytes(raw, byteorder="little", signed=False)
        except Exception:
            return None
        return None

    def _load_all_data(self, max_samples: Optional[int] = None):
        all_obs: list[np.ndarray] = []
        all_acts: list[np.ndarray] = []
        ep_starts: list[bool] = []
        total = 0

        for tfrecord_file in self.tfrecord_files:
            raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
            is_first_in_file = True
            for raw_record in raw_dataset:
                try:
                    example = tf.train.Example()
                    example.ParseFromString(raw_record.numpy())
                    features = example.features.feature

                    chunks = []
                    for key, dim in self._FEATURE_KEYS:
                        vals = np.asarray(
                            features[key].float_list.value, dtype=np.float32
                        )
                        if vals.shape[0] != dim:
                            raise ValueError(
                                f"Feature {key} has shape {vals.shape}, expected ({dim},)"
                            )
                        chunks.append(vals)
                    obs = np.concatenate(chunks)
                    action = np.asarray(
                        features["action"].float_list.value, dtype=np.float32
                    )

                    # Episode boundary detection. tf-agents step_type:
                    # 0=FIRST, 1=MID, 2=LAST. Treat 0 as start.
                    is_start = is_first_in_file
                    st_val = None
                    if "step_type" in features:
                        st_val = self._decode_step_type(features["step_type"])
                        if st_val is not None:
                            is_start = (st_val == 0)
                    is_first_in_file = False

                    # SKIP terminal rows. tf-agents Trajectory stores a row
                    # for the LAST step where the action is a placeholder /
                    # boundary value, not what the expert actually executed
                    # from the terminal state. Training a policy on
                    # (terminal_obs → boundary_action) introduces a
                    # ~episode-count fraction of noisy supervision and
                    # corrupts the BC objective.
                    if st_val == 2:  # LAST
                        continue

                    all_obs.append(obs)
                    all_acts.append(action)
                    ep_starts.append(is_start)
                    total += 1
                    if max_samples is not None and total >= max_samples:
                        break
                except Exception:
                    continue
            if max_samples is not None and total >= max_samples:
                break

        if not all_obs:
            raise ValueError(f"No valid records found in {self.tfrecord_files}")

        return (
            np.array(all_obs, dtype=np.float32),
            np.array(all_acts, dtype=np.float32),
            np.array(ep_starts, dtype=bool),
        )

    def __getitem__(self, index):
        return {"state": self.observations[index], "action": self.actions[index]}

    def __len__(self):
        return len(self.observations)


class PushingMultiDataset(PushingDataset):
    """Dataset for the IBC paper's Simulated Pushing task (Multimodal, 2 blocks + 2 targets).

    Loads the official `block_push_multimodal_states_location` TFRecord oracle
    dataset published with Florence et al. 2021 (Implicit Behavioral Cloning):
        https://storage.googleapis.com/brain-reach-public/ibc_data/block_push_multimodal_states_location.zip

    Unzip into `datasets/block_push/block_push_multimodal_states_location/`
    (same convention as the single-target dataset).

    State layout (16D before frame-stacking) — MUST stay aligned with the
    canonical ordering in `simulations.pushing_multi_env.OBS_KEYS_AND_DIMS`:
        [block_translation (2),  block_orientation (1),
         block2_translation (2), block2_orientation (1),
         effector_translation (2), effector_target_translation (2),
         target_translation (2),  target_orientation (1),
         target2_translation (2), target2_orientation (1)]
    Action (2D): xArm planar position delta — same scale as the single-target
        oracle (the multimodal oracle uses the same control envelope).
    """

    # Canonical key order. Sync with simulations.pushing_multi_env.OBS_KEYS_AND_DIMS.
    _FEATURE_KEYS = (
        ("observation/block_translation", 2),
        ("observation/block_orientation", 1),
        ("observation/block2_translation", 2),
        ("observation/block2_orientation", 1),
        ("observation/effector_translation", 2),
        ("observation/effector_target_translation", 2),
        ("observation/target_translation", 2),
        ("observation/target_orientation", 1),
        ("observation/target2_translation", 2),
        ("observation/target2_orientation", 1),
    )

    # Permissive glob — IBC ships the multimodal shards as
    # `oracle_multimodal_push_*.tfrecord`, but matching all `oracle_*.tfrecord`
    # files keeps the loader robust to future shard renames.
    _TFRECORD_GLOB = "oracle_*.tfrecord"
    _DATASET_ZIP_NAME = "block_push_multimodal_states_location.zip"

    def __init__(
        self,
        data_dir: str = "datasets/block_push/block_push_multimodal_states_location",
        frame_stack: int = 1,
        max_samples: Optional[int] = None,
        normalize_actions: bool = True,
        action_norm_range: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__(
            data_dir=data_dir,
            frame_stack=frame_stack,
            max_samples=max_samples,
            normalize_actions=normalize_actions,
            action_norm_range=action_norm_range,
        )


class PushingPixelsDataset(Dataset):
    """Dataset for the IBC paper's Simulated Pushing task (Single target, IMAGES).

    Loads the official `block_push_visual_location` TFRecord oracle dataset
    published with Florence et al. 2021 (Implicit Behavioral Cloning):
        https://storage.googleapis.com/brain-reach-public/ibc_data/block_push_visual_location.zip

    Unzip into `datasets/block_push/block_push_visual_location/` (oracle_*.tfrecord
    files at the top level — flatten any nested folder if needed).

    Storage strategy: LAZY. We scan all TFRecords at __init__ and keep the
    JPEG-encoded `observation/rgb` bytes (~14 KB/frame) in a Python list +
    the float actions and episode-start flags in numpy arrays. JPEG decode
    happens per __getitem__ call. RAM footprint:
        ~100k frames × ~14 KB = ~1.4 GB encoded
        + ~100k × 8 bytes (action) = ~800 KB
    Decode is a few ms per call so num_workers≥4 in the DataLoader keeps the
    pipeline GPU-bound.

    __getitem__ returns:
        state:  (3*frame_stack, H, W) uint8 channel-stacked image
                H=240, W=320 native env resolution. The conv encoder
                (utils.models.ConvMaxpoolEncoder) does its own bilinear
                resize to (180, 240) internally.
        action: (2,) float32 in `action_norm_range` (default [-1, 1]).

    Action normalization mirrors PushingDataset (min-max from raw oracle
    actions). The `act_min`/`act_max` and `action_norm_range` attrs are
    exposed for the eval-time simulation to invert.
    """

    _IMAGE_KEY = "observation/rgb"
    _ACTION_KEY = "action"
    _STEP_TYPE_KEY = "step_type"
    _TFRECORD_GLOB = "oracle_*.tfrecord"
    _DATASET_ZIP_NAME = "block_push_visual_location.zip"
    _IMAGE_HEIGHT = 240
    _IMAGE_WIDTH = 320
    _IMAGE_CHANNELS = 3

    def __init__(
        self,
        data_dir: str = "datasets/block_push/block_push_visual_location",
        frame_stack: int = 1,
        max_samples: Optional[int] = None,
        normalize_actions: bool = True,
        action_norm_range: tuple[float, float] = (-1.0, 1.0),
        action_chunk: int = 1,
    ):
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required to load IBC block_push TFRecord files. "
                "Install with: uv add tensorflow"
            )

        self.frame_stack = frame_stack
        self.data_dir = data_dir
        self.normalize_actions = normalize_actions
        self.action_norm_range = action_norm_range
        # K-step action chunking (1 = off). action_shape becomes K*2.
        self.action_chunk = max(1, int(action_chunk))

        pattern = os.path.join(data_dir, self._TFRECORD_GLOB)
        self.tfrecord_files = sorted(glob.glob(pattern))
        if not self.tfrecord_files:
            raise FileNotFoundError(
                f"No TFRecord files match {pattern}. Did you download "
                f"{self._DATASET_ZIP_NAME}?"
            )

        (
            self._encoded_rgb,
            raw_actions,
            self._episode_starts,
        ) = self._scan_all(max_samples=max_samples)

        # Action chunking: replace per-step targets with K-step windows BEFORE
        # stats so normalization covers the full (K*A) chunk vector. Windows
        # never cross an episode boundary (see build_chunked_actions).
        if self.action_chunk > 1:
            raw_actions = build_chunked_actions(
                raw_actions, self._episode_starts, self.action_chunk
            )

        self.act_min = raw_actions.min(axis=0).astype(np.float32)
        self.act_max = raw_actions.max(axis=0).astype(np.float32)

        if normalize_actions:
            lo, hi = float(action_norm_range[0]), float(action_norm_range[1])
            denom = self.act_max - self.act_min
            denom = np.where(denom == 0, np.ones_like(denom), denom)
            self.actions = (
                lo + (raw_actions - self.act_min) * (hi - lo) / denom
            ).astype(np.float32)
        else:
            self.actions = raw_actions

        # Pre-compute, for each step i, the indices to read for frame-stacking.
        # At episode boundaries the earliest frames are repeated (same policy
        # as `stack_frames` for flat obs — keeps position information rather
        # than zero-padding).
        self._stack_indices = self._build_stack_index_map()

        # Per-frame uint8 image is the model-facing "state". We expose its
        # shape so the training-script reads `dataset.state_shape` the same
        # way it does for flat datasets.
        self.state_shape = (
            self._IMAGE_CHANNELS * frame_stack,
            self._IMAGE_HEIGHT,
            self._IMAGE_WIDTH,
        )
        self.action_shape = self.actions.shape[1]

    def unnormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        if not self.normalize_actions:
            return np.asarray(normalized_action, dtype=np.float32)
        lo, hi = float(self.action_norm_range[0]), float(self.action_norm_range[1])
        scale = (self.act_max - self.act_min) / (hi - lo)
        return (
            self.act_min + (np.asarray(normalized_action, dtype=np.float32) - lo) * scale
        ).astype(np.float32)

    @staticmethod
    def _decode_step_type(feature) -> Optional[int]:
        try:
            if feature.int64_list.value:
                return int(feature.int64_list.value[0])
            if feature.bytes_list.value:
                raw = feature.bytes_list.value[0]
                return int.from_bytes(raw, byteorder="little", signed=False)
        except Exception:
            return None
        return None

    def _scan_all(self, max_samples: Optional[int] = None):
        encoded_rgb: list[bytes] = []
        all_acts: list[np.ndarray] = []
        ep_starts: list[bool] = []
        total = 0

        for tfrecord_file in self.tfrecord_files:
            raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
            is_first_in_file = True
            for raw_record in raw_dataset:
                try:
                    example = tf.train.Example()
                    example.ParseFromString(raw_record.numpy())
                    features = example.features.feature

                    is_start = is_first_in_file
                    st_val = None
                    if self._STEP_TYPE_KEY in features:
                        st_val = self._decode_step_type(features[self._STEP_TYPE_KEY])
                        if st_val is not None:
                            is_start = (st_val == 0)
                    is_first_in_file = False

                    # SKIP terminal rows: same logic as PushingDataset — the
                    # last step's action is a tf-agents boundary placeholder,
                    # not the executed expert action.
                    if st_val == 2:  # LAST
                        continue

                    rgb_bytes = features[self._IMAGE_KEY].bytes_list.value[0]
                    action = np.asarray(
                        features[self._ACTION_KEY].float_list.value, dtype=np.float32
                    )

                    encoded_rgb.append(rgb_bytes)
                    all_acts.append(action)
                    ep_starts.append(is_start)
                    total += 1
                    if max_samples is not None and total >= max_samples:
                        break
                except Exception:
                    continue
            if max_samples is not None and total >= max_samples:
                break

        if not encoded_rgb:
            raise ValueError(f"No valid records found in {self.tfrecord_files}")

        return (
            encoded_rgb,  # list[bytes]
            np.array(all_acts, dtype=np.float32),
            np.array(ep_starts, dtype=bool),
        )

    def _build_stack_index_map(self) -> np.ndarray:
        """For each step i, return the list of frame indices to channel-stack.

        Mirrors the boundary-repeat behavior of utils.datasets.stack_frames:
        the earliest indices are clamped to the first frame of the episode.
        Returns shape (N, frame_stack), int64.
        """
        n = len(self._encoded_rgb)
        fs = self.frame_stack
        # Episode id per step — cumulative count of episode starts.
        episode_id = np.cumsum(self._episode_starts).astype(np.int64) - 1
        # Episode-start absolute index per step.
        starts_abs = np.where(self._episode_starts)[0]
        # For each step, the absolute index of its episode start:
        ep_start_for_step = starts_abs[episode_id]

        stack = np.empty((n, fs), dtype=np.int64)
        for k in range(fs):
            # Offset k means "k frames before current" (k=fs-1 → current frame
            # in the channel-stack order, matching stack_frames' convention
            # of [oldest, ..., newest]).
            offset = fs - 1 - k
            raw = np.arange(n) - offset
            # Clamp to the episode start of the current step.
            stack[:, k] = np.maximum(raw, ep_start_for_step)
        return stack

    def _decode_jpeg(self, idx: int) -> np.ndarray:
        """Decode one frame's bytes → (H, W, 3) uint8 ndarray."""
        img = tf.io.decode_image(self._encoded_rgb[idx], channels=3).numpy()
        return img.astype(np.uint8)

    def __getitem__(self, index):
        # Decode and channel-stack `frame_stack` frames; channels-first layout
        # so the conv encoder gets (C, H, W) per sample directly.
        idxs = self._stack_indices[index]
        frames = [self._decode_jpeg(int(i)) for i in idxs]  # each (H, W, 3)
        # Channel-wise stack: [(H, W, 3), (H, W, 3)] → (H, W, 6) → (6, H, W).
        stacked = np.concatenate(frames, axis=-1)  # (H, W, 3*fs)
        stacked = np.transpose(stacked, (2, 0, 1))  # (3*fs, H, W)
        return {"state": stacked, "action": self.actions[index]}

    def __len__(self):
        return len(self._encoded_rgb)


class PushTRealPixelsDataset(Dataset):
    """Real-robot Push-T demonstrations stored in a BridgeData-style zip.

    The 2026-03-23 collection contains one transition-aligned observation for
    each policy output plus a final observation.  Each policy output is 7-D,
    but collection used ``action_mode=2trans``: only x/y translation deltas
    are nonzero.  This loader therefore trains on the first two dimensions and
    exposes their physical range through ``act_min``/``act_max`` for real-robot
    denormalization.

    Model state is channel-stacked RGB in this order::

        [oldest/camera0, oldest/camera1, ..., newest/camera0, newest/camera1]

    The archive stays compressed/on shared storage.  Every DataLoader worker
    opens its own lazy ZipFile handle, avoiding a 10+ GiB extraction and the
    unsafe sharing of one zip handle across forked workers.
    """

    _TRAJ_RE = re.compile(
        r"^(?P<root>.*?/)?raw/traj_group0/traj(?P<index>[0-9]+)/policy_out[.]pkl$"
    )

    # Train-time appearance augmentation defaults. Ranges are anchored to the
    # measured train->deploy gap (2026-07 forensics): the deploy T rendered at
    # ~0.67x the training red level while the mat was identical, so the
    # photometric ranges must cover at least a 0.6-0.7x object-level shift.
    _AUG_DEFAULTS = {
        "zoom_range": (0.85, 1.0),        # random crop scale (also shifts view)
        "channel_gain_range": (0.7, 1.3),  # per-channel gain: white balance / hue-ish
        "brightness_delta": 0.15,          # additive, in [0, 1] units
        "contrast_range": (0.7, 1.3),
        "saturation_range": (0.6, 1.4),
        "noise_std_max": 0.02,             # per-frame gaussian sensor noise
    }

    def __init__(
        self,
        archive_path: str,
        frame_stack: int = 2,
        camera_streams: tuple[str, ...] = ("images0", "images1"),
        resize_hw: tuple[int, int] = (240, 320),
        normalize_actions: bool = True,
        action_norm_range: tuple[float, float] = (-1.0, 1.0),
        max_trajectories: Optional[int] = None,
        augment: bool = False,
        aug_params: Optional[dict] = None,
    ):
        self.archive_path = os.path.abspath(os.path.expanduser(archive_path))
        if not os.path.isfile(self.archive_path):
            raise FileNotFoundError(f"Push-T archive not found: {self.archive_path}")
        if frame_stack < 1:
            raise ValueError(f"frame_stack must be >= 1, got {frame_stack}")
        if resize_hw[0] < 1 or resize_hw[1] < 1:
            raise ValueError(f"resize_hw must be positive, got {resize_hw}")
        if not camera_streams:
            raise ValueError("camera_streams must contain at least one RGB stream")
        if any(not re.fullmatch(r"images[0-9]+", name) for name in camera_streams):
            raise ValueError(
                f"camera_streams must be RGB folders like images0/images1; got {camera_streams}"
            )

        self.frame_stack = int(frame_stack)
        self.camera_streams = tuple(camera_streams)
        self.resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
        self.normalize_actions = bool(normalize_actions)
        self.action_norm_range = (
            float(action_norm_range[0]),
            float(action_norm_range[1]),
        )
        self.action_chunk = 1
        self.action_dims = (0, 1)
        self.action_semantics = "planar end-effector delta (x, y), metres per control step"
        self.augment = bool(augment)
        self.aug_params = dict(self._AUG_DEFAULTS)
        if aug_params:
            unknown = set(aug_params) - set(self._AUG_DEFAULTS)
            if unknown:
                raise ValueError(f"Unknown aug_params keys: {sorted(unknown)}")
            self.aug_params.update(aug_params)
        self._zip: zipfile.ZipFile | None = None

        trajectory_prefixes: list[tuple[int, str]] = []
        raw_actions_by_traj: list[np.ndarray] = []
        samples: list[tuple[int, int]] = []

        with zipfile.ZipFile(self.archive_path, "r") as archive:
            member_names = archive.namelist()
            member_set = set(member_names)
            for member in member_names:
                match = self._TRAJ_RE.match(member)
                if match:
                    prefix = member[: -len("policy_out.pkl")]
                    trajectory_prefixes.append((int(match.group("index")), prefix))

            trajectory_prefixes.sort(key=lambda pair: pair[0])
            if max_trajectories is not None:
                trajectory_prefixes = trajectory_prefixes[: int(max_trajectories)]
            if not trajectory_prefixes:
                raise ValueError(
                    f"No raw/traj_group0/traj*/policy_out.pkl entries in {self.archive_path}"
                )

            for traj_slot, (_, prefix) in enumerate(trajectory_prefixes):
                # This is a locally supplied demonstration archive.  policy_out
                # contains only builtins and NumPy arrays (verified for this
                # collection); agent_data.pkl is intentionally not unpickled
                # because it contains ROS message classes.
                policy_out = pickle.loads(archive.read(prefix + "policy_out.pkl"))
                actions_7d = np.asarray(
                    [step["actions"] for step in policy_out], dtype=np.float32
                )
                if actions_7d.ndim != 2 or actions_7d.shape[1] < 2:
                    raise ValueError(
                        f"Bad actions in {prefix}policy_out.pkl: {actions_7d.shape}"
                    )
                raw_actions = actions_7d[:, :2]

                # There must be a current RGB frame for every executed action.
                # The archive also contains one final frame, which has no action
                # target and is deliberately excluded from behavioral cloning.
                for stream in self.camera_streams:
                    for step in range(len(raw_actions)):
                        image_member = f"{prefix}{stream}/im_{step}.jpg"
                        if image_member not in member_set:
                            raise ValueError(
                                f"Missing action-aligned image {image_member}"
                            )

                raw_actions_by_traj.append(raw_actions)
                samples.extend((traj_slot, step) for step in range(len(raw_actions)))

        self._trajectory_prefixes = [prefix for _, prefix in trajectory_prefixes]
        self._raw_actions_by_traj = raw_actions_by_traj
        self._samples = samples

        raw_actions_all = np.concatenate(raw_actions_by_traj, axis=0)
        self.act_min = raw_actions_all.min(axis=0).astype(np.float32)
        self.act_max = raw_actions_all.max(axis=0).astype(np.float32)
        if self.normalize_actions:
            lo, hi = self.action_norm_range
            denom = np.where(
                self.act_max == self.act_min,
                np.ones_like(self.act_max),
                self.act_max - self.act_min,
            )
            self._actions_by_traj = [
                (lo + (actions - self.act_min) * (hi - lo) / denom).astype(np.float32)
                for actions in raw_actions_by_traj
            ]
        else:
            self._actions_by_traj = raw_actions_by_traj

        self._H, self._W = self.resize_hw
        self.in_channels = 3 * len(self.camera_streams) * self.frame_stack
        self.state_shape = (self.in_channels, self._H, self._W)
        self.action_shape = 2

        print(
            "PushTRealPixelsDataset: "
            f"{len(self._trajectory_prefixes)} trajectories, {len(self._samples)} transitions, "
            f"cameras={self.camera_streams}, frame_stack={self.frame_stack}, "
            f"state_shape={self.state_shape}, raw action range={self.act_min} -> {self.act_max}"
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # ZipFile objects cannot be pickled and must not be shared between
        # DataLoader workers.
        state["_zip"] = None
        return state

    def _archive(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.archive_path, "r")
        return self._zip

    def __del__(self):
        archive = getattr(self, "_zip", None)
        if archive is not None:
            archive.close()

    def _decode_rgb(self, member: str) -> np.ndarray:
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required to decode Push-T JPEG frames. "
                "Build the project environment with `uv sync`."
            )
        encoded = self._archive().read(member)
        image = tf.io.decode_jpeg(encoded, channels=3)
        if tuple(image.shape[:2]) != self.resize_hw:
            image = tf.image.resize(
                image,
                self.resize_hw,
                method=tf.image.ResizeMethod.AREA,
                antialias=True,
            )
            image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
        return image.numpy()

    def unnormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        if not self.normalize_actions:
            return np.asarray(normalized_action, dtype=np.float32)
        lo, hi = self.action_norm_range
        scale = (self.act_max - self.act_min) / (hi - lo)
        return (
            self.act_min
            + (np.asarray(normalized_action, dtype=np.float32) - lo) * scale
        ).astype(np.float32)

    def _augment_stack(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Apply one random appearance transform to every frame in the stack.

        All parameters are drawn ONCE per sample and shared across the stacked
        frames: the policy reads inter-frame differences as motion, so a
        per-frame photometric or geometric jitter would inject fake motion.
        Only the gaussian sensor noise is drawn per frame — real camera noise
        is temporally independent. Actions are EEF deltas in the robot frame,
        so the small view crop/zoom (camera-pose jitter) leaves targets valid.
        """
        p = self.aug_params
        rng = np.random
        H, W = self._H, self._W

        zoom = rng.uniform(*p["zoom_range"])
        ch, cw = max(1, round(H * zoom)), max(1, round(W * zoom))
        y0 = rng.randint(0, H - ch + 1)
        x0 = rng.randint(0, W - cw + 1)
        gains = rng.uniform(*p["channel_gain_range"], size=3).astype(np.float32)
        bright = rng.uniform(-p["brightness_delta"], p["brightness_delta"])
        contrast = rng.uniform(*p["contrast_range"])
        sat = rng.uniform(*p["saturation_range"])
        noise_std = rng.uniform(0.0, p["noise_std_max"])

        out: list[np.ndarray] = []
        for frame in frames:
            x = frame[y0:y0 + ch, x0:x0 + cw]
            if (ch, cw) != (H, W):
                x = tf.image.resize(
                    x, (H, W), method=tf.image.ResizeMethod.AREA, antialias=True
                ).numpy()
            x = x.astype(np.float32) / 255.0
            x = x * gains
            luma = x.mean(axis=-1, keepdims=True)
            x = luma + (x - luma) * sat
            x = (x - x.mean()) * contrast + x.mean()
            x = x + bright
            if noise_std > 0:
                x = x + rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
            out.append(
                np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
            )
        return out

    def __getitem__(self, index):
        traj_slot, step = self._samples[index]
        prefix = self._trajectory_prefixes[traj_slot]
        frames: list[np.ndarray] = []
        for offset in range(self.frame_stack - 1, -1, -1):
            frame_step = max(0, step - offset)
            for stream in self.camera_streams:
                frames.append(
                    self._decode_rgb(f"{prefix}{stream}/im_{frame_step}.jpg")
                )
        if self.augment:
            frames = self._augment_stack(frames)
        stacked = np.concatenate(frames, axis=-1)
        stacked = np.transpose(stacked, (2, 0, 1))
        return {
            "state": stacked,
            "action": self._actions_by_traj[traj_slot][step],
        }

    def __len__(self):
        return len(self._samples)


class LiberoGoalDataset(Dataset):
    """LIBERO-Goal multi-task dataset (state-based, language-goal-conditioned).

    Loads the 10 `libero_goal` tasks' human-teleop demos (50 each) from their
    LIBERO HDF5 files, in the benchmark's canonical task order. Every task
    shares the same scene/objects — only the language GOAL differs — which makes
    a single, unconditioned policy degenerate (same start state → 10 different
    correct actions). We therefore CONCATENATE a per-task language embedding to
    each low-dim observation, turning the problem into goal-conditioned BC.

    The model input ("state") is:
        [ frame_stacked low-dim obs | goal_embedding(task) ]

    Low-dim obs keys are DISCOVERED from the first demo file (image keys are
    dropped) and recorded in `self.libero_obs_keys` / `self.libero_obs_dims`.
    These, plus the goal-embedding matrix, are persisted by the training script
    into `norm_stats.pt` so the eval simulation rebuilds an identical vector.

    Goal embeddings come from a precomputed cache (see
    `scripts/precompute_libero_goal_embs.py`) keyed by task name.

    Action (7D): OSC delta (6) + gripper (1), natively in [-1, 1]; min-max
    re-normalized for a uniform eval-time denorm path (near-identity here).
    """

    def __init__(
        self,
        goal_embeddings_path: str,
        frame_stack: int = 1,
        max_demos_per_task: Optional[int] = None,
        max_samples: Optional[int] = None,
        normalize_actions: bool = True,
        action_norm_range: tuple[float, float] = (-1.0, 1.0),
    ):
        try:
            import h5py  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "h5py is required to load LIBERO HDF5 demos. Install the "
                "`libero` extra (uv sync --extra libero)."
            ) from e

        from utils.libero import (
            get_task_infos,
            select_lowdim_obs_keys,
            build_lowdim_vector,
            load_goal_embeddings,
            ACTION_DIM,
        )

        self.frame_stack = frame_stack
        self.normalize_actions = normalize_actions
        self.action_norm_range = action_norm_range

        # ── Goal embeddings (per task, keyed by name) ──────────────────────
        emb_names, emb_matrix, emb_instructions = load_goal_embeddings(
            goal_embeddings_path
        )
        name_to_emb = {n: emb_matrix[i] for i, n in enumerate(emb_names)}
        self.goal_emb_dim = int(emb_matrix.shape[1])

        # ── Per-task demo loading (canonical benchmark order) ──────────────
        task_infos = get_task_infos()
        self.goal_task_names = [t["name"] for t in task_infos]
        self.goal_instructions = [t["language"] for t in task_infos]
        # Goal-embedding matrix aligned to task order (row i = task i).
        self.goal_embeddings = np.stack(
            [name_to_emb[t["name"]] for t in task_infos]
        ).astype(np.float32)

        all_obs: list[np.ndarray] = []
        all_acts: list[np.ndarray] = []
        ep_starts: list[bool] = []
        task_ids: list[int] = []
        self.libero_obs_keys: list[str] | None = None
        total = 0

        import h5py

        for t in task_infos:
            task_idx = t["index"]
            demo_file = t["demo_file"]
            if not os.path.exists(demo_file):
                raise FileNotFoundError(
                    f"LIBERO demo file missing for task {t['name']!r}: {demo_file}. "
                    f"Download the libero_goal dataset (see batch README)."
                )
            with h5py.File(demo_file, "r") as f:
                data = f["data"]
                # Demos are named demo_0, demo_1, ...; sort numerically.
                demo_keys = sorted(
                    data.keys(), key=lambda k: int(k.split("_")[-1])
                )
                if max_demos_per_task is not None:
                    demo_keys = demo_keys[:max_demos_per_task]

                if self.libero_obs_keys is None:
                    obs_grp = data[demo_keys[0]]["obs"]
                    self.libero_obs_keys = select_lowdim_obs_keys(list(obs_grp.keys()))
                    if not self.libero_obs_keys:
                        raise RuntimeError(
                            f"No low-dim obs keys found in {demo_file}; available: "
                            f"{list(obs_grp.keys())}"
                        )

                for dk in demo_keys:
                    grp = data[dk]
                    obs_grp = grp["obs"]
                    # Per-step concat of the selected low-dim keys.
                    arrays = [
                        np.asarray(obs_grp[key], dtype=np.float32)
                        for key in self.libero_obs_keys
                    ]
                    ep_obs = np.concatenate(
                        [a.reshape(a.shape[0], -1) for a in arrays], axis=1
                    )
                    ep_acts = np.asarray(grp["actions"], dtype=np.float32)
                    n = min(len(ep_obs), len(ep_acts))
                    ep_obs, ep_acts = ep_obs[:n], ep_acts[:n]

                    starts = np.zeros(n, dtype=bool)
                    starts[0] = True
                    all_obs.append(ep_obs)
                    all_acts.append(ep_acts)
                    ep_starts.append(starts)
                    task_ids.extend([task_idx] * n)
                    total += n
                    if max_samples is not None and total >= max_samples:
                        break
            if max_samples is not None and total >= max_samples:
                break

        observations = np.concatenate(all_obs).astype(np.float32)
        raw_actions = np.concatenate(all_acts).astype(np.float32)
        episode_starts = np.concatenate(ep_starts)
        task_ids_arr = np.asarray(task_ids, dtype=np.int64)
        if max_samples is not None:
            observations = observations[:max_samples]
            raw_actions = raw_actions[:max_samples]
            episode_starts = episode_starts[:max_samples]
            task_ids_arr = task_ids_arr[:max_samples]

        # Per-key dims (for the eval sim to reconstruct the same layout).
        self.libero_obs_dims = [
            int(np.asarray(arr).reshape(arr.shape[0], -1).shape[1])
            for arr in arrays
        ]

        if raw_actions.shape[1] != ACTION_DIM:
            print(
                f"[LiberoGoalDataset] WARNING: action_dim={raw_actions.shape[1]} "
                f"!= expected {ACTION_DIM}."
            )

        # ── Frame-stack the low-dim obs only, then append the goal embedding ─
        if frame_stack > 1:
            observations = stack_frames(observations, episode_starts, frame_stack)
        goal_vecs = self.goal_embeddings[task_ids_arr]  # (N, goal_emb_dim)
        states = np.concatenate([observations, goal_vecs], axis=1).astype(np.float32)

        # ── Action normalization (min-max → action_norm_range) ─────────────
        self.act_min = raw_actions.min(axis=0).astype(np.float32)
        self.act_max = raw_actions.max(axis=0).astype(np.float32)
        if normalize_actions:
            lo, hi = float(action_norm_range[0]), float(action_norm_range[1])
            denom = self.act_max - self.act_min
            denom = np.where(denom == 0, np.ones_like(denom), denom)
            self.actions = (
                lo + (raw_actions - self.act_min) * (hi - lo) / denom
            ).astype(np.float32)
        else:
            self.actions = raw_actions

        # ── Standardize stats over the FULL state vector (obs + goal) ──────
        # The training script builds an ObservationNormalizer in standardize
        # mode with frame_stack=1 (no tiling) from these full-length stats.
        self.observations = states
        self.obs_mean = states.mean(axis=0).astype(np.float32)
        self.obs_std = (states.std(axis=0) + 1e-6).astype(np.float32)
        self._task_ids = task_ids_arr
        self._episode_starts = episode_starts

        self.state_shape = states.shape[1]
        self.action_shape = self.actions.shape[1]

    def unnormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        if not self.normalize_actions:
            return np.asarray(normalized_action, dtype=np.float32)
        lo, hi = float(self.action_norm_range[0]), float(self.action_norm_range[1])
        scale = (self.act_max - self.act_min) / (hi - lo)
        return (
            self.act_min + (np.asarray(normalized_action, dtype=np.float32) - lo) * scale
        ).astype(np.float32)

    def __getitem__(self, index):
        return {"state": self.observations[index], "action": self.actions[index]}

    def __len__(self):
        return len(self.observations)


class LiberoGoalPixelsDataset(Dataset):
    """LIBERO-Goal multi-task PIXEL dataset (standard protocol).

    Standard LIBERO obs: per-camera RGB (agentview + eye-in-hand, 128x128x3)
    channel-stacked, PLUS low-dim proprio (ee_pos + gripper + joint = 12), and a
    per-task language (goal) embedding. No object-state (privileged) — the policy
    infers objects from pixels.

    __getitem__ returns:
        state:  (3*2*frame_stack, H, W) uint8  — [agentview, wrist] channel-stack
        cond:   (proprio_dim*frame_stack + goal_emb_dim,) float32  — proprio | goal
        action: (7,) float32 in [-1, 1]

    The conv encoder (utils.models.ConvMaxpoolEncoder) does its own /255 + resize.
    Actions are min-max normalized to [-1, 1]; act_min/max exposed for eval denorm.

    Images held in RAM as uint8 (≈7 GB for the full suite) — needs a 32 GB node;
    keep DataLoader num_workers=0.
    """

    _IMAGE_KEYS = ("agentview_rgb", "eye_in_hand_rgb")
    # Proprio keys that have an exact live-env match (see utils.libero); NO
    # ee_ori/ee_states (euler, no live key) and NO object-state (privileged).
    _PROPRIO_KEYS = ("ee_pos", "gripper_states", "joint_states")
    _H = 128
    _W = 128

    def __init__(
        self,
        goal_embeddings_path: str,
        frame_stack: int = 1,
        max_demos_per_task: Optional[int] = None,
        max_samples: Optional[int] = None,
        normalize_actions: bool = True,
        action_norm_range: tuple[float, float] = (-1.0, 1.0),
        crop_size: int = 0,
        action_chunk: int = 1,
    ):
        try:
            import h5py  # noqa: F401
        except ImportError as e:
            raise ImportError("h5py required for LIBERO demos (uv sync --extra libero).") from e
        import h5py
        from utils.libero import get_task_infos, load_goal_embeddings

        # Random-crop augmentation (train-time only; eval center-crops to the
        # same size — see LiberoGoalPixelsSimulation). 0 = off. Standard pixel-BC
        # trick (robomimic / Diffusion Policy use ~90% crops; 116 of 128 here).
        # Note: both cameras + all stacked frames share ONE crop offset per
        # sample (they're channel-stacked); per-camera independent crops would
        # be marginally stronger aug but need a layout change.
        self.crop_size = int(crop_size)
        if self.crop_size and not (0 < self.crop_size <= self._H):
            raise ValueError(f"crop_size must be in (0, {self._H}]; got {crop_size}")
        self._rng = np.random.default_rng(0)
        # Action chunking (DP-style): each sample's target is the next K
        # actions concatenated (K*A vector); episode tails pad by repeating the
        # last action. Models treat the chunk as one big action; eval executes
        # it open-loop. K=1 keeps legacy single-step behavior.
        self.action_chunk = max(1, int(action_chunk))
        self.frame_stack = frame_stack
        self.normalize_actions = normalize_actions
        self.action_norm_range = action_norm_range

        emb_names, emb_matrix, _ = load_goal_embeddings(goal_embeddings_path)
        name_to_emb = {n: emb_matrix[i] for i, n in enumerate(emb_names)}
        self.goal_emb_dim = int(emb_matrix.shape[1])

        task_infos = get_task_infos()
        self.goal_task_names = [t["name"] for t in task_infos]
        self.goal_embeddings = np.stack(
            [name_to_emb[t["name"]] for t in task_infos]
        ).astype(np.float32)

        agv: list[np.ndarray] = []   # per-frame (H,W,3) uint8
        wrist: list[np.ndarray] = []
        proprio: list[np.ndarray] = []
        acts: list[np.ndarray] = []
        starts: list[bool] = []
        task_ids: list[int] = []
        self.libero_obs_keys = list(self._PROPRIO_KEYS)
        total = 0

        for t in task_infos:
            demo_file = t["demo_file"]
            if not os.path.exists(demo_file):
                raise FileNotFoundError(f"Missing LIBERO demo: {demo_file}")
            with h5py.File(demo_file, "r") as f:
                data = f["data"]
                demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[-1]))
                if max_demos_per_task is not None:
                    demo_keys = demo_keys[:max_demos_per_task]
                for dk in demo_keys:
                    obsg = data[dk]["obs"]
                    a = np.asarray(obsg["agentview_rgb"], dtype=np.uint8)        # (T,H,W,3)
                    w = np.asarray(obsg["eye_in_hand_rgb"], dtype=np.uint8)
                    pr = np.concatenate(
                        [np.asarray(obsg[k], dtype=np.float32).reshape(a.shape[0], -1)
                         for k in self._PROPRIO_KEYS], axis=1)               # (T,12)
                    ac = np.asarray(data[dk]["actions"], dtype=np.float32)
                    n = min(len(a), len(w), len(pr), len(ac))
                    for i in range(n):
                        agv.append(a[i]); wrist.append(w[i]); proprio.append(pr[i]); acts.append(ac[i])
                        starts.append(i == 0); task_ids.append(t["index"])
                    total += n
                    if max_samples is not None and total >= max_samples:
                        break
            if max_samples is not None and total >= max_samples:
                break

        self._agv = np.stack(agv)        # (N,H,W,3) uint8
        self._wrist = np.stack(wrist)
        self._proprio = np.stack(proprio).astype(np.float32)   # (N,12)
        raw_actions = np.stack(acts).astype(np.float32)
        self._episode_starts = np.asarray(starts, dtype=bool)
        self._task_ids = np.asarray(task_ids, dtype=np.int64)
        if max_samples is not None:
            self._agv = self._agv[:max_samples]; self._wrist = self._wrist[:max_samples]
            self._proprio = self._proprio[:max_samples]; raw_actions = raw_actions[:max_samples]
            self._episode_starts = self._episode_starts[:max_samples]
            self._task_ids = self._task_ids[:max_samples]

        self.proprio_dim = int(self._proprio.shape[1])

        if self.action_chunk > 1:
            K = self.action_chunk
            n = len(raw_actions)
            episode_id = np.cumsum(self._episode_starts) - 1
            chunks = np.empty((n, K, raw_actions.shape[1]), dtype=np.float32)
            for k in range(K):
                idx = np.minimum(np.arange(n) + k, n - 1)
                # Don't cross episode boundaries: clamp to the last step of the
                # current episode (repeat-last-action padding).
                same_ep = episode_id[idx] == episode_id
                idx = np.where(same_ep, idx, -1)
                # For crossed indices walk back to this episode's final step.
                if (idx < 0).any():
                    last_of_ep = np.zeros(n, dtype=np.int64)
                    ep_last = {}
                    for i in range(n - 1, -1, -1):
                        e = episode_id[i]
                        if e not in ep_last:
                            ep_last[e] = i
                        last_of_ep[i] = ep_last[e]
                    idx = np.where(idx < 0, last_of_ep, idx)
                chunks[:, k] = raw_actions[idx]
            raw_actions = chunks.reshape(n, K * raw_actions.shape[1])

        self.act_min = raw_actions.min(axis=0).astype(np.float32)
        self.act_max = raw_actions.max(axis=0).astype(np.float32)
        if normalize_actions:
            lo, hi = float(action_norm_range[0]), float(action_norm_range[1])
            denom = np.where((self.act_max - self.act_min) == 0, 1.0, self.act_max - self.act_min)
            self.actions = (lo + (raw_actions - self.act_min) * (hi - lo) / denom).astype(np.float32)
        else:
            self.actions = raw_actions

        self._stack_idx = self._build_stack_index_map()
        self.in_channels = 3 * len(self._IMAGE_KEYS) * frame_stack
        self.cond_dim = self.proprio_dim * frame_stack + self.goal_emb_dim
        out_hw = self.crop_size if self.crop_size else self._H
        self.state_shape = (self.in_channels, out_hw, out_hw)
        self.action_shape = self.actions.shape[1]

    def unnormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        if not self.normalize_actions:
            return np.asarray(normalized_action, dtype=np.float32)
        lo, hi = float(self.action_norm_range[0]), float(self.action_norm_range[1])
        scale = (self.act_max - self.act_min) / (hi - lo)
        return (self.act_min + (np.asarray(normalized_action, dtype=np.float32) - lo) * scale).astype(np.float32)

    def _build_stack_index_map(self) -> np.ndarray:
        n = len(self._agv)
        fs = self.frame_stack
        episode_id = np.cumsum(self._episode_starts).astype(np.int64) - 1
        starts_abs = np.where(self._episode_starts)[0]
        ep_start_for_step = starts_abs[episode_id]
        stack = np.empty((n, fs), dtype=np.int64)
        for k in range(fs):
            offset = fs - 1 - k
            raw = np.arange(n) - offset
            stack[:, k] = np.maximum(raw, ep_start_for_step)
        return stack

    def __getitem__(self, index):
        idxs = self._stack_idx[index]
        frames = []
        for i in idxs:                       # oldest -> newest
            frames.append(self._agv[int(i)])
            frames.append(self._wrist[int(i)])
        stacked = np.concatenate(frames, axis=-1)        # (H,W,3*2*fs)
        if self.crop_size:
            s = self.crop_size
            oy = int(self._rng.integers(0, stacked.shape[0] - s + 1))
            ox = int(self._rng.integers(0, stacked.shape[1] - s + 1))
            stacked = stacked[oy:oy + s, ox:ox + s]
        stacked = np.transpose(stacked, (2, 0, 1)).copy()  # (C,S,S) uint8
        proprio_stack = np.concatenate([self._proprio[int(i)] for i in idxs]).astype(np.float32)
        goal = self.goal_embeddings[self._task_ids[index]]
        cond = np.concatenate([proprio_stack, goal]).astype(np.float32)
        return {"state": stacked, "cond": cond, "action": self.actions[index]}

    def __len__(self):
        return len(self._agv)
