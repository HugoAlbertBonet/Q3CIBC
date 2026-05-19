"""Dataset classes for loading training data from various sources.

Supports frame stacking: concatenating N consecutive observations into a single
state vector to give the model temporal context.
"""

import os
import glob
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


class D4RLDataset(Dataset):
    
    def __init__(self, root: str, download: bool = True, frame_stack: int = 1):
        self.dataset = minari.load_dataset(root, download=download)
        self.frame_stack = frame_stack
        
        # Load episode data and track episode boundaries
        all_observations = []
        all_actions = []
        episode_starts = []
        
        for ep in self.dataset.iterate_episodes():
            obs = ep.observations[:-1]  # exclude terminal observation
            acts = ep.actions
            starts = np.zeros(len(obs), dtype=bool)
            starts[0] = True
            all_observations.append(obs)
            all_actions.append(acts)
            episode_starts.append(starts)
        
        self.observations = np.concatenate(all_observations)
        self.actions = np.concatenate(all_actions)
        self._episode_starts = np.concatenate(episode_starts)
        
        # Apply frame stacking
        if frame_stack > 1:
            self.observations = stack_frames(self.observations, self._episode_starts, frame_stack)
        
        self.state_shape = self.observations.shape[1]  # obs_dim * frame_stack
        self.action_shape = self.actions.shape[1]

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
