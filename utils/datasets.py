"""Dataset classes for loading training data from various sources.

Supports frame stacking: concatenating N consecutive observations into a single
state vector to give the model temporal context.
"""

import os
import glob
import numpy as np
from torch.utils.data import Dataset
import minari

try:
    import tensorflow as tf
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
                    
                    # Detect episode boundaries via step_type (0=FIRST) or file boundaries
                    is_start = is_first_in_episode
                    try:
                        if 'step_type' in features and features['step_type'].int64_list.value:
                            is_start = (features['step_type'].int64_list.value[0] == 0)
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
    """Synthetic dataset for Dummy 2D verification task.
    
    Goal: Action should point towards 0 degrees (relative to objective).
    State: Fixed relative goal position [1, 0].
    Action: Angle in [-pi, pi].
    """
    
    def __init__(self, size: int = 10000, n_dim: int = 2, frame_stack: int = 1):
        self.size = size
        self.frame_stack = frame_stack
        
        # Fixed state: Relative position of goal is always (1, 0)
        # This implies angle to goal is 0.
        self.state = np.array([1.0, 0.0], dtype=np.float32)
        
        # Generate Expert Actions: Gaussian around 0 radians
        self.mu = 0.0
        self.sigma = 0.2 # Expert is confident
        
        raw_actions = np.random.normal(self.mu, self.sigma, (size, 1)).astype(np.float32)
        self.actions = np.clip(raw_actions, -np.pi, np.pi)
        
        # Replicate fixed state
        self.observations = np.tile(self.state, (size, 1))
        
        # Apply frame stacking (trivial here)
        if frame_stack > 1:
            self.observations = np.tile(self.observations, (1, frame_stack))
            
        self.state_shape = self.observations.shape[1]
        self.action_shape = self.actions.shape[1]
        
    def __getitem__(self, index):
        return {'state': self.observations[index], 'action': self.actions[index]}

    def __len__(self):
        return self.size
