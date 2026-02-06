import torch
import json
from pathlib import Path
from typing import Optional


def wireFittingNorm(
        control_points: torch.Tensor,           # (B, N, D)
        expert_action: torch.Tensor,            # (B, D)
        control_point_values: torch.Tensor,     # (B, N)
        expert_action_value: torch.Tensor,      # (B,)
        c: torch.Tensor,                        # (B, N+1) learnable smoothing parameters
) -> torch.Tensor:
    """
    Fully vectorized wire-fitting normalization.
    Returns normalized Q-values for each candidate action (expert + control points).
    Shape: (B, N+1), first column corresponds to expert action.
    """
    # Concat expert action to control points: (B, N+1, D)
    all_points = torch.cat([expert_action.unsqueeze(1), control_points], dim=1)
    # Concat expert action value to control point values: (B, N+1)
    all_values = torch.cat([expert_action_value.unsqueeze(1), control_point_values], dim=1)

    # Squared distances: each query (row j) vs all keys (col i)
    # query: (B, N+1, 1, D), keys: (B, 1, N+1, D) -> sq_dists: (B, N+1, N+1)
    query = all_points.unsqueeze(2)
    keys = all_points.unsqueeze(1)
    sq_dists = ((query - keys) ** 2).sum(dim=-1)

    # Value penalty: max_Q - Q_i, shape (B, N+1)
    max_q = all_values.max(dim=1, keepdim=True).values
    value_penalty = max_q - all_values

    # Weights: for each query j, weight over keys i
    # c is (B, N+1), value_penalty is (B, N+1) -> broadcast to (B, 1, N+1)
    denom = sq_dists + c.unsqueeze(1) * value_penalty.unsqueeze(1) + 1e-8
    weights = 1.0 / denom  # (B, N+1, N+1)

    # Weighted average of values for each query
    norm_values = (weights * all_values.unsqueeze(1)).sum(dim=-1) / weights.sum(dim=-1)
    return norm_values  # (B, N+1)


def wireFittingInterpolation(
        control_points: torch.Tensor,           # (B, N, D)
        interpolated_points: torch.Tensor,      # (B, M, D)
        control_point_values: torch.Tensor,     # (B, N)
        c: torch.Tensor,                        # (B, N) learnable smoothing parameters
        k: int = 10,
) -> torch.Tensor:
    """
    Fully vectorized wire-fitting interpolation.
    Returns interpolated Q-values for each interpolated point.
    Shape: (B, M).
    """

    # Squared distances: each query (row j, point to interpolate) vs all keys (col i, control points)
    # query: (B, M, 1, D), keys: (B, 1, N, D) -> sq_dists: (B, M, N)
    query = interpolated_points.unsqueeze(2)
    keys = control_points.unsqueeze(1)
    sq_dists = ((query - keys) ** 2).sum(dim=-1)

    # Value penalty: max_Q - Q_i, shape (B, N)
    max_q = control_point_values.max(dim=1, keepdim=True).values
    value_penalty = max_q - control_point_values

    # Weights: for each query j, weight over keys i
    # c is (B, N), value_penalty is (B, N) -> broadcast to (B, 1, N)
    denom = sq_dists + c.unsqueeze(1) * value_penalty.unsqueeze(1) + 1e-8
    weights = 1.0 / denom  # (B, M, N)

    # Weighted average of values for each query, choose top k control points for each interpolated point based on weights
    topk_result = torch.topk(weights, k=k, dim=-1)
    top_k_weights = topk_result.values  # (B, M, k)
    top_k_indices = topk_result.indices  # (B, M, k)
    
    # Gather the control point values corresponding to top-k indices
    # control_point_values: (B, N) -> expand to (B, M, N) then gather along dim=-1
    control_point_values_expanded = control_point_values.unsqueeze(1).expand(-1, interpolated_points.shape[1], -1)  # (B, M, N)
    top_k_values = torch.gather(control_point_values_expanded, dim=-1, index=top_k_indices)  # (B, M, k)
    
    norm_values = (top_k_weights * top_k_values).sum(dim=-1) / top_k_weights.sum(dim=-1)
    return norm_values  # (B, M)


class ObservationNormalizer:
    """Normalizes observations to [0, 1] range using predefined bounds.
    
    Loads bounds from a JSON file and provides methods to normalize
    and denormalize observations.
    """

    def __init__(
        self,
        env_id: str,
        bounds_file: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the normalizer with bounds for a specific environment.
        
        Args:
            env_id: The environment ID (e.g., 'AdroitHandPen-v1').
            bounds_file: Path to the JSON file containing observation bounds.
                        Defaults to 'observation_bounds.json' in the project root.
            device: Device for tensors ('cpu' or 'cuda').
        """
        if bounds_file is None:
            bounds_file = Path(__file__).parent.parent / "config_json" / "observation_bounds.json"
        
        with open(bounds_file, "r") as f:
            all_bounds = json.load(f)
        
        if env_id not in all_bounds:
            raise ValueError(
                f"Environment '{env_id}' not found in bounds file. "
                f"Available: {list(all_bounds.keys())}"
            )
        
        env_bounds = all_bounds[env_id]
        self.env_id = env_id
        self.observation_dim = env_bounds["observation_dim"]
        
        # Load flat bounds as tensors
        flat_bounds = env_bounds["flat_bounds"]
        self.obs_min = torch.tensor(flat_bounds["min"], dtype=torch.float32, device=device)
        self.obs_max = torch.tensor(flat_bounds["max"], dtype=torch.float32, device=device)
        
        # Compute range, avoiding division by zero
        self.obs_range = self.obs_max - self.obs_min
        self.obs_range = torch.where(
            self.obs_range == 0,
            torch.ones_like(self.obs_range),
            self.obs_range
        )
        
        self.device = device

    def normalize(self, observation: torch.Tensor) -> torch.Tensor:
        """Normalize observation to [0, 1] range.
        
        Args:
            observation: Raw observation tensor of shape (..., obs_dim).
            
        Returns:
            Normalized observation in [0, 1] range, clamped.
        """
        # Ensure bounds are on same device as observation
        if observation.device != self.obs_min.device:
            self.to(observation.device)
        
        normalized = (observation - self.obs_min) / self.obs_range
        return torch.clamp(normalized, 0.0, 1.0)

    def denormalize(self, normalized_observation: torch.Tensor) -> torch.Tensor:
        """Denormalize observation from [0, 1] range back to original scale.
        
        Args:
            normalized_observation: Normalized observation tensor of shape (..., obs_dim).
            
        Returns:
            Denormalized observation in original scale.
        """
        if normalized_observation.device != self.obs_min.device:
            self.to(normalized_observation.device)
        
        return normalized_observation * self.obs_range + self.obs_min

    def to(self, device: str) -> "ObservationNormalizer":
        """Move bounds tensors to specified device.
        
        Args:
            device: Target device ('cpu' or 'cuda').
            
        Returns:
            Self for chaining.
        """
        self.obs_min = self.obs_min.to(device)
        self.obs_max = self.obs_max.to(device)
        self.obs_range = self.obs_range.to(device)
        self.device = device
        return self

    def __repr__(self) -> str:
        return f"ObservationNormalizer(env_id='{self.env_id}', dim={self.observation_dim})"


