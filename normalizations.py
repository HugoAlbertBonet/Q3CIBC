import torch


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