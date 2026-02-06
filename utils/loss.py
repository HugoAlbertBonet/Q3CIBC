import torch 

def lossInfoNCE(scores: torch.Tensor) -> torch.Tensor:
    """
    InfoNCE loss for contrastive learning between control point values and target values.
    Assumes that the first column in 'scores' corresponds to the target values.
    """
    log_probs = scores - torch.logsumexp(scores, dim=1, keepdim=True)
    loss = -log_probs[:, 0].sum()  # Sum over batch for consistent scaling
    return loss

def lossMSE(control_points: torch.Tensor, expert_action: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error loss between the closest control point and the expert action.
    """
    diffs = control_points - expert_action.unsqueeze(1)  # Shape: (batch_size, control_points, action_dim)
    dists = torch.norm(diffs, dim=2)  # Shape: (batch_size, control_points)
    min_dists, _ = torch.min(dists, dim=1)  # Shape: (batch_size,)
    loss = torch.sum(min_dists ** 2)
    return loss


def lossSeparation(control_points: torch.Tensor, epsilon: float = 1.0) -> torch.Tensor:
    """
    Separation loss to ensure control points are as uniformly distributed as possible.
    """
    B, N, D = control_points.shape
    # Compute pairwise distances and add epsilon to avoid division by zero
    diffs = control_points.unsqueeze(2) - control_points.unsqueeze(1)  # Shape: (B, N, N, D)
    dists = torch.norm(diffs, dim=3) + epsilon  # Shape: (B, N, N)
    inv_dists = 1.0 / dists  # Shape: (B, N, N)
    # Sum all inverse distances, excluding self-distances
    loss = torch.sum(inv_dists) - torch.sum(torch.diagonal(inv_dists, dim1=1, dim2=2))
    loss = loss / (N * (N - 1))  # Normalize by (batch size and) number of pairs
    return loss

    