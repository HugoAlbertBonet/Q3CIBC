import torch 

def lossInfoNCE(control_point_values, target_value):
    """
    InfoNCE loss for contrastive learning between control point values and target values.
    """
    scores = torch.cat([target_value, control_point_values], dim=1)
    log_probs = scores - torch.logsumexp(scores, dim=1, keepdim=True)
    loss = -log_probs[:, 0].sum()
    return loss

def lossMSE(control_points, expert_action):
    """
    Mean Squared Error loss between the closest control point and the expert action.
    """
    diffs = control_points - expert_action.unsqueeze(1)  # Shape: (batch_size, control_points, action_dim)
    dists = torch.norm(diffs, dim=2)  # Shape: (batch_size, control_points)
    min_dists, _ = torch.min(dists, dim=1)  # Shape: (batch_size,)
    loss = torch.mean(min_dists ** 2)
    return loss
    