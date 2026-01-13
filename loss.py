import torch 

def lossInfoNCE(control_point_values, target_value):
    scores = torch.cat([target_value, control_point_values], dim=1)
    log_probs = scores - torch.logsumexp(scores, dim=1, keepdim=True)
    loss = -log_probs[:, 0].sum()
    return loss