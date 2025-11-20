import torch

def soft_truncation(z, D_scores):
    D_scores = D_scores.view(-1)
    scaling_factors = D_scores / (1 - D_scores + 1e-6)
    scaling_factors = torch.clamp(scaling_factors, 0.5, 1.2)
    return z * scaling_factors.unsqueeze(1)
