"""Training-only imbalance strategies."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler


def class_weights(counts):
    """Sqrt inverse frequency: sqrt(N/(K*n_c)), normalized to mean 1."""
    counts = np.asarray(counts, dtype=np.float64)
    if np.any(counts <= 0):
        raise ValueError("Every class must occur in training")
    weights = np.sqrt(counts.sum() / (len(counts) * counts))
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def balanced_sampler(labels, seed=42):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels)
    weights = 1.0 / counts[labels]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(labels), replacement=True, generator=generator)


class MulticlassFocalLoss(nn.Module):
    """Mean[-alpha_y * (1 - p_y)^gamma * log(p_y)] over the batch."""

    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be nonnegative")
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32).clone())
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        log_probabilities = F.log_softmax(logits.float(), dim=1)
        log_pt = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (-torch.expm1(log_pt)).pow(self.gamma)
        return (-self.alpha[targets] * focal_factor * log_pt).mean()
