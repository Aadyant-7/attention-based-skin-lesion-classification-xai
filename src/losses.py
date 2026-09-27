"""Training-only imbalance strategies."""
import numpy as np
import torch
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
