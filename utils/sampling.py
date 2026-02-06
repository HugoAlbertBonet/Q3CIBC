"""Sampling utilities for generating random samples within specified bounds."""

import numpy as np
from typing import Union, List


def sample_uniform(
    k: int,
    batch_size: int,
    mins: Union[List[float], np.ndarray],
    maxs: Union[List[float], np.ndarray],
    seed: int = 42
) -> np.ndarray:
    """
    Generate k random samples uniformly distributed within specified bounds.

    Args:
        k: Number of samples to generate.
        mins: Minimum values for each dimension. Can be a list or numpy array.
        maxs: Maximum values for each dimension. Can be a list or numpy array.
        seed: Optional random seed for reproducibility.

    Returns:
        np.ndarray: Array of shape (k, n_dims) containing the random samples.

    Raises:
        ValueError: If mins and maxs have different lengths or if any min > max.

    Example:
        >>> samples = sample_uniform(k=100, mins=[0, -1, 0], maxs=[1, 1, 10])
        >>> samples.shape
        (100, 3)
    """
    mins = np.asarray(mins)
    maxs = np.asarray(maxs)

    if mins.shape != maxs.shape:
        raise ValueError(f"mins and maxs must have the same shape. Got {mins.shape} and {maxs.shape}")

    if np.any(mins > maxs):
        raise ValueError("All min values must be less than or equal to corresponding max values")

    if seed is not None:
        np.random.seed(seed)

    n_dims = len(mins)
    samples = np.random.uniform(low=mins, high=maxs, size=(batch_size, k, n_dims))

    return samples # (B, N, D)
