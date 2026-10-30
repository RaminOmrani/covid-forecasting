"""Time-ordered splitting and leakage-safe scaling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    train: slice
    val: slice
    test: slice


def time_split(n: int, test_frac: float = 0.2, val_frac: float = 0.1) -> Split:
    """Chronological split: [train | val | test]. The validation block is
    carved from the end of the training period and is used for early stopping;
    the test block is touched only once, for the final evaluation."""
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = n - n_test - n_val
    if n_train <= 0:
        raise ValueError("series too short for the requested split")
    return Split(slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n))


class MinMax:
    """Min-max scaler fitted on the training block only."""

    def fit(self, x: np.ndarray) -> "MinMax":
        x = np.asarray(x, dtype=float)
        self.lo = x.min(axis=0)
        self.hi = x.max(axis=0)
        self.rng = np.where(self.hi - self.lo > 0, self.hi - self.lo, 1.0)
        return self

    def transform(self, x):
        return (np.asarray(x, dtype=float) - self.lo) / self.rng

    def inverse(self, z):
        return np.asarray(z, dtype=float) * self.rng + self.lo
