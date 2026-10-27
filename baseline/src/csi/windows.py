"""Windowing and group-aware splits (project spec, section 5.4)."""
from __future__ import annotations

import numpy as np

WINDOW = 400
STEP_TRAIN = 100
STEP_TEST = 400
MIN_PURITY = 0.8


def window_starts(n: int, window: int = WINDOW, step: int = STEP_TEST, lo: int = 0, hi: int | None = None):
    hi = n if hi is None else hi
    return np.arange(lo, hi - window + 1, step)


def window_labels(y: np.ndarray, starts: np.ndarray, window: int = WINDOW, min_purity: float = MIN_PURITY):
    """Majority label per window and a keep-mask for windows with purity >= min_purity."""
    labels = np.empty(len(starts), dtype=np.int16)
    keep = np.zeros(len(starts), dtype=bool)
    for i, s in enumerate(starts):
        vals, counts = np.unique(y[s:s + window], return_counts=True)
        j = counts.argmax()
        labels[i] = vals[j]
        keep[i] = counts[j] / window >= min_purity
    return labels, keep


def time_block_split(n: int, train_frac: float = 0.7, gap: int = WINDOW):
    """Split one continuous recording (Wallhack b1) into [0, a) train and [a + gap, n) test."""
    a = int(n * train_frac)
    return (0, a), (min(a + gap, n), n)
