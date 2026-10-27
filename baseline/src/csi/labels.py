"""Global label set shared by Wallhack1.8k and 3DO (project spec, section 5.3)."""
from __future__ import annotations

import numpy as np

GLOBAL_LABELS = ["empty", "walk", "walk_wave", "sit", "lie"]
LABEL_ID = {name: i for i, name in enumerate(GLOBAL_LABELS)}

# dataset class id -> global label name
DATASET_TO_GLOBAL = {
    "wallhack": {0: "empty", 1: "walk", 2: "walk_wave"},
    "3do": {0: "empty", 1: "walk", 2: "sit", 3: "lie"},
}


def to_global(dataset: str, y: np.ndarray) -> np.ndarray:
    """Map dataset class ids to global label ids (int16); unknown ids raise."""
    mapping = DATASET_TO_GLOBAL[dataset]
    lut = np.full(max(mapping) + 1, -1, dtype=np.int16)
    for k, name in mapping.items():
        lut[k] = LABEL_ID[name]
    y = np.asarray(y)
    if y.min() < 0 or y.max() >= len(lut) or (lut[y] < 0).any():
        raise ValueError(f"unexpected class ids for {dataset}: {np.unique(y)}")
    return lut[y]
