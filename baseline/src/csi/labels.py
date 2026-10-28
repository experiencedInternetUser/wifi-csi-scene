"""Global label set shared by Wallhack1.8k and 3DO (project spec, section 5.3)."""
from __future__ import annotations

import numpy as np

GLOBAL_LABELS = ["empty", "walk", "walk_wave", "sit", "lie",
                 # OPERAnet: actions (transitions), not postures, plus a person at rest
                 "sit_down", "stand_up", "lie_down", "stand_from_lie", "body_rotate", "still"]
LABEL_ID = {name: i for i, name in enumerate(GLOBAL_LABELS)}

# dataset class id -> global label name
DATASET_TO_GLOBAL = {
    "wallhack": {0: "empty", 1: "walk", 2: "walk_wave"},
    "3do": {0: "empty", 1: "walk", 2: "sit", 3: "lie"},
    "operanet": {0: "empty", 1: "walk", 2: "sit_down", 3: "stand_up", 4: "lie_down", 5: "stand_from_lie",
                 6: "body_rotate", 7: "still"},
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
