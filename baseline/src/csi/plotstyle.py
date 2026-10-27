"""Shared matplotlib style: fixed colors per entity so every figure reads the same way."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Categorical slots in fixed order (validated reference palette, light mode).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Color follows the entity: one color per global label in every figure of both datasets.
LABEL_COLORS = {"empty": SLOTS[0], "walk": SLOTS[1], "walk_wave": SLOTS[2], "sit": SLOTS[3], "lie": SLOTS[4]}
DOMAIN_COLORS = {"LOS_BQ": SLOTS[0], "LOS_PIFA": SLOTS[1], "NLOS_BQ": SLOTS[2], "NLOS_PIFA": SLOTS[3],
                 "d1": SLOTS[0], "d2": SLOTS[1], "d3": SLOTS[2]}
DATASET_COLORS = {"wallhack": SLOTS[0], "3do": SLOTS[1]}

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

# Sequential single-hue ramp (blue 100 -> 700) for heatmaps.
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "figure.titlesize": 12,
    "lines.linewidth": 1.2, "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 140,
    "savefig.bbox": "tight",
})
