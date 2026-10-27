"""Unified on-disk format (project spec, section 5.2) and its validation.

data/processed/<dataset>/recordings.parquet  one row per continuous recording
data/processed/<dataset>/rec/<recording_id>.npz

npz keys:
  raw        int8    [T, 256]  untouched CSI bytes (L-LTF + HT-LTF); complex CSI is built
                               on load with the chosen byte pairing (docs/DECISIONS.md)
  t          float64 [T]       seconds from the first packet (32-bit wrap undone)
  y          int16   [T]       dataset class per packet
  packet_id  int64   [T]       ESP32 packet counter
  rssi       int16   [T]
  pos        float32 [T, 3]    3DO only

Usage: python -m csi.format [--dataset wallhack|3do|all]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from csi.io import threedo, wallhack
from csi.io.esp32 import LLTF_SUBCARRIERS, csi_from_raw, read_csi_csv

ROOT = Path(__file__).resolve().parents[2]
RAW = {"wallhack": ROOT / "data/raw/wallhack/wallhack1.8k", "3do": ROOT / "data/raw/3do/3DO"}
PROCESSED = ROOT / "data/processed"
LISTERS = {"wallhack": wallhack.list_recordings, "3do": threedo.list_recordings}


def timing_stats(t: np.ndarray, packet_id: np.ndarray) -> dict:
    dt = np.diff(t)
    med = float(np.median(dt))
    id_span = int(packet_id[-1] - packet_id[0] + 1)
    return {
        "fs_hz": 1.0 / med,
        "duration_s": float(t[-1]),
        "dt_median_ms": med * 1e3,
        "dt_p01_ms": float(np.percentile(dt, 1) * 1e3),
        "dt_p99_ms": float(np.percentile(dt, 99) * 1e3),
        "dt_max_ms": float(dt.max() * 1e3),
        "n_gaps_gt_3dt": int((dt > 3 * med).sum()),
        "n_nonpositive_dt": int((dt <= 0).sum()),
        # share missing according to timestamps at the median rate (main metric: the
        # receiver-side counter below does not see packets lost over the air)
        "drop_rate": 1.0 - len(t) / (t[-1] / med + 1),
        # share missing according to the ESP32 packet counter
        "drop_rate_id": 1.0 - len(t) / id_span if id_span > 0 else np.nan,
    }


def build(dataset: str) -> pd.DataFrame:
    out_dir = PROCESSED / dataset
    (out_dir / "rec").mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in tqdm(LISTERS[dataset](RAW[dataset]), desc=dataset):
        d = read_csi_csv(rec["path"])
        packet_id = d["meta"]["id"].to_numpy(np.int64)
        arrays = {
            "raw": d["raw"],
            "t": d["t"],
            "y": d["class"].astype(np.int16),
            "packet_id": packet_id,
            "rssi": d["meta"]["rssi"].to_numpy(np.int16),
        }
        if dataset == "3do":
            arrays["pos"] = np.stack([d["x"], d["y"], d["z"]], axis=1).astype(np.float32)
        np.savez(out_dir / "rec" / f"{rec['recording_id']}.npz", **arrays)
        y = arrays["y"]
        rows.append({
            **{k: v for k, v in rec.items() if k != "path"},
            "source_file": str(rec["path"].relative_to(ROOT)),
            "n_subcarriers": len(LLTF_SUBCARRIERS),
            "n_packets": len(y),
            "n_bad_rows": d["n_bad_rows"],
            "n_ts_fixed": d["n_ts_fixed"],
            "classes": ",".join(str(c) for c in np.unique(y)),
            "rssi_mean": float(arrays["rssi"].mean()),
            **timing_stats(d["t"], packet_id),
            **{f"n_class_{c}": int((y == c).sum()) for c in range(4)},
        })
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "recordings.parquet", index=False)
    return df


def load_index(dataset: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / dataset / "recordings.parquet")


def load_recording(dataset: str, recording_id: str, pairing: str = "espidf") -> dict:
    """Load one recording; `csi` is complex64 [T, 52] built with the given byte pairing."""
    with np.load(PROCESSED / dataset / "rec" / f"{recording_id}.npz") as z:
        d = {k: z[k] for k in z.files}
    d["csi"] = csi_from_raw(d["raw"], pairing)
    return d


def validate(dataset: str) -> list[str]:
    """Cheap consistency checks of the processed files; returns a list of problems."""
    problems = []
    for r in load_index(dataset).itertuples():
        d = load_recording(dataset, r.recording_id)
        T = len(d["t"])
        if d["csi"].shape != (T, 52) or len(d["y"]) != T:
            problems.append(f"{r.recording_id}: shape mismatch")
        if not np.all(np.isfinite(d["t"])) or np.any(np.diff(d["t"]) < 0):
            problems.append(f"{r.recording_id}: time not monotonic")
        if "pos" in d and d["pos"].shape != (T, 3):
            problems.append(f"{r.recording_id}: pos shape")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["wallhack", "3do", "all"])
    args = ap.parse_args()
    for ds in (["wallhack", "3do"] if args.dataset == "all" else [args.dataset]):
        df = build(ds)
        print(df[["recording_id", "n_packets", "fs_hz", "duration_s", "drop_rate", "dt_p99_ms",
                  "n_gaps_gt_3dt", "classes"]].to_string())
        print("validation:", validate(ds) or "ok")


if __name__ == "__main__":
    main()
