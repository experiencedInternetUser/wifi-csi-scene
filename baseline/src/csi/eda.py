"""Window-level feature table used by EDA plots and the threshold baseline.

Usage: python -m csi.eda [--dataset wallhack|3do|all]
writes data/processed/<dataset>/windows.parquet (window 400, step 400, purity >= 0.8).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from csi import features as F
from csi.format import PROCESSED, load_index, load_recording
from csi.labels import DATASET_TO_GLOBAL
from csi.windows import STEP_TEST, WINDOW, window_labels, window_starts


def domain_of(row) -> str:
    return f"{row.condition}_{row.antenna}" if row.dataset == "wallhack" else row.day


def build_windows(dataset: str, pairing: str = "espidf") -> pd.DataFrame:
    rows = []
    for r in tqdm(load_index(dataset).itertuples(), desc=f"windows {dataset}"):
        d = load_recording(dataset, r.recording_id, pairing)
        amp = F.amplitude(d["csi"])
        amp_clean, bad = F.remove_impulses(amp)
        ph = F.sanitized_phase(d["csi"])
        starts = window_starts(len(amp), WINDOW, STEP_TEST)
        labels, keep = window_labels(d["y"], starts)
        for s, lab, k in zip(starts, labels, keep):
            if not k:
                continue
            feats = F.window_features(amp[s:s + WINDOW], ph[s:s + WINDOW])
            clean = F.window_features(amp_clean[s:s + WINDOW])
            feats.update({f"{k}_clean": clean[k] for k in ("var1s", "cv1s", "amp_std_t")})
            feats["impulse_frac"] = float(bad[s:s + WINDOW].mean())
            row = {"recording_id": r.recording_id, "dataset": dataset, "domain": domain_of(r),
                   "session": r.session, "room": r.room, "start": int(s), "t_start": float(d["t"][s]),
                   "y": int(lab), "label": DATASET_TO_GLOBAL[dataset][int(lab)], **feats}
            if "pos" in d:
                p = d["pos"][s:s + WINDOW]
                row.update(x=float(p[:, 0].mean()), y_pos=float(p[:, 1].mean()), z=float(p[:, 2].mean()))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(PROCESSED / dataset / "windows.parquet", index=False)
    return df


def load_windows(dataset: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / dataset / "windows.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["wallhack", "3do", "all"])
    args = ap.parse_args()
    for ds in (["wallhack", "3do"] if args.dataset == "all" else [args.dataset]):
        df = build_windows(ds)
        print(ds, len(df), "windows")
        print(df.groupby(["domain", "label"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
