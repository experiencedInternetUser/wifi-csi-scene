"""Sanity check of the unified pipeline (csi.preprocess) on the team's 16 scenarios with logistic regression.

Usage: python -m csi.check_template
Writes results/template_check.csv.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from csi.baselines import GRID, metrics, recording_id
from csi.format import ROOT
from csi.preprocess import PRESETS, preprocess_recording, window_features
from csi.recording import from_processed


def main():
    plan = json.loads((GRID / "plan.json").read_text(encoding="utf-8"))
    paths = sorted({p for sc in plan["scenarios"] for p in sc["train_recordings"] + sc["test_recordings"]})
    rows = []
    for preset in ["raw", "clean", "clean_lp_centered", "clean_lp_relative", "clean_lp_scaled"]:
        cache = {}
        for p in paths:
            ds, rid = recording_id(p)
            rec = from_processed(ds, rid)
            rec.y = (rec.y != 0).astype(np.int16)
            X, y, info = preprocess_recording(rec, PRESETS[preset])
            cache[p] = (window_features(X), y, rec.meta["domain"])
        for sc in plan["scenarios"]:
            def stack(recs):
                f = [cache[p] for p in recs]
                return (np.concatenate([a for a, _, _ in f]), np.concatenate([b for _, b, _ in f]),
                        np.concatenate([[d] * len(b) for _, b, d in f]))
            xtr, ytr, _ = stack(sc["train_recordings"])
            xte, yte, _ = stack(sc["test_recordings"])
            m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)).fit(xtr, ytr)
            score = m.predict_proba(xte)[:, 1]
            rows.append({"preset": preset, "scenario": sc["id"], "n_test": len(yte),
                         **metrics(yte, (score >= 0.5).astype(int), score)})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/template_check.csv", index=False)
    wh = df[df.scenario.str.startswith("wh_") & (df.scenario != "wh_to_td") | (df.scenario == "td_to_wh")]
    print(wh.groupby("preset")[["balanced_accuracy", "roc_auc", "far"]].mean().round(3).to_string())
    print(df.pivot(index="scenario", columns="preset", values="balanced_accuracy").round(3).to_string())


if __name__ == "__main__":
    main()
