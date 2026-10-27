"""Simple models (threshold, logistic regression, LightGBM) on the team's split grid.

The 16 transfer scenarios and the 3045 windows come from the team NN grid
(results/team_grid/{plan.json,windows.csv}, branch baldezh-6.1-sol-xhigh), so the numbers
are directly comparable with results/team_grid/nn_results.csv.

Windows are rebuilt from data/processed: amplitude interpolated onto a 100 Hz grid
starting at each window's start_s, 400 samples.

Usage: python -m csi.baselines
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from csi import features as F
from csi.format import ROOT, load_recording

GRID = ROOT / "results/team_grid"
OUT = ROOT / "results/baselines_team_grid.csv"
WINDOW = 400
SEEDS = (42, 43, 44)

# input variants: amplitude source x representation
INPUTS = {
    "raw_abs": ("raw", "abs"),      # amplitude as is, absolute level kept
    "raw_rel": ("raw", "rel"),      # A / mean_time(A) per subcarrier (like team `relative`)
    "clean_abs": ("clean", "abs"),  # impulsive packets interpolated first
    "clean_rel": ("clean", "rel"),
}


def recording_id(path: str) -> tuple[str, str]:
    """Team path -> (dataset, our recording_id)."""
    parts = path.split("/")
    if parts[0] == "wallhack1.8k":
        return "wallhack", f"wallhack_{parts[1]}_{parts[2]}_{parts[3].removesuffix('.csv')}"
    return "3do", f"3do_{parts[1]}_{parts[2]}"


def build_window_features(windows: pd.DataFrame) -> pd.DataFrame:
    """Features of every team window for every input variant; one row per (window, input)."""
    rows = []
    for path, g in tqdm(windows.groupby("recording"), desc="windows"):
        ds, rid = recording_id(path)
        d = load_recording(ds, rid)
        amp = F.amplitude(d["csi"])
        sources = {"raw": amp, "clean": F.remove_impulses(amp)[0]}
        t = d["t"]
        for w in g.itertuples():
            grid = w.start_s + np.arange(WINDOW) / F.FS
            if grid[0] < t[0] - 1e-6 or grid[-1] > t[-1] + 1e-6:
                raise ValueError(f"window outside recording: {path} {w.start_s}")
            for name, (src, rep) in INPUTS.items():
                a = sources[src]
                lo, hi = np.searchsorted(t, [grid[0] - 0.2, grid[-1] + 0.2])
                win = np.stack([np.interp(grid, t[lo:hi], a[lo:hi, s]) for s in range(a.shape[1])], axis=1)
                if rep == "rel":
                    win = win / np.maximum(win.mean(axis=0, keepdims=True), 1.0)
                feats = F.window_features(win.astype(np.float32))
                rows.append({"index": w.index, "input": name, **feats})
    return pd.DataFrame(rows)


FEATURES = ["amp_mean", "amp_std_t", "var1s", "cv1s", "amp_range", "pc1_band_0_2", "pc1_band_2_10",
            "pc1_band_10_40", "pc1_logpower"]
LOG_FEATURES = ["amp_mean", "amp_std_t", "var1s", "cv1s", "amp_range"]


def design(df: pd.DataFrame) -> np.ndarray:
    x = df[FEATURES].to_numpy(dtype=float, copy=True)
    for c in LOG_FEATURES:
        j = FEATURES.index(c)
        x[:, j] = np.log10(x[:, j] + 1e-9)
    return x


class Threshold:
    """One feature (log var1s), cut chosen on train to maximize balanced accuracy."""

    def fit(self, x, y):
        v = x[:, FEATURES.index("var1s")]
        cands = np.unique(v)
        ba = [0.5 * ((v[y == 1] > c).mean() + (v[y == 0] <= c).mean()) for c in cands]
        self.cut = cands[int(np.argmax(ba))]
        return self

    def score(self, x):
        return x[:, FEATURES.index("var1s")]

    def predict(self, x):
        return (self.score(x) > self.cut).astype(int)


def make_models(seed: int) -> dict:
    return {
        "threshold": Threshold(),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
        "lightgbm": LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   class_weight="balanced", random_state=seed, verbose=-1),
    }


def metrics(y, pred, score) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    pos, neg = tp + fn, tn + fp
    rec = tp / pos if pos else np.nan
    spec = tn / neg if neg else np.nan
    prec = tp / (tp + fp) if tp + fp else 0.0
    both = pos and neg
    return {"n_windows": len(y), "n_positive": pos, "n_negative": neg, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp + tn) / len(y), "balanced_accuracy": np.nanmean([rec, spec]),
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "far": fp / neg if neg else np.nan,
            "roc_auc": roc_auc_score(y, score) if both else np.nan,
            "average_precision": average_precision_score(y, score) if both else np.nan}


def run() -> pd.DataFrame:
    plan = json.loads((GRID / "plan.json").read_text(encoding="utf-8"))
    windows = pd.read_csv(GRID / "windows.csv")
    feats = build_window_features(windows).merge(windows, on="index")
    rows = []
    for sc in tqdm(plan["scenarios"], desc="scenarios"):
        tr_rec, te_rec = set(sc["train_recordings"]), set(sc["test_recordings"])
        assert not tr_rec & te_rec
        for inp in INPUTS:
            f = feats[feats.input == inp]
            tr, te = f[f.recording.isin(tr_rec)], f[f.recording.isin(te_rec)]
            xtr, ytr, xte, yte = design(tr), tr.target.to_numpy(), design(te), te.target.to_numpy()
            for seed in SEEDS:
                for mname, model in make_models(seed).items():
                    if mname != "lightgbm" and seed != SEEDS[0]:
                        continue  # deterministic models: one seed is enough
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(xtr, ytr)
                    if mname == "threshold":
                        score, pred = model.score(xte), model.predict(xte)
                    else:
                        score = model.predict_proba(xte)[:, 1]
                        pred = (score >= 0.5).astype(int)
                    base = {"scenario": sc["id"], "family": sc["family"], "input": inp, "model": mname, "seed": seed,
                            "n_train": len(tr)}
                    doms = te.domain.to_numpy()
                    for dom in sorted(set(doms)):
                        m = doms == dom
                        rows.append({**base, "target_domain": dom, **metrics(yte[m], pred[m], score[m])})
                    if len(set(doms)) > 1:
                        rows.append({**base, "target_domain": "ALL", **metrics(yte, pred, score)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    return out


def summary(res: pd.DataFrame) -> pd.DataFrame:
    """Scenario-level row (ALL if several test domains), mean over seeds."""
    multi = res[res.target_domain == "ALL"]
    single = res[~res.set_index(["scenario", "input", "model", "seed"]).index.isin(
        multi.set_index(["scenario", "input", "model", "seed"]).index)]
    s = pd.concat([multi, single])
    return s.groupby(["scenario", "input", "model"])[["balanced_accuracy", "roc_auc", "far", "f1"]].mean().reset_index()


if __name__ == "__main__":
    res = run()
    s = summary(res)
    pd.set_option("display.width", 220)
    print(s.pivot_table(index="scenario", columns=["model", "input"], values="balanced_accuracy").round(3).to_string())
