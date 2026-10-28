"""OPERAnet (Intel 5300, 3x3 x 30 subcarriers, 5 GHz / 20 MHz, 1600 Hz) through the unified pipeline.

1. Activity types on OPERAnet itself: train on the sessions of one person, test on another person
   (background recordings split the same way).
2. Cross-hardware transfer without retraining: {empty, walk} model trained on ESP32 data
   (Wallhack + 3DO, dynamic features) applied to OPERAnet as is.

Usage: python -m csi.operanet_activity
Writes results/operanet_activity_metrics.csv, results/operanet_cross_hardware.csv,
figures/operanet_activity_confusion.png.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from csi.activity import dyn_features, make_models, SEEDS
from csi.format import PROCESSED, ROOT
from csi.io import operanet as O
from csi.labels import DATASET_TO_GLOBAL
from csi.plotstyle import SEQ, plt
from csi.preprocess import PRESETS, preprocess_recording

NAMES = DATASET_TO_GLOBAL["operanet"]


def build() -> pd.DataFrame:
    idx = pd.read_parquet(O.OUT / "recordings.parquet")
    idx = idx[idx.recording_id.str.contains("NUC1")]
    rows = []
    for r in idx.itertuples():
        rec = O.load(r.recording_id)
        X, y, info = preprocess_recording(rec, PRESETS["clean_lp_scaled"])
        for i in range(len(X)):
            w = X[i].reshape(-1, X.shape[-1]).T          # [W, 9*30]
            rows.append({"recording_id": r.recording_id, "exp": r.exp, "person_id": r.person_id,
                         "domain": r.domain, "start_s": info["start_s"][i], "y": int(y[i]),
                         "label": NAMES[int(y[i])], "impulse_frac": info["impulse_frac"][i],
                         **dyn_features(w)})
    df = pd.DataFrame(rows)
    df.to_parquet(PROCESSED / "operanet" / "activity_windows.parquet", index=False)
    return df


def fit_predict(model, xtr, ytr, xte):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(xtr, ytr)
        return model.predict(xte)


def person_transfer(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Background exps have no person: assign them to the person whose session is closest in exp number."""
    df = df.copy()
    people = df[df.label != "empty"].groupby("person_id").exp.apply(lambda s: s.astype(int).median())
    bg = df.label == "empty"
    df.loc[bg, "person_id"] = df.loc[bg, "exp"].astype(int).map(
        lambda e: (people - e).abs().idxmin())
    labels = [l for l in NAMES.values() if l in set(df.label)]
    rows, confs = [], []
    for test_person in sorted(df.person_id.unique()):
        tr, te = df.person_id != test_person, df.person_id == test_person
        common = sorted(set(df.loc[tr, "label"]) & set(df.loc[te, "label"]))
        tr, te = tr & df.label.isin(common), te & df.label.isin(common)
        for seed in SEEDS:
            for mname, m in make_models(seed).items():
                if mname == "logreg" and seed != SEEDS[0]:
                    continue
                pred = fit_predict(m, df.loc[tr, cols].to_numpy(float), df.loc[tr, "label"],
                                   df.loc[te, cols].to_numpy(float))
                yte = df.loc[te, "label"].to_numpy()
                rec = recall_score(yte, pred, labels=common, average=None, zero_division=0)
                rows.append({"split": f"->{test_person}", "model": mname, "seed": seed, "n_test": len(yte),
                             "classes": ",".join(common),
                             "macro_f1": f1_score(yte, pred, labels=common, average="macro", zero_division=0),
                             "balanced_acc": rec.mean(), **{f"recall_{l}": v for l, v in zip(common, rec)}})
                cm = confusion_matrix(yte, pred, labels=labels)
                confs += [{"split": f"->{test_person}", "model": mname, "seed": seed, "true": a, "pred": b,
                           "n": int(cm[i, j])} for i, a in enumerate(labels) for j, b in enumerate(labels)]
    return pd.DataFrame(rows), pd.DataFrame(confs)


def cross_hardware(op: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """ESP32 (Wallhack + 3DO) -> Intel 5300 (OPERAnet), {empty, walk}, no retraining."""
    src = pd.concat([pd.read_parquet(PROCESSED / ds / "activity_windows.parquet") for ds in ["wallhack", "3do"]])
    src = src[src.label.isin(["empty", "walk"])]
    tgt = op[op.label.isin(["empty", "walk"])]
    rows = []
    for train_name, s in [("wallhack+3do", src), ("wallhack", src[src.dataset == "wallhack"]),
                          ("3do", src[src.dataset == "3do"])]:
        for seed in SEEDS:
            for mname, m in make_models(seed).items():
                if mname == "logreg" and seed != SEEDS[0]:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(s[cols].to_numpy(float), s.label)
                    pred = m.predict(tgt[cols].to_numpy(float))
                    score = m.predict_proba(tgt[cols].to_numpy(float))[:, list(m.classes_).index("walk")]
                rec = recall_score(tgt.label, pred, labels=["empty", "walk"], average=None)
                from sklearn.metrics import roc_auc_score
                rows.append({"train": train_name, "test": "operanet_NUC1", "model": mname, "seed": seed,
                             "n_test": len(tgt), "balanced_acc": rec.mean(), "recall_empty": rec[0],
                             "far": 1 - rec[0], "recall_walk": rec[1],
                             "roc_auc": roc_auc_score(tgt.label == "walk", score)})
    return pd.DataFrame(rows)


def plot(conf: pd.DataFrame, labels: list[str]):
    c = conf[conf.model == "lightgbm"]
    m = c.pivot_table(index="true", columns="pred", values="n", aggfunc="sum").reindex(index=labels, columns=labels)
    m = m.div(m.sum(axis=1), axis=0).fillna(0).to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(m, cmap=SEQ, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if m[i, j] > 0.55 else "#0b0b0b")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    ax.set_title("OPERAnet (Intel 5300): activities, LightGBM, test on an unseen person")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/operanet_activity_confusion.png")
    plt.close(fig)


def main():
    df = build()
    cols = [c for c in df.columns if c not in {"recording_id", "exp", "person_id", "domain", "start_s", "y",
                                                "label", "impulse_frac"}]
    pd.set_option("display.width", 220)
    print(df.groupby(["person_id", "label"]).size().unstack(fill_value=0).to_string())
    print("impulse share mean:", round(df.impulse_frac.mean(), 4))
    met, conf = person_transfer(df, cols)
    met.to_csv(ROOT / "results/operanet_activity_metrics.csv", index=False)
    rec_cols = [c for c in met.columns if c.startswith("recall_")]
    print(met.groupby("model")[["macro_f1", "balanced_acc"] + rec_cols].mean().round(3).T.to_string())
    plot(conf, [l for l in NAMES.values() if l in set(df.label)])
    xh = cross_hardware(df, cols)
    xh.to_csv(ROOT / "results/operanet_cross_hardware.csv", index=False)
    print(xh.groupby(["train", "model"])[["balanced_acc", "roc_auc", "recall_empty", "recall_walk", "far"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
