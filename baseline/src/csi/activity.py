"""Activity-type classification (stage 3): Wallhack {empty, walk, walk_wave}, 3DO {empty, walk, sit, lie}.

Windows come from the unified pipeline (csi.preprocess). Two feature groups:
  dyn      motion features of the `clean_lp_scaled` window (unit-free): variance, PC1 spectrum in fine
           bands, periodicity, how many subcarriers co-move
  profile  static shape of the mean amplitude over subcarriers (needed for still postures, but also the
           feature that lets a model recognise a recording session -> reported separately)

Splits are by recordings / domains only:
  wallhack: within-domain 5 folds (w_k + ww_k + time block k of b1), leave-one-domain-out
  3do:      within-day folds by session number, leave-one-day-out
  cross:    wallhack <-> 3do on {empty, walk}

Usage: python -m csi.activity
Writes results/activity_metrics.csv, results/activity_confusions.csv, figures/activity_*.png.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from csi import features as F
from csi.format import PROCESSED, ROOT, load_index
from csi.labels import DATASET_TO_GLOBAL
from csi.plotstyle import SEQ, plt
from csi.preprocess import PRESETS, preprocess_recording
from csi.recording import from_processed

FS = 100.0
BANDS = [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 50)]
SEEDS = (42, 43, 44)


# ---------------------------------------------------------------- features

def dyn_features(w: np.ndarray) -> dict:
    """w [W, S] scaled window (mean removed per subcarrier, unit-free)."""
    out = {"log_var": np.log10(w.var(axis=0).mean() + 1e-12),
           "log_var_diff": np.log10(np.diff(w, axis=0).var(axis=0).mean() + 1e-12)}
    _, sv, vt = np.linalg.svd(w - w.mean(axis=0), full_matrices=False)
    ev = sv ** 2 / max((sv ** 2).sum(), 1e-12)
    out["pc1_share"], out["pc2_share"] = ev[0], ev[1]          # do subcarriers move together?
    pc = w @ vt[0]
    f, p = signal.welch(pc, fs=FS, nperseg=200, noverlap=100)
    tot = p.sum() + 1e-12
    for lo, hi in BANDS:
        out[f"band_{lo:g}_{hi:g}"] = p[(f >= lo) & (f < hi)].sum() / tot
    out["centroid_hz"] = (f * p).sum() / tot
    ac = np.correlate(pc - pc.mean(), pc - pc.mean(), "full")[len(pc) - 1:]
    ac = ac / max(ac[0], 1e-12)
    lag = slice(int(FS / 3), int(FS * 2))                      # periods 0.33-2 s (0.5-3 Hz)
    out["acf_peak"] = ac[lag].max()
    out["acf_peak_hz"] = FS / (lag.start + int(np.argmax(ac[lag])))
    d = np.diff(pc)
    out["kurt_diff"] = float(((d - d.mean()) ** 4).mean() / max(d.var() ** 2, 1e-12))
    return out


def profile_features(a: np.ndarray, n: int = 13) -> dict:
    """a [W, S] cleaned amplitude: mean profile over subcarriers, normalized, pooled to n bins."""
    prof = a.mean(axis=0)
    prof = prof / max(prof.mean(), 1e-12)
    bins = np.array_split(prof, n)
    return {f"prof_{i:02d}": float(b.mean()) for i, b in enumerate(bins)} | {"prof_slope": float(np.polyfit(
        np.arange(len(prof)), prof, 1)[0])}


def build(dataset: str) -> pd.DataFrame:
    rows = []
    idx = load_index(dataset)
    for r in tqdm(idx.itertuples(), total=len(idx), desc=dataset):
        rec = from_processed(dataset, r.recording_id)
        Xs, ys, info = preprocess_recording(rec, PRESETS["clean_lp_scaled"])
        Xc, yc, info_c = preprocess_recording(rec, PRESETS["clean"])
        assert np.allclose(info["start_s"], info_c["start_s"]) and (ys == yc).all()
        for i in range(len(Xs)):
            rows.append({"recording_id": r.recording_id, "dataset": dataset, "domain": rec.meta["domain"],
                         "session": r.session, "start_s": info["start_s"][i], "y": int(ys[i]),
                         "label": DATASET_TO_GLOBAL[dataset][int(ys[i])],
                         **dyn_features(Xs[i, 0].T), **profile_features(Xc[i, 0].T)})
    df = pd.DataFrame(rows)
    df.to_parquet(PROCESSED / dataset / "activity_windows.parquet", index=False)
    return df


DYN = None  # filled lazily from columns


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    prof = [c for c in df.columns if c.startswith("prof_")]
    meta = {"recording_id", "dataset", "domain", "session", "start_s", "y", "label", "fold"}
    dyn = [c for c in df.columns if c not in meta and c not in prof]
    return {"dyn": dyn, "dyn+profile": dyn + prof}


# ---------------------------------------------------------------- splits

def wallhack_folds(df: pd.DataFrame) -> pd.Series:
    """Fold 1..5: files w_k / ww_k; b1 split into 5 time blocks (block k -> fold k)."""
    fold = df.session.str.extract(r"(\d)$")[0].astype(int)
    b = df.session == "b1"
    for dom, g in df[b].groupby("domain"):
        q = pd.qcut(g.start_s, 5, labels=False) + 1
        fold.loc[q.index] = q
    return fold


def threedo_folds(df: pd.DataFrame) -> pd.Series:
    """Within a day: w_k, s_k, l_k share fold k; b_k -> fold k (k = 1..5, b only 1..3)."""
    return df.session.str.extract(r"(\d)$")[0].astype(int)


def splits(df: pd.DataFrame, dataset: str):
    """Yield (split_family, split_name, train_mask, test_mask)."""
    df = df.copy()
    df["fold"] = wallhack_folds(df) if dataset == "wallhack" else threedo_folds(df)
    for dom in sorted(df.domain.unique()):
        d = df.domain == dom
        for k in sorted(df.loc[d, "fold"].unique()):
            yield "within", f"{dom}|fold{k}", d & (df.fold != k), d & (df.fold == k)
    for dom in sorted(df.domain.unique()):
        yield "lodo", f"->{dom}", df.domain != dom, df.domain == dom


# ---------------------------------------------------------------- models

def make_models(seed):
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)),
        "lightgbm": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   class_weight="balanced", random_state=seed, verbose=-1),
    }


def evaluate(df, dataset, labels):
    sets = feature_sets(df)
    rows, confs = [], []
    for fam, name, tr, te in splits(df, dataset):
        if df.loc[te, "label"].nunique() < 2 or df.loc[tr, "label"].nunique() < len(labels):
            continue
        for fs_name, cols in sets.items():
            xtr, xte = df.loc[tr, cols].to_numpy(float), df.loc[te, cols].to_numpy(float)
            ytr, yte = df.loc[tr, "label"].to_numpy(), df.loc[te, "label"].to_numpy()
            for seed in SEEDS:
                for mname, m in make_models(seed).items():
                    if mname == "logreg" and seed != SEEDS[0]:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m.fit(xtr, ytr)
                        pred = m.predict(xte)
                    present = [l for l in labels if l in set(yte)]
                    rec = recall_score(yte, pred, labels=present, average=None, zero_division=0)
                    rows.append({"dataset": dataset, "family": fam, "split": name, "features": fs_name,
                                 "model": mname, "seed": seed, "n_test": len(yte),
                                 "macro_f1": f1_score(yte, pred, labels=present, average="macro", zero_division=0),
                                 "balanced_acc": rec.mean(),
                                 **{f"recall_{l}": v for l, v in zip(present, rec)}})
                    cm = confusion_matrix(yte, pred, labels=labels)
                    for i, a in enumerate(labels):
                        for j, b in enumerate(labels):
                            confs.append({"dataset": dataset, "family": fam, "split": name, "features": fs_name,
                                          "model": mname, "seed": seed, "true": a, "pred": b, "n": int(cm[i, j])})
    return pd.DataFrame(rows), pd.DataFrame(confs)


def cross_dataset(wh: pd.DataFrame, td: pd.DataFrame):
    """Train on one dataset, test on the other, classes {empty, walk} only."""
    rows = []
    a = wh[wh.label.isin(["empty", "walk"])]
    b = td[td.label.isin(["empty", "walk"])]
    for fs_name, cols in feature_sets(wh).items():
        for (src, s), (dst, t) in [(("wallhack", a), ("3do", b)), (("3do", b), ("wallhack", a))]:
            for seed in SEEDS:
                for mname, m in make_models(seed).items():
                    if mname == "logreg" and seed != SEEDS[0]:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m.fit(s[cols].to_numpy(float), s.label)
                        pred = m.predict(t[cols].to_numpy(float))
                    rec = recall_score(t.label, pred, labels=["empty", "walk"], average=None)
                    rows.append({"dataset": f"{src}->{dst}", "family": "cross", "split": f"{src}->{dst}",
                                 "features": fs_name, "model": mname, "seed": seed, "n_test": len(t),
                                 "macro_f1": f1_score(t.label, pred, average="macro"), "balanced_acc": rec.mean(),
                                 "recall_empty": rec[0], "recall_walk": rec[1]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report

def plot_confusions(conf: pd.DataFrame, dataset: str, labels: list[str]):
    sel = conf[(conf.dataset == dataset)]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, (fam, fs_name) in zip(axes.flat, [(f, s) for f in ["within", "lodo"] for s in ["dyn", "dyn+profile"]]):
        c = sel[(sel.family == fam) & (sel.features == fs_name) & (sel.model == "logreg")]
        m = c.pivot_table(index="true", columns="pred", values="n", aggfunc="sum").reindex(index=labels, columns=labels)
        m = m.div(m.sum(axis=1), axis=0).fillna(0).to_numpy()
        ax.imshow(m, cmap=SEQ, vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if m[i, j] > 0.55 else "#0b0b0b")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{'within domain' if fam == 'within' else 'leave-one-domain-out'} | {fs_name}")
        ax.grid(False)
    fig.suptitle(f"{dataset}: activity types, logistic regression, row-normalized confusion (pooled over splits)")
    fig.tight_layout()
    fig.savefig(ROOT / f"figures/activity_{dataset}_confusion.png")
    plt.close(fig)


def main():
    labels = {"wallhack": ["empty", "walk", "walk_wave"], "3do": ["empty", "walk", "sit", "lie"]}
    dfs, metrics, confs = {}, [], []
    for ds in ["wallhack", "3do"]:
        p = PROCESSED / ds / "activity_windows.parquet"
        dfs[ds] = pd.read_parquet(p) if p.exists() else build(ds)
        m, c = evaluate(dfs[ds], ds, labels[ds])
        metrics.append(m)
        confs.append(c)
    metrics.append(cross_dataset(dfs["wallhack"], dfs["3do"]))
    met, conf = pd.concat(metrics, ignore_index=True), pd.concat(confs, ignore_index=True)
    met.to_csv(ROOT / "results/activity_metrics.csv", index=False)
    conf.to_csv(ROOT / "results/activity_confusions.csv", index=False)
    for ds in labels:
        plot_confusions(conf, ds, labels[ds])
    pd.set_option("display.width", 220)
    rec_cols = [c for c in met.columns if c.startswith("recall_")]
    print(met.groupby(["dataset", "family", "features", "model"])[["macro_f1", "balanced_acc"] + rec_cols]
          .mean().round(3).to_string())
    print("\nper LODO split (logreg):")
    print(met[(met.family == "lodo") & (met.model == "logreg")].pivot_table(
        index=["dataset", "split"], columns="features", values="macro_f1").round(3).to_string())


if __name__ == "__main__":
    main()
