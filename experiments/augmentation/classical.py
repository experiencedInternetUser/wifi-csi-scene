"""Source-only augmentation checks for LogReg, PCA-LogReg and a LightGBM control."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from scipy.signal import butter, sosfilt, sosfilt_zi
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.robustness.modeling import features, fit_projection, metrics, prepare, project, remove_impulses
from .augment import Augmenter, POLICIES
from .run import CACHE, DEV, OUT, ROOT, SEEDS, digest, metadata, setup, write_csv, write_json

CLS = OUT / "classical"
MODELS = ("logreg_scaled", "logreg_pca8")


def initialize():
    setup()
    for folder in (CLS, CLS / "checkpoints", CLS / "predictions"):
        folder.mkdir(parents=True, exist_ok=True)


def plan():
    path = CLS / "plan.json"
    item = dict(models=list(MODELS), policies=list(POLICIES), seeds=list(SEEDS), copies=2, copy_weight=0.5, target_training=False, expected_logreg_fits=150, expected_lightgbm_fits=15,
                hashes={p.relative_to(ROOT).as_posix(): digest(p) for p in [Path(__file__), Path(__file__).with_name("classical_protocol.md")]}, tcn_plan_sha256=digest(OUT / "plan.json"))
    if path.exists():
        assert json.loads(path.read_text(encoding="utf8")) == item
    else:
        write_json(path, item)
    return digest(path)


def relative(dataset):
    raw_path = ROOT / "experiments/motion_results/amplitudes.npy" if dataset == "source" else ROOT / f"tmp/robustness_cache/{dataset}.npy"
    raw = np.load(raw_path, mmap_mode="r")
    path = CACHE / f"{dataset}_classic_relative.npy"
    if not path.exists():
        clean = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=raw.shape)
        for start in range(0, len(raw), 64):
            clean[start:start + 64] = prepare(np.asarray(raw[start:start + 64]), "clean_lp_scaled")
        clean.flush()
    return raw, np.load(path, mmap_mode="r")


def filtered_source(raw):
    path = CACHE / "source_classic_filtered.npy"
    if not path.exists():
        result = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=raw.shape)
        sos = butter(4, 20, fs=100, output="sos")
        for i, w in enumerate(raw):
            a = remove_impulses(w.T)[0].T
            zi = sosfilt_zi(sos)[:, None, :] * a[None, :, :1]
            result[i], _ = sosfilt(sos, a, axis=-1, zi=zi)
        result.flush()
    return np.load(path, mmap_mode="r")


def projection(rel, ids, model):
    return fit_projection(np.asarray(rel[ids]), 8) if model == "logreg_pca8" else None


def featurize(x, proj):
    return features(x if proj is None else project(x, proj))


def fit_features(stage, policy, seed, ids, raw, rel, filtered, projections):
    arrays = {}
    if policy in ("none", "gain"):
        for name in MODELS:
            path = CACHE / f"classic_{stage}__{name}__original_features.npy"
            if not path.exists():
                np.save(path, featurize(np.asarray(rel[ids]), projections[name]))
            f = np.load(path)
            arrays[name] = np.concatenate([f, f])
        return arrays
    cached = {name: CACHE / f"classic_{stage}__{name}__{policy}__s{seed}.npy" for name in MODELS}
    if all(p.exists() for p in cached.values()):
        return {name: np.load(p) for name, p in cached.items()}
    aug = Augmenter(policy, seed)
    pieces = {name: [] for name in MODELS}
    for copy in range(2):
        for start in range(0, len(ids), 64):
            selected = ids[start:start + 64]
            x = torch.from_numpy(np.asarray(filtered[selected]).astype(np.float32)).cuda()
            means = torch.from_numpy(np.asarray(raw[selected]).mean((1, 2), dtype=np.float64).astype(np.float32)[:, None, None]).cuda()
            changed = aug(x, means)
            changed = (changed - changed.mean(-1, keepdim=True)) / changed.mean((1, 2), keepdim=True).clamp_min(1e-12)
            y = changed.cpu().numpy()
            for name in MODELS:
                pieces[name].append(featurize(y, projections[name]))
    for name in MODELS:
        arrays[name] = np.concatenate(pieces[name])
        np.save(cached[name], arrays[name])
    return arrays


def fit(stage, name, policy, seed, f, target, ids, groups, proj, plan_hash):
    rid = f"{stage}__{name}__{policy}__s{seed}"
    path = CLS / "checkpoints" / f"{rid}.joblib"
    if path.exists():
        saved = joblib.load(path)
        assert saved["plan_sha256"] == plan_hash
        return rid, saved
    start = time.perf_counter()
    if name.startswith("logreg"):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
        clf.fit(f, np.tile(target, 2), logisticregression__sample_weight=np.full(len(f), 0.5))
    else:
        clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight="balanced", random_state=seed, n_jobs=1, verbose=-1).fit(f, target)
    saved = dict(name=name, policy=policy, seed=seed, model=clf, projection=proj, config=dict(kind="logreg" if name.startswith("logreg") else "lightgbm", prep="clean_lp_scaled", pca=8 if name == "logreg_pca8" else 0), threshold=0.5, train_indices=ids.tolist(), train_groups=sorted(set(groups)), train_n=len(ids), plan_sha256=plan_hash, fit_seconds=time.perf_counter() - start)
    joblib.dump(saved, path, compress=3)
    write_json(path.with_suffix(".json"), {k: v for k, v in saved.items() if k not in ("model", "projection")})
    return rid, saved


def predict_and_save(rid, saved, f, meta, ids, stage):
    probability = saved["model"].predict_proba(f)[:, 1]
    q = meta.iloc[ids].copy()
    q["index"], q["probability"], q["prediction"] = ids, probability, (probability >= 0.5).astype(int)
    write_csv(CLS / "predictions" / f"{rid}__{stage}.csv", q)
    masks = {"ALL": np.ones(len(q), bool)}
    masks.update({str(d): q.domain.to_numpy() == d for d in sorted(q.domain.unique())})
    if stage == "opera_zero_shot":
        masks["Room2/ALL"] = q.room.to_numpy() == 2
    return [dict(run_id=rid, name=saved["name"], policy=saved["policy"], seed=saved["seed"], stage=stage, domain=domain, **metrics(q.target.to_numpy()[mask], probability[mask])) for domain, mask in masks.items()]


def train_all():
    initialize()
    plan_hash = plan()
    raw, rel = relative("source")
    filtered = filtered_source(raw)
    meta = metadata("source")
    dev = meta.dataset.eq("Wallhack") & meta.domain.isin(DEV)
    stages = [("cv_" + held.replace("/", "-"), np.flatnonzero(dev & meta.domain.ne(held)), np.flatnonzero(dev & meta.domain.eq(held))) for held in DEV]
    stages += [("domain_holdout", np.flatnonzero(dev), None), ("all_wallhack", np.flatnonzero(meta.dataset.eq("Wallhack")), None)]
    rows = []
    for stage, tr, va in stages:
        projections = {name: projection(rel, tr, name) for name in MODELS}
        validation = {} if va is None else {name: featurize(np.asarray(rel[va]), projections[name]) for name in MODELS}
        for policy in POLICIES:
            for seed in SEEDS:
                arrays = fit_features(stage, policy, seed, tr, raw, rel, filtered, projections)
                for name in MODELS:
                    rid, saved = fit(stage, name, policy, seed, arrays[name], meta.target.to_numpy()[tr], tr, meta.group_id.to_numpy()[tr], projections[name], plan_hash)
                    if va is not None:
                        assert not set(saved["train_groups"]) & set(meta.iloc[va].group_id)
                        result = predict_and_save(rid, saved, validation[name], meta, va, "source_cv")
                        rows.extend(result)
                        write_csv(CLS / "source_cv.csv", rows)
                    print(rid, "complete", flush=True)
        for seed in SEEDS:
            name = "lightgbm_scaled"
            f = fit_features(stage, "none", seed, tr, raw, rel, filtered, projections)["logreg_scaled"][:len(tr)]
            rid, saved = fit(stage, name, "none", seed, f, meta.target.to_numpy()[tr], tr, meta.group_id.to_numpy()[tr], None, plan_hash)
            if va is not None:
                rows.extend(predict_and_save(rid, saved, validation["logreg_scaled"], meta, va, "source_cv"))
                write_csv(CLS / "source_cv.csv", rows)
            print(rid, "complete", flush=True)
        if stage == "cv_NLOS-BQ":
            summary = pd.DataFrame(rows).query("domain=='ALL'").groupby(["name", "policy"]).agg(mean_ba=("balanced_accuracy", "mean"), mean_auc=("roc_auc", "mean")).reset_index()
            write_csv(CLS / "source_selection.csv", summary)
            candidates = summary[summary.name.isin(MODELS) & summary.policy.ne("none")]
            best = candidates.mean_ba.max()
            eligible = candidates[candidates.mean_ba >= best - 0.005]
            ranking = {name: i for i, name in enumerate(MODELS)}
            ordered = eligible.assign(model_rank=eligible.name.map(ranking), policy_rank=eligible.policy.map({p: i for i, p in enumerate(POLICIES)})).sort_values(["model_rank", "policy_rank"])
            selected = ordered.iloc[0]
            write_json(CLS / "locked_selection.json", dict(name=selected["name"], policy=selected["policy"], best_mean_ba=float(best), selected_mean_ba=float(selected.mean_ba), external_results_used=False, plan_sha256=plan_hash))
            print("LOCKED LOGREG", selected["name"], selected["policy"], flush=True)


def evaluate_all():
    initialize()
    plan()
    assert (CLS / "locked_selection.json").exists()
    assert len(list((CLS / "checkpoints").glob("*.joblib"))) == 165
    rows = []
    for dataset in ("source", "days", "opera"):
        _, rel = relative(dataset)
        meta = metadata(dataset)
        tests = [("domain_holdout", "wallhack_holdout", np.flatnonzero(meta.dataset.eq("Wallhack") & ~meta.domain.isin(DEV))), ("all_wallhack", "3do_final", np.flatnonzero(meta.dataset.eq("3DO")))] if dataset == "source" else [("all_wallhack", dataset + "_zero_shot", np.arange(len(meta)))]
        for training, stage, ids in tests:
            for name in (*MODELS, "lightgbm_scaled"):
                first = joblib.load(CLS / "checkpoints" / f"{training}__{name}__none__s42.joblib")
                f = featurize(np.asarray(rel[ids]), first["projection"])
                for policy in (POLICIES if name in MODELS else ("none",)):
                    for seed in SEEDS:
                        rid = f"{training}__{name}__{policy}__s{seed}"
                        saved = joblib.load(CLS / "checkpoints" / f"{rid}.joblib")
                        assert saved["projection"] is None or np.array_equal(saved["projection"]["components"], first["projection"]["components"])
                        rows.extend(predict_and_save(rid, saved, f, meta, ids, stage))
                        write_csv(CLS / "evaluation.csv", rows)
                print(name, stage, "complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train", "evaluate"])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        (train_all if args.stage == "train" else evaluate_all)()
