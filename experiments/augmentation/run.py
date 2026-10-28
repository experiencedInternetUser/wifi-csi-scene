"""Fixed source-only training and frozen external evaluation of CSI augmentations."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from experiments.robustness.modeling import fit_projection, metrics, network, network_predict, prepare, project
from .augment import Augmenter, POLICIES

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CACHE = ROOT / "tmp/augmentation"
SEEDS = (17, 42, 2026)
DEV = ("LOS/BQ", "LOS/PIFA", "NLOS/BQ")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf8"))


def write_csv(path, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False, lineterminator="\n")


def setup():
    for folder in (CACHE, OUT / "checkpoints", OUT / "curves", OUT / "predictions"):
        folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def prepared(dataset):
    raw_path = ROOT / "experiments/motion_results/amplitudes.npy" if dataset == "source" else ROOT / f"tmp/robustness_cache/{dataset}.npy"
    x = np.load(raw_path, mmap_mode="r")
    output = CACHE / f"{dataset}_prepared.npy"
    if not output.exists():
        result = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=x.shape)
        for start in range(0, len(x), 128):
            result[start:start + 128] = prepare(np.asarray(x[start:start + 128]), "hampel_lp20")
        result.flush()
        del result
    result = np.load(output, mmap_mode="r")
    assert result.shape == x.shape
    return x, result


def metadata(dataset):
    path = ROOT / "experiments/motion_results/windows.csv" if dataset == "source" else ROOT / f"experiments/robustness/results/{dataset}_windows.csv"
    return pd.read_csv(path)


def make_plan():
    plan = dict(
        policies=list(POLICIES), seeds=list(SEEDS), epochs=20, threshold=0.5,
        stages=["cv_LOS-BQ", "cv_LOS-PIFA", "cv_NLOS-BQ", "domain_holdout", "all_wallhack"],
        expected_fits=75, source_only=True, augmentation_probability=0.5,
        gain=[0.25, 16.0], spectral_knots=6, spectral_log_std=0.35,
        relative_noise_rms_max=0.06, common_noise_power_fraction=0.8,
        noise_fir_taps=41, noise_fir_cutoff_hz=10, noise_fir_window="hann",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        hashes={str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in [
            HERE / "protocol.md", HERE / "augment.py", HERE / "run.py",
            ROOT / "experiments/models.py", ROOT / "experiments/final_motion/pipeline.py",
            ROOT / "experiments/motion_results/windows.csv",
            ROOT / "experiments/motion_results/amplitudes.npy",
        ]},
    )
    path = OUT / "plan.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf8"))
        # A later Git staging/commit does not change the locked scientific inputs.
        assert {k: v for k, v in previous.items() if k != "source_commit"} == {k: v for k, v in plan.items() if k != "source_commit"}
        return previous
    write_json(path, plan)
    return plan


def evaluate_prepared(saved, x):
    model = network(52).cuda().eval()
    model.load_state_dict(saved["model"])
    result = []
    for start in range(0, len(x), 256):
        z = project(np.asarray(x[start:start + 256]), saved["projection"])
        result.append(network_predict(model, torch.from_numpy(z).cuda()))
    del model
    return np.concatenate(result)


def fit(stage, policy, seed, clean, raw, meta, train_ids, val_ids, plan):
    rid = f"{stage}__{policy}__s{seed}"
    path = OUT / "checkpoints" / f"{rid}.joblib"
    plan_hash = digest(OUT / "plan.json")
    if path.exists():
        saved = joblib.load(path)
        assert saved["plan_sha256"] == plan_hash
        assert saved["train_indices"] == train_ids.tolist()
        return rid, saved
    x = np.asarray(clean[train_ids])
    projection = fit_projection(x, 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tx = torch.from_numpy(x).cuda()
    means = torch.from_numpy(np.asarray(raw[train_ids]).mean((1, 2), dtype=np.float64).astype(np.float32)[:, None, None]).cuda()
    y = meta.target.to_numpy()[train_ids].astype(np.float32)
    ty = torch.from_numpy(y).cuda()
    mu, std = (torch.from_numpy(projection[k]).cuda() for k in ("mean", "std"))
    model = network(52).cuda()
    augment = Augmenter(policy, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=0.00001)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(y == 0).sum() / (y == 1).sum()], device="cuda"))
    scaler = torch.amp.GradScaler("cuda")
    vx = None if val_ids is None else torch.from_numpy(project(np.asarray(clean[val_ids]), projection)).cuda()
    curves = []
    start = time.perf_counter()
    for epoch in range(1, 21):
        model.train()
        total = 0.0
        for ids in torch.randperm(len(tx), device="cuda").split(64):
            optimizer.zero_grad(set_to_none=True)
            batch = (augment(tx[ids], means[ids]) - mu) / std
            with torch.autocast("cuda", dtype=torch.float16):
                loss = criterion(model(batch).flatten(), ty[ids])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(ids)
        row = dict(epoch=epoch, train_loss=total / len(tx))
        if vx is not None:
            p = network_predict(model, vx)
            row.update({"validation_" + k: v for k, v in metrics(meta.target.to_numpy()[val_ids], p).items()})
        curves.append(row)
        scheduler.step()
    seconds = time.perf_counter() - start
    train_groups = sorted(set(meta.iloc[train_ids].group_id))
    validation_groups = [] if val_ids is None else sorted(set(meta.iloc[val_ids].group_id))
    assert not set(train_groups) & set(validation_groups)
    assert set(meta.iloc[train_ids].dataset) == {"Wallhack"}
    saved = dict(
        name=policy, seed=seed, augmentation_policy=policy, plan_sha256=plan_hash,
        config=dict(kind="tcn", prep="hampel_lp20", pca=0), parameters=46081,
        model={k: v.detach().cpu() for k, v in model.state_dict().items()},
        projection=projection, threshold=0.5, epochs=20, train_seconds=seconds,
        train_n=len(train_ids), train_indices=train_ids.tolist(), train_groups=train_groups,
        validation_groups=validation_groups,
    )
    joblib.dump(saved, path, compress=3)
    portable = {**saved, "projection": {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in projection.items()}}
    torch.save(portable, path.with_suffix(".pt"))
    write_json(path.with_suffix(".json"), {k: v for k, v in saved.items() if k not in ("model", "projection")})
    write_csv(OUT / "curves" / f"{rid}.csv", curves)
    del model, tx, ty, vx, optimizer, means
    torch.cuda.empty_cache()
    return rid, saved


def predictions(rid, saved, clean, meta, ids, stage):
    p = evaluate_prepared(saved, clean[ids])
    frame = meta.iloc[ids].copy()
    frame["index"] = ids
    frame["probability"] = p
    frame["prediction"] = (p >= 0.5).astype(int)
    write_csv(OUT / "predictions" / f"{rid}__{stage}.csv", frame)
    rows = []
    masks = {"ALL": np.ones(len(ids), bool)}
    masks.update({str(v): frame.domain.to_numpy() == v for v in sorted(frame.domain.unique())})
    if stage == "opera_zero_shot":
        masks["Room2/ALL"] = frame.room.to_numpy() == 2
    for domain, mask in masks.items():
        rows.append(dict(run_id=rid, policy=saved["name"], seed=saved["seed"], stage=stage, domain=domain, **metrics(frame.target.to_numpy()[mask], p[mask])))
    return rows


def train_all():
    setup()
    plan = make_plan()
    raw, clean = prepared("source")
    meta = metadata("source")
    assert clean.shape == (3052, 52, 400)
    dev = (meta.dataset == "Wallhack") & meta.domain.isin(DEV)
    rows = []
    for held in DEV:
        tr, va = np.flatnonzero(dev & (meta.domain != held)), np.flatnonzero(dev & (meta.domain == held))
        stage = "cv_" + held.replace("/", "-")
        for policy in POLICIES:
            for seed in SEEDS:
                rid, saved = fit(stage, policy, seed, clean, raw, meta, tr, va, plan)
                result = predictions(rid, saved, clean, meta, va, "source_cv")
                rows.extend(result)
                write_csv(OUT / "source_cv.csv", rows)
                print(rid, "BA", result[0]["balanced_accuracy"], "sec", round(saved["train_seconds"], 2), flush=True)
    cv = pd.DataFrame(rows).query("domain=='ALL'")
    summary = cv.groupby("policy").agg(mean_ba=("balanced_accuracy", "mean"), mean_auc=("roc_auc", "mean"), mean_fpr=("fpr", "mean"), mean_tpr=("recall", "mean"), min_ba=("balanced_accuracy", "min"), sd_ba=("balanced_accuracy", "std")).reset_index()
    write_csv(OUT / "source_selection.csv", summary)
    candidates = summary.query("policy!='none'")
    best = candidates.mean_ba.max()
    eligible = set(candidates.loc[candidates.mean_ba >= best - 0.005, "policy"])
    chosen = next(p for p in POLICIES[1:] if p in eligible)
    write_json(OUT / "locked_selection.json", dict(policy=chosen, criterion="source-only mean development BA, tolerance 0.005, fixed complexity ordering", best_mean_ba=float(best), eligible=sorted(eligible), external_results_used=False, plan_sha256=digest(OUT / "plan.json")))
    print("LOCKED SELECTION", chosen, flush=True)
    for stage, tr in [("domain_holdout", np.flatnonzero(dev)), ("all_wallhack", np.flatnonzero(meta.dataset == "Wallhack"))]:
        for policy in POLICIES:
            for seed in SEEDS:
                rid, saved = fit(stage, policy, seed, clean, raw, meta, tr, None, plan)
                print(rid, "trained", round(saved["train_seconds"], 2), flush=True)


def evaluate_all():
    setup()
    make_plan()
    assert (OUT / "locked_selection.json").exists()
    assert len(list((OUT / "checkpoints").glob("*.joblib"))) == 75
    raw, clean = prepared("source")
    meta = metadata("source")
    rows = []
    for policy in POLICIES:
        for seed in SEEDS:
            for training, stage, ids in [
                ("domain_holdout", "wallhack_holdout", np.flatnonzero((meta.dataset == "Wallhack") & ~meta.domain.isin(DEV))),
                ("all_wallhack", "3do_final", np.flatnonzero(meta.dataset == "3DO")),
            ]:
                rid = f"{training}__{policy}__s{seed}"
                saved = joblib.load(OUT / "checkpoints" / f"{rid}.joblib")
                rows.extend(predictions(rid, saved, clean, meta, ids, stage))
    for dataset in ("days", "opera"):
        raw, clean = prepared(dataset)
        meta = metadata(dataset)
        for policy in POLICIES:
            for seed in SEEDS:
                rid = f"all_wallhack__{policy}__s{seed}"
                saved = joblib.load(OUT / "checkpoints" / f"{rid}.joblib")
                rows.extend(predictions(rid, saved, clean, meta, np.arange(len(meta)), dataset + "_zero_shot"))
                write_csv(OUT / "evaluation.csv", rows)
                print(rid, dataset, "BA", rows[-len(meta.domain.unique()) - (2 if dataset == 'opera' else 1)]["balanced_accuracy"], flush=True)
    write_csv(OUT / "evaluation.csv", rows)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["stage", "domain", "policy"])[["balanced_accuracy", "roc_auc", "average_precision", "f1", "recall", "fpr"]].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(c) for c in summary.columns]
    write_csv(OUT / "summary.csv", summary.reset_index())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train", "evaluate"])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        (train_all if args.stage == "train" else evaluate_all)()
