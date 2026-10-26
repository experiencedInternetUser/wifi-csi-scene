"""Выбор предобработки на доменах Wallhack и повторное обучение кандидатов.

Запуск: python -m experiments.final_motion.train
Старая сетка не изменяется. Готовые запуски проверяются и пропускаются.
Порог и бюджет эпох фиксированы до просмотра новых контрольных результатов.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)

from .pipeline import PREPROCESSORS, make_model, parameter_count, preprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/motion_results"
SOURCE = OUT
ARCHITECTURES = ("cnngru", "tcn")
CV_SEEDS = (17, 42, 2026)
FINAL_SEEDS = (17, 42, 73, 101, 2026)
DEV_DOMAINS = ("LOS/BQ", "LOS/PIFA", "NLOS/BQ")
HOLDOUT = "NLOS/PIFA"
EPOCHS, BATCH, THRESHOLD = 20, 64, .5


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p >= THRESHOLD, labels=[0, 1]).ravel()
    return dict(n=len(y), positive=int(y.sum()), negative=int((y == 0).sum()),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                accuracy=float(accuracy_score(y, p >= THRESHOLD)),
                balanced_accuracy=float(balanced_accuracy_score(y, p >= THRESHOLD)),
                precision=float(precision_score(y, p >= THRESHOLD, zero_division=0)),
                recall=float(recall_score(y, p >= THRESHOLD, zero_division=0)),
                f1=float(f1_score(y, p >= THRESHOLD, zero_division=0)),
                roc_auc=float(roc_auc_score(y, p)),
                average_precision=float(average_precision_score(y, p)))


def predict(model, x):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(batch).flatten().sigmoid()
                          for batch in x.split(BATCH)]).cpu().numpy()


def benchmark(raw):
    """CPU: один поток, один готовый амплитудный фрагмент, FP32, без дискового I/O."""
    rows = []
    for architecture in ARCHITECTURES:
        model = make_model(architecture).eval()
        for prep in PREPROCESSORS:
            sample = raw[:1].copy()
            mean = torch.zeros(1, 52, 1)
            std = torch.ones(1, 52, 1)
            times = []
            with torch.inference_mode():
                for i in range(60):
                    begin = time.perf_counter()
                    x = torch.from_numpy(preprocess(sample, prep))
                    model((x - mean) / std).sigmoid()
                    if i >= 10:
                        times.append((time.perf_counter() - begin) * 1000)
            rows.append(dict(architecture=architecture, preprocessing=prep,
                             parameters=parameter_count(model), cpu_p50_ms=float(np.median(times)),
                             cpu_p95_ms=float(np.percentile(times, 95)), repeats=50,
                             threads=torch.get_num_threads(), batch=1, precision="float32"))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "benchmark.csv", index=False)
    return frame


def fit(run_id, architecture, prep, seed, x, meta, train_indices, validation_indices=None):
    path = OUT / "checkpoints" / (run_id + ".pt")
    curve_path = OUT / "curves" / (run_id + ".csv")
    if path.exists() and curve_path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert saved["run_id"] == run_id and saved["code_sha256"] == code_hashes()
        assert saved["train_indices"] == train_indices.tolist()
        assert saved["validation_indices"] == ([] if validation_indices is None else validation_indices.tolist())
        return saved
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_np = x[train_indices]
    mean = train_np.mean((0, 2), dtype=np.float64).reshape(1, 52, 1).astype(np.float32)
    std = train_np.std((0, 2), dtype=np.float64).clip(1e-4).reshape(1, 52, 1).astype(np.float32)
    tx = torch.from_numpy((train_np - mean) / std).to(device)
    y = meta.target.to_numpy(dtype=np.float32)
    ty = torch.from_numpy(y[train_indices]).to(device)
    if validation_indices is not None:
        assert not set(meta.iloc[train_indices].group_id) & set(meta.iloc[validation_indices].group_id)
        vx = torch.from_numpy((x[validation_indices] - mean) / std).to(device)
        vy = torch.from_numpy(y[validation_indices]).to(device)
    model = make_model(architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=.00001)
    weight = float((ty == 0).sum() / ty.sum())
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    curves = []
    if device == "cuda":
        torch.cuda.synchronize()
    begin = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(tx), device=device)
        total = 0.
        for ids in order.split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                loss = criterion(model(tx[ids]).flatten(), ty[ids])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(ids)
        row = dict(epoch=epoch, train_loss=total / len(tx))
        if validation_indices is not None:
            model.eval()
            with torch.inference_mode():
                logits = torch.cat([model(b).flatten() for b in vx.split(BATCH)])
                row["validation_loss"] = float(criterion(logits, vy))
                row["validation_balanced_accuracy"] = metrics(y[validation_indices].astype(int),
                                                               logits.sigmoid().cpu().numpy())["balanced_accuracy"]
        curves.append(row)
        scheduler.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - begin
    saved = dict(run_id=run_id, model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                 architecture=architecture, preprocessing=prep, seed=seed, epochs=EPOCHS,
                 mean=torch.from_numpy(mean), std=torch.from_numpy(std), threshold=THRESHOLD,
                 sample_rate_hz=100, window_samples=400, subcarriers=list(range(6, 32)) + list(range(33, 59)),
                 train_indices=train_indices.tolist(),
                 validation_indices=[] if validation_indices is None else validation_indices.tolist(),
                 train_recordings=sorted(meta.iloc[train_indices].group_id.unique().tolist()),
                 optimizer="AdamW", lr=.001, weight_decay=.0001, pos_weight=weight,
                 parameters=parameter_count(model), precision="CUDA AMP FP16 training; FP32 inference; TF32 allowed",
                 train_seconds=elapsed, code_sha256=code_hashes())
    torch.save(saved, path)
    pd.DataFrame(curves).to_csv(curve_path, index=False)
    print(f"Готово {run_id}: {elapsed:.1f} с", flush=True)
    del tx, ty, optimizer, scheduler, model
    gc.collect()
    return saved


def restore_predict(saved, x, indices):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(saved["architecture"]).to(device)
    model.load_state_dict(saved["model"])
    t = torch.from_numpy((x[indices] - saved["mean"].numpy()) / saved["std"].numpy()).to(device)
    probability = predict(model, t)
    del model, t
    return probability


def code_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(__file__).with_name("pipeline.py"),
                      Path(__file__).with_name("prepare_data.py"), ROOT / "experiments/models.py"]}


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for folder in [OUT, OUT / "checkpoints", OUT / "curves", OUT / "predictions"]:
        folder.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(SOURCE / "windows.csv")
    raw = np.load(SOURCE / "amplitudes.npy", mmap_mode="r")
    assert raw.shape == (len(meta), 52, 400) and np.isfinite(raw).all()
    dev = (meta.dataset == "Wallhack") & meta.domain.isin(DEV_DOMAINS)
    holdout = (meta.dataset == "Wallhack") & (meta.domain == HOLDOUT)
    external = meta.dataset == "3DO"
    plan = dict(created_date_msk="2026-10-26", architectures=list(ARCHITECTURES),
                preprocessors=list(PREPROCESSORS), cv_seeds=list(CV_SEEDS), final_seeds=list(FINAL_SEEDS),
                deployment_seed=42, development_domains=list(DEV_DOMAINS), heldout_domain=HOLDOUT,
                epochs=EPOCHS, batch=BATCH, threshold=THRESHOLD,
                selection="mean balanced accuracy of 3 domain folds x 3 seeds; within 0.005 of best: lowest CPU p50, then parameters",
                optimizer="AdamW lr=0.001 weight_decay=0.0001 cosine eta_min=0.00001 clip_grad_norm=1 dropout=0.2",
                loss="BCEWithLogitsLoss pos_weight=n0/n1; TRAIN only",
                normalization="per-subcarrier mean/std from TRAIN only; float64 accumulation; std floor 1e-4",
                checkpoint="epoch20; no early stopping, no validation threshold tuning",
                precision="CUDA AMP FP16 training, FP32 inference; TF32 allowed, fixed seeds, bitwise determinism not guaranteed",
                uncertainty="All domains appeared in prior exploratory grid. New holdout is retrospective, not an untouched independent test.",
                folds=[dict(validation_domain=d,
                            train_indices=np.flatnonzero(dev & (meta.domain != d)).tolist(),
                            validation_indices=np.flatnonzero(dev & (meta.domain == d)).tolist()) for d in DEV_DOMAINS],
                holdout_indices=np.flatnonzero(holdout).tolist(), external_indices=np.flatnonzero(external).tolist(),
                code_sha256=code_hashes(),
                data_sha256=json.loads((OUT / "data_manifest.json").read_text(encoding="utf-8"))["amplitudes_sha256"],
                versions=dict(python=platform.python_version(), torch=str(torch.__version__),
                    numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
                hardware=dict(cpu=platform.processor(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
    plan_path = OUT / "plan.json"
    if plan_path.exists():
        assert json.loads(plan_path.read_text(encoding="utf-8")) == plan, "План изменён; используйте другую папку результатов"
    else:
        write_json(plan_path, plan)
    meta.to_csv(OUT / "windows.csv", index=False)
    speed = pd.read_csv(OUT / "benchmark.csv") if (OUT / "benchmark.csv").exists() else benchmark(raw)
    rows = []
    for prep in PREPROCESSORS:
        x = preprocess(raw, prep)
        for architecture in ARCHITECTURES:
            for fold in plan["folds"]:
                ti = np.array(fold["train_indices"])
                vi = np.array(fold["validation_indices"])
                for seed in CV_SEEDS:
                    run_id = f"cv_{fold['validation_domain'].replace('/', '-')}__{prep}__{architecture}__s{seed}"
                    saved = fit(run_id, architecture, prep, seed, x, meta, ti, vi)
                    p = restore_predict(saved, x, vi)
                    truth = meta.iloc[vi].target.to_numpy()
                    np.savez_compressed(OUT / "predictions" / (run_id + ".npz"),
                                        indices=vi, truth=truth, probability=p)
                    rows.append(dict(run_id=run_id, architecture=architecture, preprocessing=prep,
                                     seed=seed, validation_domain=fold["validation_domain"],
                                     train_seconds=saved["train_seconds"], **metrics(truth, p)))
                    pd.DataFrame(rows).to_csv(OUT / "cv_runs.csv", index=False)
        del x
    cv = pd.DataFrame(rows)
    summary = cv.groupby(["architecture", "preprocessing"]).agg(
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        std_balanced_accuracy=("balanced_accuracy", "std"),
        min_balanced_accuracy=("balanced_accuracy", "min"),
        mean_f1=("f1", "mean"), mean_train_seconds=("train_seconds", "mean")).reset_index().merge(speed)
    best = float(summary.mean_balanced_accuracy.max())
    eligible = summary[summary.mean_balanced_accuracy >= best - .005]
    selected = eligible.sort_values(["cpu_p50_ms", "parameters"]).iloc[0]
    summary["eligible_within_0_005"] = summary.mean_balanced_accuracy >= best - .005
    summary.to_csv(OUT / "selection.csv", index=False)
    selection = dict(architecture=str(selected.architecture), preprocessing=str(selected.preprocessing),
                     mean_cv_balanced_accuracy=float(selected.mean_balanced_accuracy),
                     best_mean_cv_balanced_accuracy=best, cpu_p50_ms=float(selected.cpu_p50_ms),
                     parameters=int(selected.parameters), deployment_seed=42,
                     rule=plan["selection"], cv_fits=len(cv))
    write_json(OUT / "selection.json", selection)
    print("Выбор по валидации:", selection, flush=True)
    # Контрольные домены используются только после записи решения о выборе.
    x = preprocess(raw, selection["preprocessing"])
    test_rows = []
    for stage in ["domain_holdout", "all_wallhack"]:
        ti = np.flatnonzero(dev if stage == "domain_holdout" else meta.dataset == "Wallhack")
        evaluation = np.flatnonzero(holdout | external if stage == "domain_holdout" else external)
        for seed in FINAL_SEEDS:
            run_id = f"{stage}__{selection['preprocessing']}__{selection['architecture']}__s{seed}"
            saved = fit(run_id, selection["architecture"], selection["preprocessing"], seed, x, meta, ti)
            assert not set(meta.iloc[ti].group_id) & set(meta.iloc[evaluation].group_id)
            p = restore_predict(saved, x, evaluation)
            truth = meta.iloc[evaluation].target.to_numpy()
            np.savez_compressed(OUT / "predictions" / (run_id + ".npz"),
                                indices=evaluation, truth=truth, probability=p)
            for dataset in meta.iloc[evaluation].dataset.unique():
                ds = meta.iloc[evaluation].dataset.to_numpy() == dataset
                for domain in ["ALL"] + sorted(meta.iloc[evaluation][ds].domain.unique()):
                    mask = ds & ((meta.iloc[evaluation].domain.to_numpy() == domain) if domain != "ALL" else True)
                    test_rows.append(dict(run_id=run_id, stage=stage, seed=seed, dataset=dataset,
                                          domain=domain, **metrics(truth[mask], p[mask])))
            pd.DataFrame(test_rows).to_csv(OUT / "test_metrics.csv", index=False)
            if stage == "all_wallhack" and seed == 42:
                torch.save(saved, OUT / "deployment.pt")
    write_json(OUT / "status.json", dict(state="complete", cv_fits=len(cv), final_fits=10,
                                        completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    main()
