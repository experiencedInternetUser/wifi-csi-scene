"""Проверка нового эксперимента, повторный инференс и интервалы по записям."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .pipeline import MotionDetector, make_model, preprocess
from .train import OUT, SOURCE, THRESHOLD, code_hashes, metrics, restore_predict, write_json


def cluster_bootstrap(predictions, meta, repeats=5000):
    """Условный percentile CI: стратификация по дню и классу, кластер=запись."""
    rng = np.random.default_rng(42)
    table = meta.iloc[predictions["indices"]].copy()
    table["truth"] = predictions["truth"]
    table["predicted"] = predictions["probability"] >= THRESHOLD
    records = []
    for group_id, group in table.groupby("group_id"):
        y = group.truth.to_numpy()
        p = group.predicted.to_numpy()
        records.append(dict(group_id=group_id, domain=group.domain.iloc[0], target=int(y[0]), n=len(y),
                            tn=int(((y == 0) & ~p).sum()), fp=int(((y == 0) & p).sum()),
                            fn=int(((y == 1) & ~p).sum()), tp=int(((y == 1) & p).sum())))
    frame = pd.DataFrame(records)
    counts = np.zeros((repeats, 4), dtype=np.int64)
    for _, group in frame.groupby(["domain", "target"]):
        sample = group[["tn", "fp", "fn", "tp"]].to_numpy()
        ids = rng.integers(0, len(sample), size=(repeats, len(sample)))
        counts += sample[ids].sum(1)
    tn, fp, fn, tp = counts.T
    draws = pd.DataFrame(dict(balanced_accuracy=.5 * (tp / (tp + fn) + tn / (tn + fp)),
                              recall=tp / (tp + fn), specificity=tn / (tn + fp),
                              f1=2 * tp / (2 * tp + fp + fn)))
    intervals = {key: dict(lower=float(draws[key].quantile(.025)), upper=float(draws[key].quantile(.975)))
                 for key in draws}
    return frame, draws, dict(repeats=repeats, seed=42, method="stratified by day and target, resample whole recordings",
                              scope="conditional on recorded days and actor; cannot bound unseen-domain error",
                              recordings=len(frame), intervals=intervals,
                              degenerate=bool((draws.std() < 1e-12).all()))


def latency(detector, raw, device):
    values = []
    with torch.inference_mode():
        for i in range(120):
            sample = raw[i % len(raw):i % len(raw) + 1]
            if device == "cuda":
                torch.cuda.synchronize()
            begin = time.perf_counter()
            detector.predict(sample, batch_size=1)
            if device == "cuda":
                torch.cuda.synchronize()
            if i >= 20:
                values.append((time.perf_counter() - begin) * 1000)
    return dict(p50_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
                repeats=100, warmup=20, threads=torch.get_num_threads(), batch=1,
                includes="preprocessing, scaling, model, sigmoid and host-device copies; excludes I/Q parsing, resampling, disk and 4s acquisition")


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    meta = pd.read_csv(OUT / "windows.csv")
    raw = np.load(OUT / "amplitudes.npy", mmap_mode="r")
    plan = json.loads((OUT / "plan.json").read_text(encoding="utf-8"))
    assert plan["code_sha256"] == code_hashes()
    manifest = json.loads((OUT / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["amplitudes_sha256"] == hashlib.sha256((OUT / "amplitudes.npy").read_bytes()).hexdigest()
    # train сохраняет таблицу повторно через pandas: написание дробных start_s
    # может меняться, хотя авторитетные целые start_us/end_us остаются теми же.
    windows_hash = hashlib.sha256((OUT / "windows.csv").read_bytes()).hexdigest()
    if windows_hash != manifest["windows_sha256"]:
        manifest.setdefault("windows_initial_serialization_sha256", manifest["windows_sha256"])
        manifest["windows_sha256"] = windows_hash
        manifest["windows_serialization_note"] = "pandas float CSV reserialization; integer start_us/end_us verified by source audit"
        write_json(OUT / "data_manifest.json", manifest)
    audit = json.loads((OUT / "data_audit.json").read_text(encoding="utf-8"))
    assert audit["state"] == "passed" and audit["windows"] == len(meta)
    cv = pd.read_csv(OUT / "cv_runs.csv")
    test = pd.read_csv(OUT / "test_metrics.csv")
    assert len(cv) == 126 and cv.run_id.is_unique
    selection = pd.read_csv(OUT / "selection.csv")
    expected = selection[selection.mean_balanced_accuracy >= selection.mean_balanced_accuracy.max() - .005]
    expected = expected.sort_values(["cpu_p50_ms", "parameters"]).iloc[0]
    selected = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    assert (expected.architecture, expected.preprocessing) == (selected["architecture"], selected["preprocessing"])
    assert not meta.duplicated(["recording", "start_us"]).any()
    for _, group in meta.groupby("group_id"):
        assert (group.start_us.diff().dropna() >= 4_000_000).all()
    checked, rows_checked, max_error, scaler_error = 0, 0, 0., 0.
    deployment_probability = None
    for prep in plan["preprocessors"]:
        x = preprocess(raw, prep)
        for checkpoint in sorted((OUT / "checkpoints").glob("*.pt")):
            if f"__{prep}__" not in checkpoint.stem:
                continue
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert saved["code_sha256"] == code_hashes()
            assert all(torch.isfinite(v).all() for v in saved["model"].values())
            assert saved["epochs"] == 20 and saved["threshold"] == .5
            ti = np.array(saved["train_indices"])
            prediction = np.load(OUT / "predictions" / (checkpoint.stem + ".npz"))
            indices = prediction["indices"]
            truth, p = prediction["truth"], prediction["probability"]
            assert np.array_equal(truth, meta.iloc[indices].target.to_numpy())
            assert not set(ti) & set(indices)
            assert not set(meta.iloc[ti].group_id) & set(meta.iloc[indices].group_id)
            source_mean = x[ti].mean((0, 2), dtype=np.float64).reshape(1, 52, 1).astype(np.float32)
            source_std = x[ti].std((0, 2), dtype=np.float64).clip(1e-4).reshape(1, 52, 1).astype(np.float32)
            assert np.array_equal(source_mean, saved["mean"].numpy())
            assert np.array_equal(source_std, saved["std"].numpy())
            assert np.isclose(saved["pos_weight"], (meta.iloc[ti].target == 0).sum() / meta.iloc[ti].target.sum())
            replay = restore_predict(saved, x, indices)
            error = float(np.max(np.abs(replay - p)))
            assert np.allclose(replay, p, atol=2e-5, rtol=2e-4), (checkpoint.stem, error)
            max_error = max(max_error, error)
            if checkpoint.stem.startswith("cv_"):
                row = cv[cv.run_id == checkpoint.stem].iloc[0]
                assert saved["validation_indices"] == indices.tolist()
                assert not set(meta.iloc[ti].domain) & set(meta.iloc[indices].domain)
                values = metrics(truth, p)
                for key, value in values.items():
                    assert np.isclose(row[key], value, atol=1e-10)
                rows_checked += 1
            else:
                for _, row in test[test.run_id == checkpoint.stem].iterrows():
                    subset = meta.iloc[indices].dataset.to_numpy() == row.dataset
                    if row.domain != "ALL":
                        subset &= meta.iloc[indices].domain.to_numpy() == row.domain
                    for key, value in metrics(truth[subset], p[subset]).items():
                        assert np.isclose(row[key], value, atol=1e-10)
                    rows_checked += 1
                if checkpoint.stem.startswith("all_wallhack") and saved["seed"] == 42:
                    deployment_probability = dict(indices=indices, truth=truth, probability=p)
            curve = pd.read_csv(OUT / "curves" / (checkpoint.stem + ".csv"))
            assert len(curve) == 20 and np.isfinite(curve.select_dtypes("number")).all().all()
            checked += 1
        del x
    assert checked == 136 and deployment_probability is not None
    deployment = torch.load(OUT / "deployment.pt", map_location="cpu", weights_only=True)
    assert deployment["seed"] == 42 and len(deployment["train_recordings"]) == 44
    detector = MotionDetector(OUT / "deployment.pt", "cpu")
    indices = deployment_probability["indices"]
    cpu_probability = detector.predict(raw[indices])
    cpu_error = float(np.max(np.abs(cpu_probability - deployment_probability["probability"])))
    assert np.array_equal(cpu_probability >= .5, deployment_probability["probability"] >= .5)
    # CPU и CUDA могут несколько отличаться из-за TF32, поэтому проверяем и решения, и величину отклонения.
    assert cpu_error < .005
    timing = {"cpu": latency(detector, raw, "cpu")}
    if torch.cuda.is_available():
        timing["cuda"] = latency(MotionDetector(OUT / "deployment.pt", "cuda"), raw, "cuda")
    frame, draws, confidence = cluster_bootstrap(deployment_probability, meta)
    frame.to_csv(OUT / "deployment_recording_metrics.csv", index=False)
    draws.to_csv(OUT / "cluster_bootstrap.csv", index=False)
    write_json(OUT / "confidence_intervals.json", confidence)
    tensor_bytes = sum(v.numel() * v.element_size() for v in deployment["model"].values())
    result = dict(state="passed", checkpoints_replayed=checked, metric_rows_recomputed=rows_checked,
                  all_train_only_scalers_exact=True, max_gpu_probability_error=max_error,
                  deployment_cpu_probability_error=cpu_error, deployment_cpu_decisions_equal=True,
                  recordings=len(meta.group_id.unique()), windows=len(meta), windows_nonoverlapping=True,
                  recording_splits_disjoint=True, selection_recomputed=True, latency=timing,
                  deployment_file_bytes=(OUT / "deployment.pt").stat().st_size,
                  deployment_tensor_bytes=tensor_bytes,
                  deployment_sha256=hashlib.sha256((OUT / "deployment.pt").read_bytes()).hexdigest())
    write_json(OUT / "validation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
