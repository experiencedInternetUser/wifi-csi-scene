"""Numerical checks for augmentations, source-only splits and stored predictors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.signal import freqz, firwin

from experiments.robustness.modeling import predict
from .augment import Augmenter, POLICIES
from .run import CACHE, OUT, ROOT, evaluate_prepared, metadata, prepared, setup, write_json


def transforms():
    setup()
    g = torch.Generator(device="cuda").manual_seed(9101)
    x = torch.randn((128, 52, 400), generator=g, device="cuda")
    x -= x.mean(-1, keepdim=True)
    original = x.clone()
    means = torch.full((128, 1, 1), 12.0, device="cuda")
    rows = []
    for policy in POLICIES:
        a = Augmenter(policy, 42)(x, means)
        b = Augmenter(policy, 42)(x, means)
        assert torch.equal(a, b), policy
        assert torch.equal(x, original)
        assert a.shape == x.shape and torch.isfinite(a).all()
        unchanged = torch.eq(a, x).all((1, 2)).float().mean().item()
        if policy == "none":
            assert torch.equal(a, x)
        else:
            assert 0.30 < unchanged < 0.70
        mean_error = (a.mean(-1).abs().max()).item()
        assert mean_error < 2e-5, (policy, mean_error)
        if policy == "gain":
            ratio = (a * x).sum((1, 2)) / x.square().sum((1, 2))
            residual = (a - ratio[:, None, None] * x).abs().max().item()
            assert residual < 1e-4 and ratio.min() >= 0.25 and ratio.max() <= 16
        rows.append(dict(policy=policy, unchanged_fraction=unchanged, max_channel_mean=mean_error))
    coefficients = firwin(41, 10, fs=100, window="hann")
    f, h = freqz(coefficients, worN=8192, fs=100)
    energy_above20 = float(np.square(np.abs(h[f > 20])).sum() / np.square(np.abs(h)).sum())
    assert energy_above20 < 0.001
    # A zero residual tests that added noise is centered and within its RMS limit.
    noisy = Augmenter("noise", 42)(torch.zeros_like(x), means)
    rms = noisy.square().mean((1, 2)).sqrt() / 12
    assert rms.max() <= 0.060001
    write_json(OUT / "transform_verification.json", dict(status="passed", policies=rows, fir_energy_fraction_above20=energy_above20, noise_relative_rms_max=float(rms.max())))
    print("Transform invariants verified", flush=True)


def final():
    setup()
    source_meta = metadata("source")
    models = sorted((OUT / "checkpoints").glob("*.joblib"))
    assert len(models) == 75
    raw, clean = prepared("source")
    errors = []
    groups = []
    for path in models:
        saved = joblib.load(path)
        ids = saved["train_indices"]
        assert set(source_meta.iloc[ids].dataset) == {"Wallhack"}
        assert not set(saved["train_groups"]) & set(saved["validation_groups"])
        assert saved["threshold"] == 0.5 and saved["epochs"] == 20
        assert all(torch.isfinite(t).all() for t in saved["model"].values())
        projection = saved["projection"]
        expected_mu = clean[ids].mean((0, 2), dtype=np.float64)[None, :, None].astype(np.float32)
        expected_std = clean[ids].std((0, 2), dtype=np.float64).clip(1e-4)[None, :, None].astype(np.float32)
        assert np.array_equal(projection["mean"], expected_mu)
        assert np.array_equal(projection["std"], expected_std)
        sample = np.array([0, 142, 1000])
        a = predict(saved, raw[sample])
        b = evaluate_prepared(saved, clean[sample])
        errors.append(float(np.max(np.abs(a - b))))
        assert np.allclose(a, b, atol=2e-6)
        groups.append(dict(run_id=path.stem, train_groups=len(saved["train_groups"])))
    counts = []
    for path in (OUT / "predictions").glob("*.csv"):
        q = pd.read_csv(path)
        assert np.isfinite(q.probability).all() and q.probability.between(0, 1).all()
        assert np.array_equal(q.prediction.to_numpy(), (q.probability.to_numpy() >= 0.5).astype(int))
        counts.append(dict(file=path.name, windows=len(q)))
    write_json(OUT / "verification.json", dict(status="passed", checkpoints=len(models), max_cached_vs_raw_probability_error=max(errors), prediction_files=len(counts), source_only=True, source_projection_verified=True))
    print("Final verification passed", len(models), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["transforms", "final"])
    args = parser.parse_args()
    globals()[args.stage]()
