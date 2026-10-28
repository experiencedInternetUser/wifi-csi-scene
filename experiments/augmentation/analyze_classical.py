"""Matched classical-model metrics and verification of scale cancellation."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from experiments.robustness.modeling import predict
from .analyze import METRICS
from .classical import CLS, MODELS, metadata, relative
from .run import OUT, ROOT, write_csv, write_json


def analyze():
    data = pd.concat([pd.read_csv(CLS / "source_cv.csv"), pd.read_csv(CLS / "evaluation.csv")], ignore_index=True)
    summary = []
    for (stage, domain, name, policy), group in data.groupby(["stage", "domain", "name", "policy"]):
        row = dict(stage=stage, domain=domain, name=name, policy=policy, runs=len(group))
        for metric in METRICS:
            for label, value in [("mean", group[metric].mean()), ("sd", group[metric].std()), ("min", group[metric].min()), ("max", group[metric].max())]:
                row[metric + "_" + label] = float(value)
        summary.append(row)
    summary = pd.DataFrame(summary)
    write_csv(CLS / "summary.csv", summary)
    comparisons = []
    decisions = []
    for name in MODELS:
        for policy in ("gain", "spectral", "noise", "combined"):
            safety = []
            for stage, domain in [("source_cv", "ALL"), ("wallhack_holdout", "ALL"), ("3do_final", "ALL"), ("opera_zero_shot", "ALL"), ("opera_zero_shot", "Room2/ALL")]:
                q = summary[(summary.name == name) & (summary.stage == stage) & (summary.domain == domain)].set_index("policy")
                row = dict(name=name, policy=policy, stage=stage, domain=domain)
                for metric in METRICS:
                    delta = float(q.loc[policy, metric + "_mean"] - q.loc["none", metric + "_mean"])
                    row[metric + "_delta"] = delta
                    benefit = -delta if metric == "fpr" else delta
                    row[metric + "_strict"] = benefit >= -1e-12
                    row[metric + "_tolerance"] = benefit >= (-0.01 if metric in ("recall", "fpr") else -0.005) - 1e-12
                row["strict_all"] = all(row[m + "_strict"] for m in METRICS)
                row["tolerance_all"] = all(row[m + "_tolerance"] for m in METRICS)
                comparisons.append(row)
                safety.append(row)
            q = summary[(summary.name == name) & (summary.stage == "days_zero_shot") & (summary.domain == "ALL")].set_index("policy")
            delta = float(q.loc[policy, "balanced_accuracy_mean"] - q.loc["none", "balanced_accuracy_mean"])
            fpr_gain = float(q.loc["none", "fpr_mean"] - q.loc[policy, "fpr_mean"])
            practical = delta >= 0.05 and fpr_gain >= 0.10 and q.loc[policy, "roc_auc_mean"] >= q.loc["none", "roc_auc_mean"]
            strict = all(row["strict_all"] for row in safety)
            tolerant = all(row["tolerance_all"] for row in safety)
            decisions.append(dict(name=name, policy=policy, days_ba_gain=delta, days_fpr_reduction=fpr_gain, days_practical_gain=bool(practical), strict_preservation=strict, tolerant_preservation=tolerant, combined_strict=bool(practical and strict), combined_tolerance=bool(practical and tolerant)))
    selected = json.loads((CLS / "locked_selection.json").read_text(encoding="utf8"))
    write_csv(CLS / "preservation.csv", comparisons)
    write_json(CLS / "assessment.json", dict(selected_source_only=selected, decisions=decisions, selected=next(r for r in decisions if r["name"] == selected["name"] and r["policy"] == selected["policy"])))


def verify():
    torch.set_num_threads(1)
    raw, rel = relative("source")
    meta = metadata("source")
    paths = list((CLS / "checkpoints").glob("*.joblib"))
    assert len(paths) == 165
    max_error = 0.0
    for path in paths:
        saved = joblib.load(path)
        assert set(meta.iloc[saved["train_indices"]].dataset) == {"Wallhack"}
        assert saved["threshold"] == 0.5
        sample = np.array([0, 142, 1000])
        p = predict(saved, raw[sample], device="cpu")
        assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
        if saved["policy"] == "gain":
            control = joblib.load(path.with_name(path.name.replace("__gain__", "__none__")))
            q = predict(control, raw[sample], device="cpu")
            error = float(np.max(np.abs(p - q)))
            max_error = max(max_error, error)
            assert error < 1e-12
    old = ROOT / "experiments/robustness/results"
    agreement = []
    for name in MODELS:
        for stage in ("wallhack_holdout", "3do_final", "days_zero_shot", "opera_zero_shot"):
            train = "domain_holdout" if stage == "wallhack_holdout" else "all_wallhack"
            a = pd.read_csv(CLS / "predictions" / f"{train}__{name}__none__s42__{stage}.csv")
            b = pd.read_csv(old / "predictions" / f"{train}__{name}__s42__{stage}.csv")
            assert a["index"].tolist() == b["index"].tolist()
            agreement.append(dict(name=name, stage=stage, max_probability_difference=float(np.max(np.abs(a.probability - b.probability))), changed_decisions=int((a.prediction != b.prediction).sum())))
    for p in (CLS / "predictions").glob("*.csv"):
        q = pd.read_csv(p)
        assert np.array_equal(q.prediction, (q.probability >= 0.5).astype(int))
    write_csv(CLS / "control_agreement.csv", agreement)
    write_json(CLS / "verification.json", dict(status="passed", checkpoints=165, source_only=True, gain_invariance_max_probability_error=max_error, prediction_files=len(list((CLS / "predictions").glob("*.csv")))))


def paired():
    selected = json.loads((CLS / "locked_selection.json").read_text(encoding="utf8"))
    rows = []
    for stage in ("wallhack_holdout", "3do_final", "days_zero_shot", "opera_zero_shot"):
        training = "domain_holdout" if stage == "wallhack_holdout" else "all_wallhack"
        prefix = f"{training}__{selected['name']}__"
        a = pd.read_csv(CLS / "predictions" / f"{prefix}{selected['policy']}__s42__{stage}.csv")
        b = pd.read_csv(CLS / "predictions" / f"{prefix}none__s42__{stage}.csv")
        assert a["index"].tolist() == b["index"].tolist()
        for cluster in (["group_id", "day"] if stage == "days_zero_shot" else ["experiment" if stage == "opera_zero_shot" else "group_id"]):
            y, pa, pb, groups = a.target.to_numpy(), a.prediction.to_numpy(), b.prediction.to_numpy(), a[cluster].to_numpy()
            unique = np.unique(groups)
            counts = []
            for group in unique:
                m = groups == group
                counts.append([int(np.sum(m & (y == yy) & (pred == pp))) for pred in (pa, pb) for yy, pp in [(0, 0), (0, 1), (1, 0), (1, 1)]])
            counts = np.asarray(counts)
            rng = np.random.default_rng(20261028)
            sums = counts[rng.integers(len(unique), size=(2000, len(unique)))].sum(1)
            tn, fp, fn, tp, tn0, fp0, fn0, tp0 = sums.T
            keep = ((tn + fp) > 0) & ((tp + fn) > 0)
            delta = 0.5 * ((tn[keep] / (tn[keep] + fp[keep]) + tp[keep] / (tp[keep] + fn[keep])) - (tn0[keep] / (tn0[keep] + fp0[keep]) + tp0[keep] / (tp0[keep] + fn0[keep])))
            rows.append(dict(name=selected["name"], policy=selected["policy"], stage=stage, cluster=cluster, clusters=len(unique), valid_repeats=int(keep.sum()), ba_delta_lo=float(np.quantile(delta, 0.025)), ba_delta_hi=float(np.quantile(delta, 0.975))))
    write_csv(CLS / "paired_intervals.csv", rows)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        analyze()
        paired()
        verify()
    print((CLS / "assessment.json").read_text(encoding="utf8"))
