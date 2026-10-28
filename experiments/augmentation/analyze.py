"""Paired comparisons and predeclared preservation criteria; no model selection on targets."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from .run import OUT, ROOT, write_csv, write_json

METRICS = ("balanced_accuracy", "roc_auc", "average_precision", "f1", "recall", "fpr")
SCENARIOS = ("wallhack_holdout", "3do_final", "days_zero_shot", "opera_zero_shot")


def aggregate():
    evaluation = pd.read_csv(OUT / "evaluation.csv")
    cv = pd.read_csv(OUT / "source_cv.csv")
    combined = pd.concat([evaluation, cv], ignore_index=True)
    # Source CV has three repeats per domain and nine repeats in the unweighted ALL summary.
    rows = []
    for (stage, domain, policy), group in combined.groupby(["stage", "domain", "policy"]):
        row = dict(stage=stage, domain=domain, policy=policy, runs=len(group))
        for metric in METRICS:
            row.update({metric + "_" + suffix: float(value) for suffix, value in [("mean", group[metric].mean()), ("sd", group[metric].std()), ("min", group[metric].min()), ("max", group[metric].max())]})
        rows.append(row)
    write_csv(OUT / "summary.csv", rows)
    return pd.DataFrame(rows)


def paired_intervals():
    rows = []
    for stage in SCENARIOS:
        training = "domain_holdout" if stage == "wallhack_holdout" else "all_wallhack"
        control = pd.read_csv(OUT / "predictions" / f"{training}__none__s42__{stage}.csv")
        for policy in ("gain", "spectral", "noise", "combined"):
            q = pd.read_csv(OUT / "predictions" / f"{training}__{policy}__s42__{stage}.csv")
            assert control["index"].tolist() == q["index"].tolist()
            assert control.target.tolist() == q.target.tolist()
            masks = {"ALL": np.ones(len(q), bool)}
            if stage == "opera_zero_shot":
                masks["Room2/ALL"] = q.room.to_numpy() == 2
            for domain, mask in masks.items():
                for cluster_col in (["group_id", "day"] if stage == "days_zero_shot" else ["experiment" if stage == "opera_zero_shot" else "group_id"]):
                    y = q.target.to_numpy()[mask]
                    a, b = q.prediction.to_numpy()[mask], control.prediction.to_numpy()[mask]
                    clusters = q[cluster_col].to_numpy()[mask]
                    unique = np.unique(clusters)
                    counts = []
                    for cl in unique:
                        cm = clusters == cl
                        row = []
                        for pred in (a, b):
                            row.extend([int(np.sum(cm & (y == yy) & (pred == pp))) for yy, pp in [(0, 0), (0, 1), (1, 0), (1, 1)]])
                        counts.append(row)
                    counts = np.asarray(counts)
                    rng = np.random.default_rng(20261028)
                    draw = counts[rng.integers(len(unique), size=(2000, len(unique)))].sum(1)
                    tn, fp, fn, tp, tn0, fp0, fn0, tp0 = draw.T
                    valid = ((tn + fp) > 0) & ((tp + fn) > 0)
                    def values(tn, fp, fn, tp):
                        return dict(balanced_accuracy=0.5 * (tn / (tn + fp) + tp / (tp + fn)), fpr=fp / (tn + fp), recall=tp / (tp + fn), f1=2 * tp / np.maximum(2 * tp + fp + fn, 1))
                    aa = values(tn[valid], fp[valid], fn[valid], tp[valid])
                    bb = values(tn0[valid], fp0[valid], fn0[valid], tp0[valid])
                    result = dict(stage=stage, domain=domain, policy=policy, seed=42, cluster=cluster_col, clusters=len(unique), valid_repeats=int(valid.sum()), negative_clusters=len(np.unique(clusters[y == 0])))
                    for metric in aa:
                        delta = aa[metric] - bb[metric]
                        result[metric + "_lo"] = float(np.quantile(delta, 0.025))
                        result[metric + "_hi"] = float(np.quantile(delta, 0.975))
                    rows.append(result)
    write_csv(OUT / "paired_intervals.csv", rows)


def preservation(summary):
    rows = []
    for stage in ("source_cv", *SCENARIOS):
        if stage == "days_zero_shot":
            continue
        for domain in (["ALL", "Room2/ALL"] if stage == "opera_zero_shot" else ["ALL"]):
            subset = summary[(summary.stage == stage) & (summary.domain == domain)]
            control = subset[subset.policy == "none"].iloc[0]
            for _, row in subset[subset.policy != "none"].iterrows():
                item = dict(stage=stage, domain=domain, policy=row.policy)
                for metric in METRICS:
                    delta = row[metric + "_mean"] - control[metric + "_mean"]
                    item[metric + "_delta"] = float(delta)
                    benefit = -delta if metric == "fpr" else delta
                    item[metric + "_strict"] = bool(benefit >= -1e-12)
                    item[metric + "_tolerance"] = bool(benefit >= (-0.01 if metric in ("recall", "fpr") else -0.005) - 1e-12)
                item["strict_all"] = all(item[m + "_strict"] for m in METRICS)
                item["tolerance_all"] = all(item[m + "_tolerance"] for m in METRICS)
                rows.append(item)
    write_csv(OUT / "preservation.csv", rows)
    rows = pd.DataFrame(rows)
    decisions = []
    days = summary[(summary.stage == "days_zero_shot") & (summary.domain == "ALL")]
    control = days[days.policy == "none"].iloc[0]
    for _, row in days[days.policy != "none"].iterrows():
        safety = rows[rows.policy == row.policy]
        gain = row.balanced_accuracy_mean - control.balanced_accuracy_mean
        reduction = control.fpr_mean - row.fpr_mean
        practical = gain >= 0.05 and reduction >= 0.10 and row.roc_auc_mean >= control.roc_auc_mean
        decisions.append(dict(policy=row.policy, days_ba_gain=float(gain), days_fpr_reduction=float(reduction), days_practical_gain=bool(practical), strict_preservation=bool(safety.strict_all.all()), tolerant_preservation=bool(safety.tolerance_all.all()), combined_strict=bool(practical and safety.strict_all.all()), combined_tolerance=bool(practical and safety.tolerance_all.all())))
    selected = json.loads((OUT / "locked_selection.json").read_text(encoding="utf8"))["policy"]
    write_json(OUT / "assessment.json", dict(selected_source_only=selected, decisions=decisions, selected=next(r for r in decisions if r["policy"] == selected), thresholds=dict(days_ba_gain=0.05, days_fpr_reduction=0.10, no_auc_decrease=True), target_training=False, target_threshold_tuning=False))


def controls_and_references():
    old = ROOT / "experiments/robustness/results"
    old_external = pd.read_csv(old / "external_zero_shot.csv")
    old_source = pd.read_csv(old / "source_tests.csv")
    references = pd.concat([
        old_external[old_external.name.isin(["tcn", "logreg_scaled", "logreg_pca8"]) & old_external.domain.eq("ALL")],
        old_source[old_source.name.isin(["tcn", "logreg_scaled", "logreg_pca8"]) & old_source.domain.eq("ALL") & old_source.stage.isin(["3do_final", "wallhack_holdout"])],
    ])
    write_csv(OUT / "reference_metrics.csv", references)
    rows = []
    for stage in SCENARIOS:
        training = "domain_holdout" if stage == "wallhack_holdout" else "all_wallhack"
        before = pd.read_csv(old / "predictions" / f"{training}__tcn__s42__{stage}.csv")
        after = pd.read_csv(OUT / "predictions" / f"{training}__none__s42__{stage}.csv")
        assert before["index"].tolist() == after["index"].tolist()
        error = np.abs(before.probability.to_numpy() - after.probability.to_numpy())
        rows.append(dict(stage=stage, mean_probability_error=float(error.mean()), max_probability_error=float(error.max()), changed_decisions=int((before.prediction != after.prediction).sum()), windows=len(after)))
    write_csv(OUT / "control_agreement.csv", rows)
    write_json(OUT / "environment.json", dict(python=sys.version, platform=platform.platform(), torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__))


def manifest():
    files = list((ROOT / "experiments/augmentation").rglob("*")) + list((ROOT / "research/augmentation_figures").glob("*.png")) + [ROOT / "research/augmentation_report.md"]
    rows = []
    for p in sorted(files):
        if not p.is_file() or "__pycache__" in p.parts or p.name == "artifact_manifest.csv":
            continue
        rows.append(dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    write_csv(OUT / "artifact_manifest.csv", rows)


if __name__ == "__main__":
    summary = aggregate()
    paired_intervals()
    preservation(summary)
    controls_and_references()
    manifest()
    print((OUT / "assessment.json").read_text(encoding="utf8"))
