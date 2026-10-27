"""Парный анализ эффекта аугментаций по experiments/results_aug/results_aug.csv.

Для каждого (сценарий, предобработка, архитектура, seed) берётся разность метрики с аугментацией и без неё
(aug='none' из ТОГО ЖЕ скрипта и тех же seed). Дальше: среднее по seed -> сценарий -> сводка по сценариям
с бутстрэп-интервалом (ресэмплинг сценариев) и долей сценариев, где аугментация помогла.

Метрики: roc_auc (ранжирование), balanced_accuracy (с порогом 0.5), fpr и fnr (направление ошибок).
Для fpr и fnr «лучше» = меньше, поэтому знак разности перевёрнут: положительное значение всегда означает улучшение.

  python analyze_aug.py                       # сводка по всем сценариям
  python analyze_aug.py --by-family           # отдельно по семействам сценариев
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["roc_auc", "balanced_accuracy", "fpr", "fnr"]
LOWER_IS_BETTER = {"fpr", "fnr"}


def load(path):
    df = pd.read_csv(path)
    df["fpr"] = df.fp / (df.fp + df.tn)
    df["fnr"] = df.fn / (df.fn + df.tp)
    # одна строка на запуск: общая метрика ALL, если у сценария несколько тестовых доменов
    df = df.sort_values("target_domain", key=lambda s: s != "ALL").drop_duplicates("run_id")
    return df


def paired(df, metric):
    key = ["scenario", "preprocessing", "architecture", "seed"]
    base = df[df.aug == "none"].set_index(key)[metric].rename("base")
    d = df[df.aug != "none"].join(base, on=key).dropna(subset=["base"])
    d["delta"] = d[metric] - d["base"]
    if metric in LOWER_IS_BETTER:
        d["delta"] = -d["delta"]
    return d


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    g = np.random.default_rng(seed)
    m = g.choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def summarize(df, metric, by=("preprocessing", "architecture", "aug", "strength")):
    d = paired(df, metric)
    # среднее по seed внутри сценария
    sc = d.groupby(list(by) + ["scenario"], as_index=False).delta.mean()
    rows = []
    for k, g in sc.groupby(list(by)):
        lo, hi = boot_ci(g.delta)
        rows.append(dict(zip(by, k), metric=metric, mean_delta=g.delta.mean(), ci_lo=lo, ci_hi=hi,
                         share_better=(g.delta > 0).mean(), n_scenarios=len(g)))
    return pd.DataFrame(rows)


def seed_noise(df):
    """Разброс базы по seed: с ним надо сравнивать эффект аугментаций."""
    b = df[df.aug == "none"]
    s = b.groupby(["scenario", "preprocessing", "architecture"]).balanced_accuracy.std().reset_index()
    return s.groupby(["preprocessing", "architecture"]).balanced_accuracy.mean().rename("seed_std_BA").reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="experiments/results_aug/results_aug.csv")
    ap.add_argument("--by-family", action="store_true")
    ap.add_argument("--out", default=None, help="куда сохранить сводку (csv)")
    a = ap.parse_args()
    df = load(a.csv)
    print(f"запусков: {len(df)}, seed: {sorted(df.seed.unique())}, сценариев: {df.scenario.nunique()}")
    print("\nШум базы по seed (std balanced accuracy, среднее по сценариям):")
    print(seed_noise(df).round(3).to_string(index=False))
    groups = [(None, df)] if not a.by_family else list(df.groupby("family"))
    frames = []
    for fam, g in groups:
        for m in METRICS:
            t = summarize(g, m)
            if fam is not None:
                t.insert(0, "family", fam)
            frames.append(t)
    full = pd.concat(frames, ignore_index=True)
    pd.set_option("display.width", 220)
    for m in METRICS:
        print(f"\n=== Δ{m} (положительное = аугментация лучше), парно с aug=none ===")
        print(full[full.metric == m].drop(columns="metric").round(3).to_string(index=False))
    if a.out:
        full.to_csv(a.out, index=False)
