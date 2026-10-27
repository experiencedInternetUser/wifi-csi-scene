"""Simple models vs the team NN grid on the same 16 scenarios.

Usage: python -m csi.compare_grid
Writes results/compare_simple_vs_nn.csv and figures/compare_simple_vs_nn.png.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from csi.baselines import GRID, OUT, summary
from csi.format import ROOT
from csi.plotstyle import INK2, MUTED, SEQ, plt

GROUPS = {
    "3DO internal": lambda s: s.startswith("td_one") or s.startswith("td_lodo"),
    "Wallhack 1->3": lambda s: s.startswith("wh_one"),
    "Wallhack 3->1": lambda s: s.startswith("wh_lodo"),
    "Wallhack->3DO": lambda s: s == "wh_to_td",
    "3DO->Wallhack": lambda s: s == "td_to_wh",
}


def nn_summary() -> pd.DataFrame:
    res = pd.read_csv(GRID / "nn_results.csv")
    res["far"] = res.fp / res.n_negative
    multi = res[res.target_domain == "ALL"]
    single = res[~res.run_id.isin(multi.run_id)]
    s = pd.concat([multi, single])
    s["model"] = "nn_" + s.architecture
    s["input"] = s.preprocessing
    return s.groupby(["scenario", "input", "model"])[["balanced_accuracy", "roc_auc", "far", "f1"]].mean().reset_index()


def group_of(s: str) -> str:
    return next(g for g, f in GROUPS.items() if f(s))


def main():
    simple = summary(pd.read_csv(OUT))
    nn = nn_summary()
    allr = pd.concat([simple, nn], ignore_index=True)
    allr["group"] = allr.scenario.map(group_of)
    allr.to_csv(ROOT / "results/compare_simple_vs_nn.csv", index=False)

    configs = [("threshold", "clean_abs"), ("logreg", "raw_abs"), ("logreg", "clean_abs"), ("lightgbm", "clean_abs"),
               ("nn_tinycnn", "lp20_centered"), ("nn_transformer", "lp20_centered"),
               ("nn_cnngru", "lp20_centered"), ("nn_tcn", "lp20_centered")]
    pd.set_option("display.width", 220)
    for metric in ["balanced_accuracy", "roc_auc", "far"]:
        t = pd.DataFrame({f"{m}|{i}": allr[(allr.model == m) & (allr.input == i)].groupby("group")[metric].mean()
                          for m, i in configs}).T[list(GROUPS)]
        t["Wallhack all (10)"] = [allr[(allr.model == m) & (allr.input == i) & allr.scenario.str.contains("wh")].query(
            "group != 'Wallhack->3DO'")[metric].mean() for m, i in configs]
        print(f"\n== {metric} (mean over scenarios in group)\n", t.round(3).to_string())

    # best config per family overall
    print("\n== best input per simple model (mean balanced acc over the 10 Wallhack-target scenarios)")
    wh = allr[allr.scenario.str.startswith("wh_") & (allr.scenario != "wh_to_td") | (allr.scenario == "td_to_wh")]
    print(wh.groupby(["model", "input"]).balanced_accuracy.mean().unstack().round(3).to_string())

    # figure: per scenario balanced accuracy, selected configs
    scen = [s for g in GROUPS for s in sorted(allr.scenario.unique()) if group_of(s) == g]
    m = np.array([[allr[(allr.model == mo) & (allr.input == i) & (allr.scenario == s)].balanced_accuracy.mean()
                   for s in scen] for mo, i in configs])
    fig, ax = plt.subplots(figsize=(13, 4.2))
    im = ax.imshow(m, cmap=SEQ, vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if m[i, j] > 0.8 else "#0b0b0b")
    ax.set_yticks(range(len(configs)), [f"{mo.replace('nn_', 'NN ')} | {i}" for mo, i in configs])
    ax.set_xticks(range(len(scen)), scen, rotation=45, ha="right")
    ax.axhline(3.5, color="white", lw=3)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="balanced accuracy", shrink=0.8)
    ax.set_title("Simple models (top, 3 seeds for LightGBM) vs team NN grid (bottom, 1 seed) — same windows and splits")
    fig.savefig(ROOT / "figures/compare_simple_vs_nn.png")
    print("saved figures/compare_simple_vs_nn.png")


if __name__ == "__main__":
    main()
