"""Empty room vs a motionless person (lying / sitting / at rest), binary, dynamic features.

  3DO:      leave-one-day-out, empty vs lie and empty vs sit (also a variance-only baseline)
  OPERAnet: empty vs still, train on one person (+ one background recording), test on the other
  cross:    trained on 3DO (empty vs sit+lie, ESP32) -> OPERAnet empty vs still (Intel 5300), no retraining

Needs data/processed/{3do,operanet}/activity_windows.parquet (python -m csi.activity; python -m csi.operanet_activity).
Usage: python -m csi.presence_check  -> results/presence_empty_vs_still.csv
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from csi.activity import feature_sets
from csi.format import PROCESSED, ROOT


def fit_score(tr_x, tr_y, te_x):
    m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(tr_x, tr_y)
    return m.predict_proba(te_x)[:, 1]


def row(name, test, y, p, var=None):
    pred = p >= 0.5
    out = {"task": name, "test": test, "n_empty": int((~y).sum()), "n_person": int(y.sum()),
           "roc_auc": roc_auc_score(y, p), "recall_empty": float((~pred[~y]).mean()),
           "recall_person": float(pred[y].mean())}
    if var is not None:
        out["roc_auc_variance_only"] = roc_auc_score(y, var)
    return out


def main():
    td = pd.read_parquet(PROCESSED / "3do" / "activity_windows.parquet")
    op = pd.read_parquet(PROCESSED / "operanet" / "activity_windows.parquet")
    cols = feature_sets(td)["dyn"]
    rows = []
    for other in ["lie", "sit"]:
        d = td[td.label.isin(["empty", other])]
        for day in sorted(d.domain.unique()):
            tr, te = d[d.domain != day], d[d.domain == day]
            y = (te.label == other).to_numpy()
            rows.append(row(f"3do empty vs {other}", f"day {day}", y,
                            fit_score(tr[cols], tr.label == other, te[cols]), te["log_var"].to_numpy()))
    d = op[op.label.isin(["empty", "still"])].copy()
    d.loc[d.exp == "001", "person_id"] = "One"   # background recordings go with the nearest session
    d.loc[d.exp == "019", "person_id"] = "Five"
    for test in ["One", "Five"]:
        tr, te = d[d.person_id != test], d[d.person_id == test]
        y = (te.label == "still").to_numpy()
        rows.append(row("operanet empty vs still", f"person {test}", y,
                        fit_score(tr[cols], tr.label == "still", te[cols]), te["log_var"].to_numpy()))
    s = td[td.label.isin(["empty", "lie", "sit"])]
    y = (d.label == "still").to_numpy()
    rows.append(row("3do (ESP32) -> operanet (Intel 5300) empty vs still", "operanet all", y,
                    fit_score(s[cols], s.label != "empty", d[cols]), d["log_var"].to_numpy()))
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results/presence_empty_vs_still.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string())
    med = pd.concat([td[td.label.isin(["empty", "lie", "sit"])].assign(ds="3do"),
                     op[op.label.isin(["empty", "still"])].assign(ds="operanet")])
    print("\nmedian pc2_share (share of variance in the 2nd principal component over subcarriers):")
    print(med.groupby(["ds", "label"]).pc2_share.median().round(3).to_string())


if __name__ == "__main__":
    main()
