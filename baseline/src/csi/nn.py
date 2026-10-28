"""Multiclass neural nets for activity types, on the same splits as csi.activity.

Architectures follow the team binary grid (Causal TCN 32 ch, dilations 1..64; CNN-GRU 32/64 + GRU 64),
with a multiclass head. Input: `clean_lp_centered` windows [52, 400], per-subcarrier standardization
fitted on the training part only. AdamW, cosine LR, class-weighted cross-entropy, 20 epochs, last epoch.

Usage: python -m csi.nn [--datasets wallhack 3do] [--archs tcn cnngru] [--seeds 42]
Writes results/activity_nn_metrics.csv (+ confusions).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from torch import nn
from torch.nn import functional as Fn

from csi.activity import splits
from csi.format import PROCESSED, ROOT, load_index
from csi.labels import DATASET_TO_GLOBAL
from csi.preprocess import PRESETS, preprocess_recording
from csi.recording import from_processed

LABELS = {"wallhack": ["empty", "walk", "walk_wave"], "3do": ["empty", "walk", "sit", "lie"]}


# ---------------------------------------------------------------- data

def window_arrays(dataset: str, preset: str = "clean_lp_centered"):
    """Cached [N, S, W] float16 windows + meta aligned with activity_windows.parquet (same starts)."""
    path = PROCESSED / dataset / f"nn_windows_{preset}.npz"
    meta_path = PROCESSED / dataset / f"nn_windows_{preset}.parquet"
    if path.exists():
        return np.load(path)["X"], pd.read_parquet(meta_path)
    X, rows = [], []
    for r in load_index(dataset).itertuples():
        rec = from_processed(dataset, r.recording_id)
        x, y, info = preprocess_recording(rec, PRESETS[preset])
        X.append(x[:, 0].astype(np.float16))
        for i in range(len(x)):
            rows.append({"recording_id": r.recording_id, "domain": rec.meta["domain"], "session": r.session,
                         "start_s": info["start_s"][i], "y": int(y[i]),
                         "label": DATASET_TO_GLOBAL[dataset][int(y[i])]})
    X, meta = np.concatenate(X), pd.DataFrame(rows)
    np.savez(path, X=X)
    meta.to_parquet(meta_path, index=False)
    return X, meta


# ---------------------------------------------------------------- models

class CausalConv(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(ch, ch, 3, dilation=dilation)

    def forward(self, x):
        return self.conv(Fn.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.c1, self.c2 = CausalConv(ch, dilation), CausalConv(ch, dilation)
        self.n1, self.n2 = nn.LayerNorm(ch), nn.LayerNorm(ch)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        y = Fn.relu(self.n1(self.c1(x).transpose(1, 2)).transpose(1, 2))
        y = self.n2(self.c2(self.drop(y)).transpose(1, 2)).transpose(1, 2)
        return Fn.relu(x + self.drop(y))


class TCN(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.proj = nn.Conv1d(n_in, 32, 1)
        self.blocks = nn.Sequential(*[TCNBlock(32, d) for d in (1, 2, 4, 8, 16, 32, 64)])
        self.head = nn.Linear(32, n_out)

    def forward(self, x):
        return self.head(self.blocks(self.proj(x))[:, :, -1])


class CNNGRU(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(n_in, 32, 5, stride=2, padding=2), nn.ReLU(),
                                  nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU())
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, n_out))

    def forward(self, x):
        _, h = self.gru(self.conv(x).transpose(1, 2))
        return self.head(h[-1])


ARCHS = {"tcn": TCN, "cnngru": CNNGRU}


def train_eval(arch, Xtr, ytr, Xte, n_classes, seed, device, epochs=20, bs=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu = Xtr.astype(np.float64).mean(axis=(0, 2), keepdims=True)
    sd = np.maximum(Xtr.astype(np.float64).std(axis=(0, 2), keepdims=True), 1e-4)
    xtr = torch.tensor(((Xtr - mu) / sd).astype(np.float32))
    xte = torch.tensor(((Xte - mu) / sd).astype(np.float32))
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    counts = np.bincount(ytr, minlength=n_classes).astype(float)
    w = torch.tensor(np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / n_classes, 0.0),
                     dtype=torch.float32, device=device)
    model = ARCHS[arch](xtr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(xtr) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(xtr), generator=g).split(bs):
            xb, yb = xtr[idx].to(device), ytr_t[idx].to(device)
            loss = Fn.cross_entropy(model(xb), yb, weight=w)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(xte[i:i + 256].to(device)).cpu() for i in range(0, len(xte), 256)])
    return logits.argmax(1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["wallhack", "3do"])
    ap.add_argument("--archs", nargs="+", default=["tcn", "cnngru"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--families", nargs="+", default=["within", "lodo"])
    ap.add_argument("--tag", default="", help="suffix of the output files")
    args = ap.parse_args()
    out = ROOT / f"results/activity_nn_metrics{args.tag}.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)
    rows, confs = [], []
    for ds in args.datasets:
        labels = LABELS[ds]
        X, meta = window_arrays(ds)
        lab_id = {l: i for i, l in enumerate(labels)}
        y_all = meta.label.map(lab_id).to_numpy()
        for fam, name, tr, te in splits(meta, ds):
            if fam not in args.families:
                continue
            tr, te = tr.to_numpy(), te.to_numpy()
            if meta.loc[te, "label"].nunique() < 2 or meta.loc[tr, "label"].nunique() < len(labels):
                continue
            for arch in args.archs:
                for seed in args.seeds:
                    t0 = time.time()
                    pred = train_eval(arch, X[tr].astype(np.float32), y_all[tr], X[te].astype(np.float32),
                                      len(labels), seed, device)
                    yte = y_all[te]
                    present = sorted(set(yte))
                    rec = recall_score(yte, pred, labels=present, average=None, zero_division=0)
                    rows.append({"dataset": ds, "family": fam, "split": name, "features": "clean_lp_centered",
                                 "model": f"nn_{arch}", "seed": seed, "n_test": len(yte),
                                 "macro_f1": f1_score(yte, pred, labels=present, average="macro", zero_division=0),
                                 "balanced_acc": rec.mean(),
                                 **{f"recall_{labels[c]}": v for c, v in zip(present, rec)},
                                 "train_s": time.time() - t0})
                    cm = confusion_matrix(yte, pred, labels=range(len(labels)))
                    confs += [{"dataset": ds, "family": fam, "split": name, "model": f"nn_{arch}", "seed": seed,
                               "true": labels[i], "pred": labels[j], "n": int(cm[i, j])}
                              for i in range(len(labels)) for j in range(len(labels))]
                    print(f"{ds} {fam} {name} {arch} s{seed}: macroF1 {rows[-1]['macro_f1']:.3f} "
                          f"({rows[-1]['train_s']:.0f}s)", flush=True)
                    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(confs).to_csv(ROOT / f"results/activity_nn_confusions{args.tag}.csv", index=False)
    met = pd.DataFrame(rows)
    rec_cols = [c for c in met.columns if c.startswith("recall_")]
    print(met.groupby(["dataset", "family", "model"])[["macro_f1", "balanced_acc"] + rec_cols].mean().round(3)
          .to_string())


if __name__ == "__main__":
    main()
