"""Сетка «аугментация x предобработка x архитектура» на тех же сценариях, что и базовая сетка.

Положите файл рядом с models.py и train_grid.py (experiments/). Файлы базовой сетки не меняются,
поэтому проверка хешей в train_grid.verify_code остаётся рабочей.

Что совпадает с базой: сценарии из plan.json, модели из models.py, AdamW, lr, косинусное расписание,
20 эпох, batch 64, pos_weight по источнику, порог 0.5, веса последней эпохи, без валидации,
стандартизация по поднесущим по НЕаугментированному train, метрики из train_grid.metrics.
Что отличается: несколько seed, аугментация сырой амплитуды на лету (на GPU), результаты пишутся в
experiments/results_aug/, TF32 для matmul выключен (нужно для точного ФНЧ; cuDNN TF32 как в базе).

Примеры:
  python train_aug_grid.py --selfcheck
  python train_aug_grid.py                          # первый раунд: по умолчанию, 3 seed
  python train_aug_grid.py --augs none snr --preps relative --archs tcn --seeds 42 43 44 45 46
  python train_aug_grid.py --families all --augs none all   # проверка «ничего не сломали» на 3DO
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import aug_torch as AT
from models import make_model, parameter_count
from train_grid import metrics, save_csv, write_json   # те же метрики, что в базовой сетке

EPOCHS, BATCH = 20, 64
LR = {"transformer": 3e-4}
DEFAULT_PREPS = ("logamp", "relative", "lp20_centered")
DEFAULT_ARCHS = ("tinycnn", "tcn")
DEFAULT_AUGS = ("none", "calib", "snr", "snr+warp", "core")
DEFAULT_FAMILIES = ("Wallhack 1→3", "Wallhack 3→1", "3DO→Wallhack")   # где у базы есть запас по качеству


def fit_predict(arch, prep_fn, raw_train, train_y, mean, std, test_x, aug_names, strength, p,
                seed, epochs, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    model = make_model(arch).to(device)
    lr = LR.get(arch, 1e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr * 0.01)
    pos = float(train_y.sum().item())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(train_y) - pos) / pos], device=device))
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    n, losses = len(train_y), []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n, device=device)
        total = torch.zeros((), device=device)
        for left in range(0, n, BATCH):
            ids = order[left:left + BATCH]
            with torch.no_grad():
                xb = (prep_fn(AT.augment(raw_train[ids], aug_names, strength, p)) - mean) / std
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(xb).flatten(), train_y[ids])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += loss.detach() * len(ids)
        losses.append(float((total / n).item()))
        assert math.isfinite(losses[-1])
        sched.step()
    model.eval()
    out = []
    with torch.no_grad():
        for left in range(0, len(test_x), BATCH):
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                out.append(torch.sigmoid(model(test_x[left:left + BATCH]).flatten().float()).cpu().numpy())
    return np.concatenate(out), losses, parameter_count(model)


def main(a):
    root = Path(a.root) if a.root else Path(__file__).resolve().parents[1]
    res = root / "experiments/results"
    out = root / "experiments/results_aug"
    for d in ("", "predictions"):
        (out / d).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False      # точный ФНЧ; в базе было True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(False)

    meta = pd.read_csv(res / "windows.csv")
    plan = json.loads((res / "plan.json").read_text(encoding="utf8"))
    amp = torch.from_numpy(np.load(res / "amplitudes.npy")).to(device)
    y_all = torch.as_tensor(meta.target.to_numpy(), dtype=torch.float32, device=device)
    assert amp.shape == (len(meta), AT.CHANNELS, AT.SAMPLES)

    scenarios = [s for s in plan["scenarios"]
                 if (a.families == ["all"] or s["family"] in a.families)
                 and (not a.scenarios or any(k in s["id"] for k in a.scenarios))]
    jobs_total = len(scenarios) * len(a.preps) * len(a.archs) * len(a.augs) * len(a.strengths) * len(a.seeds)
    runs_p, res_p = out / "runs_aug.csv", out / "results_aug.csv"
    runs = pd.read_csv(runs_p).to_dict("records") if runs_p.exists() else []
    results = pd.read_csv(res_p).to_dict("records") if res_p.exists() else []
    done = {r["run_id"] for r in runs if r["status"] == "complete"}
    print(f"Устройство: {device}; сценариев {len(scenarios)}; обучений в этом запуске до {jobs_total}; "
          f"уже готово {len(done)}", flush=True)

    for prep in a.preps:
        pp = AT.PP[prep]
        for sc in scenarios:
            tm = (meta.dataset == sc["train_dataset"]) & meta.domain.isin(sc["train_domains"])
            sm = (meta.dataset == sc["test_dataset"]) & meta.domain.isin(sc["test_domains"])
            assert not set(meta.loc[tm, "group_id"]) & set(meta.loc[sm, "group_id"])
            ti = torch.tensor(np.flatnonzero(tm), device=device)
            si = torch.tensor(np.flatnonzero(sm), device=device)
            raw_train, train_y = amp[ti], y_all[ti]
            with torch.no_grad():
                tr_pp = pp(raw_train)
                mean = tr_pp.mean((0, 2), keepdim=True)
                std = tr_pp.std((0, 2), unbiased=False, keepdim=True).clamp_min(1e-4)
                test_x = (pp(amp[si]) - mean) / std
                del tr_pp
            truth = meta.loc[sm, "target"].to_numpy(dtype=int)
            domains = meta.loc[sm, "domain"].to_numpy()
            targets = sc["test_domains"] + (["ALL"] if len(sc["test_domains"]) > 1 else [])
            for arch in a.archs:
                for aug in a.augs:
                    for st in ([1.0] if aug == "none" else a.strengths):
                        for seed in a.seeds:
                            run_id = f"{sc['id']}__{prep}__{arch}__{aug}__st{st:g}__s{seed}"
                            if run_id in done:
                                continue
                            t0 = time.perf_counter()
                            prob, losses, n_par = fit_predict(arch, pp, raw_train, train_y, mean, std, test_x,
                                                              AT.AUG_SETS[aug], st, a.p, seed, a.epochs, device)
                            sec = time.perf_counter() - t0
                            np.savez_compressed(out / "predictions" / (run_id + ".npz"),
                                                indices=np.flatnonzero(sm), truth=truth, probability=prob)
                            for tg in targets:
                                sel = np.ones(len(truth), bool) if tg == "ALL" else domains == tg
                                results.append(dict(run_id=run_id, scenario=sc["id"], family=sc["family"],
                                                    preprocessing=prep, architecture=arch, aug=aug, strength=st,
                                                    p=a.p, seed=seed, target_domain=tg,
                                                    test_dataset=sc["test_dataset"], **metrics(prob[sel], truth[sel])))
                            runs.append(dict(run_id=run_id, scenario=sc["id"], family=sc["family"],
                                             preprocessing=prep, architecture=arch, aug=aug, strength=st, p=a.p,
                                             seed=seed, epochs=a.epochs, train_windows=int(tm.sum()),
                                             test_windows=int(sm.sum()), n_parameters=n_par, train_seconds=sec,
                                             loss_first=losses[0], loss_last=losses[-1], status="complete"))
                            done.add(run_id)
                            save_csv(res_p, results)
                            save_csv(runs_p, runs)
                            write_json(out / "status.json", dict(completed=len(done), updated_utc=datetime.now(
                                timezone.utc).isoformat(), last_run=run_id))
                            print(f"{len(done)}: {run_id}; {sec:.1f} с; loss {losses[0]:.4f}→{losses[-1]:.4f}", flush=True)
            del raw_train, train_y, test_x, mean, std
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print("Готово.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Сетка аугментаций для CSI (детекция движения).")
    ap.add_argument("--root", help="корень проекта (по умолчанию родитель папки experiments)")
    ap.add_argument("--preps", nargs="+", default=list(DEFAULT_PREPS), choices=list(AT.PP))
    ap.add_argument("--archs", nargs="+", default=list(DEFAULT_ARCHS))
    ap.add_argument("--augs", nargs="+", default=list(DEFAULT_AUGS), choices=list(AT.AUG_SETS))
    ap.add_argument("--strengths", nargs="+", type=float, default=[1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES), help="'all' = все 16 сценариев")
    ap.add_argument("--scenarios", nargs="*", help="подстроки id сценариев, например wh_one td_to_wh")
    ap.add_argument("--p", type=float, default=0.5, help="вероятность каждой аугментации для каждого окна")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        AT.selfcheck("cuda" if torch.cuda.is_available() else "cpu")
    else:
        main(args)
