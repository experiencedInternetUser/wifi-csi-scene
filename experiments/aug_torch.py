"""Аугментации сырой амплитуды CSI и предобработка в torch (на GPU, по батчам).

Окно: тензор [B, 52, 400] с амплитудами в исходных единицах устройства (как в amplitudes.npy).
Предобработки PP повторяют models.PREPROCESSORS, но работают в torch, потому что аугментация
должна идти ДО предобработки (она имитирует физический сдвиг сырого сигнала).

Порядок в обучении: сырая амплитуда -> augment() -> PP[name] -> стандартизация по статистикам
неаугментированного train (как в базовой сетке) -> модель.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F
from scipy.signal import butter, sosfilt, sosfilt_zi

CHANNELS, SAMPLES, FS = 52, 400, 100.0


# ------------------------------------------------------------------ предобработка
def pp_logamp(x):
    return torch.log1p(x)


def pp_relative(x):
    return x / x.mean(-1, keepdim=True).clamp_min(1.0) - 1.0


def pp_relative_diff(x):
    r = pp_relative(x)
    return torch.diff(r, dim=-1, prepend=r[..., :1])


_LP = {}


def _lp20_operator(device, dtype):
    """ФНЧ Баттерворта как линейный оператор на окне: y = x @ H.T + x[..., :1] * h_zi.
    Точно воспроизводит sosfilt с начальным состоянием sosfilt_zi * x[0] (проверено: отн. ошибка ~3e-7)."""
    key = (str(device), dtype)
    if key not in _LP:
        sos = butter(4, 20.0, btype="lowpass", fs=FS, output="sos")
        H = sosfilt(sos, np.eye(SAMPLES), axis=0)                       # столбец j: отклик на импульс в j
        hz = sosfilt(sos, np.zeros(SAMPLES), zi=sosfilt_zi(sos))[0]     # вклад начального состояния
        _LP[key] = (torch.tensor(H.T, dtype=dtype, device=device), torch.tensor(hz, dtype=dtype, device=device))
    return _LP[key]


def pp_lp20_centered(x):
    HT, hz = _lp20_operator(x.device, x.dtype)
    y = x @ HT + x[..., :1] * hz
    return y - y.mean(-1, keepdim=True)


PP = {"logamp": pp_logamp, "relative": pp_relative,
      "relative_diff": pp_relative_diff, "lp20_centered": pp_lp20_centered}


# ------------------------------------------------------------------ аугментации (батч, у каждого окна свои параметры)
def _on(B, p, dev):
    return torch.rand(B, device=dev) < p


def _u(n, lo, hi, dev):
    return lo + (hi - lo) * torch.rand(n, device=dev)


def a_gain(x, s, p):
    """Общий множитель амплитуды окна (AGC, мощность). Для relative это no-op."""
    B = x.shape[0]
    f = torch.exp(torch.randn(B, device=x.device) * 0.3 * s)
    f = torch.where(_on(B, p, x.device), f, torch.ones_like(f))
    return x * f[:, None, None]


def a_profile(x, s, p):
    """Гладкий профиль усиления по поднесущим (дрейф статической многолучёвости). Для relative это no-op."""
    B, Fc, _ = x.shape
    k = torch.randn(B, 1, 5, device=x.device) * 0.15 * s
    c = F.interpolate(k, size=Fc, mode="linear", align_corners=True)          # [B, 1, F]
    c = c * _on(B, p, x.device)[:, None, None]
    return x * torch.exp(c).transpose(1, 2)                                   # [B, F, 1]


def a_noise(x, s, p):
    """Аддитивный шум, пропорциональный средней амплитуде окна."""
    B = x.shape[0]
    sig = _u(B, 0.01 * s, 0.1 * s, x.device) * x.mean((1, 2)) * _on(B, p, x.device)
    return x + torch.randn_like(x) * sig[:, None, None]


def a_dynscale(x, s, p):
    """Масштаб динамической части: mean + k * (x - mean). Глубина модуляции (LOS <-> NLOS), статика не меняется."""
    B = x.shape[0]
    lim = 0.7 * s
    k = torch.exp(_u(B, -lim, lim, x.device))
    k = torch.where(_on(B, p, x.device), k, torch.ones_like(k))
    m = x.mean(-1, keepdim=True)
    return x + (k[:, None, None] - 1.0) * (x - m)      # то же, что m + k*(x-m); при k=1 даёт ровно x


def a_glitch(x, s, p):
    """Отдельные пакеты с общим множителем по всем поднесущим (вертикальные штрихи в Wallhack)."""
    B, _, T = x.shape
    dev = x.device
    rate = _u(B, 0.0, 0.08 * min(s, 2.0), dev) * _on(B, p, dev)
    mask = torch.rand(B, 1, T, device=dev) < rate[:, None, None]
    lim = 0.9 * s
    f = torch.exp(_u(B * T, -lim, lim, dev)).view(B, 1, T)
    return x * torch.where(mask, f, torch.ones_like(f))


def a_warp(x, s, p):
    """Изменение скорости движения / частоты дискретизации со случайным сдвигом (зеркальные границы)."""
    B, _, T = x.shape
    dev = x.device
    R = 1.0 + 0.25 * s
    on = _on(B, p, dev)
    r = torch.where(on, torch.exp(_u(B, -math.log(R), math.log(R), dev)), torch.ones(B, device=dev))
    off = torch.where(on, _u(B, 0.0, float(T), dev), torch.zeros(B, device=dev))
    pos = off[:, None] + torch.arange(T, device=dev)[None, :] * r[:, None]
    period = 2 * (T - 1)
    pos = torch.remainder(pos, period)
    pos = torch.where(pos > T - 1, period - pos, pos)
    i0 = pos.floor().long().clamp(0, T - 1)
    i1 = (i0 + 1).clamp(max=T - 1)
    fr = (pos - i0.float())[:, None, :]
    return x.gather(-1, i0[:, None, :].expand_as(x)) * (1 - fr) + x.gather(-1, i1[:, None, :].expand_as(x)) * fr


def a_subdrop(x, s, p):
    """Часть поднесущих заменяется своим средним по времени (нейтрально: динамика пропадает)."""
    B, Fc, _ = x.shape
    rate = _u(B, 0.0, 0.2 * min(s, 2.5), x.device) * _on(B, p, x.device)
    mask = torch.rand(B, Fc, 1, device=x.device) < rate[:, None, None]
    return torch.where(mask, x.mean(-1, keepdim=True).expand_as(x), x)


AUGS = {"gain": a_gain, "profile": a_profile, "noise": a_noise, "dynscale": a_dynscale,
        "glitch": a_glitch, "warp": a_warp, "subdrop": a_subdrop}
ORDER = ["warp", "dynscale", "gain", "profile", "subdrop", "glitch", "noise"]   # фиксированный порядок применения

AUG_SETS = {
    "none":      [],
    "gain":      ["gain"],
    "profile":   ["profile"],
    "noise":     ["noise"],
    "dynscale":  ["dynscale"],
    "glitch":    ["glitch"],
    "warp":      ["warp"],
    "subdrop":   ["subdrop"],
    "calib":     ["gain", "profile"],                          # сдвиг уровня; для relative это no-op (контроль)
    "snr":       ["dynscale", "noise", "glitch"],              # LOS <-> NLOS: глубина модуляции и шум
    "snr+warp":  ["dynscale", "noise", "glitch", "warp"],
    "core":      ["gain", "profile", "noise", "glitch"],
    "all":       ["gain", "profile", "noise", "glitch", "warp", "subdrop", "dynscale"],
}


def augment(x, names, strength=1.0, p=0.5):
    """Применяет набор аугментаций; каждая включается для каждого окна независимо с вероятностью p."""
    if not names:
        return x
    for n in ORDER:
        if n in names:
            x = AUGS[n](x, strength, p)
    return x.clamp_min(0.0)


# ------------------------------------------------------------------ самопроверка
def selfcheck(device="cpu"):
    """Сверка предобработки с models.PREPROCESSORS и базовые свойства аугментаций."""
    from models import PREPROCESSORS
    g = np.random.default_rng(0)
    x_np = (g.uniform(5, 60, (8, CHANNELS, SAMPLES)) * (1 + 0.2 * np.sin(np.arange(SAMPLES) / 9))).astype(np.float32)
    x = torch.tensor(x_np, device=device)
    for name, fn in PP.items():
        ref, got = PREPROCESSORS[name](x_np), fn(x).cpu().numpy()
        err = float(np.abs(ref - got).max() / (np.abs(ref).max() + 1e-9))
        print(f"предобработка {name:14s} отн. расхождение с models.py: {err:.2e}")
        assert err < 1e-4, name
    for n in AUGS:
        y = augment(x, [n], 1.0, 1.0)
        assert y.shape == x.shape and torch.isfinite(y).all() and (y >= 0).all(), n
        print(f"аугментация {n:9s} ok, средняя |Δ| = {float((y - x).abs().mean()):.3f}")
    ident = augment(x, AUG_SETS["all"], 1.0, 0.0)
    assert torch.allclose(ident, x.clamp_min(0), atol=1e-4), "при p=0 окно не должно меняться"
    r0, r1 = pp_relative(x), pp_relative(augment(x, ["gain", "profile"], 1.0, 1.0))
    print("relative инвариантен к gain/profile:", bool(torch.allclose(r0, r1, atol=1e-3)))
    l0, l1 = pp_lp20_centered(x), pp_lp20_centered(augment(x, ["gain"], 1.0, 1.0))
    print("lp20_centered инвариантен к gain (ожидаем False):", bool(torch.allclose(l0, l1, atol=1e-3)))
    print("самопроверка пройдена")


if __name__ == "__main__":
    selfcheck("cuda" if torch.cuda.is_available() else "cpu")
