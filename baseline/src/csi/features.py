"""Signal features: amplitude, sanitized phase, rolling variance, spectra, window statistics."""
from __future__ import annotations

import numpy as np
from scipy import signal

FS = 100.0

# OFDM subcarrier numbers of the 52 L-LTF columns (-26..-1, 1..26) for phase detrending.
SUBCARRIER_NUMBERS = np.r_[-26:0, 1:27].astype(np.float64)


def amplitude(csi: np.ndarray) -> np.ndarray:
    return np.abs(csi).astype(np.float32)


def raw_phase(csi: np.ndarray) -> np.ndarray:
    return np.angle(csi).astype(np.float32)


def sanitized_phase(csi: np.ndarray) -> np.ndarray:
    """Unwrap across subcarriers and remove the per-packet linear trend (slope + offset).

    Slope comes from timing offsets (STO/SFO), offset from CFO / random initial phase.
    """
    p = np.unwrap(np.angle(csi).astype(np.float64), axis=1)
    k = SUBCARRIER_NUMBERS
    kc = k - k.mean()
    slope = (p - p.mean(axis=1, keepdims=True)) @ kc / (kc @ kc)
    detrended = p - p.mean(axis=1, keepdims=True) - slope[:, None] * kc[None, :]
    return detrended.astype(np.float32)


def remove_impulses(amp: np.ndarray, win: int = 51, tol: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Replace impulsive packets (mean amplitude off its rolling median by > tol) by interpolation.

    Wallhack has 3-5% such packets in every class, empty room included (hardware/link artifact).
    Returns (cleaned amplitude, bad-packet mask).
    """
    from scipy.ndimage import median_filter

    m = amp.mean(axis=1)
    rel = m / np.maximum(median_filter(m, size=win, mode="nearest"), 1e-6)
    bad = np.abs(rel - 1) > tol
    if not bad.any() or bad.all():
        return amp, bad
    good = np.flatnonzero(~bad)
    out = amp.copy()
    for s in range(amp.shape[1]):
        out[bad, s] = np.interp(np.flatnonzero(bad), good, amp[good, s])
    return out, bad


def rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average along axis 0 (edges use the available samples)."""
    x = np.asarray(x, dtype=np.float64)
    c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), x]), axis=0)
    n = len(x)
    lo = np.clip(np.arange(n) - win // 2, 0, n)
    hi = np.clip(np.arange(n) + (win - win // 2), 0, n)
    shape = (-1,) + (1,) * (x.ndim - 1)
    return (c[hi] - c[lo]) / (hi - lo).reshape(shape)


def rolling_var(x: np.ndarray, win: int = 100) -> np.ndarray:
    """Centered moving variance along time, x is [T] or [T, S]."""
    m = rolling_mean(x, win)
    m2 = rolling_mean(np.asarray(x, dtype=np.float64) ** 2, win)
    return np.maximum(m2 - m ** 2, 0.0)


def motion_score(amp: np.ndarray, win: int = 100, normalize: bool = False) -> np.ndarray:
    """Rolling variance of amplitude averaged over subcarriers, [T].

    normalize=True divides each subcarrier by its rolling mean first (squared coefficient
    of variation), which removes overall gain differences between recordings and domains.
    """
    v = rolling_var(amp, win)
    if normalize:
        v = v / np.maximum(rolling_mean(amp, win), 1e-6) ** 2
    return v.mean(axis=1)


def first_pc(amp: np.ndarray) -> np.ndarray:
    """Projection on the first principal component of the mean-removed amplitude, [T]."""
    x = amp - amp.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x[:: max(1, len(x) // 5000)], full_matrices=False)
    return x @ vt[0]


def stft_db(x: np.ndarray, fs: float = FS, nperseg: int = 256, noverlap: int = 224):
    f, t, z = signal.stft(x - x.mean(), fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, 20 * np.log10(np.abs(z) + 1e-9)


def psd(x: np.ndarray, fs: float = FS, nperseg: int = 512):
    return signal.welch(x - x.mean(), fs=fs, nperseg=min(nperseg, len(x)))


BANDS = [(0.0, 2.0), (2.0, 10.0), (10.0, 40.0)]


def window_features(amp: np.ndarray, phase_s: np.ndarray | None = None, fs: float = FS) -> dict:
    """Simple per-window statistics for one window amp [W, S] (and sanitized phase [W, S])."""
    sub = amp.reshape(-1, int(fs), amp.shape[1]) if len(amp) % int(fs) == 0 else amp[None]
    var1s = sub.var(axis=1).mean()
    mean = amp.mean()
    out = {
        "amp_mean": mean,
        "amp_std_t": amp.std(axis=0).mean(),
        "var1s": var1s,
        "cv1s": (sub.var(axis=1) / np.maximum(sub.mean(axis=1), 1e-6) ** 2).mean(),
        "amp_range": (amp.max(axis=0) - amp.min(axis=0)).mean(),
    }
    pc = first_pc(amp)
    f, p = signal.welch(pc, fs=fs, nperseg=min(128, len(pc)))
    total = p.sum() + 1e-12
    for lo, hi in BANDS:
        out[f"pc1_band_{lo:g}_{hi:g}"] = p[(f >= lo) & (f < hi)].sum() / total
    out["pc1_logpower"] = np.log10(total)
    if phase_s is not None:
        subp = phase_s.reshape(sub.shape) if sub.shape[0] > 1 else phase_s[None]
        out["phase_var1s"] = subp.var(axis=1).mean()
    return out
