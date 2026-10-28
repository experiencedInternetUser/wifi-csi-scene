"""Single preprocessing pipeline for every dataset and model (stage 2 template).

    Recording -> amplitude -> split at gaps -> remove impulsive packets (native rate)
              -> anti-alias + resample to 100 Hz -> low-pass -> windows -> per-window normalization
              -> [N, A, S, W] (+ window labels)
    then, fitted on train only: Standardizer; or hardware-agnostic window features for simple models.

Presets reproduce the variants compared in stage 2 (docs/RESULTS_BINARY.md):
    raw                 amplitude only
    clean               + impulse removal
    clean_lp            + causal low-pass 20 Hz
    clean_lp_centered   + subtract window mean per subcarrier (team `lp20_centered` + cleaning)
    clean_lp_relative   + A / window mean - 1 per subcarrier: unit-free, but amplifies noise on
                          deeply faded subcarriers (worse on Wallhack: BA 0.83 vs 0.92)
    clean_lp_scaled     + subtract window mean per subcarrier, divide by ONE window mean: unit-free
                          (ESP32 int8 vs Intel/Atheros units differ) without that problem. DEFAULT

Usage:
    from csi.preprocess import preprocess_recording, PRESETS
    X, y, info = preprocess_recording(rec, PRESETS["clean_lp_relative"])
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from csi import features as F
from csi.recording import Recording

FS = 100.0


@dataclass(frozen=True)
class PreprocConfig:
    name: str
    fs: float = FS                 # target packet rate after resampling
    max_gap_s: float = 0.05        # larger gaps split the recording into segments
    impulses: bool = False         # interpolate impulsive (drop-out) packets
    impulse_win_s: float = 0.5     # median window, seconds (51 packets at 100 Hz)
    impulse_tol: float = 0.3       # relative deviation of packet mean amplitude
    lowpass_hz: float | None = None
    norm: str = "none"             # none | centered | relative | scaled
    window: int = 400
    step: int = 400
    min_purity: float = 0.8


PRESETS = {p.name: p for p in [
    PreprocConfig("raw"),
    PreprocConfig("clean", impulses=True),
    PreprocConfig("clean_lp", impulses=True, lowpass_hz=20.0),
    PreprocConfig("clean_lp_centered", impulses=True, lowpass_hz=20.0, norm="centered"),
    PreprocConfig("clean_lp_relative", impulses=True, lowpass_hz=20.0, norm="relative"),
    PreprocConfig("clean_lp_scaled", impulses=True, lowpass_hz=20.0, norm="scaled"),
]}
DEFAULT = PRESETS["clean_lp_scaled"]


# ---------------------------------------------------------------- steps on a continuous segment

def segments(t: np.ndarray, y: np.ndarray, max_gap_s: float) -> list[tuple[int, int]]:
    """[start, end) packet ranges without gaps > max_gap_s, non-increasing time or label changes."""
    dt = np.diff(t)
    cut = np.flatnonzero((dt <= 0) | (dt > max_gap_s) | (np.diff(y) != 0)) + 1
    b = np.r_[0, cut, len(t)]
    return [(int(s), int(e)) for s, e in zip(b[:-1], b[1:]) if e - s > 1]


def native_rate(t: np.ndarray) -> float:
    """Mean packet rate of a segment (robust to quantized timestamps, unlike the median interval)."""
    return (len(t) - 1) / max(float(t[-1] - t[0]), 1e-9)


def resample(t: np.ndarray, x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample x [T, ...] of one continuous segment onto a uniform grid at fs, starting at t[0].

    If the native packet rate is well above fs (Intel 5300 at 1-1.6 kHz, Nexmon ~170 Hz), the signal is
    first put on a uniform native-rate grid and low-passed at 0.4*fs (anti-aliasing), otherwise
    everything above fs/2 would fold into the working band.
    """
    flat = x.reshape(len(t), -1)
    fs_native = native_rate(t)
    grid = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * fs)) + 1) / fs
    out = np.empty((len(grid), flat.shape[1]), np.float32)
    if fs_native > 1.5 * fs:
        # channel by channel: a [T_native, 270] float64 intermediate would not fit in memory
        g = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * fs_native)) + 1) / fs_native
        sos = butter(8, 0.4 * fs, btype="lowpass", fs=fs_native, output="sos")
        zi0 = sosfilt_zi(sos)
        for j in range(flat.shape[1]):
            u = np.interp(g, t, flat[:, j].astype(np.float64))
            u, _ = sosfilt(sos, u, zi=zi0 * u[0])
            out[:, j] = np.interp(grid, g, u)
    else:
        for j in range(flat.shape[1]):
            out[:, j] = np.interp(grid, t, flat[:, j].astype(np.float64))
    return grid, out.reshape((len(grid),) + x.shape[1:])


def remove_impulses(amp: np.ndarray, win: int, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """amp [T, A, S]: per antenna, see csi.features.remove_impulses. Returns (clean, bad [T])."""
    out = amp.copy()
    bad_any = np.zeros(len(amp), dtype=bool)
    for a in range(amp.shape[1]):
        out[:, a], bad = F.remove_impulses(amp[:, a], win, tol)
        bad_any |= bad
    return out, bad_any


def lowpass(x: np.ndarray, cutoff: float, fs: float) -> np.ndarray:
    """Causal 4th-order Butterworth along time, initial state = first sample (no start transient)."""
    sos = butter(4, cutoff, btype="lowpass", fs=fs, output="sos")
    zi = sosfilt_zi(sos)
    zi = zi.reshape(zi.shape + (1,) * (x.ndim - 1)) * x[:1][None]
    y, _ = sosfilt(sos, x, axis=0, zi=zi)
    return y.astype(np.float32)


def normalize_window(w: np.ndarray, norm: str) -> np.ndarray:
    """w [W, A, S] -> normalized over time, per antenna and subcarrier."""
    if norm == "none":
        return w
    m = w.mean(axis=0, keepdims=True)
    if norm == "centered":
        return w - m
    if norm == "relative":
        return w / np.maximum(m, 1e-6 * max(float(np.abs(m).max()), 1e-12)) - 1.0
    if norm == "scaled":
        # centered, divided by ONE number (window mean over time, antennas, subcarriers):
        # unit-free like `relative`, but does not blow up noise on deeply faded subcarriers
        return (w - m) / max(float(m.mean()), 1e-12)
    raise ValueError(f"unknown norm {norm!r}")


# ---------------------------------------------------------------- whole recording

def preprocess_recording(rec: Recording, cfg: PreprocConfig = DEFAULT):
    """Recording -> X [N, A, S, W] float32, y [N] int16 (majority label), info DataFrame-able dict.

    Windows never cross a gap or a label change; windows with label purity < min_purity are dropped
    (purity is always 1 here because segments are cut at label changes, kept for clarity).
    """
    X, Y, starts, impulse_frac = [], [], [], []
    for s, e in segments(rec.t, rec.y, cfg.max_gap_s):
        t, amp, lab = rec.t[s:e], rec.amp[s:e], rec.y[s:e]
        if (t[-1] - t[0]) * cfg.fs < cfg.window:
            continue
        # impulses are removed at the native packet rate, before anti-aliasing would smear them
        bad_native = np.zeros(len(t), dtype=bool)
        if cfg.impulses:
            win = max(3, int(round(cfg.impulse_win_s * native_rate(t))) | 1)
            amp, bad_native = remove_impulses(amp, win, cfg.impulse_tol)
        grid, a = resample(t, amp, cfg.fs)
        bad = np.interp(grid, t, bad_native.astype(float)) > 0
        if cfg.lowpass_hz:
            a = lowpass(a, cfg.lowpass_hz, cfg.fs)
        for k in range(0, len(a) - cfg.window + 1, cfg.step):
            w = normalize_window(a[k:k + cfg.window], cfg.norm)
            X.append(np.transpose(w, (1, 2, 0)))       # [A, S, W]
            Y.append(lab[0])
            starts.append(float(grid[k]))
            impulse_frac.append(float(bad[k:k + cfg.window].mean()))
    if not X:
        empty = np.zeros((0, rec.n_antennas, rec.n_subcarriers, cfg.window), np.float32)
        return empty, np.zeros(0, np.int16), {"start_s": [], "impulse_frac": []}
    info = {"recording_id": [rec.recording_id] * len(X), "start_s": starts, "impulse_frac": impulse_frac,
            **{k: [v] * len(X) for k, v in rec.meta.items() if np.isscalar(v)}}
    return np.stack(X).astype(np.float32), np.asarray(Y, np.int16), info


# ---------------------------------------------------------------- after windowing

class Standardizer:
    """Per-(antenna, subcarrier) mean/std over windows and time, fitted on train windows only."""

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean = X.mean(axis=(0, 3), keepdims=True)[0]
        self.std = np.maximum(X.std(axis=(0, 3), keepdims=True)[0], 1e-4)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)


FEATURE_NAMES = ["log_var1s", "log_std", "log_range", "pc1_band_0_2", "pc1_band_2_10",
                 "pc1_band_10_40", "log_pc1_power"]


def window_features(X: np.ndarray, fs: float = FS) -> np.ndarray:
    """Hardware-agnostic features of windows X [N, A, S, W] (FEATURE_NAMES), aggregated over antennas
    and subcarriers, so one simple model applies to any number of subcarriers / antennas.

    Compute them on `relative`-normalized windows to make them unit-free across hardware.
    """
    out = np.zeros((len(X), len(FEATURE_NAMES)), np.float32)
    for i, w in enumerate(X):
        a = w.reshape(-1, w.shape[-1]).T                  # [W, A*S]
        f = F.window_features(a, fs=fs)
        out[i] = [np.log10(f["var1s"] + 1e-12), np.log10(f["amp_std_t"] + 1e-12), np.log10(f["amp_range"] + 1e-12),
                  f["pc1_band_0_2"], f["pc1_band_2_10"], f["pc1_band_10_40"], f["pc1_logpower"]]
    return out


def with_overrides(cfg: PreprocConfig, **kw) -> PreprocConfig:
    return replace(cfg, **kw)
