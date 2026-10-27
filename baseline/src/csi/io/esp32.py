"""Parsing of ESP-IDF `csi_recv` CSV rows (ESP32-S3), shared by Wallhack1.8k and 3DO.

Each row's `data` field is a list of int8 values. With `len == 256` it holds
128 subcarriers x 2 bytes: L-LTF (subcarriers 0..63) then HT-LTF (64..127).

Byte pairing (see docs/DECISIONS.md, 2026-10-23):
- "espidf":  subcarrier k -> imag = buf[2k], real = buf[2k+1]  (ESP-IDF docs)
- "authors": subcarrier k -> real = buf[2k], imag = buf[2k-1]  (StrohmayerJ loaders;
  mixes bytes of subcarriers k and k-1, kept to reproduce their numbers)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_BYTES = 256

# 52 L-LTF data+pilot subcarriers, same indices as in the authors' loaders.
LLTF_SUBCARRIERS = np.r_[6:32, 33:59]

PAIRINGS = ("espidf", "authors")

META_COLUMNS = ["id", "rssi", "noise_floor", "local_timestamp", "len", "sig_mode",
                "bandwidth", "channel", "secondary_channel", "ant"]


def parse_data_column(data: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Parse the `data` strings into int8 [N, 256].

    Corrupted rows (wrong length or values outside int8, a few per dataset) are dropped.
    Returns (raw, ok_mask) where ok_mask marks rows of the input that were kept.
    """
    body = data.str.slice(1, -1)
    n_values = body.str.count(",") + 1
    ok = (n_values == N_BYTES).to_numpy().copy()
    flat = np.array(",".join(body[ok]).split(","), dtype=np.int32)
    raw = flat.reshape(-1, N_BYTES)
    in_range = ((raw >= -128) & (raw <= 127)).all(axis=1)
    ok[np.flatnonzero(ok)[~in_range]] = False
    return raw[in_range].astype(np.int8), ok


def csi_from_raw(raw: np.ndarray, pairing: str = "espidf",
                 subcarriers: np.ndarray = LLTF_SUBCARRIERS) -> np.ndarray:
    """Build complex CSI [N, S] from raw int8 bytes [N, 256]."""
    r = raw.astype(np.float32)
    k = np.asarray(subcarriers)
    if pairing == "espidf":
        real, imag = r[:, 2 * k + 1], r[:, 2 * k]
    elif pairing == "authors":
        real, imag = r[:, 2 * k], r[:, 2 * k - 1]
    else:
        raise ValueError(f"pairing must be one of {PAIRINGS}, got {pairing!r}")
    return (real + 1j * imag).astype(np.complex64)


def unwrap_timestamps_us(ts: np.ndarray, spike_s: float = 1.0) -> tuple[np.ndarray, int]:
    """Turn the 32-bit wrapping `local_timestamp` (us) into seconds from the first packet.

    Single-packet timestamp glitches (jump > spike_s forward and straight back, seen once in
    3DO d2/s4) are replaced by the midpoint of the neighbours. Returns (t, n_fixed).
    """
    ts = np.asarray(ts, dtype=np.int64) % (1 << 32)
    d = np.diff(ts)
    d[d < -(1 << 31)] += 1 << 32
    t = np.concatenate([[0.0], np.cumsum(d) * 1e-6])
    dt = np.diff(t)
    spikes = np.flatnonzero((dt[:-1] > spike_s) & (dt[1:] < -spike_s)) + 1
    t[spikes] = 0.5 * (t[spikes - 1] + t[spikes + 1])
    return t, len(spikes)


def read_csi_csv(path) -> dict:
    """Read one ESP32 CSI CSV into numpy arrays.

    Returns dict with raw [N,256] int8, t [N] s, meta DataFrame, extra columns
    (`class`, `x`, `y`, `z` when present) and `n_bad_rows`.
    """
    df = pd.read_csv(path)
    raw, ok = parse_data_column(df["data"])
    df = df.loc[ok].reset_index(drop=True)
    t, n_fixed = unwrap_timestamps_us(df["local_timestamp"].to_numpy())
    out = {
        "raw": raw,
        "t": t,
        "meta": df[[c for c in META_COLUMNS if c in df.columns]],
        "n_bad_rows": int((~ok).sum()),
        "n_ts_fixed": n_fixed,
    }
    for c in ("class", "x", "y", "z"):
        if c in df.columns:
            out[c] = df[c].to_numpy()
    return out
