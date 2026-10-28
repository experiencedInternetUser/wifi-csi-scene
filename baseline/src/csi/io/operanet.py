"""OPERAnet WiFi CSI (Intel 5300, 3x3 MIMO, 30 subcarriers, 5 GHz ch 149, ~1600 packets/s).

Raw files: data/raw/operanet/wificsi{1,2}/wificsi{1,2}_expNNN.mat, MATLAB 7.3 `table` objects
(one row per packet: timestamp, activity, exp_no, person_id, room_no, tx1rx1_sub1..tx3rx3_sub30).
Read with `mat-io` (scipy/h5py cannot decode MATLAB tables).

Converted once to data/processed/operanet/rec/<id>.npz (amplitude [T, 9, 30] float16 at the native rate,
t, y) + recordings.parquet, because reading a 1 GB table takes minutes. Cleaning, anti-aliasing and
resampling to 100 Hz are done by csi.preprocess. Receiver NUC1 faces the transmitter (LoS), NUC2 is at 90 degrees (bistatic).

Usage: python -m csi.io.operanet   (converts every downloaded .mat)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from csi.format import PROCESSED, ROOT

RAW = ROOT / "data/raw/operanet"
OUT = PROCESSED / "operanet"
FS = 100.0  # target rate of csi.preprocess

ACTIVITY_ID = {"background": 0, "walk": 1, "sit": 2, "stand": 3, "liedown": 4, "standfromlie": 5,
               "bodyrotate": 6, "noactivity": 7}
CSI_COLS = [f"tx{t}rx{r}_sub{s}" for t in (1, 2, 3) for r in (1, 2, 3) for s in range(1, 31)]
# Intel 5300 reports 30 of the 56 subcarriers of a 20 MHz channel (grouping as in the 802.11n CSI Tool)
SUBCARRIER_INDEX = np.array([-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1,
                             1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28], dtype=np.int16)


def _scalar(v):
    while isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1:
        v = v[0]
    return v


def parse_time_of_day(values) -> np.ndarray:
    """'HH:MM:SS.ffffff' strings -> seconds from the first packet (midnight wrap handled)."""
    s = pd.Series([str(_scalar(v)).strip("[]' ") for v in values])
    parts = s.str.extract(r"(\d+):(\d+):(\d+(?:\.\d+)?)").astype(float)
    sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
    sec = sec.to_numpy()
    d = np.diff(sec)
    d[d < -43200] += 86400
    return dequantize(np.concatenate([[0.0], np.cumsum(d)]))


def dequantize(t: np.ndarray) -> np.ndarray:
    """Timestamps have 1 ms resolution while packets arrive at ~1.6 kHz, so ~40% repeat the previous
    value. Spread every run of equal timestamps evenly over the interval up to the next distinct value."""
    if len(t) < 2 or (np.diff(t) > 0).all():
        return t
    new = np.r_[True, np.diff(t) != 0]
    run_start = np.flatnonzero(new)
    run_len = np.diff(np.r_[run_start, len(t)])
    nxt = np.r_[t[run_start[1:]], t[-1] + np.median(np.diff(t[run_start]))]
    step = (nxt - t[run_start]) / run_len
    k = np.arange(len(t)) - np.repeat(run_start, run_len)
    return t + k * np.repeat(step, run_len)


def read_mat(path: Path) -> tuple[dict, np.ndarray]:
    """-> (string columns as object arrays, amplitude [T, 9, 30] float16).

    The table is loaded with raw_data=True: mat-io's own conversion stacks all 270 complex columns into
    one complex128 matrix, which does not fit in memory for the large files (exp001: 2.5 GB). Here each
    CSI column is turned into float16 amplitude and released right away.
    """
    import matio  # mat-io
    from matio.utils.converters.matstring import mat_to_string

    d = matio.load_from_mat(str(path), raw_data=True)
    props = next(iter(d.values())).properties
    names = [str(np.asarray(n).ravel()[0]) for n in props["varnames"].ravel()]
    data = props["data"]
    col = {n: i for i, n in enumerate(names)}
    meta = {}
    for n in ("timestamp", "activity", "exp_no", "person_id", "room_no"):
        v = data[0, col[n]]
        meta[n] = (mat_to_string(v.properties, "<") if getattr(v, "classname", "") == "string" else
                   np.asarray(v)).ravel()
    T = len(meta["timestamp"])
    amp = np.empty((T, 270), np.float16)
    for k, c in enumerate(CSI_COLS):
        amp[:, k] = np.abs(np.asarray(data[0, col[c]]).ravel())
        data[0, col[c]] = None
    return meta, amp.reshape(T, 9, 30)


def convert(path: Path) -> dict:
    meta, amp = read_mat(path)
    receiver = "NUC1" if path.name.startswith("wificsi1") else "NUC2"
    exp = re.search(r"exp(\d+\w*)", path.stem).group(1)
    t = parse_time_of_day(meta["timestamp"])
    act = pd.Series([str(_scalar(v)).strip() for v in meta["activity"]])
    unknown = sorted(set(act) - set(ACTIVITY_ID))
    y = act.map(ACTIVITY_ID).fillna(-1).to_numpy(np.int16)   # -1: localization labels etc., not used
    order = np.argsort(t, kind="stable")
    t, amp, y = t[order], amp[order], y[order]
    rid = f"operanet_{receiver}_exp{exp}"
    OUT.joinpath("rec").mkdir(parents=True, exist_ok=True)
    # native rate kept (float16 amplitude); cleaning, anti-aliasing and resampling happen in csi.preprocess
    np.savez(OUT / "rec" / f"{rid}.npz", amp=amp, t=t, y=y)
    room = int(float(str(_scalar(meta["room_no"][0]))))
    dt = np.diff(t)
    return {"recording_id": rid, "dataset": "operanet", "receiver": receiver, "exp": exp, "room": room,
            "person_id": str(_scalar(meta["person_id"][0])), "domain": f"room{room}_{receiver}",
            "n_packets": len(t), "fs_native_hz": float(1 / np.median(dt)), "duration_s": float(t[-1]),
            "dt_p99_ms": float(np.percentile(dt, 99) * 1e3), "n_gaps_gt_50ms": int((dt > 0.05).sum()),
            "n_nonpositive_dt": int((dt <= 0).sum()), "unknown_activities": ",".join(unknown),
            "source_file": str(path.resolve().relative_to(ROOT)),
            **{f"sec_{k}": float(np.diff(t)[y[:-1] == v].sum()) for k, v in ACTIVITY_ID.items()}}


def load(recording_id: str):
    """-> csi.recording.Recording with amp [T, 9, 30] at 100 Hz."""
    from csi.recording import Recording

    row = pd.read_parquet(OUT / "recordings.parquet").set_index("recording_id").loc[recording_id]
    with np.load(OUT / "rec" / f"{recording_id}.npz") as z:
        amp, t, y = z["amp"].astype(np.float32), z["t"], z["y"]
    keep = y >= 0
    return Recording(recording_id, t[keep], amp[keep], y[keep], SUBCARRIER_INDEX,
                     meta={"dataset": "operanet", "domain": row.domain, "hardware": "intel5300",
                           "bandwidth_mhz": 20, "band_ghz": 5.0, "fs_nominal_hz": FS, "session": f"exp{row.exp}",
                           "person_id": row.person_id, "room": int(row.room), "receiver": row.receiver})


def main():
    import sys

    paths = sorted(RAW.glob("wificsi*/*.mat"))
    OUT.mkdir(parents=True, exist_ok=True)
    idx_path = OUT / "recordings.parquet"
    force = "--force" in sys.argv
    done = pd.read_parquet(idx_path) if idx_path.exists() and not force else pd.DataFrame()
    rows = done.to_dict("records")
    have = set(done.source_file) if len(done) else set()
    for p in paths:
        if str(p.relative_to(ROOT)) in have:
            continue
        print("converting", p.name, flush=True)
        rows.append(convert(p))
        pd.DataFrame(rows).to_parquet(idx_path, index=False)
        print(pd.DataFrame(rows).tail(1).T.to_string(), flush=True)


if __name__ == "__main__":
    main()
