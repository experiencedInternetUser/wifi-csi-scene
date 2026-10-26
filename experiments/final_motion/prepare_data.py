"""Подготовка амплитуд с проверкой разрывов в целых микросекундах.

Локальный большой кэш исключён из Git; все его окна восстанавливаются из ZIP.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/motion_results"
SELECTED = np.r_[6:32, 33:59]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    old_meta = pd.read_csv(ROOT / "experiments/results/windows.csv")
    arrays, rows, recordings, archives = [], [], [], {}
    for dataset, archive in [("Wallhack", "wallhack1.8k.zip"), ("3DO", "3DO.zip")]:
        source = ROOT / "datasets" / archive
        archives[archive] = sha256(source)
        with zipfile.ZipFile(source) as z:
            names = sorted(n for n in z.namelist() if n.endswith(".csv") and
                           (("spectrograms" not in n) if dataset == "Wallhack" else
                            (n.endswith("/csiposreg.csv") and n.split("/")[2].startswith(("b", "w")))))
            for name in names:
                frame = pd.read_csv(z.open(name))
                total = len(frame)
                good = (frame.data.str.count(",") + 1 == frame["len"]) & (frame.rx_state == 0)
                frame = frame[good].reset_index(drop=True)
                iq = np.stack([np.fromstring(s.strip()[1:-1], sep=",", dtype=np.float32) for s in frame.data])
                assert iq.shape[1] == 256
                amp = np.abs(iq[:, 2 * SELECTED + 1] + 1j * iq[:, 2 * SELECTED])
                ticks = frame.local_timestamp.to_numpy(dtype=np.int64)
                wraps = np.r_[0, (np.diff(ticks) < -(2 ** 31)).astype(np.int64)]
                corrected = ticks + np.cumsum(wraps) * 2 ** 32
                elapsed_us = corrected - corrected[0]
                dt = np.diff(elapsed_us)
                time_s = elapsed_us.astype(np.float64) * 1e-6
                labels = frame["class"].to_numpy(dtype=int)
                domain = "/".join(name.split("/")[1:3]) if dataset == "Wallhack" else name.split("/")[1]
                sequence = Path(name).stem if dataset == "Wallhack" else name.split("/")[2]
                label = 0 if sequence.startswith("b") else (2 if sequence.startswith("ww") else 1)
                assert set(labels) == {label}
                cuts = np.r_[0, np.flatnonzero((dt <= 0) | (dt > 50_000) | (np.diff(labels) != 0)) + 1, len(frame)]
                legacy_gap = np.diff(time_s) > .05
                spurious_50ms = int(((dt == 50_000) & legacy_gap).sum())
                before = len(rows)
                for segment, (left, right) in enumerate(zip(cuts[:-1], cuts[1:])):
                    if right - left < 400 or elapsed_us[right - 1] - elapsed_us[left] < 3_990_000:
                        continue
                    local = (elapsed_us[left:right] - elapsed_us[left]).astype(np.float64) * 1e-6
                    grid_us = np.arange(0, int(elapsed_us[right - 1] - elapsed_us[left]) + 1, 10_000, dtype=np.int64)
                    grid = grid_us.astype(np.float64) * 1e-6
                    regular = np.stack([np.interp(grid, local, amp[left:right, k]) for k in range(52)]).astype(np.float32)
                    for start in range(0, len(grid) - 399, 400):
                        arrays.append(regular[:, start:start + 400])
                        rows.append(dict(index=len(rows), dataset=dataset, domain=domain, sequence=sequence,
                                         recording=name, group_id=dataset + "::" + name, original_label=label,
                                         target=int(label != 0), segment=segment,
                                         start_us=int(elapsed_us[left] + grid_us[start]),
                                         end_us=int(elapsed_us[left] + grid_us[start + 399]),
                                         start_s=float((elapsed_us[left] + grid_us[start]) * 1e-6),
                                         end_s=float((elapsed_us[left] + grid_us[start + 399]) * 1e-6)))
                starts = np.array([r["start_us"] for r in rows[before:]], dtype=np.int64)
                old_starts = np.rint(old_meta.loc[old_meta.recording == name, "start_s"].to_numpy() * 1e6).astype(np.int64)
                recordings.append(dict(dataset=dataset, domain=domain, sequence=sequence, recording=name,
                                       target=int(label != 0), original_label=label, raw_packets=total,
                                       valid_packets=len(frame), discarded_packets=total - len(frame), wraps=int(wraps.sum()),
                                       segments=len(cuts) - 1, exact_50ms_gaps=int((dt == 50_000).sum()),
                                       spurious_legacy_50ms_cuts=spurious_50ms, nonpositive_dt=int((dt <= 0).sum()),
                                       gaps_over_50ms=int((dt > 50_000).sum()),
                                       median_interval_us=float(np.median(dt[dt > 0])),
                                       windows=len(rows) - before, legacy_windows=len(old_starts),
                                       identical_start_windows=int(np.isin(starts, old_starts).sum()),
                                       source_crc32=z.getinfo(name).CRC, source_bytes=z.getinfo(name).file_size))
                print(f"Подготовлена {name}: {len(rows) - before} окон", flush=True)
    x = np.stack(arrays)
    meta = pd.DataFrame(rows)
    rec = pd.DataFrame(recordings)
    assert len(rec) == 67 and len(set(zip(rec.source_crc32, rec.source_bytes))) == 67
    assert x.shape == (len(meta), 52, 400) and np.isfinite(x).all() and (x >= 0).all()
    for _, group in meta.groupby("group_id"):
        assert (group.start_us.diff().dropna() >= 4_000_000).all()
    meta.to_csv(OUT / "windows.csv", index=False)
    rec.to_csv(OUT / "recordings.csv", index=False)
    np.save(OUT / "amplitudes.npy", x)
    info = dict(state="passed", parser="real=data[2*k+1], imaginary=data[2*k]", subcarriers=SELECTED.tolist(),
                time="int64 microseconds with uint32 unwrapping; dt<=0 or dt>50000 splits; no float threshold",
                resampling="linear amplitude interpolation on integer 10000us grid within continuous segment",
                sample_rate_hz=100, window_samples=400, stride_samples=400, std_floor=1e-4,
                recordings=len(rec), windows=len(meta), shape=list(x.shape), dtype=str(x.dtype),
                target="0=empty scene,1=walking/walking+arm-waving; sitting and lying excluded",
                spurious_legacy_50ms_cuts=int(rec.spurious_legacy_50ms_cuts.sum()),
                legacy_windows=int(rec.legacy_windows.sum()), identical_start_windows=int(rec.identical_start_windows.sum()),
                raw_packets=int(rec.raw_packets.sum()), valid_packets=int(rec.valid_packets.sum()),
                discarded_packets=int(rec.discarded_packets.sum()), wraps=int(rec.wraps.sum()),
                archive_sha256=archives, amplitudes_sha256=sha256(OUT / "amplitudes.npy"),
                preparation_code_sha256=sha256(Path(__file__)), windows_sha256=sha256(OUT / "windows.csv"))
    (OUT / "data_manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
    return x, meta


if __name__ == "__main__":
    prepare()
