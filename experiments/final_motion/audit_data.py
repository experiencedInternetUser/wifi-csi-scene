"""Независимая сверка всех сохранённых окон с исходными I/Q CSV в ZIP."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/motion_results"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(OUT / "windows.csv")
    recordings = pd.read_csv(OUT / "recordings.csv")
    cached = np.load(OUT / "amplitudes.npy", mmap_mode="r")
    selected = np.r_[6:32, 33:59]
    checks, hashes = [], {}
    for dataset, archive in [("Wallhack", "wallhack1.8k.zip"), ("3DO", "3DO.zip")]:
        archive_path = ROOT / "datasets" / archive
        hashes[archive] = sha256(archive_path)
        with zipfile.ZipFile(archive_path) as z:
            for rec in recordings[recordings.dataset == dataset].itertuples():
                frame = pd.read_csv(z.open(rec.recording))
                raw_count = len(frame)
                good = (frame.data.str.count(",") + 1 == frame["len"]) & (frame.rx_state == 0)
                frame = frame[good].reset_index(drop=True)
                iq = np.stack([np.fromstring(s[1:-1], sep=",", dtype=np.float32) for s in frame.data])
                assert iq.shape[1] == 256
                # np.hypot независимо от комплексного np.abs в исходном prepare.py.
                amplitude = np.hypot(iq[:, 2 * selected], iq[:, 2 * selected + 1])
                ticks = frame.local_timestamp.to_numpy(dtype=np.int64)
                wraps = np.r_[0, np.diff(ticks) < -(2 ** 31)]
                time_us = ticks + np.cumsum(wraps) * 2 ** 32
                time_us = time_us - time_us[0]
                labels = frame["class"].to_numpy(dtype=int)
                assert set(labels) == {rec.original_label}
                assert rec.target == int(rec.original_label != 0)
                cuts = np.r_[0, np.flatnonzero((np.diff(time_us) <= 0) | (np.diff(time_us) > 50_000)
                                               | (np.diff(labels) != 0)) + 1, len(frame)]
                expected_rows = meta[meta.recording == rec.recording]
                reconstructed = []
                starts = []
                for left, right in zip(cuts[:-1], cuts[1:]):
                    if right - left < 400 or time_us[right - 1] - time_us[left] < 3_990_000:
                        continue
                    local_t = time_us[left:right] - time_us[left]
                    grid = np.arange(0, local_t[-1] + 1, 10_000, dtype=np.int64)
                    uniform = np.stack([np.interp(grid, local_t, amplitude[left:right, k]) for k in range(52)])
                    for start in range(0, len(grid) - 399, 400):
                        reconstructed.append(uniform[:, start:start + 400])
                        starts.append(time_us[left] + start * 10_000)
                rebuilt = np.asarray(reconstructed, np.float32)
                assert len(rebuilt) == len(expected_rows) == rec.windows
                assert np.array_equal(starts, expected_rows.start_us)
                reference = cached[expected_rows["index"].to_numpy()]
                maximum_error = float(np.max(np.abs(rebuilt - reference)))
                assert np.allclose(rebuilt, reference, rtol=2e-6, atol=2e-5), rec.recording
                assert (expected_rows.start_us.diff().dropna() >= 4_000_000).all()
                checks.append(dict(dataset=dataset, recording=rec.recording, raw_packets=raw_count,
                                   valid_packets=len(frame), discarded_packets=raw_count - len(frame),
                                   windows=len(rebuilt), wraps=int(wraps.sum()), segments=len(cuts) - 1,
                                   maximum_amplitude_error=maximum_error, source_crc32=z.getinfo(rec.recording).CRC))
                print(f"Сверена {rec.recording}: {len(rebuilt)} окон", flush=True)
    assert len(checks) == 67 and sum(r["windows"] for r in checks) == len(meta)
    pd.DataFrame(checks).to_csv(OUT / "source_audit.csv", index=False)
    inventories = []
    for archive in sorted((ROOT / "datasets").glob("*.zip")):
        with zipfile.ZipFile(archive) as z:
            files = [i for i in z.infolist() if not i.is_dir()]
            inventories.append(dict(archive=archive.name, bytes=archive.stat().st_size, files=len(files),
                                    extensions=pd.Series([Path(i.filename).suffix for i in files]).value_counts().to_dict()))
    materials = []
    paths = [ROOT / "README.md", ROOT / "csi_datasets_exploration.ipynb"]
    paths += sorted((ROOT / "research").rglob("*.md"))
    paths += sorted((ROOT / "research/analysis").glob("*.csv"))
    paths += sorted((ROOT / "research/analysis").glob("*.json"))
    paths += sorted((ROOT / "experiments").glob("*.py"))
    paths += [ROOT / "experiments/protocol.md", ROOT / "experiments/results/results.csv", ROOT / "experiments/results/runs.csv"]
    for path in paths:
        row = dict(path=path.relative_to(ROOT).as_posix(), sha256=sha256(path), bytes=path.stat().st_size)
        if path.suffix == ".csv":
            table = pd.read_csv(path)
            row["rows"] = len(table)
            row["columns"] = ",".join(table.columns)
        materials.append(row)
    pd.DataFrame(materials).to_csv(OUT / "materials.csv", index=False)
    result = dict(state="passed", recordings=len(checks), windows=len(meta),
                  raw_packets=sum(r["raw_packets"] for r in checks),
                  valid_packets=sum(r["valid_packets"] for r in checks),
                  discarded_packets=sum(r["discarded_packets"] for r in checks),
                  wraps=sum(r["wraps"] for r in checks),
                  maximum_amplitude_error=max(r["maximum_amplitude_error"] for r in checks),
                  amplitude_sha256=sha256(OUT / "amplitudes.npy"),
                  windows_sha256=sha256(OUT / "windows.csv"),
                  archive_sha256=hashes, archive_inventory=inventories,
                  all_windows_reconstructed=True, nonoverlapping=True, labels_verified=True,
                  materials=len(materials))
    (OUT / "data_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
