"""Численные свидетельства для интерпретации сигнала и предобработки."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.signal import welch

from .pipeline import preprocess
from .train import OUT


def main():
    meta = pd.read_csv(OUT / "windows.csv")
    raw = np.load(OUT / "amplitudes.npy", mmap_mode="r")
    features, spectra = [], []
    for start in range(0, len(raw), 128):
        amp = np.asarray(raw[start:start + 128])
        relative = amp / np.maximum(amp.mean(-1, keepdims=True), 1) - 1
        filtered = preprocess(amp, "lp20_centered")
        hz, psd = welch(relative, fs=100, nperseg=200, noverlap=100, axis=-1, detrend="constant")
        mean_psd = psd.mean(1)
        for i in range(len(amp)):
            row = dict(index=start + i, mean_amplitude=float(amp[i].mean()),
                       relative_rms=float(np.sqrt(np.mean(relative[i] ** 2))),
                       centered_rms=float(np.sqrt(np.mean((amp[i] - amp[i].mean(-1, keepdims=True)) ** 2))),
                       lp20_rms=float(np.sqrt(np.mean(filtered[i] ** 2))),
                       psd_above_20_fraction=float(mean_psd[i, hz > 20].sum() / max(mean_psd[i].sum(), 1e-12)))
            features.append(row)
        spectra.append(mean_psd)
    feature = meta.merge(pd.DataFrame(features), on="index")
    feature.to_csv(OUT / "signal_features.csv", index=False)
    medians = feature.groupby(["dataset", "domain", "target"])[
        ["mean_amplitude", "relative_rms", "centered_rms", "lp20_rms", "psd_above_20_fraction"]].median()
    medians.to_csv(OUT / "signal_medians.csv")
    spectrum = np.concatenate(spectra)
    rows = []
    for (dataset, domain, target), group in meta.groupby(["dataset", "domain", "target"]):
        values = spectrum[group["index"]].mean(0)
        rows.extend(dict(dataset=dataset, domain=domain, target=int(target), frequency_hz=float(f),
                         relative_amplitude_psd=float(p)) for f, p in zip(hz, values))
    pd.DataFrame(rows).to_csv(OUT / "amplitude_psd.csv", index=False)
    if (OUT / "test_metrics.csv").exists():
        tests = pd.read_csv(OUT / "test_metrics.csv")
        tests.groupby(["stage", "dataset", "domain"]).agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            balanced_accuracy_min=("balanced_accuracy", "min"),
            balanced_accuracy_max=("balanced_accuracy", "max"),
            f1_mean=("f1", "mean"), f1_std=("f1", "std"),
            fp_min=("fp", "min"), fp_max=("fp", "max"), fn_min=("fn", "min"), fn_max=("fn", "max")
        ).to_csv(OUT / "seed_summary.csv")
    summary = dict(feature_windows=len(feature), psd_method="Welch on relative amplitude; Hann, nperseg200, noverlap100, fs100, mean over52 channels",
                   interpretation="Amplitude fluctuation spectrum, not a signed Doppler/velocity estimate",
                   nn_mac_per_window=dict(cnngru=5_145_664, tcn=17_868_832),
                   mac_scope="Conv1d, GRU input/recurrent matrices, Linear; excludes norm, activation, bias, filtering and memory transfers")
    (OUT / "analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(medians.to_string())


if __name__ == "__main__":
    main()
