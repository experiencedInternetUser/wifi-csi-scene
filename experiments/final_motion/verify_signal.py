"""Численная проверка свойств предобработки на контрольных сигналах."""
import json

import numpy as np

from .pipeline import PREPROCESSORS, preprocess
from .train import OUT


def main():
    constant = np.full((2, 52, 400), 10, dtype=np.float32)
    centered = [p for p in PREPROCESSORS if p != "logamp"]
    maximum_dc = 0.
    for prep in centered:
        error = float(np.max(np.abs(preprocess(constant, prep))))
        assert error < 1e-5, (prep, error)
        maximum_dc = max(maximum_dc, error)
    rng = np.random.default_rng(42)
    signal = rng.uniform(1, 20, (3, 52, 400)).astype(np.float32)
    batch_error = 0.
    for prep in PREPROCESSORS:
        full = preprocess(signal, prep)
        separate = np.concatenate([preprocess(s[None], prep) for s in signal])
        error = float(np.max(np.abs(full - separate)))
        assert error < 1e-6, (prep, error)
        assert full.shape == signal.shape and np.isfinite(full).all()
        batch_error = max(batch_error, error)
    gains = {}
    t = np.arange(400) / 100
    for prep, cutoff in [("lp10_centered", 10), ("lp20_centered", 20)]:
        wave = 10 + np.sin(2 * np.pi * cutoff * t)
        amplitude = np.broadcast_to(wave, (1, 52, 400)).astype(np.float32).copy()
        filtered = preprocess(amplitude, prep)[0, 0, 200:]
        gain = float(filtered.std() / wave[200:].std())
        assert abs(gain - 2 ** -.5) < 1e-4, (prep, gain)
        gains[prep] = gain
    spike = constant[:1].copy()
    spike[:, :, 200] = 100
    assert np.max(np.abs(preprocess(spike, "hampel_lp20"))) < 1e-5
    invalid_rejected = 0
    for invalid in [np.full((1, 52, 400), -1), np.full((1, 52, 400), np.nan), np.ones((1, 400, 52))]:
        try:
            preprocess(invalid, "lp20_centered")
        except ValueError:
            invalid_rejected += 1
    assert invalid_rejected == 3
    result = dict(state="passed", preprocessors=len(PREPROCESSORS), maximum_constant_residual=maximum_dc,
                  batch_vs_single_window_error=batch_error, cutoff_amplitude_gains=gains,
                  expected_cutoff_gain=2 ** -.5, isolated_spike_removed=True, invalid_inputs_rejected=invalid_rejected)
    (OUT / "signal_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
