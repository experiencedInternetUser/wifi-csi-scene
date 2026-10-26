"""Оконная предобработка и загрузка воспроизводимого детектора движения."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfilt, sosfilt_zi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import make_model, parameter_count  # noqa: E402

PREPROCESSORS = (
    "logamp", "relative", "centered", "lp20_centered", "lp10_centered",
    "relative_lp20", "hampel_lp20",
)


def preprocess(amplitudes: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(amplitudes, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (52, 400):
        raise ValueError("Ожидалась амплитуда формы [N,52,400]")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Амплитуда должна быть конечной и неотрицательной")
    if name not in PREPROCESSORS:
        raise ValueError(f"Неизвестная предобработка: {name}")
    if name == "logamp":
        return np.log1p(x)
    if name == "relative":
        return (x / np.maximum(x.mean(-1, keepdims=True), 1) - 1).astype(np.float32)
    if name == "hampel_lp20":
        median = median_filter(x, size=(1, 1, 5), mode="reflect")
        mad = median_filter(np.abs(x - median), size=(1, 1, 5), mode="reflect")
        x = np.where(np.abs(x - median) > 3 * np.maximum(1.4826 * mad, .5), median, x)
    if name != "centered":
        sos = butter(4, 10 if name == "lp10_centered" else 20,
                     fs=100, btype="lowpass", output="sos")
        zi = sosfilt_zi(sos)[:, None, None, :] * x[None, ..., :1]
        x, _ = sosfilt(sos, x, axis=-1, zi=zi)
    mean = x.mean(-1, keepdims=True)
    x = x - mean
    if name == "relative_lp20":
        x = x / np.maximum(mean, 1)
    return x.astype(np.float32)


class MotionDetector:
    """Принимает уже восстановленные амплитуды, не сырые I/Q и не PNG."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.preprocessing = saved["preprocessing"]
        self.threshold = saved["threshold"]
        self.mean = saved["mean"].to(self.device)
        self.std = saved["std"].to(self.device)
        self.model = make_model(saved["architecture"]).to(self.device).eval()
        self.model.load_state_dict(saved["model"])

    @torch.inference_mode()
    def predict(self, amplitudes: np.ndarray, batch_size: int = 64) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size должен быть положительным")
        x = preprocess(amplitudes, self.preprocessing)
        probabilities = []
        for first in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[first:first + batch_size]).to(self.device)
            probabilities.append(self.model((batch - self.mean) / self.std).flatten().sigmoid().cpu().numpy())
        return np.concatenate(probabilities) if probabilities else np.empty(0, np.float32)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Прогноз движения по амплитуде CSI [N,52,400]")
    parser.add_argument("amplitudes", type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).resolve().parents[1] / "motion_results/deployment.pt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    detector = MotionDetector(args.checkpoint)
    p = detector.predict(np.load(args.amplitudes, allow_pickle=False))
    np.savez_compressed(args.output, probability=p, prediction=(p >= detector.threshold).astype(np.int8))
    print(f"Сохранены {len(p)} прогнозов: {args.output}")


if __name__ == "__main__":
    main()
