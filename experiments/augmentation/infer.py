"""Apply an augmentation-trained checkpoint without test-time augmentation."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch

from experiments.robustness.modeling import predict


def load(path):
    path = Path(path)
    if path.suffix == ".joblib":
        return joblib.load(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    saved["projection"] = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in saved["projection"].items()}
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("amplitudes", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = load(args.model)
    x = np.load(args.amplitudes, allow_pickle=False)
    p = predict(saved, x, device=args.device)
    np.savez_compressed(args.output, probability=p, prediction=(p >= saved["threshold"]).astype(np.int8))
    print(f"Saved {len(p)} predictions")


if __name__ == "__main__":
    main()
