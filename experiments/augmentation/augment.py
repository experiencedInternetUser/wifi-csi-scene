"""Class-independent randomization of cleaned CSI residuals during training."""
from __future__ import annotations

import math

import torch
from scipy.signal import firwin
from torch.nn import functional as F

POLICIES = ("none", "gain", "spectral", "noise", "combined")


class Augmenter:
    def __init__(self, policy: str, seed: int, device: str = "cuda"):
        if policy not in POLICIES:
            raise ValueError(policy)
        self.policy = policy
        self.generator = torch.Generator(device=device).manual_seed(seed + 50000)
        taps = firwin(41, 10, fs=100, window="hann")
        self.kernel = torch.tensor(taps, dtype=torch.float32, device=device)[None, None]

    def uniform(self, shape, x):
        return torch.rand(shape, generator=self.generator, device=x.device, dtype=x.dtype)

    def normal(self, shape, x):
        return torch.randn(shape, generator=self.generator, device=x.device, dtype=x.dtype)

    def __call__(self, x: torch.Tensor, raw_mean: torch.Tensor) -> torch.Tensor:
        if self.policy == "none":
            return x
        n, channels, length = x.shape
        enabled = self.uniform((n, 1, 1), x) < 0.5
        out = x
        if self.policy in ("spectral", "combined"):
            knots = self.normal((n, 1, 6), x) * 0.35
            profile = F.interpolate(knots, size=channels, mode="linear", align_corners=True)
            profile = (profile - profile.mean(-1, keepdim=True)).exp().transpose(1, 2)
            out = out * profile
        if self.policy in ("noise", "combined"):
            white = self.normal((n, channels + 1, length), x)
            filtered = F.conv1d(F.pad(white.reshape(-1, 1, length), (20, 20), mode="reflect"), self.kernel)
            filtered = filtered.reshape(n, channels + 1, length)
            noise = math.sqrt(0.8) * filtered[:, :1] + math.sqrt(0.2) * filtered[:, 1:]
            noise = noise - noise.mean(-1, keepdim=True)
            noise = noise / noise.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-6)
            relative_rms = 0.06 * self.uniform((n, 1, 1), x)
            out = out + relative_rms * raw_mean * noise
        if self.policy in ("gain", "combined"):
            log_gain = math.log(0.25) + math.log(64) * self.uniform((n, 1, 1), x)
            out = out * log_gain.exp()
        return torch.where(enabled, out, x)
