"""Canonical in-memory recording: the common format every dataset adapter produces.

One `Recording` = one continuous capture from one device setup. Shapes are hardware-agnostic:
A = antenna / TX-RX pairs, S = subcarriers, T = packets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Recording:
    recording_id: str
    t: np.ndarray                      # [T] float64, seconds from first packet
    amp: np.ndarray                    # [T, A, S] float32 amplitude
    y: np.ndarray                      # [T] int16, dataset class per packet
    subcarrier_index: np.ndarray       # [S] OFDM subcarrier numbers (e.g. -26..26 without 0)
    csi: np.ndarray | None = None      # [T, A, S] complex64 when the dataset ships phase
    pos: np.ndarray | None = None      # [T, 3] float32 when available
    meta: dict = field(default_factory=dict)  # dataset, domain, hardware, bandwidth_mhz, fs_nominal_hz, ...

    def __post_init__(self):
        if self.amp.ndim != 3:
            raise ValueError(f"amp must be [T, A, S], got {self.amp.shape}")
        T, _, S = self.amp.shape
        if len(self.t) != T or len(self.y) != T:
            raise ValueError("t, y and amp disagree on T")
        if len(self.subcarrier_index) != S:
            raise ValueError("subcarrier_index length != S")
        if self.csi is not None and self.csi.shape != self.amp.shape:
            raise ValueError("csi and amp shapes differ")

    @property
    def n_antennas(self) -> int:
        return self.amp.shape[1]

    @property
    def n_subcarriers(self) -> int:
        return self.amp.shape[2]


def from_processed(dataset: str, recording_id: str, pairing: str = "espidf") -> Recording:
    """Adapter for data/processed (Wallhack1.8k, 3DO: ESP32-S3, 1 antenna, 52 L-LTF subcarriers)."""
    from csi.features import SUBCARRIER_NUMBERS
    from csi.format import load_index, load_recording

    d = load_recording(dataset, recording_id, pairing)
    row = load_index(dataset).set_index("recording_id").loc[recording_id]
    domain = f"{row.condition}_{row.antenna}" if dataset == "wallhack" else row.day
    csi = d["csi"][:, None, :]
    return Recording(
        recording_id=recording_id, t=d["t"], amp=np.abs(csi).astype(np.float32), y=d["y"],
        subcarrier_index=SUBCARRIER_NUMBERS.astype(np.int16), csi=csi, pos=d.get("pos"),
        meta={"dataset": dataset, "domain": domain, "hardware": "esp32s3", "bandwidth_mhz": 20,
              "fs_nominal_hz": 100.0, "session": row.session},
    )
