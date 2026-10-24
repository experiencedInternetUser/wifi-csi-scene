"""Модели для сравнения и предобработка CSI без сохранения состояния.

Все модели принимают float32-массивы [размер пакета, 52 поднесущие, 400 отсчётов]
и возвращают логиты формы [размер пакета, 1]. Окно содержит последние четыре
секунды сигнала с частотой 100 Гц. Модели и функции предобработки в этом модуле
не оценивают статистику датасета. Параметры стандартизации рассчитываются
в скрипте обучения только по обучающим записям.

Предобработка принимает амплитуды в исходных единицах устройства, до
стандартизации. Относительная амплитуда рассчитывается отдельно в каждом
полностью доступном окне. Это не означает, что вся цепочка предобработки
и модели работает причинно для каждого отдельного отсчёта.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
import torch
from torch import nn
from torch.nn import functional as F

CHANNELS = 52
SAMPLES = 400
SAMPLE_RATE_HZ = 100.0


def _amplitude(windows: np.ndarray) -> np.ndarray:
    x = np.asarray(windows, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (CHANNELS, SAMPLES):
        raise ValueError(f"Ожидалась форма [N,{CHANNELS},{SAMPLES}], получена {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Амплитуда CSI содержит NaN или бесконечные значения")
    if np.any(x < 0):
        raise ValueError("Для предобработки нужны неотрицательные амплитуды CSI")
    return x


def logamp(windows: np.ndarray) -> np.ndarray:
    """P1: log(1+A) с сохранением статической составляющей и изменений во времени."""
    return np.log1p(_amplitude(windows)).astype(np.float32, copy=False)


def relative(windows: np.ndarray) -> np.ndarray:
    """P2: A/max(mean_time(A), 1)-1 отдельно для каждого окна и поднесущей."""
    x = _amplitude(windows)
    mean = np.maximum(x.mean(axis=-1, keepdims=True), 1.0)
    return np.asarray(x / mean - 1.0, dtype=np.float32)


def relative_diff(windows: np.ndarray) -> np.ndarray:
    """P3: первая разность P2 по времени; первый выходной отсчёт равен нулю."""
    x = relative(windows)
    return np.diff(x, axis=-1, prepend=x[..., :1]).astype(np.float32, copy=False)


def lp20_centered(windows: np.ndarray) -> np.ndarray:
    """P4: причинный ФНЧ Баттерворта 4-го порядка, 20 Гц, затем центрирование окна.

    Начальное состояние фильтра соответствует первой амплитуде каждого канала:
    так не возникает переходного процесса от нулевого начального состояния.
    Окна обрабатываются независимо, поэтому состояние и отсчёты не переходят
    между записями, доменами или выборками. Для центрирования используется
    целиком доступное окно из последних отсчётов.
    """
    x = _amplitude(windows)
    sos = butter(4, 20.0, btype="lowpass", fs=SAMPLE_RATE_HZ, output="sos")
    zi = sosfilt_zi(sos)[:, None, None, :] * x[None, ..., :1]
    filtered, _ = sosfilt(sos, x, axis=-1, zi=zi)
    return np.asarray(filtered - filtered.mean(axis=-1, keepdims=True), dtype=np.float32)


PREPROCESSORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "logamp": logamp,
    "relative": relative,
    "relative_diff": relative_diff,
    "lp20_centered": lp20_centered,
}
PREPROCESSOR_DESCRIPTIONS = {
    "logamp": "log(1+A); static amplitude and dynamics retained",
    "relative": "A/max(window time mean per subcarrier,1)-1",
    "relative_diff": "first time difference of relative; first sample zero",
    "lp20_centered": "causal 4th-order Butterworth LP20Hz at 100Hz; window-centered",
}
DEFAULT_PREPROCESSORS = ("logamp", "relative", "relative_diff")


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, (3, 7), padding=(1, 3)),
            nn.GroupNorm(4, 16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, (3, 5), padding=(1, 2)),
            nn.GroupNorm(4, 32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x.unsqueeze(1)))


class CausalConv(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = CausalConv(channels, dilation)
        self.conv2 = CausalConv(channels, dilation)
        # Нормализуем каналы в каждом отсчёте отдельно, без усреднения по времени.
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = F.relu(self.norm1(y.transpose(1, 2)).transpose(1, 2))
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.norm2(y.transpose(1, 2)).transpose(1, 2)
        return F.relu(x + self.dropout(y))


class CausalTCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Conv1d(CHANNELS, 32, 1)
        self.blocks = nn.Sequential(
            *[ResidualTCNBlock(32, dilation) for dilation in (1, 2, 4, 8, 16, 32, 64)]
        )
        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Рецептивное поле в 509 отсчётов охватывает всё окно длиной 400 отсчётов.
        return self.head(self.blocks(self.project(x))[:, :, -1])


class CNNGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(CHANNELS, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
        )
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, state = self.gru(self.conv(x).transpose(1, 2))
        return self.head(state[-1])


class TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokenize = nn.Conv1d(CHANNELS, 64, 8, stride=8)
        self.position = nn.Parameter(torch.zeros(1, SAMPLES // 8, 64))
        block = nn.TransformerEncoderLayer(
            64, 4, dim_feedforward=128, dropout=0.2,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, 2, enable_nested_tensor=False)
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenize(x).transpose(1, 2) + self.position
        return self.head(self.encoder(x).mean(dim=1))


MODELS: dict[str, type[nn.Module]] = {
    "tinycnn": TinyCNN,
    "tcn": CausalTCN,
    "cnngru": CNNGRU,
    "transformer": TinyTransformer,
}
MODEL_DESCRIPTIONS = {
    "tinycnn": "2D CNN16/32/64 + GroupNorm + GAP + linear",
    "tcn": "causal residual TCN32, dilations1..64, RF509, last-state head",
    "cnngru": "strided 1D CNN32/64 + GRU64 + last-state head",
    "transformer": "8-sample patch embedding64 + 2 encoder layers/4 heads + mean pooling",
}


def make_model(name: str) -> nn.Module:
    aliases = {"causaltcn": "tcn", "motiontcn": "tcn", "tinytransformer": "transformer"}
    name = aliases.get(name.lower(), name.lower())
    if name not in MODELS:
        raise KeyError(f"Неизвестная архитектура {name!r}; доступные варианты: {tuple(MODELS)}")
    return MODELS[name]()


def parameter_count(model: nn.Module) -> int:
    """Точное количество обучаемых скалярных параметров."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)



MotionTCN = CausalTCN
