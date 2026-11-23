"""Контракт моделей шума + тривиальная модель постоянного смещения.

Побочные эффекты (ГСЧ) изолированы: ``numpy.random.Generator`` всегда
передаётся **извне**, модели шума сами его не создают и не хранят.
Благодаря этому :mod:`mag_nav.noise` остаётся набором чистых преобразований.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..convention import broadcast_sigma, safe_norm

__all__ = [
    "NoiseModel",
    "AdditiveNoise",
    "ConstantBias",
    "SigmaMap",
]

SigmaMap = NDArray[np.float64]


class NoiseModel(ABC):
    """Базовый контракт: ``signal -> noisy_signal``."""

    @abstractmethod
    def apply(
        self, signal: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Применить шум к сигналу. Чистая функция от ``(signal, rng.state)``."""

    @property
    @abstractmethod
    def effective_sigma(self) -> float:
        """Характерное СКО [в тех же единицах, что сигнал] для отчётов."""

    def describe(self) -> str:  # pragma: no cover - косметика
        return f"{type(self).__name__}(sigma={self.effective_sigma:.3e})"


class AdditiveNoise(NoiseModel):
    """Аддитивная помеха: ``apply(s) = s + sample(shape(s))``."""

    @abstractmethod
    def sample(
        self,
        shape: tuple[int, ...],
        rng: np.random.Generator,
        *,
        reference: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Сгенерировать реализацию помехи заданной формы."""

    def apply(
        self, signal: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        return np.asarray(signal, dtype=np.float64) + self.sample(
            signal.shape, rng, reference=signal
        )


@dataclass(frozen=True)
class ConstantBias(AdditiveNoise):
    """Детерминированное смещение (offset АЦП, остаточный фон).

    ``bias`` может быть скаляром или вектором формы ``(3,)`` — тогда
    смещение применяется покомпонентно к последней оси сигнала.
    """

    bias: ArrayLike = 0.0

    def sample(
        self,
        shape: tuple[int, ...],
        rng: np.random.Generator,
        *,
        reference: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        b = np.asarray(self.bias, dtype=np.float64)
        return np.broadcast_to(b, shape).astype(np.float64, copy=True)

    @property
    def effective_sigma(self) -> float:
        b = np.asarray(self.bias, dtype=np.float64)
        return float(np.linalg.norm(b))