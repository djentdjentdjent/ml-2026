"""Композиция моделей шума.

``SumNoise`` (публичный алиас ``CompositeNoise``)
    Аддитивная сумма независимых помех — собственная схема ГСЧ для каждой.

``NoiseChain``
    Последовательное применение: выход предыдущего звена — вход следующего
    (например, «шум датчика → квантование АЦП → ограничение диапазона»).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .base import AdditiveNoise, NoiseModel

__all__ = ["SumNoise", "CompositeNoise", "NoiseChain", "QuantizationNoise", "ClipNoise"]


@dataclass(frozen=True)
class SumNoise(AdditiveNoise):
    """Сумма независимых аддитивных помех."""

    models: tuple[AdditiveNoise, ...] = ()

    def __init__(self, models: Sequence[AdditiveNoise]) -> None:
        for m in models:
            if not isinstance(m, AdditiveNoise):
                raise TypeError(
                    f"SumNoise принимает только AdditiveNoise, получено {type(m).__name__}. "
                    "Для последовательного применения используйте NoiseChain."
                )
        object.__setattr__(self, "models", tuple(models))

    def sample(
        self,
        shape: tuple[int, ...],
        rng: np.random.Generator,
        *,
        reference: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        out = np.zeros(shape, dtype=np.float64)
        for m in self.models:
            out += m.sample(shape, rng, reference=reference)
        return out

    @property
    def effective_sigma(self) -> float:
        return float(np.sqrt(sum(m.effective_sigma**2 for m in self.models)))

    def describe(self) -> str:  # pragma: no cover
        return "SumNoise[" + ", ".join(m.describe() for m in self.models) + "]"


CompositeNoise = SumNoise  # публичный алиас, соответствующий имени файла модуля


@dataclass(frozen=True)
class NoiseChain(NoiseModel):
    """Последовательное применение моделей."""

    models: tuple[NoiseModel, ...] = ()

    def __init__(self, models: Sequence[NoiseModel]) -> None:
        if not models:
            raise ValueError("NoiseChain требует хотя бы одну модель")
        object.__setattr__(self, "models", tuple(models))

    def apply(
        self, signal: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        out = np.asarray(signal, dtype=np.float64).copy()
        for m in self.models:
            out = m.apply(out, rng)
        return out

    @property
    def effective_sigma(self) -> float:
        return float(np.sqrt(sum(m.effective_sigma**2 for m in self.models)))

    def describe(self) -> str:  # pragma: no cover
        return "NoiseChain[" + " -> ".join(m.describe() for m in self.models) + "]"


# --------------------------------------------------------------------------- #
# Дополнительные детерминированные звенья цепи
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QuantizationNoise(NoiseModel):
    """Квантование АЦП с шагом ``lsb`` (детерминированное, без ГСЧ)."""

    lsb: float

    def apply(
        self, signal: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        if self.lsb <= 0:
            return np.asarray(signal, dtype=np.float64).copy()
        return np.round(np.asarray(signal, dtype=np.float64) / self.lsb) * self.lsb

    @property
    def effective_sigma(self) -> float:
        """СКП равномерного распределения на шаге квантования."""
        return float(self.lsb / np.sqrt(12.0))


@dataclass(frozen=True)
class ClipNoise(NoiseModel):
    """Ограничение диапазона измерений (насыщение входного каскада)."""

    limit: float

    def apply(
        self, signal: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        return np.clip(np.asarray(signal, dtype=np.float64), -abs(self.limit), abs(self.limit))

    @property
    def effective_sigma(self) -> float:
        return 0.0