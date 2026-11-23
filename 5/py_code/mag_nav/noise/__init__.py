"""Модели шума: чистые преобразования сигнала, ГСЧ инжектируется извне."""

from .base import AdditiveNoise, ConstantBias, NoiseModel
from .composite import (
    ClipNoise,
    CompositeNoise,
    NoiseChain,
    QuantizationNoise,
    SumNoise,
)
from .gaussian import GaussianNoise

__all__ = [
    "NoiseModel",
    "AdditiveNoise",
    "ConstantBias",
    "GaussianNoise",
    "SumNoise",
    "CompositeNoise",
    "NoiseChain",
    "QuantizationNoise",
    "ClipNoise",
]