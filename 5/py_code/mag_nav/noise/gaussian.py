"""Гауссов аддитивный шум магнитометра.

Два режима:

``relative=False`` (по умолчанию)
    Абсолютное СКО в Тесла — модель собственного шума датчика.

``relative=True``
    СКО пропорционально модулю измеряемого вектора:
    ``sigma_i = sigma * ||B_i||``. Модель мультипликативной погрешности
    (дрейб чувствительности, scale-factor error).

Дополнительно поддерживается **межосевая корреляция** через ``cov``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..convention import broadcast_sigma, safe_norm
from .base import AdditiveNoise

__all__ = ["GaussianNoise"]


@dataclass(frozen=True)
class GaussianNoise(AdditiveNoise):
    """Аддитивный гауссов шум.

    Parameters
    ----------
    sigma:
        СКП. Скаляр (одинаков по всем осям), либо массив формы ``(3,)`` /
        ``(..., 3)`` для покомпонентного задания.
    relative:
        Если ``True`` — ``sigma`` трактуется как относительная величина
        (доля от ``||signal||``).
    cov:
        Ковариационная матрица ``(3, 3)``. Если задана, ``sigma`` игнорируется
        для формирования формы корреляции (используется как ``L @ z``).
    clip_sigma:
        Множитель обрезки выбросов (``0`` — без обрезки). Полезен, чтобы
        редкие ``5*sigma``-выбросы не ломали статистику.
    """

    sigma: ArrayLike = 0.0
    relative: bool = False
    cov: NDArray[np.float64] | None = None
    clip_sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.cov is not None:
            c = np.asarray(self.cov, dtype=np.float64)
            if c.shape != (3, 3):
                raise ValueError(f"cov должна быть (3, 3), получено {c.shape}")
            if not np.allclose(c, c.T, atol=1e-15):
                raise ValueError("cov должна быть симметричной")
            ev = np.linalg.eigvalsh(c)
            if np.min(ev) < -1e-18:
                raise ValueError("cov не является неотрицательно определённой")
            object.__setattr__(self, "cov", c)

    # -- генерация ----------------------------------------------------------- #
    def _white(self, shape: tuple[int, ...], rng: np.random.Generator) -> NDArray[np.float64]:
        z = rng.standard_normal(shape)
        if self.clip_sigma > 0:
            z = np.clip(z, -self.clip_sigma, self.clip_sigma)
        return z

    def sample(
        self,
        shape: tuple[int, ...],
        rng: np.random.Generator,
        *,
        reference: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        if self.cov is not None:
            L = np.linalg.cholesky(self.cov + 1e-300 * np.eye(3))
            z = self._white(shape, rng)                # (..., 3)
            out = z @ L.T
        else:
            z = self._white(shape, rng)
            sigma = np.asarray(self.sigma, dtype=np.float64)
            if sigma.ndim == 0 and not self.relative:
                # Для обычного скалярного шума не нужен массив sigma размером
                # со всем сигналом (временной ряд × точки × оси).
                out = float(sigma) * z
            else:
                out = broadcast_sigma(sigma, shape) * z

        if self.relative:
            if reference is None:
                raise ValueError("relative=True требует reference (исходный сигнал)")
            scale = safe_norm(np.asarray(reference, dtype=np.float64), axis=-1, keepdims=True)
            if self.cov is not None:
                out = out * scale / max(float(np.sqrt(np.trace(self.cov) / 3.0)), 1e-300)
            else:
                out = out * scale / np.maximum(
                    broadcast_sigma(self.sigma, shape), 1e-300
                ) * broadcast_sigma(self.sigma, shape)
        return out

    # -- мета ---------------------------------------------------------------- #
    @property
    def effective_sigma(self) -> float:
        if self.cov is not None:
            return float(np.sqrt(np.trace(self.cov) / 3.0))
        s = np.asarray(self.sigma, dtype=np.float64)
        return float(np.sqrt(np.mean(s**2)))

    def sigma_map(
        self, reference: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Эффективное покомпонентное СКО для заданного сигнала."""
        if self.cov is not None:
            base = np.sqrt(np.diag(self.cov))
        else:
            base = np.asarray(self.sigma, dtype=np.float64)
        if self.relative and reference is not None:
            scale = safe_norm(reference, axis=-1, keepdims=True)
            return base * scale
        return base

    def describe(self) -> str:  # pragma: no cover
        kind = "relative" if self.relative else "absolute"
        return f"GaussianNoise(sigma={self.effective_sigma:.3e}, {kind})"