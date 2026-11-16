"""Конвенции проекта ``mag_nav``.

Единственная ответственность модуля — зафиксировать *соглашения*:
систему координат, физические константы, формы массивов и лёгкие
утилиты приведения/валидации. Физики здесь нет намеренно.

Соглашения
----------
1. Система координат маяка (beacon frame) — правая декартова ``(x, y, z)``,
   начало отсчёта — в центре маяка.
2. Магнитный момент маяка вращается в плоскости **XY**, ось вращения — **Z**::

       m(t) = m0 * ( cos(w t + phi0) * ex + sin(w t + phi0) * ey )

3. Рабочая область — **1-й квадрант (октант)**: ``x >= 0, y >= 0, z >= 0``.
   Именно это предположение снимает знаковую неоднозначность обратного
   решения (см. :mod:`mag_nav.navigation.position`).
4. Точки и поля задаются массивами формы ``(..., 3)``; поле — в Тесла.
5. ``a``- и ``b``-параметры — векторные коэффициенты первой гармоники::

       B(t) = a_vec * cos(w t) + b_vec * sin(w t)

   Для точечного диполя это тождественно ``B(phi=0)`` и ``B(phi=pi/2)``.
6. Дипольная константа маяка ``B0 = mu0 * m0 / (4*pi)`` [Тл*м^3] —
   калибровочный параметр, необходимый только для определения *дальности*.
   Направление (x:y:z) от калибровки не зависит.
"""

from __future__ import annotations

from enum import Enum
from typing import Final, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

# --------------------------------------------------------------------------- #
# Константы
# --------------------------------------------------------------------------- #
MU_0: Final[float] = 4.0e-7 * np.pi
"""Магнитная постоянная, Гн/м (точно как в MATLAB: ``4*pi*1e-7``)."""

TWO_PI: Final[float] = 2.0 * np.pi
QUARTER_PI: Final[float] = 0.5 * np.pi

EPS: Final[float] = 1e-300
"""Защита от деления на ноль (не порог физический, а машинный)."""

AXIS_X: Final[NDArray[np.float64]] = np.array([1.0, 0.0, 0.0])
AXIS_Y: Final[NDArray[np.float64]] = np.array([0.0, 1.0, 0.0])
AXIS_Z: Final[NDArray[np.float64]] = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------- #
# Лёгкие типы для скалярного API
# --------------------------------------------------------------------------- #
class Position(NamedTuple):
    """Точка в пространстве, метры."""

    x: float
    y: float
    z: float

    def as_array(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y, self.z], dtype=float)

    @classmethod
    def from_array(cls, arr: ArrayLike) -> "Position":
        a = np.asarray(arr, dtype=float).ravel()
        return cls(float(a[0]), float(a[1]), float(a[2]))


class FieldVec(NamedTuple):
    """Вектор магнитной индукции, Тесла."""

    bx: float
    by: float
    bz: float

    def as_array(self) -> NDArray[np.float64]:
        return np.array([self.bx, self.by, self.bz], dtype=float)

    @classmethod
    def from_array(cls, arr: ArrayLike) -> "FieldVec":
        a = np.asarray(arr, dtype=float).ravel()
        return cls(float(a[0]), float(a[1]), float(a[2]))


# --------------------------------------------------------------------------- #
# Политика первого квадранта
# --------------------------------------------------------------------------- #
class QuadrantPolicy(str, Enum):
    """Что делать с точками вне первого октанта."""

    STRICT = "strict"   # бросить исключение
    CLIP = "clip"       # прижать к >= 0
    RELAXED = "relaxed"  # допустить y < 0 (знак восстанавливается через sign(a·b))
    OFF = "off"         # никакой обработки


def dipole_constant(moment_amplitude: float) -> float:
    """``B0 = mu0 * m0 / (4*pi)`` — из амплитуды магнитного момента [А*м^2]."""
    return MU_0 * float(moment_amplitude) / (4.0 * np.pi)


def moment_from_constant(b0: float) -> float:
    """Обратное преобразование к :func:`dipole_constant`."""
    return float(b0) * 4.0 * np.pi / MU_0


# --------------------------------------------------------------------------- #
# Утилиты работы с массивами (чистые функции)
# --------------------------------------------------------------------------- #
def as_points(points: ArrayLike) -> NDArray[np.float64]:
    """Привести вход к ``float64``-массиву формы ``(..., 3)``."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.shape[-1] != 3:
        raise ValueError(f"Ожидалась последняя ось размера 3, получено {arr.shape}")
    return arr


def ensure_shape(arr: ArrayLike, ndim_last: int = 3) -> NDArray[np.float64]:
    a = np.asarray(arr, dtype=np.float64)
    if a.shape[-1] != ndim_last:
        raise ValueError(f"Ожидалась последняя ось размера {ndim_last}, получено {a.shape}")
    return a


def norm(vec: ArrayLike, *, axis: int = -1, keepdims: bool = False) -> NDArray[np.float64]:
    """Евклидова норма по последней оси (без ``NaN`` при нулевом векторе)."""
    a = np.asarray(vec, dtype=np.float64)
    return np.sqrt(np.sum(a * a, axis=axis, keepdims=keepdims))


def safe_norm(vec: ArrayLike, *, axis: int = -1, keepdims: bool = False) -> NDArray[np.float64]:
    """То же, что :func:`norm`, но нули заменяются на ``EPS`` (защита от 0/0)."""
    n = norm(vec, axis=axis, keepdims=keepdims)
    return np.maximum(n, EPS)


def unit(vec: ArrayLike, *, axis: int = -1) -> NDArray[np.float64]:
    """Единичный вектор вдоль ``axis``."""
    a = np.asarray(vec, dtype=np.float64)
    return a / safe_norm(a, axis=axis, keepdims=True)


def direction_cosines(rel_points: ArrayLike) -> NDArray[np.float64]:
    """``(u, v, w) = (x, y, z)/r`` — направляющие косинусы радиус-вектора."""
    p = as_points(rel_points)
    return p / safe_norm(p, keepdims=True)


def enforce_quadrant(
    points: ArrayLike,
    policy: QuadrantPolicy | str = QuadrantPolicy.STRICT,
    *,
    tol: float = 0.0,
) -> NDArray[np.float64]:
    """Применить политику первого квадранта к массиву точек ``(..., 3)``.

    Parameters
    ----------
    policy:
        * ``STRICT``  — исключение, если ``x < -tol`` или ``z < -tol``
          (``y`` допускается отрицательным: его знак восстанавливается
          аналитически через ``sign(a·b)``);
        * ``CLIP``    — отрицательные ``x, z`` обнуляются;
        * ``RELAXED`` — только предупреждающая проверка ``x, z``;
        * ``OFF``     — массив возвращается как есть.
    """
    pol = QuadrantPolicy(policy)
    p = as_points(points).copy()
    if pol is QuadrantPolicy.OFF:
        return p

    checked = np.stack([p[..., 0], p[..., 1], p[..., 2]], axis=-1)
    bad_xz = (checked[..., 0] < -tol) | (checked[..., 2] < -tol)

    if pol is QuadrantPolicy.STRICT and np.any(bad_xz):
        idx = np.argwhere(bad_xz)
        raise ValueError(
            f"Точки вне первого квадранта (x>=0, z>=0): {len(idx)} шт., "
            f"первая — индекс {tuple(idx[0])}. Используйте QuadrantPolicy.CLIP/RELAXED."
        )
    if pol is QuadrantPolicy.CLIP:
        p[..., 0] = np.maximum(p[..., 0], 0.0)
        p[..., 2] = np.maximum(p[..., 2], 0.0)
    return p


def broadcast_sigma(sigma: ArrayLike, shape: tuple[int, ...]) -> NDArray[np.float64]:
    """Развернуть скаляр/вектор СКО в массив формы ``shape``."""
    s = np.asarray(sigma, dtype=np.float64)
    if s.ndim == 0:
        return np.full(shape, float(s))
    return np.broadcast_to(s, shape).astype(np.float64, copy=True)