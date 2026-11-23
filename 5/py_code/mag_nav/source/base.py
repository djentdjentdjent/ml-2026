"""Протокол и базовые примитивы источника магнитного поля.

Ответственность модуля — определить *контракт* источника и общие
чистые утилиты (трассировка силовых линий, построение локального базиса).
Конкретная физика живёт в :mod:`mag_nav.source.dipole` и
:mod:`mag_nav.source.solenoid`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..convention import EPS, TWO_PI, as_points, norm, safe_norm, unit

__all__ = [
    "MagneticSource",
    "orthonormal_frame",
    "rotation_to_frame",
    "trace_field_line",
]


class MagneticSource(ABC):
    """Контракт вращающегося магнитного маяка.

    Attributes
    ----------
    origin:
        Положение центра маяка в мировой системе, ``(3,)``, метры.
    dipole_constant:
        ``B0 = mu0 * m0 / (4*pi)``, Тл*м^3. Калибровочная константа,
        используемая обратным решением навигационной задачи.
    phase_offset:
        Начальная фаза ``phi0`` в ``m(t) = m0*(cos(wt+phi0) e1 + sin(wt+phi0) e2)``.
    is_quadrature_exact:
        ``True``, если поле линейно по ``(cos phi, sin phi)`` и потому
        ``a_vec = B(0)``, ``b_vec = B(pi/2)`` *точно*. Для источников
        конечных размеров — ``False`` (нужна гармоническая экстракция).
    """

    origin: NDArray[np.float64]
    dipole_constant: float
    phase_offset: float
    is_quadrature_exact: bool

    @abstractmethod
    def field_at_phase(
        self, points: ArrayLike, phase: float = 0.0
    ) -> NDArray[np.float64]:
        """Мгновенное поле ``B`` [Тл] в точках ``points`` при фазе ``phase`` [рад].

        Точки задаются в **мировой** системе; источник сам вычитает ``origin``.
        Форма входа ``(..., 3)`` → форма выхода ``(..., 3)``.
        """

    def field(self, points: ArrayLike) -> NDArray[np.float64]:
        """Поле при ``phase = phase_offset`` (удобная обёртка)."""
        return self.field_at_phase(points, self.phase_offset)

    def magnitude(self, points: ArrayLike, phase: float = 0.0) -> NDArray[np.float64]:
        """Модуль ``|B|`` [Тл]."""
        return norm(self.field_at_phase(points, phase))

    def relative_points(self, points: ArrayLike) -> NDArray[np.float64]:
        """Радиус-векторы точек относительно ``origin``."""
        return as_points(points) - self.origin

    def __repr__(self) -> str:  # pragma: no cover - косметика
        return f"{type(self).__name__}(origin={self.origin.tolist()}, B0={self.dipole_constant:.6e})"


# --------------------------------------------------------------------------- #
# Геометрические утилиты
# --------------------------------------------------------------------------- #
def orthonormal_frame(axis: ArrayLike) -> NDArray[np.float64]:
    """Построить правую ортонормированную тройку ``(i, j, k)`` с ``k || axis``.

    Returns
    -------
    R : ndarray, shape (3, 3)
        Матрица, **столбцы** которой — ``i, j, k``. Преобразование
        ``p_world = R @ p_local``.
    """
    k = unit(np.asarray(axis, dtype=np.float64))
    # Для оси ~Z helper = Y, чтобы (i, j) = (X, Y), как обещано в конвенции.
    helper = np.array([0.0, 0.0, 1.0]) if abs(k[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    i = unit(np.cross(helper, k))
    j = np.cross(k, i)
    return np.stack([i, j, k], axis=1)


def rotation_to_frame(axis: ArrayLike) -> NDArray[np.float64]:
    """Матрица поворота локального базиса в мировой для оси, заданной углом в XY.

    ``axis`` — ``(3,)`` единичный вектор (ось катушки/момента).
    """
    return orthonormal_frame(axis)


# --------------------------------------------------------------------------- #
# Трассировка силовых линий (аналог MagneticLine.m)
# --------------------------------------------------------------------------- #
def _rk4_step(
    field_dir: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    p: NDArray[np.float64],
    h: float,
) -> NDArray[np.float64]:
    k1 = field_dir(p)
    k2 = field_dir(p + 0.5 * h * k1)
    k3 = field_dir(p + 0.5 * h * k2)
    k4 = field_dir(p + h * k3)
    return p + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def trace_field_line(
    field_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    start: ArrayLike,
    *,
    step: float = 5e-3,
    n_steps: int = 800,
    bidirectional: bool = True,
    stop_radius: float | None = None,
    max_radius: float | None = None,
) -> NDArray[np.float64]:
    """Интегрировать ``dr/ds = B(r)/|B(r)|`` методом Рунге–Кутты 4-го порядка.

    Parameters
    ----------
    field_fn:
        Чистая функция ``B(r)`` для одной точки ``(3,) -> (3,)``.
    bidirectional:
        Трассировать в обе стороны и склеить (линия проходит через ``start``).
    stop_radius:
        Прервать, если ``|r| < stop_radius`` (уход в источник).
    max_radius:
        Прервать, если ``|r| > max_radius`` (уход на бесконечность).

    Returns
    -------
    ndarray, shape (n, 3) — полилиния.
    """

    def direction(p: NDArray[np.float64]) -> NDArray[np.float64]:
        b = np.asarray(field_fn(p), dtype=np.float64).ravel()
        nb = np.linalg.norm(b)
        if nb < EPS:
            return np.zeros(3)
        return b / nb

    def integrate(p0: NDArray[np.float64], sign: float) -> list[NDArray[np.float64]]:
        pts = [p0.copy()]
        p = p0.copy()
        for _ in range(n_steps):
            p_new = _rk4_step(direction, p, sign * step)
            if not np.all(np.isfinite(p_new)) or np.allclose(p_new, p):
                break
            p = p_new
            pts.append(p.copy())
            rn = float(np.linalg.norm(p))
            if stop_radius is not None and rn < stop_radius:
                break
            if max_radius is not None and rn > max_radius:
                break
        return pts

    p_start = np.asarray(start, dtype=np.float64).ravel().astype(float)
    forward = integrate(p_start, +1.0)
    if not bidirectional:
        return np.asarray(forward)
    backward = integrate(p_start, -1.0)
    # backward[0] == start уже есть в forward — без него, иначе нулевой сегмент
    return np.asarray(list(reversed(backward[1:])) + forward)


def field_line_tracer(
    source: "MagneticSource", phase: float = 0.0
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Замкнуть ``source.field_at_phase`` в функцию одной точки."""

    def _f(p: NDArray[np.float64]) -> NDArray[np.float64]:
        return source.field_at_phase(p.reshape(1, 3), phase).ravel()

    return _f


def safe_norm_of(v: ArrayLike) -> NDArray[np.float64]:
    return safe_norm(v)