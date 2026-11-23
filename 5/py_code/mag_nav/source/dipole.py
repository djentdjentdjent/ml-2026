"""Точечный вращающийся магнитный диполь — основная модель маяка.

Воспроизводит ``MagneticMoment.m`` / ``MgMomentField.m``:

.. math::

    \\vec B(\\vec r) = \\frac{\\mu_0}{4\\pi}
        \\left[ \\frac{3\\,\\vec r\\,(\\vec m\\cdot\\vec r)}{|\\vec r|^5}
              - \\frac{\\vec m}{|\\vec r|^3} \\right]
      = \\frac{B_0}{r^3}\\left[3(\\hat m\\cdot\\hat r)\\hat r - \\hat m\\right]

где :math:`B_0 = \\mu_0 m_0/(4\\pi)`, :math:`\\hat m` — единичный вектор момента.

Для момента, вращающегося в плоскости XY:

.. math::

    \\vec B(\\vec r,\\varphi) = \\frac{B_0}{r^3}\\big[\\vec a\\cos\\varphi
                              + \\vec b\\sin\\varphi\\big],\\qquad
    \\vec a = 3u\\hat r - \\hat x,\\quad \\vec b = 3v\\hat r - \\hat y

с направляющими косинусами :math:`u=x/r,\\;v=y/r,\\;w=z/r`.
Отсюда — инварианты, используемые обратным решением:

.. math::

    |\\vec a|^2 = \\kappa(3u^2+1),\\quad
    |\\vec b|^2 = \\kappa(3v^2+1),\\quad
    \\vec a\\cdot\\vec b = 3\\kappa u v,\\quad \\kappa = B_0^2/r^6 .
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..convention import (
    AXIS_X,
    AXIS_Y,
    AXIS_Z,
    MU_0,
    TWO_PI,
    QuadrantPolicy,
    as_points,
    dipole_constant as _b0_from_moment,
    direction_cosines,
    enforce_quadrant,
    safe_norm,
    unit,
)
from .base import MagneticSource, orthonormal_frame

__all__ = [
    "RotatingDipole",
    "dipole_field",
    "dipole_ab_unit_vectors",
    "dipole_ab_vectors",
    "dipole_invariants",
]


def dipole_field(
    rel_points: ArrayLike,
    moment_vec: ArrayLike,
    b0: float,
) -> NDArray[np.float64]:
    """Чистая функция поля точечного диполя в начале координат.

    Parameters
    ----------
    rel_points : (..., 3)
        Радиус-векторы точек **относительно диполя**, м.
    moment_vec : (3,)
        **Единичный** вектор направления магнитного момента.
    b0 : float
        Дипольная константа ``mu0*m0/(4*pi)``, Тл*м^3.

    Returns
    -------
    (..., 3) — вектор ``B``, Тл.
    """
    p = as_points(rel_points)
    m_hat = unit(np.asarray(moment_vec, dtype=np.float64))
    r = safe_norm(p, keepdims=True)
    r_hat = p / r
    m_dot_r = np.sum(m_hat * r_hat, axis=-1, keepdims=True)
    return (b0 / r**3) * (3.0 * m_dot_r * r_hat - m_hat)


def dipole_ab_unit_vectors(uvw: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Безразмерные ``(a_hat, b_hat)`` по направляющим косинусам.

    ``a_hat = 3*u*r_hat - x_hat``, ``b_hat = 3*v*r_hat - y_hat``.
    """
    u_v_w = as_points(uvw)
    u = u_v_w[..., 0:1]
    v = u_v_w[..., 1:2]
    a_hat = 3.0 * u * u_v_w - AXIS_X
    b_hat = 3.0 * v * u_v_w - AXIS_Y
    return a_hat, b_hat


def dipole_ab_vectors(
    rel_points: ArrayLike, b0: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Измеряемые (масштабированные) ``a_vec``, ``b_vec`` [Тл] для точечного диполя.

    Точный аналитический аналог ``field_at_phase(0)`` и ``field_at_phase(pi/2)``.
    """
    p = as_points(rel_points)
    r = safe_norm(p, keepdims=True)
    a_hat, b_hat = dipole_ab_unit_vectors(p / r)
    scale = b0 / r**3
    return scale * a_hat, scale * b_hat


def dipole_invariants(
    rel_points: ArrayLike, b0: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``(kappa, asq, bsq, ab)`` — теоретические инварианты для самопроверки."""
    p = as_points(rel_points)
    r = safe_norm(p, keepdims=True)
    uvw = p / r
    kappa = (b0 / r**3) ** 2
    asq = kappa * (3.0 * uvw[..., 0] ** 2 + 1.0)
    bsq = kappa * (3.0 * uvw[..., 1] ** 2 + 1.0)
    ab = 3.0 * kappa[..., 0] * uvw[..., 0] * uvw[..., 1]
    return kappa[..., 0], asq, bsq, ab


@dataclass(frozen=True)
class RotatingDipole(MagneticSource):
    """Точечный диполь с моментом, вращающимся вокруг заданной оси.

    Parameters
    ----------
    origin:
        Центр маяка, ``(3,)``, м.
    moment_amplitude:
        ``m0`` [А*м^2]. Альтернатива — задать ``b0`` напрямую.
    b0:
        Дипольная константа ``mu0*m0/(4*pi)`` [Тл*м^3]. Если ``None``,
        вычисляется из ``moment_amplitude``.
    rotation_axis:
        Ось вращения момента. По умолчанию ``Z`` (момент вращается в XY) —
        это соглашение, под которое выведено обратное решение.
    phase_offset:
        Начальная фаза ``phi0``, рад.
    quadrant_policy:
        Политика обработки точек вне первого квадранта.

    Notes
    -----
    При ``rotation_axis = Z`` базис вращения ``(e1, e2) = (X, Y)`` и
    ``is_quadrature_exact = True``.
    """

    origin: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    moment_amplitude: float = 1.0
    b0: float | None = None
    rotation_axis: NDArray[np.float64] = field(
        default_factory=lambda: AXIS_Z.copy()
    )
    phase_offset: float = 0.0
    quadrant_policy: QuadrantPolicy = QuadrantPolicy.STRICT

    is_quadrature_exact: Final[bool] = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", as_points(self.origin).reshape(3))
        object.__setattr__(self, "rotation_axis", unit(np.asarray(self.rotation_axis, float)))
        object.__setattr__(
            self, "quadrant_policy", QuadrantPolicy(self.quadrant_policy)
        )
        if self.b0 is None:
            object.__setattr__(self, "b0", _b0_from_moment(self.moment_amplitude))

    # -- свойства контракта -------------------------------------------------- #
    @property
    def dipole_constant(self) -> float:
        assert self.b0 is not None
        return float(self.b0)

    @property
    def moment_amplitude_value(self) -> float:
        """``m0`` [А*м^2], соответствующий текущему ``b0``."""
        return float(self.dipole_constant * 4.0 * np.pi / MU_0)

    @property
    def rotation_frame(self) -> NDArray[np.float64]:
        """Матрица ``(3, 3)``, столбцы — ``(e1, e2, axis)``."""
        r = orthonormal_frame(self.rotation_axis)
        # orthonormal_frame даёт (i, j, k) с k || axis -> переставим так,
        # чтобы ось была третьей, а e1, e2 лежали в плоскости вращения.
        return np.stack([r[:, 0], r[:, 1], r[:, 2]], axis=1)

    # -- физика -------------------------------------------------------------- #
    def moment_direction(self, phase: float) -> NDArray[np.float64]:
        """Единичный вектор момента ``m_hat(phi)`` в мировой системе."""
        e1, e2 = self.rotation_frame[:, 0], self.rotation_frame[:, 1]
        return np.cos(phase) * e1 + np.sin(phase) * e2

    def field_at_phase(
        self, points: ArrayLike, phase: float = 0.0
    ) -> NDArray[np.float64]:
        rel = self.relative_points(points)
        rel = enforce_quadrant(rel, self.quadrant_policy) if self._check_quadrant else rel
        m_hat = self.moment_direction(phase + self.phase_offset)
        return dipole_field(rel, m_hat, self.dipole_constant)

    @property
    def _check_quadrant(self) -> bool:
        return self.quadrant_policy is not QuadrantPolicy.OFF

    # -- a/b параметры ------------------------------------------------------- #
    def ab_vectors(
        self, points: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(a_vec, b_vec)`` в системе маяка, Тл. Точная аналитика."""
        rel = self.relative_points(points)
        if self.quadrant_policy is QuadrantPolicy.STRICT:
            rel = enforce_quadrant(rel, QuadrantPolicy.STRICT)
        if not np.allclose(self.rotation_axis, AXIS_Z):
            raise NotImplementedError(
                "Аналитические a/b выведены для оси вращения Z. "
                "Используйте ab_from_source() для произвольной оси."
            )
        return dipole_ab_vectors(rel, self.dipole_constant)

    def time_series(
        self, points: ArrayLike, phases: ArrayLike
    ) -> NDArray[np.float64]:
        """``B(t)`` формы ``(n_phases, ..., 3)`` — сигнал на приёмнике."""
        a_vec, b_vec = self.ab_vectors(points)
        ph = np.asarray(phases, dtype=np.float64).reshape(-1, 1, 1)
        return np.cos(ph) * a_vec[np.newaxis, ...] + np.sin(ph) * b_vec[np.newaxis, ...]

    # -- производные величины ------------------------------------------------ #
    def magnitude_envelope(
        self, points: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(|B|min, |B|max)`` — полуоси эллипса поляризации поля."""
        a_vec, b_vec = self.ab_vectors(points)
        asq = np.sum(a_vec**2, axis=-1)
        bsq = np.sum(b_vec**2, axis=-1)
        ab = np.sum(a_vec * b_vec, axis=-1)
        tr = asq + bsq
        det = asq * bsq - ab**2
        disc = np.sqrt(np.maximum(tr**2 - 4.0 * det, 0.0))
        s_max = np.sqrt(np.maximum(0.5 * (tr + disc), 0.0))
        s_min = np.sqrt(np.maximum(0.5 * (tr - disc), 0.0))
        return s_min, s_max

    def with_moment(self, moment_amplitude: float) -> "RotatingDipole":
        """Копия с другой амплитудой момента (иммутабельность сохранена)."""
        return RotatingDipole(
            origin=self.origin.copy(),
            moment_amplitude=moment_amplitude,
            b0=_b0_from_moment(moment_amplitude),
            rotation_axis=self.rotation_axis.copy(),
            phase_offset=self.phase_offset,
            quadrant_policy=self.quadrant_policy,
        )


# --------------------------------------------------------------------------- #
# Convenience-конструкторы
# --------------------------------------------------------------------------- #
def dipole_from_current_loop(
    origin: ArrayLike,
    area: float,
    n_turns: int,
    current: float,
    **kwargs,
) -> RotatingDipole:
    """Диполь из эквивалентного витка: ``m0 = N * I * S``."""
    m0 = float(n_turns) * float(current) * float(area)
    return RotatingDipole(
        origin=as_points(origin).reshape(3),
        moment_amplitude=m0,
        b0=_b0_from_moment(m0),
        **kwargs,
    )