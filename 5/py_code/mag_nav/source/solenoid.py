"""Конечный соленоид (катушка) как уточнённая модель маяка.

Поле считается суперпозицией полей круговых витков (точные формулы через
полные эллиптические интегралы I и II рода) с квадратурой Гаусса–Лежандра
по длине намотки:

.. math::

    \\vec B_{sol}(\\rho,\\zeta) = \\frac{N I}{2}\\sum_{i=1}^{n_q} w_i\\,
        \\vec B_{loop}^{(I=1)}\\big(\\rho,\\;\\zeta-\\tfrac{L}{2}t_i\\big)

Поле одиночного витка радиуса ``a`` с током ``I`` в цилиндрических
координатах (виток в плоскости z=0, ось — z):

.. math::

    B_\\zeta = \\frac{\\mu_0 I}{2\\pi}\\frac{1}{\\alpha}
        \\left[K(m) + \\frac{a^2-\\rho^2-\\zeta^2}{\\beta^2}E(m)\\right]

    B_\\rho = \\frac{\\mu_0 I}{2\\pi}\\frac{\\zeta}{\\rho\\,\\alpha}
        \\left[-K(m) + \\frac{a^2+\\rho^2+\\zeta^2}{\\beta^2}E(m)\\right]

где ``alpha^2=(a+rho)^2+zeta^2``, ``beta^2=(a-rho)^2+zeta^2``, ``m=k^2=4*a*rho/alpha^2``.

Вращение катушки как жёсткого тела вокруг оси Z приводит к *несинусоидальной*
зависимости ``B(phi)``; поэтому ``is_quadrature_exact = False`` и a/b
извлекаются первой гармоникой (см. :mod:`mag_nav.signal.ab_params`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ellipe, ellipk

from ..convention import (
    AXIS_Z,
    MU_0,
    TWO_PI,
    QuadrantPolicy,
    as_points,
    direction_cosines,
    enforce_quadrant,
    safe_norm,
    unit,
)
from .base import MagneticSource, orthonormal_frame
from .dipole import RotatingDipole, dipole_field

__all__ = ["Solenoid", "loop_field_cylindrical", "gauss_legendre"]


# --------------------------------------------------------------------------- #
# Квадратура
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def gauss_legendre(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Узлы и веса Гаусса–Лежандра на ``[-1, 1]`` (кэшируются)."""
    t, w = np.polynomial.legendre.leggauss(int(n))
    return t.astype(np.float64), w.astype(np.float64)


# --------------------------------------------------------------------------- #
# Поле одиночного витка
# --------------------------------------------------------------------------- #
_SMALL_RHO: Final[float] = 1e-6


def loop_field_cylindrical(
    rho: ArrayLike,
    zeta: ArrayLike,
    radius: float,
    current: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Точное поле кругового витка в цилиндрических координатах.

    Returns
    -------
    (B_rho, B_zeta), Тл.

    Notes
    -----
    В окрестности оси (``rho -> 0``) используется ряд, т.к. прямая формула
    даёт ``0/0`` с катастрофической потерей значащих цифр:

    .. math:: B_\\rho \\approx \\frac{3\\mu_0 I a^2 \\rho\\,\\zeta}{4(a^2+\\zeta^2)^{5/2}}
    """
    rho_a = np.asarray(rho, dtype=np.float64)
    z_a = np.asarray(zeta, dtype=np.float64)
    # Только читаем broadcast-view: копировать входные массивы здесь не нужно.
    rho_b, z_b = np.broadcast_arrays(rho_a, z_a)

    a = float(radius)
    a2 = a * a
    s = a2 + rho_b**2 + z_b**2  # a^2 + rho^2 + zeta^2
    alpha_sq = s + 2.0 * a * rho_b  # (a+rho)^2 + zeta^2
    beta_sq = np.maximum(s - 2.0 * a * rho_b, 1e-300)  # (a-rho)^2 + zeta^2
    alpha = np.sqrt(alpha_sq)

    m = np.clip(4.0 * a * rho_b / np.maximum(alpha_sq, 1e-300), 0.0, 1.0 - 1e-14)
    K = ellipk(m)
    E = ellipe(m)

    pref = MU_0 * float(current) / TWO_PI

    b_zeta = (pref / alpha) * (K + (a2 - rho_b**2 - z_b**2) / beta_sq * E)

    small = rho_b <= _SMALL_RHO * a
    rho_safe = np.where(small, 1.0, rho_b)
    b_rho_full = (pref * z_b / (rho_safe * alpha)) * (-K + s / beta_sq * E)

    denom_series = (a2 + z_b**2) ** 2.5
    b_rho_series = np.where(
        denom_series > 0.0,
        3.0 * MU_0 * float(current) * a2 * rho_b * z_b / (4.0 * np.maximum(denom_series, 1e-300)),
        0.0,
    )
    b_rho = np.where(small, b_rho_series, b_rho_full)

    return b_rho, b_zeta


# --------------------------------------------------------------------------- #
# Соленоид
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Solenoid(MagneticSource):
    """Тонкостенный соленоид конечной длины, вращающийся вокруг оси Z.

    Parameters
    ----------
    origin:
        Центр катушки, ``(3,)``, м.
    n_turns:
        Число витков ``N``.
    current:
        Ток ``I``, А.
    radius:
        Радиус намотки ``R_s``, м.
    length:
        Длина намотки ``L``, м.
    rotation_axis:
        Ось вращения катушки как жёсткого тела (по умолчанию ``Z``).
    phase_offset:
        Начальная фаза, рад. При ``phase = phi`` ось катушки направлена вдоль
        ``cos(phi)*e1 + sin(phi)*e2``.
    n_quad:
        Число узлов Гаусса–Лежандра по длине (``>= 32`` достаточно;
        ``128`` — машинная точность для точек вне намотки).
    quadrant_policy:
        Политика первого квадранта.
    """

    origin: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    n_turns: int = 100
    current: float = 1.0
    radius: float = 0.05
    length: float = 0.10
    rotation_axis: NDArray[np.float64] = field(default_factory=lambda: AXIS_Z.copy())
    phase_offset: float = 0.0
    n_quad: int = 96
    quadrant_policy: QuadrantPolicy = QuadrantPolicy.STRICT

    is_quadrature_exact: Final[bool] = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", as_points(self.origin).reshape(3))
        object.__setattr__(self, "rotation_axis", unit(np.asarray(self.rotation_axis, float)))
        object.__setattr__(self, "quadrant_policy", QuadrantPolicy(self.quadrant_policy))
        if self.radius <= 0 or self.length <= 0 or self.n_turns <= 0:
            raise ValueError("radius, length и n_turns должны быть положительными")

    # -- контракт ------------------------------------------------------------ #
    @property
    def magnetic_moment(self) -> float:
        """``m = N*I*pi*R_s^2``, А*м^2."""
        return float(self.n_turns) * float(self.current) * np.pi * self.radius**2

    @property
    def dipole_constant(self) -> float:
        """``B0 = mu0*m/(4*pi) = mu0*N*I*R_s^2/4``, Тл*м^3."""
        return MU_0 * self.magnetic_moment / (4.0 * np.pi)

    @property
    def turns_density(self) -> float:
        """``n = N/L``, витков/м."""
        return float(self.n_turns) / float(self.length)

    @property
    def rotation_frame(self) -> NDArray[np.float64]:
        return orthonormal_frame(self.rotation_axis)

    def coil_axis(self, phase: float) -> NDArray[np.float64]:
        """Единичный вектор оси катушки при фазе ``phase``."""
        e1, e2 = self.rotation_frame[:, 0], self.rotation_frame[:, 1]
        return np.cos(phase + self.phase_offset) * e1 + np.sin(phase + self.phase_offset) * e2

    # -- поле ---------------------------------------------------------------- #
    def field_local(
        self, points_local: ArrayLike, *, n_quad: int | None = None
    ) -> NDArray[np.float64]:
        """Поле в **локальной** системе катушки (ось катушки = локальная Z).

        Катушка центрирована в начале координат, занимает ``z' in [-L/2, +L/2]``.
        """
        p = as_points(points_local)
        px, py, pz = p[..., 0], p[..., 1], p[..., 2]

        rho = np.hypot(px, py)
        cos_phi = np.divide(px, rho, out=np.ones_like(px), where=rho > 0.0)
        sin_phi = np.divide(py, rho, out=np.zeros_like(py), where=rho > 0.0)

        nq = int(n_quad or self.n_quad)
        t_nodes, weights = gauss_legendre(nq)

        # Одним векторизованным вызовом обрабатываем все узлы квадратуры.
        # Форма: (n_quad, *shape_points); loop_field_cylindrical уже поддерживает
        # broadcasting и применяет ellipk/ellipe к массивам.
        half_L = 0.5 * self.length
        point_ndim = pz.ndim
        node_shape = (nq,) + (1,) * point_ndim
        rho_all = np.broadcast_to(rho[np.newaxis, ...], (nq,) + rho.shape)
        zeta_all = pz[np.newaxis, ...] - half_L * t_nodes.reshape(node_shape)
        br_all, bz_all = loop_field_cylindrical(rho_all, zeta_all, self.radius, current=1.0)
        weight_shape = weights.reshape(node_shape)
        b_rho = np.sum(br_all * weight_shape, axis=0)
        b_zeta = np.sum(bz_all * weight_shape, axis=0)

        factor = 0.5 * float(self.n_turns) * float(self.current)
        b_rho *= factor
        b_zeta *= factor

        return np.stack([b_rho * cos_phi, b_rho * sin_phi, b_zeta], axis=-1)

    def field_at_phase(self, points: ArrayLike, phase: float = 0.0) -> NDArray[np.float64]:
        rel = self.relative_points(points)
        if self.quadrant_policy is QuadrantPolicy.STRICT:
            rel = enforce_quadrant(rel, QuadrantPolicy.STRICT)

        axis = self.coil_axis(phase)
        R = orthonormal_frame(axis)  # столбцы: e1_perp, e2_perp, axis
        p_local = rel @ R  # R.T @ p  (для строк)
        b_local = self.field_local(p_local)
        return b_local @ R.T

    # -- сравнение с диполем ------------------------------------------------- #
    def as_dipole(self) -> RotatingDipole:
        """Эквивалентный точечный диполь (тот же момент, та же ось вращения)."""
        return RotatingDipole(
            origin=self.origin.copy(),
            moment_amplitude=self.magnetic_moment,
            b0=self.dipole_constant,
            rotation_axis=self.rotation_axis.copy(),
            phase_offset=self.phase_offset,
            quadrant_policy=self.quadrant_policy,
        )

    def field_dipole_approx(self, points: ArrayLike, phase: float = 0.0) -> NDArray[np.float64]:
        """Дипольное приближение поля соленоида."""
        rel = self.relative_points(points)
        return dipole_field(rel, self.coil_axis(phase), self.dipole_constant)

    def relative_deviation(self, points: ArrayLike, phase: float = 0.0) -> NDArray[np.float64]:
        """Относительное расхождение моделей ``||B_sol - B_dip|| / ||B_dip||``."""
        b_sol = self.field_at_phase(points, phase)
        b_dip = self.field_dipole_approx(points, phase)
        return safe_norm(b_sol - b_dip) / safe_norm(b_dip)

    def amplitude_deviation(self, points: ArrayLike, phase: float = 0.0) -> NDArray[np.float64]:
        """Расхождение **амплитуд** ``|B_sol| / |B_dip| - 1``."""
        return (
            safe_norm(self.field_at_phase(points, phase))
            / safe_norm(self.field_dipole_approx(points, phase))
            - 1.0
        )

    def on_axis_field(self, zeta: ArrayLike, *, phase: float = 0.0) -> NDArray[np.float64]:
        """Аналитическое поле на оси катушки (для контроля квадратуры).

        .. math:: B_\\zeta = \\frac{\\mu_0 n I}{2}
            \\left[\\frac{\\zeta+L/2}{\\sqrt{R^2+(\\zeta+L/2)^2}}
                 -\\frac{\\zeta-L/2}{\\sqrt{R^2+(\\zeta-L/2)^2}}\\right]
        """
        z = np.asarray(zeta, dtype=np.float64)
        half = 0.5 * self.length
        term1 = (z + half) / np.sqrt(self.radius**2 + (z + half) ** 2)
        term2 = (z - half) / np.sqrt(self.radius**2 + (z - half) ** 2)
        b = 0.5 * MU_0 * self.turns_density * self.current * (term1 - term2)
        axis = self.coil_axis(phase)
        return b[..., np.newaxis] * axis

    # ---------- прямые адаптеры: момент / b0 -> Solenoid ---------- #
    @classmethod
    def from_moment(
        cls,
        moment: float,
        radius: float,
        length: float,
        *,
        n_turns: int = 100,
        **kwargs,
    ) -> "Solenoid":
        """Создать соленоид по заданному магнитному моменту.

        Число витков фиксируется (по умолчанию 100), а ток подбирается так,
        чтобы произведение N*I дало ровно заданный момент:
            I = m / (N * pi * R^2).
        """
        if moment <= 0:
            raise ValueError("moment должен быть положительным")
        current = float(moment) / (int(n_turns) * np.pi * radius**2)
        return cls(
            radius=radius,
            length=length,
            n_turns=int(n_turns),
            current=current,
            **kwargs,
        )

    @classmethod
    def from_b0(
        cls,
        b0: float,
        radius: float,
        length: float,
        *,
        n_turns: int = 100,
        **kwargs,
    ) -> "Solenoid":
        """Создать соленоид по заданной дипольной константе ``b0`` [Тл*м^3].

        Использует связь m = 4*pi*b0 / mu0, т.е. ``b0`` здесь — ровно та же
        величина, что и ``b0`` у ``RotatingDipole``. Поэтому дальние поля
        (и излучаемая мощность на одной частоте) совпадают.
        """
        moment = 4.0 * np.pi * float(b0) / MU_0
        return cls.from_moment(moment, radius, length, n_turns=n_turns, **kwargs)

    # ---------- обратные адаптеры: Solenoid -> момент / b0 ---------- #
    @property
    def moment(self) -> float:
        """Магнитный момент соленоида ``m = N*I*pi*R^2``, А*м^2."""
        return self.magnetic_moment  # уже существует

    @property
    def b0(self) -> float:
        """Дипольная константа ``b0 = mu0*m/(4*pi)``, Тл*м^3.

        Это та же величина, что ``b0`` у ``RotatingDipole``: ``Solenoid.b0``
        равен ``RotatingDipole.b0`` для согласованных маяков.
        """
        return self.dipole_constant  # уже существует
