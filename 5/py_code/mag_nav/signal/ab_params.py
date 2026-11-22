"""Генерация и анализ a/b-параметров (аналог ``CreateABParameterGenerator.m``).

Ответственность модуля — **прямая** задача: по геометрии и модели источника
получить измеряемые ``a_vec``, ``b_vec`` и их инварианты. Обратная задача —
в :mod:`mag_nav.navigation.position`.

Определение
-----------
При вращении момента поле приёмника имеет вид

.. math:: \\vec B(\\varphi) = \\vec a\\cos\\varphi + \\vec b\\sin\\varphi

Для точечного диполя это тождество; для соленоида конечных размеров
``B(phi)`` несинусоидально, и ``a``, ``b`` — коэффициенты **первой гармоники**:

.. math::
    \\vec a = \\frac{2}{N}\\sum_{i} \\vec B(\\varphi_i)\\cos\\varphi_i,\\quad
    \\vec b = \\frac{2}{N}\\sum_{i} \\vec B(\\varphi_i)\\sin\\varphi_i
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..convention import TWO_PI, as_points, direction_cosines, safe_norm
from ..source.base import MagneticSource
from ..source.dipole import dipole_ab_vectors, dipole_invariants

__all__ = [
    "ABParams",
    "ABInvariants",
    "ab_from_source",
    "ab_from_dipole",
    "ab_invariants",
    "invariants_from_geometry",
    "first_harmonic",
    "harmonic_residual",
    "polarization_ellipse",
    "phase_grid",
]


# --------------------------------------------------------------------------- #
# Типы-контейнеры
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ABParams:
    """Пара векторов a/b (Тл) и служебная информация."""

    a: NDArray[np.float64]
    b: NDArray[np.float64]
    exact: bool = True
    """``True`` — получено квадратурами точно; ``False`` — первая гармоника."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", as_points(self.a))
        object.__setattr__(self, "b", as_points(self.b))
        if self.a.shape != self.b.shape:
            raise ValueError(f"Формы a и b не совпадают: {self.a.shape} vs {self.b.shape}")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.a.shape[:-1]

    def invariants(self) -> "ABInvariants":
        return ab_invariants(self.a, self.b)

    def as_tuple(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.a, self.b


@dataclass(frozen=True)
class ABInvariants:
    """Скалярные инварианты, используемые обратным решением."""

    asq: NDArray[np.float64]   # |a|^2
    bsq: NDArray[np.float64]   # |b|^2
    ab: NDArray[np.float64]    # a·b
    discriminant: NDArray[np.float64]  # (asq-bsq)^2 + 4 ab^2
    denominator: NDArray[np.float64]   # D = asq + bsq - sqrt(discriminant) = 2*kappa

    @property
    def kappa(self) -> NDArray[np.float64]:
        """``kappa = B0^2 / r^6 = D/2``."""
        return 0.5 * self.denominator


# --------------------------------------------------------------------------- #
# Инварианты
# --------------------------------------------------------------------------- #
def ab_invariants(a_vec: ArrayLike, b_vec: ArrayLike) -> ABInvariants:
    """Вычислить ``(asq, bsq, ab, D)`` — чисто, векторизованно."""
    a = as_points(a_vec)
    b = as_points(b_vec)
    asq = np.sum(a * a, axis=-1)
    bsq = np.sum(b * b, axis=-1)
    ab = np.sum(a * b, axis=-1)
    disc = (asq - bsq) ** 2 + 4.0 * ab**2
    denom = asq + bsq - np.sqrt(disc)
    return ABInvariants(asq=asq, bsq=bsq, ab=ab, discriminant=disc, denominator=denom)


def invariants_from_geometry(
    rel_points: ArrayLike, b0: float
) -> ABInvariants:
    """Теоретические инварианты напрямую из геометрии (эталон для тестов)."""
    kappa, asq, bsq, ab = dipole_invariants(rel_points, b0)
    disc = (asq - bsq) ** 2 + 4.0 * ab**2
    return ABInvariants(asq=asq, bsq=bsq, ab=ab, discriminant=disc, denominator=2.0 * kappa)


# --------------------------------------------------------------------------- #
# Прямая генерация a/b
# --------------------------------------------------------------------------- #
def ab_from_dipole(
    rel_points: ArrayLike, b0: float
) -> ABParams:
    """Аналитические a/b точечного диполя (ось вращения — Z)."""
    a, b = dipole_ab_vectors(rel_points, b0)
    return ABParams(a=a, b=b, exact=True)


def phase_grid(n_harmonic: int, phase_offset: float = 0.0) -> NDArray[np.float64]:
    """Равномерная сетка фаз на периоде, ``(n_harmonic,)``."""
    n = int(n_harmonic)
    if n < 4:
        raise ValueError("n_harmonic >= 4 необходимо для корректного выделения 1-й гармоники")
    return TWO_PI * np.arange(n) / n + phase_offset


def first_harmonic(
    samples: ArrayLike, phases: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Квадратурная проекция на ``(cos phi, sin phi)``.

    Parameters
    ----------
    samples : (n_phase, ..., 3)
        Мгновенные векторы поля; ось 0 — фаза.
    phases : (n_phase,)
        Фазы, рад (равномерная полная сетка).

    Returns
    -------
    (a_vec, b_vec) той же формы ``(..., 3)``, что и ``samples[0]``.
    """
    s = np.asarray(samples, dtype=np.float64)
    ph = np.asarray(phases, dtype=np.float64).reshape(-1)
    if s.shape[0] != ph.size:
        raise ValueError("Размер оси фаз не совпадает с samples.shape[0]")
    n = ph.size
    shape = (n,) + (1,) * (s.ndim - 2) + (1,)
    cos_w = (np.cos(ph).reshape(shape)) * (2.0 / n)
    sin_w = (np.sin(ph).reshape(shape)) * (2.0 / n)
    a = np.sum(s * cos_w, axis=0)
    b = np.sum(s * sin_w, axis=0)
    return a, b


_AB_CACHE_MAXSIZE = 8
_AB_CACHE: OrderedDict[tuple[object, ...], ABParams] = OrderedDict()


def _source_fingerprint(source: MagneticSource) -> tuple[object, ...] | None:
    """Fingerprint exact dataclass state; unsupported custom objects skip caching."""
    if not is_dataclass(source):
        return None
    values: list[object] = [type(source).__module__, type(source).__qualname__]
    for item in fields(source):
        value = getattr(source, item.name)
        if isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(value)
            values.append((item.name, arr.dtype.str, arr.shape, arr.tobytes()))
        elif isinstance(value, Enum):
            values.append((item.name, type(value).__qualname__, value.value))
        elif isinstance(value, (str, bytes, int, float, bool, type(None))):
            values.append((item.name, type(value).__name__, value))
        else:
            return None
    return tuple(values)


def _copy_ab(ab: ABParams) -> ABParams:
    """Не отдавать вызывающему коду изменяемые массивы из глобального кэша."""
    return ABParams(a=ab.a.copy(), b=ab.b.copy(), exact=ab.exact)


def ab_from_source(
    source: MagneticSource,
    points: ArrayLike,
    *,
    n_harmonic: int = 64,
    force_harmonic: bool = False,
) -> ABParams:
    """Получить a/b-параметры для произвольного источника.

    Если ``source.is_quadrature_exact`` и ``force_harmonic=False``,
    используется быстрый точный путь ``a = B(0)``, ``b = B(pi/2)``.
    Иначе — дискретное выделение первой гармоники по ``n_harmonic`` фазам.
    """
    pts = as_points(points)
    source_key = _source_fingerprint(source)
    # Ключ содержит точные байты геометрии и все параметры dataclass-источника.
    # Если состояние источника нельзя надёжно сериализовать, вычисляем без кэша.
    cache_key: tuple[object, ...] | None = None
    if source_key is not None:
        points_bytes = np.ascontiguousarray(pts, dtype=np.float64).tobytes()
        cache_key = (
            source_key,
            pts.shape,
            hashlib.md5(points_bytes).digest(),
            n_harmonic,
            bool(force_harmonic),
        )
        cached = _AB_CACHE.get(cache_key)
        if cached is not None:
            _AB_CACHE.move_to_end(cache_key)
            return _copy_ab(cached)

    if source.is_quadrature_exact and not force_harmonic:
        a = source.field_at_phase(pts, 0.0)
        b = source.field_at_phase(pts, np.pi / 2.0)
        result = ABParams(a=a, b=b, exact=True)
    else:
        phases = phase_grid(n_harmonic, source.phase_offset)
        samples = np.empty((phases.size,) + pts.shape, dtype=np.float64)
        for i, ph in enumerate(phases):
            samples[i] = source.field_at_phase(pts, ph)
        a, b = first_harmonic(samples, phases)
        result = ABParams(a=a, b=b, exact=False)

    if cache_key is not None:
        _AB_CACHE[cache_key] = _copy_ab(result)
        _AB_CACHE.move_to_end(cache_key)
        while len(_AB_CACHE) > _AB_CACHE_MAXSIZE:
            _AB_CACHE.popitem(last=False)
    return result


def harmonic_residual(
    source: MagneticSource,
    points: ArrayLike,
    *,
    n_harmonic: int = 128,
    ab: ABParams | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Относительная несинусоидальность сигнала источника.

    Returns
    -------
    (rms_residual, relative) :
        ``rms_residual`` — СКП остатка ``B(phi) - a cos - b sin``, Тл;
        ``relative``     — то же, нормированное на СКП самого ``B(phi)``.

    Для точечного диполя ``relative ~ 1e-16``. Для соленоида — заметная
    величина в ближней зоне (именно она порождает систематическую
    погрешность навигации).
    """
    pts = as_points(points)
    phases = phase_grid(n_harmonic, source.phase_offset)
    samples = np.stack([source.field_at_phase(pts, ph) for ph in phases], axis=0)
    if ab is None:
        a, b = first_harmonic(samples, phases)
    else:
        a, b = ab.a, ab.b

    ph = phases.reshape((-1,) + (1,) * (samples.ndim - 2) + (1,))
    model = a[np.newaxis, ...] * np.cos(ph) + b[np.newaxis, ...] * np.sin(ph)
    resid = samples - model

    rms_res = np.sqrt(np.mean(np.sum(resid**2, axis=-1), axis=0))
    rms_sig = np.sqrt(np.mean(np.sum(samples**2, axis=-1), axis=0))
    return rms_res, rms_res / safe_norm(rms_sig[..., np.newaxis], axis=-1).reshape(rms_res.shape) if False else rms_res / np.maximum(rms_sig, 1e-300)


# --------------------------------------------------------------------------- #
# Поляризация (аналог AmplPhaseMap_mlx)
# --------------------------------------------------------------------------- #
def polarization_ellipse(
    a_vec: ArrayLike, b_vec: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Параметры эллипса, описываемого концом вектора ``B(t)``.

    Returns
    -------
    (s_min, s_max, tilt) :
        полуоси эллипса [Тл] и угол наклона большой полуоси [рад]
        в плоскости, натянутой на ``(a, b)``.
    """
    a = as_points(a_vec)
    b = as_points(b_vec)
    asq = np.sum(a * a, axis=-1)
    bsq = np.sum(b * b, axis=-1)
    ab = np.sum(a * b, axis=-1)

    tr = asq + bsq
    det = asq * bsq - ab**2
    disc = np.sqrt(np.maximum(tr**2 - 4.0 * det, 0.0))
    s_max = np.sqrt(np.maximum(0.5 * (tr + disc), 0.0))
    s_min = np.sqrt(np.maximum(0.5 * (tr - disc), 0.0))
    tilt = 0.5 * np.arctan2(2.0 * ab, asq - bsq)
    return s_min, s_max, tilt