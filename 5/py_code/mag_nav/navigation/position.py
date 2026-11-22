"""Обратная навигационная задача: ``(a_vec, b_vec, B0) -> (x, y, z)``.

Векторизованный порт ``CalcAll.m`` (скалярные ``CalcR.m``, ``CalcX.m``,
``CalcY.m``, ``CalcZ.m`` сохранены как обёртки для трассируемости).

Вывод
-----
Пусть ``u = x/r``, ``v = y/r``, ``w = z/r`` и ``kappa = B0^2 / r^6``. Тогда

.. math::
    A = |\\vec a|^2 = \\kappa(3u^2+1),\\quad
    B = |\\vec b|^2 = \\kappa(3v^2+1),\\quad
    C = \\vec a\\cdot\\vec b = 3\\kappa u v

Отсюда

.. math::
    (A-B)^2 + 4C^2 = 9\\kappa^2(u^2+v^2)^2,\\qquad
    D \\equiv A + B - \\sqrt{(A-B)^2+4C^2} = 2\\kappa

.. math::
    r = \\left(\\frac{2 B_0^2}{D}\\right)^{1/6},\\qquad
    \\text{temp} = \\frac{r^6}{3 B_0^2} = \\frac{2}{3D}

.. math::
    u = \\sqrt{\\text{temp}\\,A - \\tfrac13},\\quad
    v = \\operatorname{sign}(C)\\sqrt{\\text{temp}\\,B - \\tfrac13},\\quad
    w = \\sqrt{\\tfrac53 - \\text{temp}\\,(A+B)}

.. note::
   **Ключевое свойство:** ``u, v, w`` (направление) зависят только от
   отношений ``A:B:C`` и потому *не зависят от калибровки* ``B0``.
   Дальность ``r`` пропорциональна ``B0^(1/3)``. Это проверено тестом
   ``tests/primitives/nav_checks.check_range_calibration_scaling``.

Знак ``v`` восстанавливается через ``sign(C)``, что корректно при ``u >= 0``
(первый квадрант). Знаки ``u`` и ``w`` берутся неотрицательными — это и есть
ограничение «квадрант 1».
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..convention import (
    EPS,
    QuadrantPolicy,
    as_points,
    norm,
    safe_norm,
)
from ..signal.ab_params import ABInvariants, ab_invariants

__all__ = [
    "PositionSolution",
    "solve_position",
    "solve_from_invariants",
    "calc_r",
    "calc_u",
    "calc_v",
    "calc_w",
    "calc_x",
    "calc_y",
    "calc_z",
    "recompose_position",
    "position_error",
]

_DEFAULT_TOL: Final[float] = 1e-14
_DEFAULT_W_CLIP_TOL: Final[float] = 0.05
"""Насколько отрицательным может быть ``w^2``, чтобы точку прижать к z=0, а не отбросить."""


# --------------------------------------------------------------------------- #
# Результат
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PositionSolution:
    """Результат решения обратной задачи + диагностика.

    Attributes
    ----------
    points : (..., 3)
        Оценки координат в системе маяка, м. Недействительные элементы — ``NaN``.
    r : (...)
        Дальность, м.
    uvw : (..., 3)
        Направляющие косинусы.
    valid : (...)
        Маска точек, где решение математически корректно
        (``D > 0``, подкоренные выражения неотрицательны, ``uvw`` нормирован).
    invariants : ABInvariants
        Промежуточные скаляры для отладки.
    uvw_norm : (...)
        ``sqrt(u^2+v^2+w^2)`` — должно быть ``1`` (контроль согласованности).
    reasons : (...)
        Код причины недействительности (0 = OK, битовая маска ошибок).
    """

    points: NDArray[np.float64]
    r: NDArray[np.float64]
    uvw: NDArray[np.float64]
    valid: NDArray[np.bool_]
    invariants: ABInvariants
    uvw_norm: NDArray[np.float64]
    reasons: NDArray[np.int32]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.points.shape[:-1]

    @property
    def n_valid(self) -> int:
        return int(np.count_nonzero(self.valid))

    @property
    def validity_ratio(self) -> float:
        return float(self.valid.mean()) if self.valid.size else 0.0

    def x(self) -> NDArray[np.float64]:
        return self.points[..., 0]

    def y(self) -> NDArray[np.float64]:
        return self.points[..., 1]

    def z(self) -> NDArray[np.float64]:
        return self.points[..., 2]

    def valid_only(self) -> NDArray[np.float64]:
        """Только действительные точки, форма ``(n_valid, 3)``."""
        return self.points.reshape(-1, 3)[self.valid.ravel()]

    def where_invalid(self) -> NDArray[np.int64]:
        """Индексы недействительных точек."""
        return np.argwhere(~self.valid)


# Биты причин отказа
REASON_OK: Final[int] = 0
REASON_DENOM: Final[int] = 1       # D <= 0
REASON_U_NEG: Final[int] = 2       # temp*A - 1/3 < 0
REASON_V_NEG: Final[int] = 4       # temp*B - 1/3 < 0
REASON_W_NEG: Final[int] = 8       # 5/3 - temp*(A+B) < 0
REASON_UVW: Final[int] = 16        # |uvw| != 1
REASON_NONFINITE: Final[int] = 32  # NaN/Inf на входе


# --------------------------------------------------------------------------- #
# Скалярные примитивы (1:1 с MATLAB-файлами)
# --------------------------------------------------------------------------- #
def calc_r(denominator: ArrayLike, b0: float) -> NDArray[np.float64]:
    """``CalcR.m``: ``r = (2*B0^2 / D)^(1/6)``."""
    d = np.asarray(denominator, dtype=np.float64)
    return np.power(2.0 * float(b0) ** 2 / np.where(d > 0, d, np.nan), 1.0 / 6.0)


def calc_temp(denominator: ArrayLike) -> NDArray[np.float64]:
    """``temp = r^6/(3*B0^2) = 2/(3*D)`` — форма без переполнения."""
    d = np.asarray(denominator, dtype=np.float64)
    return 2.0 / (3.0 * np.where(d > 0, d, np.nan))


def calc_u(asq: ArrayLike, denominator: ArrayLike) -> NDArray[np.float64]:
    """``CalcX.m`` (домен u): ``u = sqrt(temp*A - 1/3)``."""
    return np.sqrt(np.maximum(calc_temp(denominator) * np.asarray(asq, float) - 1.0 / 3.0, 0.0))


def calc_v(
    bsq: ArrayLike, denominator: ArrayLike, ab: ArrayLike
) -> NDArray[np.float64]:
    """``CalcY.m`` (домен v): ``v = sign(C)*sqrt(temp*B - 1/3)``."""
    raw = np.sqrt(np.maximum(calc_temp(denominator) * np.asarray(bsq, float) - 1.0 / 3.0, 0.0))
    return np.sign(np.asarray(ab, dtype=np.float64)) * raw


def calc_w(asq: ArrayLike, bsq: ArrayLike, denominator: ArrayLike) -> NDArray[np.float64]:
    """``CalcZ.m`` (домен w): ``w = sqrt(5/3 - temp*(A+B))``."""
    t = calc_temp(denominator)
    return np.sqrt(
        np.maximum(5.0 / 3.0 - t * (np.asarray(asq, float) + np.asarray(bsq, float)), 0.0)
    )


def calc_x(r: ArrayLike, u: ArrayLike) -> NDArray[np.float64]:
    """``CalcX.m``: ``x = r*u``."""
    return np.asarray(r, dtype=np.float64) * np.asarray(u, dtype=np.float64)


def calc_y(r: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
    """``CalcY.m``: ``y = r*v``."""
    return np.asarray(r, dtype=np.float64) * np.asarray(v, dtype=np.float64)


def calc_z(r: ArrayLike, w: ArrayLike) -> NDArray[np.float64]:
    """``CalcZ.m``: ``z = r*w``."""
    return np.asarray(r, dtype=np.float64) * np.asarray(w, dtype=np.float64)


def recompose_position(r: ArrayLike, uvw: ArrayLike) -> NDArray[np.float64]:
    """``(r, uvw) -> (x, y, z)`` формы ``(..., 3)``."""
    rr = np.asarray(r, dtype=np.float64)[..., np.newaxis]
    return rr * as_points(uvw)


# --------------------------------------------------------------------------- #
# Основной решатель
# --------------------------------------------------------------------------- #
def _solve_from_invariants_full(
    inv: ABInvariants,
    b0: float,
    *,
    tol: float = _DEFAULT_TOL,
    uvw_tol: float = 1e-6,
    w_clip_tol: float = _DEFAULT_W_CLIP_TOL,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.int32]
]:
    """Внутреннее ядро: возвращает координаты, uvw, дальность и диагностику."""
    asq, bsq, ab = inv.asq, inv.bsq, inv.ab
    denom = inv.denominator

    reasons = np.zeros(np.broadcast(asq, bsq, ab).shape, dtype=np.int32)

    finite_in = np.isfinite(asq) & np.isfinite(bsq) & np.isfinite(ab)
    reasons |= np.where(finite_in, 0, REASON_NONFINITE)

    # D имеет масштаб квадрата измеренного поля. Абсолютный порог,
    # привязанный к max(B0**2, 1), отбрасывал практически все реальные поля
    # (обычно 1e-12..1e-18 Тл²). Используем относительный допуск к A+B.
    denom_scale = np.maximum(asq + bsq, EPS)
    denom_ok = np.isfinite(denom) & (denom > tol * denom_scale)
    reasons |= np.where(denom_ok, 0, REASON_DENOM)

    safe_denom = np.where(denom_ok, denom, np.nan)
    temp = 2.0 / (3.0 * safe_denom)

    u2_raw = temp * asq - 1.0 / 3.0
    v2_raw = temp * bsq - 1.0 / 3.0
    w2_raw = 5.0 / 3.0 - temp * (asq + bsq)

    reasons |= np.where(u2_raw >= -uvw_tol, 0, REASON_U_NEG)
    reasons |= np.where(v2_raw >= -uvw_tol, 0, REASON_V_NEG)
    # w^2 < 0 у плоскости вращения (z ~ 0) — следствие шума или недипольности
    # источника. Ближайшее допустимое решение — w = 0, поэтому небольшой
    # «недолёт» прижимаем к плоскости, а не отбрасываем точку.
    reasons |= np.where(w2_raw >= -max(uvw_tol, w_clip_tol), 0, REASON_W_NEG)

    u = np.sqrt(np.maximum(u2_raw, 0.0))
    # Не np.isclose(ab, 0): его atol=1e-8 на порядки больше реальных a·b (~1e-14 Тл²).
    v = np.where(ab < 0.0, -1.0, 1.0) * np.sqrt(np.maximum(v2_raw, 0.0))
    w = np.sqrt(np.maximum(w2_raw, 0.0))

    # u^2 + v^2 + w^2 = 1 тождественно, поэтому после w -> 0 имеем
    # u^2 + v^2 = 1 - w2_raw > 1: перенормируем (u, v) на единичную окружность.
    clipped = w2_raw < 0.0
    uv_norm = np.sqrt(u * u + v * v)
    uv_scale = np.where(clipped & (uv_norm > 0.0), 1.0 / np.where(uv_norm > 0.0, uv_norm, 1.0), 1.0)
    u = u * uv_scale
    v = v * uv_scale

    uvw = np.stack([u, v, w], axis=-1)
    uvw_norm = safe_norm(uvw, axis=-1)
    reasons |= np.where(np.abs(uvw_norm - 1.0) <= uvw_tol, 0, REASON_UVW)

    r6 = 2.0 * float(b0) ** 2 / safe_denom
    r = np.power(np.maximum(r6, 0.0), 1.0 / 6.0)

    points = r[..., np.newaxis] * uvw
    valid = reasons == REASON_OK
    points = np.where(valid[..., np.newaxis], points, np.nan)
    uvw = np.where(valid[..., np.newaxis], uvw, np.nan)
    r = np.where(valid, r, np.nan)
    return points, uvw, r, valid, reasons


def solve_from_invariants(
    inv: ABInvariants,
    b0: float,
    *,
    tol: float = _DEFAULT_TOL,
    uvw_tol: float = 1e-6,
    w_clip_tol: float = _DEFAULT_W_CLIP_TOL,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.int32]]:
    """Ядро решения по инвариантам: ``(points, r, valid, reasons)``."""
    points, _uvw, r, valid, reasons = _solve_from_invariants_full(
        inv, b0, tol=tol, uvw_tol=uvw_tol, w_clip_tol=w_clip_tol
    )
    return points, r, valid, reasons


def solve_position(
    a_vec: ArrayLike,
    b_vec: ArrayLike,
    b0: float,
    *,
    tol: float = _DEFAULT_TOL,
    uvw_tol: float = 1e-6,
    w_clip_tol: float = _DEFAULT_W_CLIP_TOL,
    quadrant_policy: QuadrantPolicy | str = QuadrantPolicy.RELAXED,
    return_uvw_norm: bool = True,
) -> PositionSolution:
    """Решить обратную навигационную задачу.

    Parameters
    ----------
    a_vec, b_vec : (..., 3)
        Измеренные векторные параметры (Тл) — коэффициенты при ``cos(wt)``
        и ``sin(wt)`` в сигнале трёхосного магнитометра, **в рамке маяка**.
    b0 : float
        Дипольная константа маяка ``mu0*m0/(4*pi)``, Тл*м^3.
    tol:
        Относительный порог положительности знаменателя ``D``.
    uvw_tol:
        Допуск на отрицательность подкоренных выражений и на
        ``|uvw| = 1`` (компенсирует шум измерения).
    w_clip_tol:
        Если ``-w_clip_tol <= w^2 < 0`` (точка у плоскости вращения), ``w``
        прижимается к 0, а ``(u, v)`` перенормируются; точка остаётся
        действительной. ``0`` — прежнее строгое поведение.
    quadrant_policy:
        ``RELAXED`` (по умолчанию) — допускает ``y < 0``;
        ``STRICT`` — требует ``y >= 0`` и обнуляет отрицательный ``y``.

    Returns
    -------
    PositionSolution

    Raises
    ------
    ValueError
        При несогласованных формах или ``b0 <= 0``.
    """
    if not (b0 > 0.0):
        raise ValueError(f"Дипольная константа b0 должна быть > 0, получено {b0}")

    a = as_points(a_vec)
    b = as_points(b_vec)
    if a.shape != b.shape:
        raise ValueError(f"Формы a_vec и b_vec не совпадают: {a.shape} vs {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        # не падаем: просто пометим такие точки как недействительные
        pass

    inv = ab_invariants(a, b)
    points, uvw, r, valid, reasons = _solve_from_invariants_full(
        inv, b0, tol=tol, uvw_tol=uvw_tol, w_clip_tol=w_clip_tol
    )

    pol = QuadrantPolicy(quadrant_policy)
    if pol is QuadrantPolicy.STRICT:
        neg_y = points[..., 1] < 0.0
        if np.any(neg_y):
            reasons = reasons | np.where(neg_y, REASON_UVW, 0)
            valid = reasons == REASON_OK
            points = np.where(valid[..., np.newaxis], points, np.nan)
            uvw = np.where(valid[..., np.newaxis], uvw, np.nan)
    elif pol is QuadrantPolicy.CLIP:
        points = np.where(np.isnan(points), np.nan, np.maximum(points, 0.0))
        # CLIP меняет координаты, так что в этом специальном режиме направление
        # восстанавливаем из уже обрезанных точек.
        uvw = points / np.where(r > 0, r, np.nan)[..., np.newaxis]

    uvw_n = norm(uvw, axis=-1)

    return PositionSolution(
        points=points,
        r=r,
        uvw=uvw,
        valid=valid,
        invariants=inv,
        uvw_norm=uvw_n,
        reasons=reasons,
    )


# --------------------------------------------------------------------------- #
# Метрики погрешности
# --------------------------------------------------------------------------- #
def position_error(
    true_points: ArrayLike,
    estimated_points: ArrayLike,
    *,
    relative: bool = True,
) -> dict[str, NDArray[np.float64]]:
    """Абсолютные и относительные погрешности по координатам.

    Returns
    -------
    dict с ключами ``abs`` (..., 3), ``abs_norm`` (...), ``rel`` (..., 3),
    ``rel_norm`` (...), ``range_abs``, ``range_rel``.
    """
    t = as_points(true_points)
    e = as_points(estimated_points)
    if t.shape != e.shape:
        raise ValueError(f"Формы не совпадают: {t.shape} vs {e.shape}")

    diff = e - t
    abs_norm = norm(diff, axis=-1)
    r_true = safe_norm(t, axis=-1)

    rel = diff / r_true[..., np.newaxis]
    rel_norm = abs_norm / r_true

    range_abs = np.abs(safe_norm(e, axis=-1) - r_true)
    range_rel = range_abs / r_true

    return {
        "abs": diff,
        "abs_norm": abs_norm,
        "rel": rel,
        "rel_norm": rel_norm,
        "range_abs": range_abs,
        "range_rel": range_rel,
    }


def error_statistics(
    true_points: ArrayLike,
    estimated_points: ArrayLike,
    valid: NDArray[np.bool_] | None = None,
) -> dict[str, float | NDArray[np.float64]]:
    """Сводная статистика: RMSE, среднее, медиана, максимум, 95-й процентиль."""
    err = position_error(true_points, estimated_points)
    mask = np.ones(err["abs_norm"].shape, dtype=bool) if valid is None else np.asarray(valid, bool)
    mask &= np.isfinite(err["abs_norm"])

    def _stat(key: str) -> NDArray[np.float64] | float:
        arr = np.asarray(err[key])[mask] if np.asarray(err[key]).ndim == mask.ndim else np.asarray(err[key])[..., :][mask]
        return arr

    flat_abs = err["abs"][mask]
    flat_norm = err["abs_norm"][mask]
    flat_rel = err["rel_norm"][mask]
    return {
        "n": int(flat_norm.size),
        "mean_abs": flat_abs.mean(axis=0) if flat_abs.size else np.zeros(3),
        "rms_abs": np.sqrt(np.mean(flat_abs**2, axis=0)) if flat_abs.size else np.zeros(3),
        "mean_norm": float(flat_norm.mean()) if flat_norm.size else float("nan"),
        "rms_norm": float(np.sqrt(np.mean(flat_norm**2))) if flat_norm.size else float("nan"),
        "median_norm": float(np.median(flat_norm)) if flat_norm.size else float("nan"),
        "max_norm": float(flat_norm.max()) if flat_norm.size else float("nan"),
        "p95_norm": float(np.percentile(flat_norm, 95)) if flat_norm.size else float("nan"),
        "mean_rel": float(flat_rel.mean()) if flat_rel.size else float("nan"),
        "p95_rel": float(np.percentile(flat_rel, 95)) if flat_rel.size else float("nan"),
    }