"""Генераторы траекторий и сеток наблюдения.

Все функции **чистые**: на входе — параметры, на выходе — ``ndarray (n, 3)``
в метрах. ГСЧ передаётся через явный ``seed``/``rng``.
Точки генерируются в первом квадранте (``x, y, z >= 0``).
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..convention import (
    QuadrantPolicy,
    TWO_PI,
    as_points,
    enforce_quadrant,
    norm,
    unit,
)

__all__ = [
    "line",
    "waypoint_path",
    "arc",
    "helix",
    "circle",
    "grid3d",
    "plane_grid",
    "radial_rays",
    "random_first_octant",
    "uniform_moving_observer",
    "PlaneName",
]

PlaneName = Literal["xy", "xz", "yz"]


# --------------------------------------------------------------------------- #
# Базовые кривые
# --------------------------------------------------------------------------- #
def line(start: ArrayLike, end: ArrayLike, n: int, *, endpoint: bool = True) -> NDArray[np.float64]:
    """Равномерная polyline-траектория из ``start`` в ``end``."""
    if n < 2:
        raise ValueError("n >= 2")
    s = as_points(start).reshape(3)
    e = as_points(end).reshape(3)
    t = np.linspace(0.0, 1.0, int(n), endpoint=endpoint)
    return s + t[:, np.newaxis] * (e - s)


def waypoint_path(
    waypoints: Sequence[ArrayLike], n_per_segment: int = 50, *, closed: bool = False
) -> NDArray[np.float64]:
    """Ломаная через список опорных точек."""
    pts = [as_points(w).reshape(3) for w in waypoints]
    if len(pts) < 2:
        raise ValueError("Нужно минимум 2 опорные точки")
    if closed:
        pts = pts + [pts[0]]
    segs = [line(a, b, n_per_segment, endpoint=False) for a, b in zip(pts[:-1], pts[1:])]
    segs.append(np.array([pts[-1]]))
    return np.concatenate(segs, axis=0)


def arc(
    center: ArrayLike,
    radius: float,
    start_angle: float,
    end_angle: float,
    n: int,
    *,
    plane: PlaneName = "xy",
    fixed_offset: float = 0.0,
) -> NDArray[np.float64]:
    """Дуга окружности в заданной координатной плоскости."""
    c = as_points(center).reshape(3)
    th = np.linspace(start_angle, end_angle, int(n))
    pts = np.zeros((int(n), 3))
    if plane == "xy":
        pts[:, 0] = c[0] + radius * np.cos(th)
        pts[:, 1] = c[1] + radius * np.sin(th)
        pts[:, 2] = c[2] + fixed_offset
    elif plane == "xz":
        pts[:, 0] = c[0] + radius * np.cos(th)
        pts[:, 2] = c[2] + radius * np.sin(th)
        pts[:, 1] = c[1] + fixed_offset
    elif plane == "yz":
        pts[:, 1] = c[1] + radius * np.cos(th)
        pts[:, 2] = c[2] + radius * np.sin(th)
        pts[:, 0] = c[0] + fixed_offset
    else:
        raise ValueError(f"Неизвестная плоскость: {plane!r}")
    return pts


def circle(
    center: ArrayLike, radius: float, n: int, *, plane: PlaneName = "xy", fixed_offset: float = 0.0
) -> NDArray[np.float64]:
    """Замкнутая окружность."""
    return arc(center, radius, 0.0, TWO_PI, n, plane=plane, fixed_offset=fixed_offset)


def helix(
    start: ArrayLike,
    axis_dir: ArrayLike,
    radius: float,
    pitch: float,
    n_turns: float,
    n: int,
) -> NDArray[np.float64]:
    """Винтовая траектория вокруг оси ``axis_dir``."""
    s = as_points(start).reshape(3)
    k = unit(np.asarray(axis_dir, dtype=np.float64))
    helper = np.array([0.0, 0.0, 1.0]) if abs(k[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = unit(np.cross(helper, k))
    e2 = np.cross(k, e1)
    t = np.linspace(0.0, float(n_turns), int(n))
    th = TWO_PI * t
    return (
        s
        + radius * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
        + (pitch * t)[:, None] * k
    )


# --------------------------------------------------------------------------- #
# Сетки
# --------------------------------------------------------------------------- #
def grid3d(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    n: tuple[int, int, int],
    *,
    exclude_radius: float = 0.0,
    origin: ArrayLike = (0.0, 0.0, 0.0),
) -> NDArray[np.float64]:
    """Прямоугольная 3D-сетка, опционально с вырезанной сферой вокруг ``origin``."""
    xs = np.linspace(*x_range, int(n[0]))
    ys = np.linspace(*y_range, int(n[1]))
    zs = np.linspace(*z_range, int(n[2]))
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    if exclude_radius > 0:
        o = as_points(origin).reshape(3)
        keep = norm(pts - o, axis=-1) >= exclude_radius
        pts = pts[keep]
    return pts


def plane_grid(
    plane: PlaneName,
    fixed: float,
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    n: tuple[int, int],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], tuple[str, str]]:
    """Двумерная сетка в координатной плоскости.

    Returns
    -------
    (points, U, V, labels) :
        ``points`` — ``(n[0], n[1], 3)``; ``U, V`` — meshgrid-координаты;
        ``labels`` — подписи осей, например ``("x, м", "y, м")``.
    """
    u = np.linspace(*u_range, int(n[0]))
    v = np.linspace(*v_range, int(n[1]))
    U, V = np.meshgrid(u, v, indexing="ij")
    pts = np.zeros(U.shape + (3,))
    if plane == "xy":
        pts[..., 0], pts[..., 1], pts[..., 2] = U, V, fixed
        labels = ("x, м", "y, м")
    elif plane == "xz":
        pts[..., 0], pts[..., 2], pts[..., 1] = U, V, fixed
        labels = ("x, м", "z, м")
    elif plane == "yz":
        pts[..., 1], pts[..., 2], pts[..., 0] = U, V, fixed
        labels = ("y, м", "z, м")
    else:
        raise ValueError(f"Неизвестная плоскость: {plane!r}")
    return pts, U, V, labels


def radial_rays(
    n_rays: int,
    radii: ArrayLike,
    *,
    elevation_deg: float = 45.0,
    azimuth_span_deg: float = 90.0,
    azimuth_start_deg: float = 0.0,
    origin: ArrayLike = (0.0, 0.0, 0.0),
) -> NDArray[np.float64]:
    """Семейство лучей из ``origin`` — для исследования погрешности vs. дальности."""
    rr = np.asarray(radii, dtype=np.float64).reshape(-1)
    az = np.deg2rad(
        np.linspace(azimuth_start_deg, azimuth_start_deg + azimuth_span_deg, int(n_rays))
    )
    el = np.deg2rad(elevation_deg)
    dirs = np.stack(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.full_like(az, np.sin(el))],
        axis=-1,
    )
    pts = dirs[:, np.newaxis, :] * rr[np.newaxis, :, np.newaxis]
    o = as_points(origin).reshape(3)
    return (pts.reshape(-1, 3) + o)


def random_first_octant(
    n: int,
    r_min: float,
    r_max: float,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    origin: ArrayLike = (0.0, 0.0, 0.0),
) -> NDArray[np.float64]:
    """Равномерно по объёму в первом октанте сферического слоя ``[r_min, r_max]``."""
    if r_min < 0 or r_max <= r_min:
        raise ValueError("Требуется 0 <= r_min < r_max")
    gen = rng if rng is not None else np.random.default_rng(seed)
    n = int(n)

    # равномерное распределение по объёму: r = (u*(rmax^3-rmin^3)+rmin^3)^(1/3)
    u = gen.random(n)
    r = np.cbrt(u * (r_max**3 - r_min**3) + r_min**3)

    # равномерно по телесному углу в октанте
    cos_theta = gen.random(n)              # theta in [0, pi/2] -> cos in [0,1]
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = 0.5 * np.pi * gen.random(n)      # phi in [0, pi/2]

    pts = np.stack(
        [r * sin_theta * np.cos(phi), r * sin_theta * np.sin(phi), r * cos_theta], axis=-1
    )
    return pts + as_points(origin).reshape(3)


def uniform_moving_observer(
    start: ArrayLike,
    velocity: ArrayLike,
    times: ArrayLike,
) -> NDArray[np.float64]:
    """``p(t) = start + velocity * t`` — прямолинейное равномерное движение."""
    s = as_points(start).reshape(3)
    v = as_points(velocity).reshape(3)
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    return s + t[:, np.newaxis] * v