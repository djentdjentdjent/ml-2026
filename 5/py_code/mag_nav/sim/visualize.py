"""Визуализация. Все побочные эффекты (matplotlib) изолированы здесь.

Правило модуля: **ни одна функция не вызывает ``plt.show()`` и не пишет
файлы**. Функции принимают данные и возвращают ``Figure``/``Axes``;
показ и сохранение — ответственность вызывающего кода (скрипта, ноутбука).
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import ArtistAnimation, FuncAnimation
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from ..convention import TWO_PI, as_points, norm, safe_norm
from ..navigation.position import PositionSolution
from ..signal.ab_params import ab_from_source, polarization_ellipse
from ..source.base import MagneticSource, field_line_tracer, trace_field_line
from .scenario import MonteCarloSummary, SimResult, model_discrepancy_map
from .trajectory import PlaneName, plane_grid

__all__ = [
    "apply_style",
    "plot_field_slice",
    "plot_field_magnitude",
    "plot_field_lines",
    "plot_trajectory_3d",
    "plot_error_components",
    "plot_error_histogram",
    "plot_error_vs_range",
    "plot_ab_invariants",
    "plot_polarization",
    "plot_model_discrepancy",
    "plot_boundary_radius",
    "plot_monte_carlo",
    "plot_monte_carlo_error_vs_range",
    "animate_navigation",
    "save_figures",
]

_MAG_LOG_FLOOR = 1e-14


def apply_style(style: str = "default") -> None:
    """Единая настройка стиля (изолированный побочный эффект)."""
    plt.style.use(style)
    plt.rcParams.update(
        {
            "figure.autolayout": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
            "legend.frameon": True,
        }
    )


# --------------------------------------------------------------------------- #
# Поле
# --------------------------------------------------------------------------- #
def _slice_points(
    plane: PlaneName,
    fixed: float,
    extent: tuple[float, float, float, float],
    n: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], tuple[str, str]]:
    return plane_grid(plane, fixed, (extent[0], extent[1]), (extent[2], extent[3]), (n, n))


def plot_field_magnitude(
    source: MagneticSource,
    *,
    plane: PlaneName = "xz",
    fixed: float = 0.0,
    extent: tuple[float, float, float, float] = (-0.6, 0.6, 0.0, 1.2),
    n: int = 200,
    phase: float = 0.0,
    log: bool = True,
    ax: plt.Axes | None = None,
    cmap: str = "inferno",
    colorbar: bool = True,
    title: str | None = None,
) -> Figure:
    """Цветовая карта ``|B|`` в сечении."""
    pts, U, V, labels = _slice_points(plane, fixed, extent, n)
    b = source.field_at_phase(pts.reshape(-1, 3), phase).reshape(pts.shape)
    mag = np.linalg.norm(b, axis=-1)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
    else:
        fig = ax.figure

    data = np.log10(np.maximum(mag, _MAG_LOG_FLOOR)) if log else mag
    pcm = ax.pcolormesh(U, V, data.T, shading="auto", cmap=cmap)
    if colorbar:
        cb = fig.colorbar(pcm, ax=ax)
        cb.set_label("log10|B|, Тл" if log else "|B|, Тл")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title or f"Модуль поля {type(source).__name__} (срез {plane}, phase={phase:g})")
    return fig


def plot_field_slice(
    source: MagneticSource,
    *,
    plane: PlaneName = "xz",
    fixed: float = 0.0,
    extent: tuple[float, float, float, float] = (-0.6, 0.6, 0.0, 1.2),
    n: int = 160,
    phase: float = 0.0,
    ax: plt.Axes | None = None,
    log: bool = True,
    stream: bool = True,
    quiver_step: int = 8,
    title: str | None = None,
) -> Figure:
    """Карта ``|B|`` + линии тока (``streamplot``) и векторы (``quiver``)."""
    pts, U, V, labels = _slice_points(plane, fixed, extent, n)
    flat = pts.reshape(-1, 3)
    b = source.field_at_phase(flat, phase).reshape(pts.shape)
    mag = np.linalg.norm(b, axis=-1)

    # компоненты в координатах среза
    idx = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]
    bu = b[..., idx[0]]
    bv = b[..., idx[1]]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 6.0))
    else:
        fig = ax.figure

    data = np.log10(np.maximum(mag, _MAG_LOG_FLOOR)) if log else mag
    pcm = ax.pcolormesh(U, V, data.T, shading="auto", cmap="inferno", alpha=0.9)
    fig.colorbar(pcm, ax=ax).set_label("log10|B|, Тл" if log else "|B|, Тл")

    if stream:
        speed = np.hypot(bu, bv)
        lw = 1.6 * speed / np.nanmax(speed) if np.nanmax(speed) > 0 else None
        ax.streamplot(
            U.T, V.T, bu.T, bv.T, color="white", linewidth=0.7, density=1.4, arrowsize=0.7
        )

    ax.quiver(
        U.T[::quiver_step, ::quiver_step],
        V.T[::quiver_step, ::quiver_step],
        bu.T[::quiver_step, ::quiver_step],
        bv.T[::quiver_step, ::quiver_step],
        color="cyan",
        alpha=0.55,
        scale=np.nanmax(np.hypot(bu, bv)) * 40,
        width=0.0025,
    )

    o = source.origin
    ax.plot([o[idx[0]]], [o[idx[1]]], "w*", ms=14, mec="k", label="маяк")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title or f"Поле {type(source).__name__}: срез {plane}, phase={phase:g}")
    return fig


def plot_field_lines(
    source: MagneticSource,
    *,
    seeds: ArrayLike,
    phase: float = 0.0,
    step: float = 4e-3,
    n_steps: int = 900,
    plane: PlaneName = "xz",
    fixed: float = 0.0,
    extent: tuple[float, float, float, float] = (-0.6, 0.6, -0.6, 0.6),
    ax: plt.Axes | None = None,
    color: str = "tab:blue",
    title: str | None = None,
) -> Figure:
    """Силовые линии, трассированные РК4 (аналог ``MagneticLine.m``)."""
    idx = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]
    tracer = field_line_tracer(source, phase)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 6.0))
    else:
        fig = ax.figure

    for s in as_points(seeds).reshape(-1, 3):
        poly = trace_field_line(
            tracer,
            s,
            step=step,
            n_steps=n_steps,
            bidirectional=True,
            stop_radius=0.02,
            max_radius=4.0,
        )
        ax.plot(poly[:, idx[0]], poly[:, idx[1]], color=color, lw=1.0, alpha=0.8)

    o = source.origin
    ax.plot([o[idx[0]]], [o[idx[1]]], "r*", ms=15, mec="k", label="маяк")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(["x, м", "y, м", "y, м"][idx[0]])
    ax.set_ylabel(["y, м", "z, м", "z, м"][idx[1]])
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title or f"Силовые линии (срез {plane}, phase={phase:g})")
    return fig


# --------------------------------------------------------------------------- #
# Результаты навигации
# --------------------------------------------------------------------------- #
def plot_trajectory_3d(
    result: SimResult,
    *,
    ax: Any | None = None,
    max_points: int = 3000,
    marker_size: float = 8.0,
    title: str | None = None,
) -> Figure:
    """3D: истинная и восстановленная траектории."""
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(7.6, 6.6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    t = result.true_positions
    e = result.estimated_positions
    m = result.valid
    if t.shape[0] > max_points:
        sel = np.linspace(0, t.shape[0] - 1, max_points).astype(int)
        t, e, m = t[sel], e[sel], m[sel]

    ax.scatter(
        t[:, 0], t[:, 1], t[:, 2], c="tab:blue", s=marker_size, label="истинная", depthshade=False
    )
    ax.scatter(
        e[m, 0],
        e[m, 1],
        e[m, 2],
        c="tab:red",
        s=marker_size,
        alpha=0.75,
        label="оценка",
        depthshade=False,
    )
    if np.any(~m):
        ax.scatter(
            e[~m, 0], e[~m, 1], e[~m, 2], c="k", marker="x", s=marker_size * 2, label="отказ"
        )
    for i in range(0, len(t), max(1, len(t) // 12)):
        if m[i]:
            ax.plot(
                [t[i, 0], e[i, 0]],
                [t[i, 1], e[i, 1]],
                [t[i, 2], e[i, 2]],
                color="grey",
                lw=0.6,
                alpha=0.6,
            )

    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_zlabel("z, м")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title or "Траектория: истинная vs. восстановленная")
    return fig


def plot_error_components(
    result: SimResult,
    *,
    against: Literal["index", "range"] = "range",
    logy: bool = True,
    figsize: tuple[float, float] = (11.0, 3.4),
) -> Figure:
    """Три панели: ``Δx``, ``Δy``, ``Δz``."""
    err = result.error_abs
    x_axis = result.r_true if against == "range" else np.arange(err.shape[0])
    xlab = "r, м" if against == "range" else "№ точки"

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
    for k, (ax, lbl) in enumerate(zip(axes, "xyz")):
        m = result.valid
        ax.scatter(
            x_axis[m], err[m, k], s=5, alpha=0.5, color=f"tab:{['blue', 'orange', 'green'][k]}"
        )
        if logy:
            ax.set_yscale("symlog", linthresh=1e-9)
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_xlabel(xlab)
        ax.set_ylabel(f"Δ{lbl}, м" if k == 0 else "")
        ax.set_title(f"Погрешность по {lbl.upper()}")
    fig.tight_layout()
    return fig


def plot_error_histogram(
    result: SimResult | MonteCarloSummary,
    *,
    bins: int = 80,
    log: bool = False,
    figsize: tuple[float, float] = (11.0, 3.4),
) -> Figure:
    """Гистограммы погрешностей по осям и по модулю."""
    if hasattr(result, "errors") and hasattr(result, "valid"):
        # MonteCarloResult из MonteCarloSimulator (не требует хранить SimResult
        # для каждого прогона, поэтому подходит для keep_results=False).
        errors = np.asarray(result.errors, dtype=np.float64)
        mask = np.asarray(result.valid, dtype=bool) & np.all(np.isfinite(errors), axis=-1)
        stack = errors[mask]
        norms = np.linalg.norm(stack, axis=-1)
    elif isinstance(result, MonteCarloSummary):
        if not result.results:
            raise ValueError("Для MonteCarloSummary нужны results; либо используйте MonteCarloResult")
        stack = np.concatenate([r.error_abs[r.valid] for r in result.results], axis=0)
        norms = np.concatenate([r.error_norm[r.valid] for r in result.results], axis=0)
    else:
        stack = result.error_abs[result.valid]
        norms = result.error_norm[result.valid]

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    if stack.size == 0:
        for ax, title in zip(axes, ("ΔX", "ΔY", "ΔZ", "|Δr|")):
            ax.text(0.5, 0.5, "Нет валидных решений", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
    else:
        for k, (ax, lbl, col) in enumerate(
            zip(axes[:3], "xyz", ["tab:blue", "tab:orange", "tab:green"])
        ):
            ax.hist(stack[:, k], bins=bins, color=col, alpha=0.8)
            if log:
                ax.set_yscale("log")
            ax.set_xlabel(f"Δ{lbl}, м")
            ax.set_title(f"{lbl.upper()}: σ={np.std(stack[:, k]):.2e}")
        axes[3].hist(norms, bins=bins, color="tab:red", alpha=0.8)
        if log:
            axes[3].set_yscale("log")
        axes[3].set_xlabel("|Δr|, м")
        axes[3].set_title(f"|Δr|: RMS={np.sqrt(np.mean(norms**2)):.2e}")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_error_vs_range(
    result: SimResult,
    *,
    relative: bool = True,
    loglog: bool = True,
    trend: bool = True,
    ax: plt.Axes | None = None,
) -> Figure:
    """Погрешность как функция дальности (главный график для отчётов)."""
    r = result.r_true
    y = result.error_rel_norm if relative else result.error_norm
    m = result.valid & np.isfinite(y) & (r > 0)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
    else:
        fig = ax.figure

    ax.scatter(r[m], y[m], s=8, alpha=0.45, color="tab:blue", label="точки")
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")

    if trend and np.count_nonzero(m) > 8:
        order = np.argsort(r[m])
        rr, yy = r[m][order], y[m][order]
        if loglog:
            k, c = np.polyfit(np.log(rr), np.log(np.maximum(yy, 1e-300)), 1)
            ax.plot(rr, np.exp(c) * rr**k, "r-", lw=1.6, label=f"аппроксимация: ~ r^{{{k:.2f}}}")
        else:
            k, c = np.polyfit(rr, yy, 1)
            ax.plot(rr, k * rr + c, "r-", lw=1.6, label=f"лин. тренд k={k:.2e}")
        ax.legend(fontsize=8)

    ax.set_xlabel("r, м")
    ax.set_ylabel("относительная погрешность" if relative else "|Δr|, м")
    ax.set_title("Погрешность навигации vs. дальность")
    return fig


# --------------------------------------------------------------------------- #
# a/b-параметры и поляризация
# --------------------------------------------------------------------------- #
def plot_ab_invariants(
    result: SimResult,
    *,
    against: Literal["index", "range"] = "range",
    figsize: tuple[float, float] = (11.0, 3.4),
) -> Figure:
    """Инварианты ``|a|², |b|², a·b`` — истинные и измеренные."""
    inv_t = result.solution.invariants
    a_t, b_t = result.a_true, result.b_true
    asq_t = np.sum(a_t**2, -1)
    bsq_t = np.sum(b_t**2, -1)
    ab_t = np.sum(a_t * b_t, -1)

    x = result.r_true if against == "range" else np.arange(len(result.r_true))
    xlab = "r, м" if against == "range" else "№ точки"

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)
    for ax, yt, ym, lbl in zip(
        axes, [asq_t, bsq_t, ab_t], [inv_t.asq, inv_t.bsq, inv_t.ab], ["|a|²", "|b|²", "a·b"]
    ):
        ax.semilogy(x, np.abs(yt), "b.", ms=4, label="истина")
        ax.semilogy(x, np.abs(ym), "r.", ms=4, alpha=0.6, label="измерение")
        ax.set_xlabel(xlab)
        ax.set_title(lbl)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_polarization(
    source: MagneticSource,
    *,
    points: ArrayLike,
    plane: PlaneName = "xz",
    fixed: float = 0.0,
    extent: tuple[float, float, float, float] = (-0.6, 0.6, 0.0, 1.2),
    n: int = 120,
    ax: plt.Axes | None = None,
) -> Figure:
    """Карта большой полуоси эллипса поляризации ``|B(t)|``."""
    pts, U, V, labels = _slice_points(plane, fixed, extent, n)
    ab = ab_from_source(source, pts.reshape(-1, 3))
    s_min, s_max, _ = polarization_ellipse(ab.a, ab.b)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.6, 5.6))
    else:
        fig = ax.figure

    pcm = ax.pcolormesh(
        U,
        V,
        np.log10(np.maximum(s_max.reshape(U.shape), _MAG_LOG_FLOOR)).T,
        shading="auto",
        cmap="viridis",
    )
    fig.colorbar(pcm, ax=ax).set_label("log10(|B|max), Тл")
    cnt = ax.contour(
        U,
        V,
        np.log10(np.maximum(s_min.reshape(U.shape), _MAG_LOG_FLOOR)).T,
        levels=8,
        colors="w",
        linewidths=0.6,
    )
    ax.clabel(cnt, inline=True, fontsize=7, fmt="%.1f")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Амплитудно-фазовая карта (полуоси эллипса поляризации)")
    return fig


# --------------------------------------------------------------------------- #
# Расхождение моделей / граничные радиусы
# --------------------------------------------------------------------------- #
def plot_model_discrepancy(
    source_truth: MagneticSource,
    source_model: MagneticSource,
    *,
    plane: PlaneName = "xz",
    fixed: float = 0.0,
    u_range: tuple[float, float] = (0.03, 1.0),
    v_range: tuple[float, float] = (0.03, 1.0),
    n: tuple[int, int] = (90, 90),
    relative: bool = True,
    log: bool = True,
    component: Literal["norm", "x", "y", "z"] = "norm",
    n_harmonic: int = 64,
    ax: plt.Axes | None = None,
) -> Figure:
    """Карта систематической погрешности «соленоид vs. диполь»."""
    data = model_discrepancy_map(
        source_truth,
        source_model,
        plane=plane,
        fixed=fixed,
        u_range=u_range,
        v_range=v_range,
        n=n,
        n_harmonic=n_harmonic,
    )
    if component == "norm":
        val = data["error_rel"] if relative else data["error_norm"]
    else:
        k = {"x": 0, "y": 1, "z": 2}[component]
        val = (
            np.abs(data["error_abs"][..., k]) / np.maximum(data["r_true"], 1e-300)
            if relative
            else np.abs(data["error_abs"][..., k])
        )

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
    else:
        fig = ax.figure

    plot_val = np.log10(np.maximum(np.abs(val), 1e-16)) if log else val
    pcm = ax.pcolormesh(data["U"], data["V"], plot_val.T, shading="auto", cmap="magma")
    cb = fig.colorbar(pcm, ax=ax)
    unit_str = " (доля от r)" if relative else ", м"
    cb.set_label(("log10 погрешность" if log else "погрешность") + unit_str)

    cs = ax.contour(
        data["U"],
        data["V"],
        np.abs(val).T,
        levels=np.geomspace(max(np.nanpercentile(np.abs(val), 1), 1e-9), np.nanmax(np.abs(val)), 6),
        colors="cyan",
        linewidths=0.8,
    )
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1e")

    ax.set_xlabel(data["labels"][0])
    ax.set_ylabel(data["labels"][1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"Систематическая погрешность ({component}), "
        f"{type(source_truth).__name__} → {type(source_model).__name__}"
    )
    return fig


def plot_boundary_radius(
    source_truth: MagneticSource,
    source_model: MagneticSource,
    *,
    thresholds: Sequence[float] = (0.01, 0.05, 0.1),
    elevations: Sequence[float] = (15.0, 30.0, 45.0, 60.0, 75.0),
    azimuth: float = 45.0,
    r_range: tuple[float, float] = (0.02, 3.0),
    n_r: int = 300,
    relative: bool = True,
    figsize: tuple[float, float] = (11.5, 4.6),
) -> Figure:
    """Семейство кривых погрешности + граничные радиусы (``BoundaryR.m``)."""
    from .scenario import boundary_radius_map

    r_grid = np.geomspace(*r_range, n_r)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    cmap = plt.get_cmap("plasma")
    boundaries: dict[float, list[float]] = {th: [] for th in thresholds}

    for i, el in enumerate(elevations):
        col = cmap(i / max(1, len(elevations) - 1))
        res = boundary_radius_map(
            source_truth,
            source_model,
            threshold=float("inf"),
            elevation_deg=el,
            azimuth_deg=azimuth,
            r_grid=r_grid,
            relative=relative,
        )
        axes[0].semilogx(res["r"], res["metric"], color=col, lw=1.4, label=f"el={el:g}°")

        for th in thresholds:
            ok = np.isfinite(res["metric"]) & res["valid"] & (res["metric"] < th)
            br = np.nan
            if np.any(ok):
                for k in range(len(r_grid)):
                    if np.all(ok[k:]):
                        br = r_grid[k]
                        break
            boundaries[th].append(br)

    axes[0].set_xlabel("r, м")
    axes[0].set_ylabel("относительная погрешность" if relative else "погрешность, м")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"Погрешность vs. дальность (az={azimuth:g}°)")

    bmat = np.asarray([boundaries[t] for t in thresholds])  # (n_th, n_el)
    el_arr = np.asarray(elevations, dtype=float)
    for i, th in enumerate(thresholds):
        axes[1].plot(el_arr, bmat[i, :], "o-", ms=4, label=f"порог={th:g}")
    axes[1].set_xlabel("угол места, °")
    axes[1].set_ylabel("граничный радиус r*, м")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8, title="порог")
    axes[1].set_title("Граничная поверхность расхождения")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Монте-Карло
# --------------------------------------------------------------------------- #
def plot_monte_carlo(
    mc: Any, *, figsize: tuple[float, float] = (11.5, 4.0)
) -> Figure:
    """Средняя погрешность по точкам траектории + разброс.

    Принимает как ``MonteCarloSummary`` (Scenario API), так и
    ``MonteCarloResult`` (MonteCarloSimulator API).
    """
    if hasattr(mc, "errors") and hasattr(mc, "l2_per_point"):
        err_norm = np.linalg.norm(np.asarray(mc.errors), axis=-1)
        valid = np.asarray(mc.valid, dtype=bool) & np.isfinite(err_norm)
        counts = np.sum(valid, axis=0)
        sums = np.sum(np.where(valid, err_norm, 0.0), axis=0)
        mean = np.full(err_norm.shape[1], np.nan, dtype=np.float64)
        np.divide(sums, counts, out=mean, where=counts > 0)
        squared_dev = np.sum(np.where(valid, (err_norm - mean[np.newaxis, :]) ** 2, 0.0), axis=0)
        variance = np.full_like(mean, np.nan)
        np.divide(squared_dev, counts, out=variance, where=counts > 0)
        std = np.sqrt(variance)
        rms = np.asarray(mc.l2_per_point, dtype=np.float64)
        n_trials = int(mc.n_trials)
        errors = np.asarray(mc.errors, dtype=np.float64)
        n_valid = int(np.count_nonzero(valid))
        if n_valid:
            per_component_rms = np.sqrt(
                np.sum(np.where(valid[..., np.newaxis], errors**2, 0.0), axis=(0, 1)) / n_valid
            )
        else:
            per_component_rms = np.full(3, np.nan)
    else:
        mean = np.asarray(mc.error_norm_mean)
        std = np.asarray(mc.error_norm_std)
        rms = np.asarray(mc.rms_per_point)
        n_trials = int(mc.n_trials)
        per_component_rms = np.asarray(mc.per_component_rms)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    idx = np.arange(mean.size)
    axes[0].plot(idx, mean, "b-", lw=1.4, label="среднее")
    axes[0].fill_between(
        idx,
        np.maximum(mean - std, 1e-16),
        mean + std,
        color="b",
        alpha=0.2,
        label="±1σ",
    )
    axes[0].plot(idx, rms, "r--", lw=1.0, label="RMS / L2")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("№ точки")
    axes[0].set_ylabel("|Δr|, м")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"Монте-Карло, {n_trials} реализаций")

    axes[1].bar(np.arange(3), per_component_rms, color=["tab:blue", "tab:orange", "tab:green"])
    axes[1].set_xticks(np.arange(3), ["Δx", "Δy", "Δz"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("RMS, м")
    axes[1].set_title("Погрешность по компонентам")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_monte_carlo_error_vs_range(
    mc: Any,
    trajectory: ArrayLike,
    *,
    origin: ArrayLike = (0.0, 0.0, 0.0),
    relative: bool = True,
    loglog: bool = True,
    trend: bool = True,
    ax: plt.Axes | None = None,
) -> Figure:
    """Разброс ошибки ансамбля по дальности без сохранения полных SimResult."""
    if not (hasattr(mc, "errors") and hasattr(mc, "valid")):
        raise TypeError("Ожидается результат MonteCarloSimulator с полями errors и valid")
    pts = as_points(trajectory)
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    r = norm(pts - o, axis=-1)
    errors = np.linalg.norm(np.asarray(mc.errors, dtype=np.float64), axis=-1)
    valid = np.asarray(mc.valid, dtype=bool) & np.isfinite(errors) & (r[np.newaxis, :] > 0.0)
    y = errors / r[np.newaxis, :] if relative else errors
    x_grid = np.broadcast_to(r, y.shape)
    valid &= np.isfinite(y) & (y > 0.0 if loglog else True)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
    else:
        fig = ax.figure

    x_values, y_values = x_grid[valid], y[valid]
    # Ограничиваем число маркеров только для очень больших ансамблей.
    if x_values.size > 50000:
        take = np.linspace(0, x_values.size - 1, 50000).astype(int)
        x_values, y_values = x_values[take], y_values[take]
    ax.scatter(x_values, y_values, s=5, alpha=0.10, color="tab:blue", label="реализации")
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")

    trend_y = np.asarray(mc.l2_per_point, dtype=np.float64)
    if relative:
        trend_y = trend_y / np.maximum(r, 1e-300)
    trend_mask = np.isfinite(r) & np.isfinite(trend_y) & (r > 0.0) & (trend_y > 0.0 if loglog else True)
    if trend and np.count_nonzero(trend_mask) > 8:
        rr, yy = r[trend_mask], trend_y[trend_mask]
        order = np.argsort(rr)
        rr, yy = rr[order], yy[order]
        if loglog:
            k, c = np.polyfit(np.log(rr), np.log(np.maximum(yy, 1e-300)), 1)
            ax.plot(rr, np.exp(c) * rr**k, "r-", lw=1.6, label=f"L2 ~ r^{{{k:.2f}}}")
        else:
            k, c = np.polyfit(rr, yy, 1)
            ax.plot(rr, k * rr + c, "r-", lw=1.6, label=f"линейный тренд k={k:.2e}")
        ax.legend(fontsize=8)

    ax.set_xlabel("r, м")
    ax.set_ylabel("относительная погрешность" if relative else "|Δr|, м")
    ax.set_title("Погрешность Монте-Карло vs. дальность")
    return fig


# --------------------------------------------------------------------------- #
# Анимация
# --------------------------------------------------------------------------- #
def animate_navigation(
    result: SimResult,
    *,
    interval_ms: int = 40,
    trail: int = 60,
    source: MagneticSource | None = None,
    figsize: tuple[float, float] = (8.0, 7.0),
) -> FuncAnimation:
    """Покадровая анимация движения приёмника вдоль траектории."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    t = result.true_positions
    e = result.estimated_positions
    n = len(t)
    step = max(1, n // 600)
    idx = np.arange(0, n, step)

    ax.set_xlim(0, max(t[:, 0].max(), np.nanmax(e[:, 0])) * 1.15 + 1e-9)
    ax.set_ylim(0, max(t[:, 1].max(), np.nanmax(e[:, 1])) * 1.15 + 1e-9)
    ax.set_zlim(0, max(t[:, 2].max(), np.nanmax(e[:, 2])) * 1.15 + 1e-9)
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_zlabel("z, м")
    title = ax.set_title("")

    (true_line,) = ax.plot([], [], [], "b-", lw=1.0, alpha=0.6, label="истинная")
    (est_line,) = ax.plot([], [], [], "r-", lw=1.0, alpha=0.6, label="оценка")
    (true_pt,) = ax.plot([], [], [], "bo", ms=7)
    (est_pt,) = ax.plot([], [], [], "ro", ms=7)
    (link,) = ax.plot([], [], [], "k-", lw=0.7, alpha=0.5)
    if source is not None:
        ax.scatter(*source.origin, c="gold", s=140, marker="*", edgecolors="k", label="маяк")
    ax.legend(loc="upper left", fontsize=8)

    def update(k: int) -> list:
        i = idx[k]
        lo = max(0, i - trail)
        true_line.set_data(t[lo : i + 1, 0], t[lo : i + 1, 1])
        true_line.set_3d_properties(t[lo : i + 1, 2])
        if result.valid[i]:
            est_line.set_data(e[lo : i + 1, 0], e[lo : i + 1, 1])
            est_line.set_3d_properties(e[lo : i + 1, 2])
            est_pt.set_data([e[i, 0]], [e[i, 1]])
            est_pt.set_3d_properties([e[i, 2]])
            link.set_data([t[i, 0], e[i, 0]], [t[i, 1], e[i, 1]])
            link.set_3d_properties([t[i, 2], e[i, 2]])
        else:
            est_pt.set_data([], [])
            est_pt.set_3d_properties([])
        true_pt.set_data([t[i, 0]], [t[i, 1]])
        true_pt.set_3d_properties([t[i, 2]])
        title.set_text(
            f"шаг {i}/{n - 1}   r={np.linalg.norm(t[i]):.3f} м   |Δr|={result.error_norm[i]:.2e} м"
        )
        return [true_line, est_line, true_pt, est_pt, link, title]

    return FuncAnimation(fig, update, frames=len(idx), interval=interval_ms, blit=False)


# --------------------------------------------------------------------------- #
# Сохранение
# --------------------------------------------------------------------------- #
def save_figures(
    figures: dict[str, Figure], out_dir: str, *, dpi: int = 150, ext: str = "png"
) -> list[str]:
    """Сохранить набор фигур на диск (изолированный побочный эффект)."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, fig in figures.items():
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    return paths
