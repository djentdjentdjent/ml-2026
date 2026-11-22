"""Оркестрация прогона симуляции.

Единственная ответственность модуля — собрать пайплайн
«источник → a/b → (шум) → демодуляция → решение → погрешности»
и вернуть неизменяемый :class:`SimResult`. Математика делегируется
соответствующим модулям; визуализация — :mod:`mag_nav.sim.visualize`.

Два режима внесения шума
------------------------
``inject_at="samples"`` (реалистичный)
    Шум применяется к мгновенным отсчетам ``B(t_i)``, затем МНК-демодуляция.
    Корректно воспроизводит усреднение шума за период.

``inject_at="ab"`` (быстрый)
    Шум применяется напрямую к ``a_vec``/``b_vec``. Аналог MATLAB-скриптов
    ``NoizSimulationCordCord.m`` / ``NoizeSimulationAnglesCord.m``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..convention import TWO_PI, as_points, norm, safe_norm
from ..navigation.orientation import IDENTITY, RotationMatrix
from ..navigation.position import PositionSolution, error_statistics, position_error, solve_position
from ..noise.base import NoiseModel
from ..signal.ab_params import ABParams, ab_from_source, harmonic_residual
from ..signal.recovery import (
    DemodulationResult,
    demodulate_lstsq,
    make_phase_grid,
    synthesize_series,
)
from ..source.base import MagneticSource
from .trajectory import PlaneName, plane_grid

__all__ = [
    "SensorConfig",
    "Scenario",
    "SimResult",
    "MonteCarloSummary",
    "monte_carlo",
    "model_discrepancy_map",
    "boundary_radius_map",
    "calibration_sweep",
]


# --------------------------------------------------------------------------- #
# Конфигурация приёмника
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SensorConfig:
    """Параметры тракта измерения."""

    samples_per_period: int = 64
    n_periods: int = 4
    noise: NoiseModel | None = None
    demod_method: Literal["lstsq", "projection"] = "lstsq"
    with_dc: bool = True
    phase_offset: float = 0.0

    @property
    def n_samples(self) -> int:
        return int(self.samples_per_period) * int(self.n_periods)

    def phases(self) -> NDArray[np.float64]:
        return make_phase_grid(self.samples_per_period, self.n_periods, self.phase_offset)


# --------------------------------------------------------------------------- #
# Результат
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SimResult:
    """Полный неизменяемый снимок одного прогона."""

    true_positions: NDArray[np.float64]  # (n, 3) в системе маяка
    estimated_positions: NDArray[np.float64]  # (n, 3)
    true_positions_abs: NDArray[np.float64]
    estimated_positions_abs: NDArray[np.float64]
    a_true: NDArray[np.float64]
    b_true: NDArray[np.float64]
    a_meas: NDArray[np.float64]
    b_meas: NDArray[np.float64]
    solution: PositionSolution
    b0: float
    valid: NDArray[np.bool_]
    inject_at: str
    meta: dict[str, Any] = field(default_factory=dict)

    # -- погрешности --------------------------------------------------------- #
    @property
    def error_abs(self) -> NDArray[np.float64]:
        return self.estimated_positions - self.true_positions

    @property
    def error_norm(self) -> NDArray[np.float64]:
        return np.linalg.norm(self.error_abs, axis=-1)

    @property
    def error_rel(self) -> NDArray[np.float64]:
        return self.error_abs / safe_norm(self.true_positions, keepdims=True)

    @property
    def error_rel_norm(self) -> NDArray[np.float64]:
        return self.error_norm / safe_norm(self.true_positions, axis=-1)

    @property
    def r_true(self) -> NDArray[np.float64]:
        return norm(self.true_positions, axis=-1)

    @property
    def r_est(self) -> NDArray[np.float64]:
        return self.solution.r

    def statistics(self) -> dict[str, Any]:
        return error_statistics(self.true_positions, self.estimated_positions, self.valid)

    def summary(self) -> str:
        s = self.statistics()
        return (
            f"SimResult: N={s['n']}/{self.valid.size} (valid), "
            f"RMS={s['rms_norm']:.3e} м, mean={s['mean_norm']:.3e} м, "
            f"p95={s['p95_norm']:.3e} м, rel(mean)={s['mean_rel']:.3e}"
        )

    def ab_error(self) -> dict[str, NDArray[np.float64]]:
        """Погрешность восстановления a/b относительно истинных."""
        return {
            "abs_a": norm(self.a_meas - self.a_true, axis=-1),
            "abs_b": norm(self.b_meas - self.b_true, axis=-1),
            "rel_a": norm(self.a_meas - self.a_true, axis=-1) / safe_norm(self.a_true, axis=-1),
            "rel_b": norm(self.b_meas - self.b_true, axis=-1) / safe_norm(self.b_true, axis=-1),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()


# --------------------------------------------------------------------------- #
# Сценарий
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scenario:
    """Один прогон: источник + траектория + (опц.) тракт измерения с шумом.

    Parameters
    ----------
    source:
        Модель маяка (:class:`~mag_nav.source.dipole.RotatingDipole`,
        :class:`~mag_nav.source.solenoid.Solenoid`, ...).
    trajectory:
        Точки наблюдения ``(n, 3)`` в **мировой** системе, м.
    sensor:
        Конфигурация тракта. ``None`` — шум вносится напрямую в a/b
        (или отсутствует), без синтеза отсчётов.
    noise:
        Модель шума. При ``sensor is not None`` приоритет у ``sensor.noise``.
    inject_at:
        ``"samples"`` | ``"ab"`` | ``"none"``.
    frame_alignment:
        Поворот «рамка приёмника → рамка маяка». По умолчанию единичный
        (углы не определяются).
    b0_override:
        Переопределить дипольную константу для обратного решения
        (исследование ошибки калибровки).
    seed:
        Seed ГСЧ для воспроизводимости.
    """

    source: MagneticSource
    trajectory: NDArray[np.float64]
    sensor: SensorConfig | None = None
    noise: NoiseModel | None = None
    inject_at: Literal["samples", "ab", "none"] = "samples"
    frame_alignment: RotationMatrix = IDENTITY
    b0_override: float | None = None
    seed: int | None = None
    n_harmonic: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "trajectory", as_points(self.trajectory))
        object.__setattr__(self, "frame_alignment", self.frame_alignment or IDENTITY)

    # -- вспомогательное ----------------------------------------------------- #
    @property
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    @property
    def b0_solve(self) -> float:
        return float(
            self.b0_override if self.b0_override is not None else self.source.dipole_constant
        )

    @property
    def active_noise(self) -> NoiseModel | None:
        if self.sensor is not None and self.sensor.noise is not None:
            return self.sensor.noise
        return self.noise

    # -- основной прогон ----------------------------------------------------- #
    def run(self, *, seed: int | None = None) -> SimResult:
        """Выполнить симуляцию. Чистая функция от ``(self, seed)``."""
        rng = np.random.default_rng(self.seed if seed is None else seed)
        pts = self.trajectory
        rel_true = self.source.relative_points(pts)

        ab_true = ab_from_source(self.source, pts, n_harmonic=self.n_harmonic)
        a_true, b_true = (
            self.frame_alignment.apply_inverse(ab_true.a),
            self.frame_alignment.apply_inverse(ab_true.b),
        )

        noise = self.active_noise
        mode = self.inject_at
        if noise is None:
            mode = "none"

        meta: dict[str, Any] = {
            "mode": mode,
            "source": type(self.source).__name__,
            "b0_true": float(self.source.dipole_constant),
            "b0_solve": self.b0_solve,
            "n_harmonic": self.n_harmonic,
            "exact_ab": ab_true.exact,
        }

        if mode == "none":
            a_meas, b_meas = a_true.copy(), b_true.copy()

        elif mode == "ab":
            a_meas = noise.apply(a_true, rng)
            b_meas = noise.apply(b_true, rng)
            meta["noise"] = noise.describe()

        elif mode == "samples":
            sensor = self.sensor or SensorConfig()
            phases = sensor.phases()
            series = synthesize_series(a_true, b_true, phases)  # (n_s, n_p, 3)
            if noise is not None:
                series = noise.apply(series, rng)
                meta["noise"] = noise.describe()
            demod: DemodulationResult = (
                demodulate_lstsq(series, phases, with_dc=sensor.with_dc)
                if sensor.demod_method == "lstsq"
                else _demodulate_projection(series, phases)
            )
            a_meas, b_meas = demod.a, demod.b
            meta.update(
                demod_method=sensor.demod_method,
                n_samples=sensor.n_samples,
                samples_per_period=sensor.samples_per_period,
                residual_rms=demod.residual_rms,
            )
        else:
            raise ValueError(f"Неизвестный inject_at: {mode!r}")

        a_solve = self.frame_alignment.apply(a_meas)
        b_solve = self.frame_alignment.apply(b_meas)

        sol = solve_position(a_solve, b_solve, self.b0_solve)
        est_rel = sol.points
        est_abs = est_rel + self.source.origin

        return SimResult(
            true_positions=rel_true,
            estimated_positions=est_rel,
            true_positions_abs=pts,
            estimated_positions_abs=est_abs,
            a_true=a_true,
            b_true=b_true,
            a_meas=a_meas,
            b_meas=b_meas,
            solution=sol,
            b0=self.b0_solve,
            valid=sol.valid,
            inject_at=mode,
            meta=meta,
        )

    # -- варианты ------------------------------------------------------------ #
    def without_noise(self) -> "Scenario":
        return Scenario(
            source=self.source,
            trajectory=self.trajectory,
            sensor=(self.sensor.with_noise(None) if self.sensor else None),
            noise=None,
            inject_at="none",
            frame_alignment=self.frame_alignment,
            b0_override=self.b0_override,
            seed=self.seed,
            n_harmonic=self.n_harmonic,
        )

    def with_trajectory(self, trajectory: ArrayLike) -> "Scenario":
        return Scenario(
            source=self.source,
            trajectory=as_points(trajectory),
            sensor=self.sensor,
            noise=self.noise,
            inject_at=self.inject_at,
            frame_alignment=self.frame_alignment,
            b0_override=self.b0_override,
            seed=self.seed,
            n_harmonic=self.n_harmonic,
        )


def _demodulate_projection(
    series: NDArray[np.float64], phases: NDArray[np.float64]
) -> DemodulationResult:
    from ..signal.recovery import demodulate

    return demodulate(series, phases)


# СенсорConfig.with_noise — добавим как метод через dataclass-замену
def _sensor_with_noise(self: SensorConfig, noise: NoiseModel | None) -> SensorConfig:
    return SensorConfig(
        samples_per_period=self.samples_per_period,
        n_periods=self.n_periods,
        noise=noise,
        demod_method=self.demod_method,
        with_dc=self.with_dc,
        phase_offset=self.phase_offset,
    )


SensorConfig.with_noise = _sensor_with_noise  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Монте-Карло
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MonteCarloSummary:
    """Агрегированная статистика по множеству реализаций шума."""

    n_trials: int
    error_norm_mean: NDArray[np.float64]  # (n_points,) среднее по реализациям
    error_norm_std: NDArray[np.float64]
    rms_per_point: NDArray[np.float64]
    overall_rms: float
    overall_p95: float
    validity_ratio: float
    per_component_rms: NDArray[np.float64]  # (3,)
    results: tuple[SimResult, ...]

    def summary(self) -> str:
        return (
            f"MonteCarlo[{self.n_trials}]: RMS={self.overall_rms:.3e} м, "
            f"p95={self.overall_p95:.3e} м, valid={100 * self.validity_ratio:.1f}%, "
            f"RMS(xyz)=({self.per_component_rms[0]:.2e}, "
            f"{self.per_component_rms[1]:.2e}, {self.per_component_rms[2]:.2e})"
        )


def monte_carlo(
    scenario: Scenario,
    n_trials: int,
    *,
    base_seed: int = 0,
    keep_results: bool = True,
) -> MonteCarloSummary:
    """Прогнать сценарий ``n_trials`` раз с разными seed и агрегировать."""
    if n_trials < 1:
        raise ValueError("n_trials >= 1")
    results: list[SimResult] = []
    for k in range(int(n_trials)):
        results.append(scenario.run(seed=base_seed + k))

    n_pts = scenario.trajectory.shape[0]
    err_stack = np.full((len(results), n_pts, 3), np.nan)
    valid_stack = np.zeros((len(results), n_pts), dtype=bool)
    for i, res in enumerate(results):
        err_stack[i] = res.error_abs
        valid_stack[i] = res.valid

    err_norm = np.linalg.norm(err_stack, axis=-1)
    valid_norm = valid_stack & np.isfinite(err_norm)
    counts = np.sum(valid_norm, axis=0)
    sum_norm = np.sum(np.where(valid_norm, err_norm, 0.0), axis=0)
    sum_norm_sq = np.sum(np.where(valid_norm, err_norm**2, 0.0), axis=0)
    mean_norm = np.full(n_pts, np.nan, dtype=np.float64)
    mean_sq_norm = np.full(n_pts, np.nan, dtype=np.float64)
    np.divide(sum_norm, counts, out=mean_norm, where=counts > 0)
    np.divide(sum_norm_sq, counts, out=mean_sq_norm, where=counts > 0)
    rms_pt = np.sqrt(mean_sq_norm)
    variance_num = np.sum(
        np.where(valid_norm, (err_norm - mean_norm[np.newaxis, :]) ** 2, 0.0), axis=0
    )
    variance = np.full(n_pts, np.nan, dtype=np.float64)
    np.divide(variance_num, counts, out=variance, where=counts > 0)
    std_norm = np.sqrt(variance)
    n_valid = int(np.count_nonzero(valid_norm))
    component_sum = np.sum(np.where(valid_norm[..., np.newaxis], err_stack**2, 0.0), axis=(0, 1))
    rms_comp = np.sqrt(component_sum / n_valid) if n_valid else np.full(3, np.nan)

    all_vals = err_norm[valid_norm]
    return MonteCarloSummary(
        n_trials=len(results),
        error_norm_mean=mean_norm,
        error_norm_std=std_norm,
        rms_per_point=rms_pt,
        overall_rms=float(np.sqrt(np.mean(all_vals**2))) if all_vals.size else float("nan"),
        overall_p95=float(np.percentile(all_vals, 95)) if all_vals.size else float("nan"),
        validity_ratio=float(valid_stack.mean()),
        per_component_rms=rms_comp,
        results=tuple(results) if keep_results else (),
    )


# --------------------------------------------------------------------------- #
# Анализ расхождения моделей (аналог BoundaryR.m / CoordErrorSurface.m)
# --------------------------------------------------------------------------- #
def model_discrepancy_map(
    source_truth: MagneticSource,
    source_model: MagneticSource,
    *,
    plane: PlaneName = "xz",
    fixed: float = 0.0,
    u_range: tuple[float, float] = (0.05, 1.0),
    v_range: tuple[float, float] = (0.05, 1.0),
    n: tuple[int, int] = (80, 80),
    b0: float | None = None,
    n_harmonic: int = 64,
) -> dict[str, Any]:
    """Карта систематической погрешности координат из-за подмены модели.

    «Истина» — ``source_truth`` (например, соленоид), «модель приёмника» —
    ``source_model`` (например, точечный диполь с той же ``B0``).
    Обратное решение всегда использует аналитику диполя.

    Returns
    -------
    dict с ключами ``points, U, V, labels, error_abs (n_u, n_v, 3),
    error_norm, error_rel, valid, r_true``.
    """
    pts, U, V, labels = plane_grid(plane, fixed, u_range, v_range, n)
    flat = pts.reshape(-1, 3)

    ab_truth = ab_from_source(source_truth, flat, n_harmonic=n_harmonic)
    b0_solve = float(b0 if b0 is not None else source_model.dipole_constant)
    sol = solve_position(ab_truth.a, ab_truth.b, b0_solve)

    rel_true = source_truth.relative_points(flat)
    err = sol.points - rel_true
    err_norm = np.linalg.norm(err, axis=-1)

    shape = U.shape
    return {
        "points": pts,
        "U": U,
        "V": V,
        "labels": labels,
        "plane": plane,
        "fixed": fixed,
        "error_abs": err.reshape(shape + (3,)),
        "error_norm": err_norm.reshape(shape),
        "error_rel": (err_norm / safe_norm(rel_true, axis=-1)).reshape(shape),
        "valid": sol.valid.reshape(shape),
        "r_true": norm(rel_true, axis=-1).reshape(shape),
        "b0": b0_solve,
    }


def boundary_radius_map(
    source_truth: MagneticSource,
    source_model: MagneticSource,
    *,
    threshold: float,
    elevation_deg: float = 45.0,
    azimuth_deg: float = 45.0,
    r_grid: ArrayLike | None = None,
    relative: bool = True,
    n_harmonic: int = 64,
) -> dict[str, NDArray[np.float64]]:
    """Найти граничные дальности, на которых погрешность превышает порог.

    Для каждого направления (набора ``(elevation, azimuth)``) выполняется
    радиальное сканирование; возвращается минимальное ``r``, начиная с
    которого погрешность **устойчиво** ниже порога (аналог ``BoundaryR.m``,
    ``Aprox.m``).

    Parameters
    ----------
    threshold:
        Порог. При ``relative=True`` — доля от истинной дальности,
        иначе — метры.
    """
    r = (
        np.asarray(r_grid, dtype=np.float64).reshape(-1)
        if r_grid is not None
        else np.geomspace(0.02, 5.0, 400)
    )
    el = np.deg2rad(elevation_deg)
    az = np.deg2rad(azimuth_deg)
    dirs = np.array(
        [[np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]], dtype=np.float64
    )
    pts = dirs * r[:, np.newaxis]

    ab_truth = ab_from_source(source_truth, pts, n_harmonic=n_harmonic)
    sol = solve_position(ab_truth.a, ab_truth.b, source_model.dipole_constant)
    rel_true = source_truth.relative_points(pts)

    err_norm = np.linalg.norm(sol.points - rel_true, axis=-1)
    metric = err_norm / r if relative else err_norm
    ok = np.isfinite(metric) & sol.valid & (metric < threshold)

    boundary = np.nan
    if np.any(ok):
        idx = np.where(ok)[0]
        # первое r, после которого все точки удовлетворяют порогу
        tail_ok = np.all(ok[idx[-1] :])
        if tail_ok:
            for k in range(len(r)):
                if np.all(ok[k:]):
                    boundary = float(r[k])
                    break

    return {
        "r": r,
        "error_norm": err_norm,
        "metric": metric,
        "valid": sol.valid,
        "boundary_r": np.array([boundary]),
    }


def calibration_sweep(
    scenario: Scenario,
    b0_factors: ArrayLike,
    *,
    seed: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Влияние ошибки калибровки ``B0`` на точность.

    ``b0_factors`` — множители относительно истинного ``B0``.
    Направление не зависит от калибровки; дальность ∝ ``factor^(1/3)``.
    """
    fac = np.asarray(b0_factors, dtype=np.float64).reshape(-1)
    base_b0 = scenario.source.dipole_constant
    rms, rms_range, ang_err = [], [], []
    for f in fac:
        sc = Scenario(
            source=scenario.source,
            trajectory=scenario.trajectory,
            sensor=scenario.sensor,
            noise=scenario.noise,
            inject_at=scenario.inject_at,
            frame_alignment=scenario.frame_alignment,
            b0_override=base_b0 * float(f),
            seed=scenario.seed,
            n_harmonic=scenario.n_harmonic,
        )
        res = sc.run(seed=seed)
        stats = res.statistics()
        rms.append(stats["rms_norm"])
        with np.errstate(invalid="ignore"):
            rr = res.r_est / np.maximum(res.r_true, 1e-300)
            rms_range.append(float(np.sqrt(np.nanmean((rr[res.valid] - 1.0) ** 2))))
            cosang = np.sum(
                unit_safe(res.estimated_positions[res.valid])
                * unit_safe(res.true_positions[res.valid]),
                axis=-1,
            )
            ang_err.append(float(np.degrees(np.arccos(np.clip(cosang, -1, 1)).mean())))
    return {
        "factors": fac,
        "rms_position": np.asarray(rms),
        "rms_range_relative": np.asarray(rms_range),
        "angular_error_deg": np.asarray(ang_err),
    }


def unit_safe(v: NDArray[np.float64]) -> NDArray[np.float64]:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)
