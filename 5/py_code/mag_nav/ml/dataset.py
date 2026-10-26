"""Получение обучающих записей из существующего API симулятора."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import logging
import pickle
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..convention import safe_norm
from ..navigation.orientation import RotationMatrix
from ..noise.gaussian import GaussianNoise
from ..signal.ab_params import ab_invariants
from ..sim.scenario import Scenario
from ..sim.trajectory import line, random_first_octant
from ..source.base import MagneticSource
from .config import MLConfig, available_cpu_count
from .source_factory import make_source

logger = logging.getLogger(__name__)

GEOMETRY_FEATURES = ["dip_x", "dip_y", "dip_z", "dip_r", "dip_inv_r", "dip_inv_r2"]
INVARIANT_FEATURES = ["S_aa", "S_bb", "S_ab", "log_S_aa", "log_S_bb"]
RAW_AB_FEATURES = ["a_x", "b_x", "a_y", "b_y", "a_z", "b_z"]
TRUE_COLUMNS = ["true_x", "true_y", "true_z"]
DIP_COLUMNS = ["dip_x", "dip_y", "dip_z"]


def source_descriptor(source: MagneticSource) -> tuple[list[str], NDArray[np.float64]]:
    """Представить параметры источника числовым вектором.

    Схема извлекается из числовых полей класса-источника. Чтобы поменять
    физическую модель, достаточно подменить SOURCE_FACTORY. Для by_beacon
    все экземпляры в одном запуске должны иметь одинаковую схему полей.
    """
    names: list[str] = ["source_dipole_constant"]
    values: list[float] = [float(source.dipole_constant)]
    candidates = (
        [(item.name, getattr(source, item.name)) for item in fields(source)]
        if is_dataclass(source)
        else (sorted(vars(source).items()) if hasattr(source, "__dict__") else [])
    )
    for field_name, value in candidates:
        if isinstance(value, Enum) or isinstance(value, (str, bytes, type(None))):
            continue
        if isinstance(value, (bool, int, float, np.number)):
            names.append(f"source_{field_name}")
            values.append(float(value))
        elif isinstance(value, (np.ndarray, list, tuple)):
            array = np.asarray(value)
            if array.dtype.kind not in "iuf" or array.size == 0:
                continue
            for idx, item in enumerate(array.reshape(-1)):
                names.append(f"source_{field_name}_{idx}")
                values.append(float(item))
    return names, np.asarray(values, dtype=np.float64)


def _frame_alignment(rng: np.random.Generator) -> RotationMatrix:
    """Случайная ориентация датчика, постоянная в пределах одной траектории."""
    angles = rng.uniform(-np.pi, np.pi, size=3)
    return RotationMatrix.from_euler(*angles)


def _generate_trajectory_records(task: tuple) -> list[dict[str, Any]]:
    """Worker-функция уровня модуля, совместимая с multiprocessing spawn."""
    (
        source, desc_names, desc_values, beacon_idx, trajectory_idx, seed,
        n_points, r_min, r_max, noise_sigma, n_harmonic, eps,
    ) = task
    rng = np.random.default_rng(seed)
    endpoints = random_first_octant(2, r_min, r_max, rng=rng, origin=np.zeros(3))
    relative_trajectory = line(endpoints[0], endpoints[1], n_points)
    world_trajectory = relative_trajectory + np.asarray(source.origin, dtype=np.float64).reshape(3)
    rotation = _frame_alignment(rng)
    noise = GaussianNoise(sigma=noise_sigma, relative=True) if noise_sigma > 0 else None
    sample_seed = int(rng.integers(0, np.iinfo(np.int32).max))

    sim = Scenario(
        source=source,
        trajectory=world_trajectory,
        noise=noise,
        inject_at="ab" if noise is not None else "none",
        frame_alignment=rotation,
        seed=sample_seed,
        n_harmonic=n_harmonic,
    ).run()

    a = np.asarray(sim.a_meas, dtype=np.float64)
    b = np.asarray(sim.b_meas, dtype=np.float64)
    invariants = ab_invariants(a, b)
    p_true = np.asarray(sim.true_positions, dtype=np.float64)
    p_dip = np.asarray(sim.estimated_positions, dtype=np.float64)
    r_true = safe_norm(p_true, axis=-1)
    r_dip = safe_norm(p_dip, axis=-1)
    trajectory_id = f"traj-b{beacon_idx:04d}-{trajectory_idx:05d}"
    beacon_id = f"beacon-{beacon_idx:04d}"
    rows: list[dict[str, Any]] = []

    for sample_idx in range(n_points):
        p_t, p_d = p_true[sample_idx], p_dip[sample_idx]
        delta = p_t - p_d
        row: dict[str, Any] = {
            "trajectory_id": trajectory_id,
            "beacon_id": beacon_id,
            "sample_idx": sample_idx,
            "source_type": type(source).__name__,
            "solver_valid": bool(sim.valid[sample_idx]),
            "solver_reason": int(sim.solution.reasons[sample_idx]),
            "solver_r": float(sim.solution.r[sample_idx]),
            "solver_uvw_norm": float(sim.solution.uvw_norm[sample_idx]),
            "true_x": p_t[0], "true_y": p_t[1], "true_z": p_t[2],
            "dip_x": p_d[0], "dip_y": p_d[1], "dip_z": p_d[2],
            "dip_r": r_dip[sample_idx],
            "dip_inv_r": 1.0 / (r_dip[sample_idx] + eps),
            "dip_inv_r2": 1.0 / (r_dip[sample_idx] ** 2 + eps),
            "r_true": r_true[sample_idx], "r_dip": r_dip[sample_idx],
            "delta_x": delta[0], "delta_y": delta[1], "delta_z": delta[2],
            "S_aa": invariants.asq[sample_idx],
            "S_bb": invariants.bsq[sample_idx],
            "S_ab": invariants.ab[sample_idx],
            "log_S_aa": np.log(max(float(invariants.asq[sample_idx]), eps)),
            "log_S_bb": np.log(max(float(invariants.bsq[sample_idx]), eps)),
            "a_x": a[sample_idx, 0], "b_x": b[sample_idx, 0],
            "a_y": a[sample_idx, 1], "b_y": b[sample_idx, 1],
            "a_z": a[sample_idx, 2], "b_z": b[sample_idx, 2],
        }
        for name, value in zip(desc_names, desc_values, strict=True):
            row[name] = float(value)
        rows.append(row)
    return rows


def generate_dataset(
    *,
    config: MLConfig | None = None,
    n_trajectories: int = 30,
    n_points_per_trajectory: int = 64,
    n_beacons: int = 8,
    r_min: float = 0.12,
    r_max: float = 0.9,
    relative_noise_sigma: float = 0.003,
    n_harmonic: int = 32,
) -> pd.DataFrame:
    """Симулировать траектории параллельно и вернуть строку на кадр.

    ``n_workers=0`` (по умолчанию) использует доступные CPU-ядерные ресурсы;
    ``n_workers=1`` запускает последовательный режим для отладки. Порядок и
    случайные seed не зависят от количества workers.
    """
    cfg = config or MLConfig()
    if n_trajectories < 3:
        raise ValueError("Для группового сплита нужно не менее 3 траекторий")
    if n_points_per_trajectory < 2:
        raise ValueError("n_points_per_trajectory должен быть >= 2")
    if cfg.split_mode == "by_beacon" and n_beacons < 3:
        raise ValueError("split_mode='by_beacon' требует как минимум 3 маяка")
    if r_min <= 0 or r_max <= r_min:
        raise ValueError("Требуется 0 < r_min < r_max")
    if relative_noise_sigma < 0:
        raise ValueError("relative_noise_sigma не может быть отрицательным")

    beacon_count = n_beacons if cfg.split_mode == "by_beacon" else 1
    tasks: list[tuple] = []
    child_seeds = np.random.SeedSequence(cfg.random_seed).spawn(beacon_count * n_trajectories)
    child_index = 0
    descriptor_schema: list[str] | None = None

    for beacon_idx in range(beacon_count):
        source = make_source(beacon_idx, cfg.random_seed)
        descriptor_names, descriptor_values = source_descriptor(source)
        if descriptor_schema is None:
            descriptor_schema = descriptor_names
        elif descriptor_names != descriptor_schema:
            raise ValueError(
                "SOURCE_FACTORY создал несовместимые схемы параметров маяка. "
                "Все источники одного запуска должны иметь одинаковые числовые поля."
            )
        for trajectory_idx in range(n_trajectories):
            seed = int(child_seeds[child_index].generate_state(1, dtype=np.uint32)[0])
            child_index += 1
            tasks.append((
                source, descriptor_names, descriptor_values, beacon_idx, trajectory_idx, seed,
                n_points_per_trajectory, r_min, r_max, relative_noise_sigma, n_harmonic, cfg.eps,
            ))

    requested_workers = cfg.n_workers
    if requested_workers == 0:
        requested_workers = available_cpu_count()
    worker_count = max(1, min(requested_workers, len(tasks)))

    # Некоторые пользовательские классы источников могут содержать объекты,
    # непригодные для pickle. В таком случае сохраняем совместимость и явно
    # переключаемся на последовательный режим, не ломая генерацию данных.
    if worker_count > 1:
        try:
            for source_task in tasks[::max(1, len(tasks) // max(1, beacon_count))]:
                pickle.dumps(source_task[0])
        except Exception as exc:
            logger.warning(
                "Источник не сериализуется для multiprocessing (%r); "
                "генерация будет выполнена последовательно.", exc
            )
            worker_count = 1

    logger.info(
        "Генерация датасета: trajectories=%d, beacons=%d, workers=%d (requested=%d)",
        len(tasks), beacon_count, worker_count, cfg.n_workers,
    )
    if worker_count == 1:
        results = map(_generate_trajectory_records, tasks)
        records = [row for trajectory_rows in results for row in trajectory_rows]
    else:
        # map возвращает результаты в порядке исходных задач; порядок строк и
        # воспроизводимость не зависят от того, какой процесс завершился первым.
        chunksize = max(1, len(tasks) // (worker_count * 4))
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp.get_context("spawn")) as pool:
            results = pool.map(_generate_trajectory_records, tasks, chunksize=chunksize)
            records = [row for trajectory_rows in results for row in trajectory_rows]
    frame = pd.DataFrame.from_records(records)
    frame.attrs["n_workers_used"] = worker_count
    return frame


@dataclass
class FeatureExtractor:
    """Сборка векторов X/Y и временных окон строго внутри траектории."""

    config: MLConfig

    @property
    def frame_feature_columns(self) -> list[str]:
        columns = GEOMETRY_FEATURES.copy()
        if self.config.use_invariants:
            columns.extend(INVARIANT_FEATURES)
        if self.config.use_raw_ab:
            columns.extend(RAW_AB_FEATURES)
        return columns

    def transform(
        self,
        raw_frame: pd.DataFrame,
        eligible_frame: pd.DataFrame,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], pd.DataFrame, list[str]]:
        """Вернуть ``X, Y, sample_weight, aligned_rows, feature_names``."""
        frame_cols = self.frame_feature_columns
        absent = sorted(set(frame_cols) - set(raw_frame.columns))
        if absent:
            raise ValueError(f"В датасете отсутствуют признаки: {absent}")
        if self.config.include_beacon_params:
            beacon_cols = sorted(
                [c for c in raw_frame.columns if c.startswith("source_") and pd.api.types.is_numeric_dtype(raw_frame[c])],
                key=lambda s: (s.split("_")[-1].isdigit(), s),
            )
        else:
            beacon_cols = []

        x_rows: list[NDArray[np.float64]] = []
        y_rows: list[NDArray[np.float64]] = []
        weight_rows: list[float] = []
        metadata_rows: list[pd.Series] = []
        feature_names = [f"t{offset}_{name}" for offset in range(self.config.window_size) for name in frame_cols]
        feature_names.extend(beacon_cols)
        eligible_keys = set(zip(eligible_frame["trajectory_id"].astype(str), eligible_frame["sample_idx"].astype(int)))

        for trajectory_id, sequence in raw_frame.groupby("trajectory_id", sort=False):
            sequence = sequence.sort_values("sample_idx", kind="stable").reset_index(drop=True)
            indices = sequence["sample_idx"].astype(int).to_numpy()
            by_idx = {int(idx): sequence.iloc[pos] for pos, idx in enumerate(indices)}
            sequence_features = sequence[frame_cols].to_numpy(dtype=np.float64)
            finite_rows = np.all(np.isfinite(sequence_features), axis=1)
            valid_rows = sequence.get("solver_valid", pd.Series(True, index=sequence.index)).to_numpy(bool)
            acceptable_first = np.flatnonzero(finite_rows & valid_rows)
            if acceptable_first.size == 0:
                acceptable_first = np.flatnonzero(finite_rows)
            if acceptable_first.size == 0:
                continue
            first_valid_idx = int(indices[acceptable_first[0]])

            for _, row in sequence.iterrows():
                sample_idx = int(row["sample_idx"])
                if (str(trajectory_id), sample_idx) not in eligible_keys:
                    continue
                if self.config.padding == "drop" and sample_idx < self.config.window_size - 1:
                    continue
                frame_window: list[float] = []
                for offset in range(self.config.window_size):
                    wanted = sample_idx - offset
                    if wanted < int(indices[0]):
                        wanted = first_valid_idx
                    hist_row = by_idx.get(wanted)
                    if hist_row is None or not np.all(np.isfinite(hist_row[frame_cols].to_numpy(dtype=float))):
                        previous = [int(idx) for idx in indices if idx <= wanted]
                        hist_row = None
                        for prev_idx in reversed(previous):
                            candidate = by_idx[prev_idx]
                            if np.all(np.isfinite(candidate[frame_cols].to_numpy(dtype=float))):
                                hist_row = candidate
                                break
                        if hist_row is None:
                            hist_row = by_idx[first_valid_idx]
                    frame_window.extend(hist_row[frame_cols].to_numpy(dtype=np.float64).tolist())
                if beacon_cols:
                    frame_window.extend(row[beacon_cols].to_numpy(dtype=np.float64).tolist())
                x_rows.append(np.asarray(frame_window, dtype=np.float64))
                delta = row[TRUE_COLUMNS].to_numpy(dtype=np.float64) - row[DIP_COLUMNS].to_numpy(dtype=np.float64)
                if self.config.target_mode == "relative":
                    delta = delta / max(float(row["r_dip"]), self.config.eps)
                y_rows.append(delta)
                radius = max(float(row["r_dip"]), self.config.eps)
                weight_rows.append(radius ** (-self.config.sample_weight_alpha))
                metadata_rows.append(row)

        if not x_rows:
            raise ValueError("После фильтрации и построения окон не осталось обучающих примеров")
        X = np.asarray(x_rows, dtype=np.float64)
        Y = np.asarray(y_rows, dtype=np.float64)
        weights = np.asarray(weight_rows, dtype=np.float64)
        weights /= max(float(np.mean(weights)), self.config.eps)
        aligned = pd.DataFrame(metadata_rows).reset_index(drop=True)
        return X, Y, weights, aligned, feature_names
