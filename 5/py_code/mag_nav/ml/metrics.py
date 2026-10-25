"""Метрики позиционирования до/после ML-коррекции."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def _summary(error: np.ndarray) -> dict[str, float]:
    e = np.asarray(error, dtype=np.float64).reshape(-1)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"n": 0, "mean_m": np.nan, "median_m": np.nan, "rmse_m": np.nan, "p95_m": np.nan, "max_m": np.nan}
    return {
        "n": int(e.size),
        "mean_m": float(np.mean(e)),
        "median_m": float(np.median(e)),
        "rmse_m": float(np.sqrt(np.mean(e**2))),
        "p95_m": float(np.percentile(e, 95)),
        "max_m": float(np.max(e)),
    }


def evaluate_predictions(
    true_positions: ArrayLike,
    dipole_positions: ArrayLike,
    corrected_positions: ArrayLike,
) -> pd.DataFrame:
    """Сводные метрики базовой и скорректированной 3D-ошибки в метрах."""
    true = np.asarray(true_positions, dtype=np.float64).reshape(-1, 3)
    baseline = np.asarray(dipole_positions, dtype=np.float64).reshape(-1, 3)
    corrected = np.asarray(corrected_positions, dtype=np.float64).reshape(-1, 3)
    if true.shape != baseline.shape or true.shape != corrected.shape:
        raise ValueError("true, baseline и corrected должны иметь одинаковую форму (n, 3)")
    keep = np.all(np.isfinite(true), axis=1) & np.all(np.isfinite(baseline), axis=1) & np.all(np.isfinite(corrected), axis=1)
    e_base = np.linalg.norm(true[keep] - baseline[keep], axis=1)
    e_corr = np.linalg.norm(true[keep] - corrected[keep], axis=1)
    rows = [{"estimator": "Baseline (Dipole)", **_summary(e_base)}, {"estimator": "Corrected (Dipole + ML)", **_summary(e_corr)}]
    return pd.DataFrame(rows)


def distance_bin_metrics(
    true_positions: ArrayLike,
    dipole_positions: ArrayLike,
    corrected_positions: ArrayLike,
    *,
    n_bins: int = 6,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Метрики Mean/Median/P95 по бинам истинного расстояния."""
    true = np.asarray(true_positions, dtype=np.float64).reshape(-1, 3)
    baseline = np.asarray(dipole_positions, dtype=np.float64).reshape(-1, 3)
    corrected = np.asarray(corrected_positions, dtype=np.float64).reshape(-1, 3)
    if n_bins < 1:
        raise ValueError("n_bins должен быть >= 1")
    keep = np.all(np.isfinite(true), axis=1) & np.all(np.isfinite(baseline), axis=1) & np.all(np.isfinite(corrected), axis=1)
    true, baseline, corrected = true[keep], baseline[keep], corrected[keep]
    if not len(true):
        return pd.DataFrame(columns=["bin", "r_left_m", "r_right_m", "n", "baseline_mean_m", "baseline_median_m", "baseline_p95_m", "corrected_mean_m", "corrected_median_m", "corrected_p95_m"])
    r = np.linalg.norm(true, axis=1)
    if strategy == "quantile":
        edges = np.unique(np.quantile(r, np.linspace(0, 1, n_bins + 1)))
    elif strategy == "uniform":
        lo, hi = float(np.min(r)), float(np.max(r))
        edges = np.linspace(lo, hi if hi > lo else lo + 1e-9, n_bins + 1)
    else:
        raise ValueError("strategy должен быть 'quantile' или 'uniform'")
    if len(edges) < 2:
        edges = np.array([float(r[0]) - 1e-9, float(r[0]) + 1e-9])
    bin_index = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, len(edges) - 2)
    e_base = np.linalg.norm(true - baseline, axis=1)
    e_corr = np.linalg.norm(true - corrected, axis=1)
    rows = []
    for idx in range(len(edges) - 1):
        mask = bin_index == idx
        if not mask.any():
            continue
        b, c = e_base[mask], e_corr[mask]
        rows.append({
            "bin": idx,
            "r_left_m": float(edges[idx]), "r_right_m": float(edges[idx + 1]), "n": int(mask.sum()),
            "baseline_mean_m": float(np.mean(b)), "baseline_median_m": float(np.median(b)), "baseline_p95_m": float(np.percentile(b, 95)),
            "corrected_mean_m": float(np.mean(c)), "corrected_median_m": float(np.median(c)), "corrected_p95_m": float(np.percentile(c, 95)),
        })
    return pd.DataFrame(rows)
