"""Фильтрация невалидных решений солвера и выбросов октанта."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import MLConfig

logger = logging.getLogger(__name__)


def annotate_filtering(
    frame: pd.DataFrame, config: MLConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Добавить к каждой точке булевы флаги диагностики и вернуть отчёт.

    Категории причин в отчёте могут перекрываться: один образец способен
    одновременно содержать невалидный статус солвера и NaN/Inf. ``filter_retained``
    — итоговая маска, вычисленная объединением всех условий.
    """
    cfg = config or MLConfig()
    if frame.empty:
        raise ValueError("В датасете нет строк")
    required = {
        "solver_valid", "true_x", "true_y", "true_z", "dip_x", "dip_y", "dip_z",
        "r_true", "r_dip", "delta_x", "delta_y", "delta_z",
        "a_x", "b_x", "a_y", "b_y", "a_z", "b_z", "S_aa", "S_bb", "S_ab",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Для фильтрации не хватает столбцов: {missing}")

    true_p = frame[["true_x", "true_y", "true_z"]].to_numpy(dtype=float)
    dip_p = frame[["dip_x", "dip_y", "dip_z"]].to_numpy(dtype=float)
    raw = frame[["a_x", "b_x", "a_y", "b_y", "a_z", "b_z", "S_aa", "S_bb", "S_ab"]].to_numpy(dtype=float)
    deltas = frame[["delta_x", "delta_y", "delta_z"]].to_numpy(dtype=float)
    r_true = frame["r_true"].to_numpy(dtype=float)

    finite = (
        np.all(np.isfinite(true_p), axis=1)
        & np.all(np.isfinite(dip_p), axis=1)
        & np.all(np.isfinite(raw), axis=1)
        & np.all(np.isfinite(deltas), axis=1)
        & np.isfinite(r_true)
        & (r_true > cfg.eps)
    )
    solver_valid = frame["solver_valid"].to_numpy(dtype=bool)
    octant_mismatch = np.zeros(len(frame), dtype=bool)
    tol = cfg.coordinate_zero_tol_m
    for axis in range(3):
        true_axis = true_p[:, axis]
        dip_axis = dip_p[:, axis]
        mismatch = ((true_axis > tol) & (dip_axis < -tol)) | ((true_axis < -tol) & (dip_axis > tol))
        octant_mismatch |= mismatch
    octant_mismatch &= finite

    residual_norm = np.linalg.norm(deltas, axis=1)
    residual_outlier = finite & (residual_norm > cfg.octant_outlier_tau * r_true)
    invalid_solver = ~solver_valid
    retained_mask = finite & solver_valid & ~octant_mismatch & ~residual_outlier

    annotated = frame.copy()
    annotated["filter_finite"] = finite
    annotated["filter_solver_valid"] = solver_valid
    annotated["filter_octant_mismatch"] = octant_mismatch
    annotated["filter_residual_outlier"] = residual_outlier
    annotated["filter_retained"] = retained_mask

    total = int(len(frame))
    report_rows = []

    def add_report(category: str, mask: np.ndarray, overlap: bool = True) -> None:
        count = int(np.count_nonzero(mask))
        report_rows.append({
            "category": category,
            "count": count,
            "percent_of_total": 100.0 * count / max(total, 1),
            "total": total,
            "overlap_possible": overlap,
        })

    for reason, mask in [
        ("solver_invalid", invalid_solver),
        ("nan_or_inf", ~finite),
        ("octant_mismatch", octant_mismatch),
        ("residual_outlier", residual_outlier),
        ("octant_or_residual_outlier", octant_mismatch | residual_outlier),
        ("retained", retained_mask),
        ("excluded_union", ~retained_mask),
    ]:
        add_report(reason, mask, reason not in {"retained", "excluded_union"})

    # Битовые причины из PositionSolution.reasons: 1=DENOM, 2=U, 4=V,
    # 8=W, 16=UVW normalization, 32=nonfinite input.
    reason_labels = {
        1: "solver_reason_denominator",
        2: "solver_reason_u_negative",
        4: "solver_reason_v_negative",
        8: "solver_reason_w_negative",
        16: "solver_reason_uvw_norm",
        32: "solver_reason_nonfinite",
    }
    if "solver_reason" in frame.columns:
        reasons = frame["solver_reason"].fillna(0).to_numpy(dtype=np.int64)
        for bit, label in reason_labels.items():
            add_report(label, (reasons & bit) != 0)

    report = pd.DataFrame(report_rows)
    logger.info(
        "ML filtering: total=%d, retained=%d (%.1f%%), NaN/Inf=%.1f%%, "
        "invalid_solver=%.1f%%, octant/outlier=%.1f%%",
        total,
        int(retained_mask.sum()),
        100 * float(retained_mask.mean()),
        100 * float((~finite).mean()),
        100 * float(invalid_solver.mean()),
        100 * float((octant_mismatch | residual_outlier).mean()),
    )
    if not retained_mask.any():
        raise ValueError(
            "Фильтры удалили все точки. Проверьте модель источника, tau, диапазон расстояний "
            "и статус PositionSolution.valid."
        )
    return annotated, report


def filter_dataset(
    frame: pd.DataFrame, config: MLConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Вернуть только пригодные обучающие точки и CSV-совместимый отчёт."""
    annotated, report = annotate_filtering(frame, config)
    filtered = annotated.loc[annotated["filter_retained"]].copy().reset_index(drop=True)
    return filtered, report
