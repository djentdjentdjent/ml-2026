"""Group-based Train/Val/Test splits without trajectory or beacon leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .config import MLConfig


def split_dataset(
    aligned_rows: pd.DataFrame,
    split_mode: str = "by_trajectory",
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    """Вернуть индексы строк Train/Val/Test с непересекающимися группами."""
    if split_mode not in ("by_trajectory", "by_beacon"):
        raise ValueError("split_mode должен быть 'by_trajectory' или 'by_beacon'")
    group_col = "trajectory_id" if split_mode == "by_trajectory" else "beacon_id"
    if group_col not in aligned_rows:
        raise ValueError(f"Нет столбца для группового разбиения: {group_col}")
    groups = aligned_rows[group_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if unique_groups.size < 3:
        raise ValueError(
            f"Для split_mode={split_mode!r} нужно минимум 3 группы {group_col}, "
            f"найдено {unique_groups.size}. Для by_beacon увеличьте n_beacons."
        )
    all_indices = np.arange(len(aligned_rows))
    first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=random_seed)
    train_val_idx, test_idx = next(first.split(all_indices, groups=groups))
    # Validation — 15% всей выборки: 0.15 / 0.85 оставшейся доли.
    val_fraction_remaining = 0.15 / 0.85
    second = GroupShuffleSplit(
        n_splits=1,
        test_size=val_fraction_remaining,
        random_state=random_seed + 1,
    )
    local_train, local_val = next(
        second.split(train_val_idx, groups=groups[train_val_idx])
    )
    train_idx = train_val_idx[local_train]
    val_idx = train_val_idx[local_val]
    output = {"train": train_idx, "val": val_idx, "test": test_idx}

    group_sets = {name: set(groups[idx]) for name, idx in output.items()}
    if group_sets["train"] & group_sets["val"]:
        raise RuntimeError("Утечка групп: пересечение Train и Val")
    if group_sets["train"] & group_sets["test"]:
        raise RuntimeError("Утечка групп: пересечение Train и Test")
    if group_sets["val"] & group_sets["test"]:
        raise RuntimeError("Утечка групп: пересечение Val и Test")
    return output


def split_summary(
    aligned_rows: pd.DataFrame,
    split_indices: dict[str, np.ndarray],
    split_mode: str,
) -> pd.DataFrame:
    """CSV-совместимая сводка размеров выборок и списков групп."""
    group_col = "trajectory_id" if split_mode == "by_trajectory" else "beacon_id"
    rows = []
    for name, idx in split_indices.items():
        part = aligned_rows.iloc[idx]
        rows.append({
            "split": name,
            "n_samples": len(part),
            "n_trajectories": part["trajectory_id"].nunique(),
            "n_beacons": part["beacon_id"].nunique(),
            "group_column": group_col,
            "group_ids": ";".join(sorted(part[group_col].astype(str).unique())),
        })
    return pd.DataFrame(rows)


def split_dataset_from_config(
    aligned_rows: pd.DataFrame, config: MLConfig
) -> dict[str, np.ndarray]:
    return split_dataset(aligned_rows, config.split_mode, config.random_seed)
