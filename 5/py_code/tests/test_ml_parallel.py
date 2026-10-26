"""Проверки CPU-параллелизма генератора ML-датасета."""
from __future__ import annotations

import pandas as pd

from mag_nav.ml.config import MLConfig, TrainingConfig
from mag_nav.ml.dataset import generate_dataset


def test_parallel_generation_is_reproducible_and_order_stable() -> None:
    common = dict(
        n_trajectories=3,
        n_points_per_trajectory=8,
        n_beacons=3,
        r_min=0.15,
        r_max=0.4,
        relative_noise_sigma=0.001,
        n_harmonic=8,
    )
    sequential = generate_dataset(
        config=MLConfig(split_mode="by_trajectory", random_seed=42, n_workers=1),
        **common,
    )
    parallel = generate_dataset(
        config=MLConfig(split_mode="by_trajectory", random_seed=42, n_workers=2),
        **common,
    )
    pd.testing.assert_frame_equal(sequential, parallel)
    assert sequential.attrs["n_workers_used"] == 1
    assert parallel.attrs["n_workers_used"] == 2


def test_cpu_threads_are_not_limited_to_four() -> None:
    config = TrainingConfig(model_type="mlp", cpu_threads=12)
    assert config.cpu_threads == 12
    # 0 leaves the framework's own hardware-aware default intact.
    assert TrainingConfig(model_type="mlp", cpu_threads=0).cpu_threads == 0
