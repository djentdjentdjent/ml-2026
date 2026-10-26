#!/usr/bin/env python3
"""Обучить ML-компенсатор для дипольной оценки позиции."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Запуск без предварительной установки пакета: uv run scripts/train_ml_corrector.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mag_nav.ml.config import MLConfig, TrainingConfig
from mag_nav.ml.training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mlp", "catboost"), default="mlp")
    parser.add_argument("--split-mode", choices=("by_trajectory", "by_beacon"), default="by_beacon")
    parser.add_argument(
        "--trajectories", type=int, default=50, help="Количество траекторий на каждый маяк"
    )
    parser.add_argument("--points", type=int, default=64, help="Кадров в траектории")
    parser.add_argument("--beacons", type=int, default=64, help="Число маяков для by_beacon")
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--padding", choices=("repeat_first", "drop"), default="repeat_first")
    parser.add_argument("--target-mode", choices=("meters", "relative"), default="meters")
    parser.add_argument(
        "--use-raw-ab", action="store_true", help="Добавить raw a/b, зависящие от ориентации"
    )
    parser.add_argument(
        "--no-invariants", action="store_true", help="Исключить инвариантные признаки"
    )
    parser.add_argument("--exclude-beacon-params", action="store_true")
    parser.add_argument("--tau", type=float, default=0.8, help="Порог остатка относительно r_true")
    parser.add_argument(
        "--weight-alpha", type=float, default=0.0, help="sample_weight = 1 / r_dip**alpha"
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Параллельная генерация: 0=авто, 1=последовательно"
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="Потоки CatBoost/PyTorch: 0=штатное число CPU-ядер",
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=0,
        help="DataLoader workers для PyTorch (обычно 0 достаточно)",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu", help="Устройство PyTorch MLP"
    )
    parser.add_argument("--catboost-task-type", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--catboost-devices", default=None, help="GPU IDs для CatBoost, например 0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--noise-sigma", type=float, default=0.003, help="Относительное СКО шума коэффициентов a/b"
    )
    parser.add_argument("--r-min", type=float, default=0.12)
    parser.add_argument("--r-max", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    ml_config = MLConfig(
        window_size=args.window_size,
        use_raw_ab=args.use_raw_ab,
        use_invariants=not args.no_invariants,
        include_beacon_params=not args.exclude_beacon_params,
        target_mode=args.target_mode,
        padding=args.padding,
        split_mode=args.split_mode,
        random_seed=42,
        octant_outlier_tau=args.tau,
        sample_weight_alpha=args.weight_alpha,
        n_workers=args.workers,
    )
    training_config = TrainingConfig(
        model_type=args.model,
        random_seed=42,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        cpu_threads=args.cpu_threads,
        dataloader_num_workers=args.loader_workers,
        torch_device=args.device,
        catboost_task_type=args.catboost_task_type,
        catboost_devices=args.catboost_devices,
        verbose=not args.quiet,
    )
    result = run_training(
        ml_config=ml_config,
        training_config=training_config,
        n_trajectories=args.trajectories,
        n_points_per_trajectory=args.points,
        n_beacons=args.beacons,
        r_min=args.r_min,
        r_max=args.r_max,
        relative_noise_sigma=args.noise_sigma,
        output_root=args.output,
        run_name=args.run_name,
    )
    print(f"\nЭксперимент сохранён: {result.resolve()}")
    print(f"Метрики: {(result / 'test_metrics.csv').resolve()}")
    print(f"История обучения: {(result / 'training_history.csv').resolve()}")
    print(f"Графики: {(result / 'plots').resolve()}")


if __name__ == "__main__":
    main()
