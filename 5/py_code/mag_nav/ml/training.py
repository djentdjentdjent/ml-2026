"""Воспроизводимый pipeline обучения, оценки и сохранения артефактов."""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MLConfig, TrainingConfig, available_cpu_count
from .dataset import DIP_COLUMNS, TRUE_COLUMNS, FeatureExtractor, generate_dataset
from .metrics import distance_bin_metrics, evaluate_predictions
from .models import NearFieldCorrector
from .preprocessing import annotate_filtering
from .splitting import split_dataset, split_summary
from .source_factory import SOURCE_FACTORY
from .visualize import visualize_run

logger = logging.getLogger(__name__)


def run_training(
    *,
    ml_config: MLConfig | None = None,
    training_config: TrainingConfig | None = None,
    n_trajectories: int = 30,
    n_points_per_trajectory: int = 64,
    n_beacons: int = 8,
    r_min: float = 0.12,
    r_max: float = 0.9,
    relative_noise_sigma: float = 0.003,
    n_harmonic: int = 32,
    output_root: str | Path = "runs",
    run_name: str | None = None,
) -> Path:
    """Выполнить эксперимент с групповой валидацией и сохранить все артефакты."""
    cfg = ml_config or MLConfig()
    train_cfg = training_config or TrainingConfig()
    if cfg.split_mode == "by_trajectory" and cfg.include_beacon_params:
        logger.info("by_trajectory: отключаю include_beacon_params для фиксированного маяка")
        cfg = replace(cfg, include_beacon_params=False)
    elif cfg.split_mode == "by_beacon" and not cfg.include_beacon_params:
        logger.warning("by_beacon: включаю include_beacon_params для обобщения на новые маяки")
        cfg = replace(cfg, include_beacon_params=True)
    if cfg.random_seed != train_cfg.random_seed:
        logger.warning("MLConfig.seed=%d отличается от TrainingConfig.seed=%d", cfg.random_seed, train_cfg.random_seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or f"{timestamp}_{cfg.split_mode}_{train_cfg.model_type}"
    run_dir = Path(output_root) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    log_handler = logging.FileHandler(run_dir / "training.log", encoding="utf-8")
    log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(log_handler)
    try:
        generation_params = {
            "n_trajectories_per_beacon": n_trajectories,
            "n_points_per_trajectory": n_points_per_trajectory,
            "n_beacons": n_beacons,
            "n_beacons_generated": n_beacons if cfg.split_mode == "by_beacon" else 1,
            "r_min_m": r_min,
            "r_max_m": r_max,
            "relative_noise_sigma": relative_noise_sigma,
            "n_harmonic": n_harmonic,
            "n_workers_requested": cfg.n_workers,
            "n_workers_effective": None,
        }
        factory_name = SOURCE_FACTORY if isinstance(SOURCE_FACTORY, str) else f"{SOURCE_FACTORY.__module__}:{SOURCE_FACTORY.__qualname__}"

        def save_config() -> None:
            payload = {
                "ml_config": cfg.to_dict(),
                "training_config": train_cfg.to_dict(),
                "generation": generation_params,
                "source_factory": factory_name,
            }
            (run_dir / "config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        save_config()
        logger.info("Запуск ML: %s", run_dir)
        logger.info("Available CPU count=%s; generation workers requested=%s", available_cpu_count(), cfg.n_workers)
        logger.info("MLConfig=%s", cfg)
        logger.info("TrainingConfig=%s", train_cfg)

        raw_frame = generate_dataset(
            config=cfg,
            n_trajectories=n_trajectories,
            n_points_per_trajectory=n_points_per_trajectory,
            n_beacons=n_beacons,
            r_min=r_min,
            r_max=r_max,
            relative_noise_sigma=relative_noise_sigma,
            n_harmonic=n_harmonic,
        )
        generation_params["n_workers_effective"] = int(raw_frame.attrs.get("n_workers_used", 1))
        save_config()

        source_cols = [c for c in raw_frame.columns if c.startswith("source_") and pd.api.types.is_numeric_dtype(raw_frame[c])]
        beacon_cols = ["beacon_id", "source_type", *source_cols]
        raw_frame[beacon_cols].drop_duplicates(subset=["beacon_id"]).to_csv(run_dir / "beacon_parameters.csv", index=False)
        logger.info("Сгенерировано точек: %d; траекторий=%d; маяков=%d", len(raw_frame), raw_frame.trajectory_id.nunique(), raw_frame.beacon_id.nunique())

        raw_frame, report = annotate_filtering(raw_frame, cfg)
        raw_frame.to_csv(run_dir / "raw_dataset.csv", index=False)
        report.to_csv(run_dir / "filter_report.csv", index=False)
        filtered_frame = raw_frame.loc[raw_frame["filter_retained"]].copy().reset_index(drop=True)
        filtered_frame.to_csv(run_dir / "filtered_dataset.csv", index=False)
        logger.info("После фильтрации осталось точек: %d (%.1f%%)", len(filtered_frame), 100 * len(filtered_frame) / len(raw_frame))

        extractor = FeatureExtractor(cfg)
        X, Y, weights, aligned, feature_names = extractor.transform(raw_frame, filtered_frame)
        (run_dir / "feature_columns.json").write_text(json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8")
        indices = split_dataset(aligned, cfg.split_mode, cfg.random_seed)
        split_summary(aligned, indices, cfg.split_mode).to_csv(run_dir / "split_summary.csv", index=False)
        aligned_export = aligned.copy()
        aligned_export["sample_weight"] = weights
        aligned_export["split"] = "unassigned"
        for split_name, split_idx in indices.items():
            aligned_export.loc[split_idx, "split"] = split_name
        aligned_export.to_csv(run_dir / "aligned_dataset.csv", index=False)
        for label, idx in indices.items():
            part = aligned.iloc[idx]
            logger.info("%s: n=%d, trajectories=%d, beacons=%d", label, len(idx), part.trajectory_id.nunique(), part.beacon_id.nunique())

        corrector = NearFieldCorrector(train_cfg, cfg, feature_names=feature_names)
        corrector.fit(
            X[indices["train"]], Y[indices["train"]],
            X[indices["val"]], Y[indices["val"]],
            sample_weight=weights[indices["train"]],
        )
        pd.DataFrame(corrector.history_).to_csv(run_dir / "training_history.csv", index=False)
        corrector.save(run_dir)
        best_display = None if corrector.best_iteration_ is None else int(corrector.best_iteration_) + 1
        logger.info("Лучшая эпоха/итерация (нумерация с 1): %s", best_display)

        test_idx = indices["test"]
        test_rows = aligned.iloc[test_idx].reset_index(drop=True)
        true = test_rows[TRUE_COLUMNS].to_numpy(dtype=float)
        dipole = test_rows[DIP_COLUMNS].to_numpy(dtype=float)
        delta_hat = corrector.predict_delta(X[test_idx], r_dip=test_rows["r_dip"].to_numpy(dtype=float))
        corrected = dipole + delta_hat
        predictions = test_rows[["trajectory_id", "beacon_id", "sample_idx", "r_true", "r_dip"]].copy()
        predictions[["true_x", "true_y", "true_z"]] = true
        predictions[["dip_x", "dip_y", "dip_z"]] = dipole
        predictions[["pred_delta_x", "pred_delta_y", "pred_delta_z"]] = delta_hat
        predictions[["corrected_x", "corrected_y", "corrected_z"]] = corrected
        predictions["baseline_error_m"] = np.linalg.norm(true - dipole, axis=1)
        predictions["corrected_error_m"] = np.linalg.norm(true - corrected, axis=1)
        predictions.to_csv(run_dir / "test_predictions.csv", index=False)

        metrics = evaluate_predictions(true, dipole, corrected)
        metrics.to_csv(run_dir / "test_metrics.csv", index=False)
        bins = distance_bin_metrics(true, dipole, corrected, n_bins=6, strategy="quantile")
        bins.to_csv(run_dir / "distance_bins.csv", index=False)
        logger.info("Test metrics:\n%s", metrics.to_string(index=False))
        plots = visualize_run(run_dir)
        logger.info("Сохранено графиков: %d", len(plots))
        (run_dir / "run_summary.json").write_text(json.dumps({
            "run_dir": str(run_dir),
            "n_raw": int(len(raw_frame)),
            "n_filtered": int(len(filtered_frame)),
            "n_train": int(len(indices["train"])),
            "n_val": int(len(indices["val"])),
            "n_test": int(len(indices["test"])),
            "best_iteration": corrector.best_iteration_,
            "metrics": metrics.to_dict(orient="records"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return run_dir
    finally:
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()
