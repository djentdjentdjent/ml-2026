"""Графики из сохранённых CSV — обучение повторно не запускается."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_run(run_dir: str | Path) -> list[Path]:
    """Построить диагностические PNG для каталога запуска."""
    root = Path(run_dir)
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    history_path = root / "training_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        if not history.empty:
            xcol = "epoch" if "epoch" in history else history.columns[0]
            fig, ax = plt.subplots(figsize=(9, 5))
            for column, label in [("train_loss", "Train"), ("val_loss", "Validation")]:
                if column in history and np.isfinite(history[column].to_numpy(dtype=float)).any():
                    ax.plot(history[xcol], history[column], label=label, linewidth=1.8)
            ax.set_title("Learning curve: train vs validation")
            ax.set_xlabel("Epoch / boosting iteration")
            ax.set_ylabel("Loss / MultiRMSE")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            path = plots / "training_loss.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)

            metric = "val_rmse" if "val_rmse" in history else "val_loss"
            if metric in history:
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.plot(history[xcol], history[metric], label=metric, linewidth=1.8)
                if "best_iteration" in history and len(history):
                    best = int(history["best_iteration"].iloc[-1])
                    ax.axvline(best, linestyle="--", alpha=0.7, label=f"best={best}")
                ax.set_title("Validation metric / early stopping")
                ax.set_xlabel("Epoch / boosting iteration")
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.25)
                ax.legend()
                fig.tight_layout()
                path = plots / "validation_metrics.png"
                fig.savefig(path, dpi=150)
                plt.close(fig)
                written.append(path)

    pred_path = root / "test_predictions.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        if not pred.empty:
            if {"baseline_error_m", "corrected_error_m"}.issubset(pred.columns):
                baseline = pred["baseline_error_m"].to_numpy(dtype=float)
                corrected = pred["corrected_error_m"].to_numpy(dtype=float)
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.hist(baseline[np.isfinite(baseline)], bins=50, alpha=0.60, label="Baseline")
                ax.hist(corrected[np.isfinite(corrected)], bins=50, alpha=0.60, label="Corrected")
                ax.set_title("Test position-error distribution")
                ax.set_xlabel("Euclidean position error, m")
                ax.set_ylabel("Count")
                ax.grid(True, alpha=0.2)
                ax.legend()
                fig.tight_layout()
                path = plots / "error_distribution.png"
                fig.savefig(path, dpi=150)
                plt.close(fig)
                written.append(path)

            if {"r_true", "baseline_error_m", "corrected_error_m"}.issubset(pred.columns):
                rr = pred["r_true"].to_numpy(dtype=float)
                eb = pred["baseline_error_m"].to_numpy(dtype=float)
                ec = pred["corrected_error_m"].to_numpy(dtype=float)
                good = np.isfinite(rr) & np.isfinite(eb) & np.isfinite(ec)
                order = np.argsort(rr[good])
                rr, eb, ec = rr[good][order], eb[good][order], ec[good][order]
                fig, ax = plt.subplots(figsize=(10, 5.5))
                ax.scatter(rr, eb, s=12, alpha=0.35, label="Baseline")
                ax.scatter(rr, ec, s=12, alpha=0.35, label="Corrected")
                # Running median is a diagnostic guide, not a fitted model.
                if len(rr) >= 20:
                    block = max(10, len(rr) // 40)
                    for values, label in [(eb, "Baseline median"), (ec, "Corrected median")]:
                        x_med, y_med = [], []
                        for start in range(0, len(rr), block):
                            stop = min(start + block, len(rr))
                            x_med.append(float(np.median(rr[start:stop])))
                            y_med.append(float(np.median(values[start:stop])))
                        ax.plot(x_med, y_med, linewidth=2.0, label=label)
                ax.set_title("Position error vs true distance")
                ax.set_xlabel("True distance from beacon, m")
                ax.set_ylabel("Euclidean position error, m")
                ax.grid(True, alpha=0.2)
                ax.legend()
                fig.tight_layout()
                path = plots / "error_vs_distance.png"
                fig.savefig(path, dpi=150)
                plt.close(fig)
                written.append(path)

    bins_path = root / "distance_bins.csv"
    if bins_path.exists():
        bins = pd.read_csv(bins_path)
        if not bins.empty:
            x = 0.5 * (bins["r_left_m"].to_numpy(dtype=float) + bins["r_right_m"].to_numpy(dtype=float))
            fig, ax = plt.subplots(figsize=(9, 5))
            for prefix in ("baseline", "corrected"):
                for metric, style in [("mean", "-o"), ("p95", "--s")]:
                    col = f"{prefix}_{metric}_m"
                    if col in bins:
                        ax.plot(x, bins[col], style, label=f"{prefix.title()} {metric}")
            ax.set_title("Distance-bin position error")
            ax.set_xlabel("True distance, m")
            ax.set_ylabel("Error, m")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            path = plots / "distance_bins.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)
    return written
