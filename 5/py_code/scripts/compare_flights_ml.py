#!/usr/bin/env python3
"""Сравнение навигации с ML-корректором и без него. Запуск из корня проекта."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mag_nav.ml.dataset import FeatureExtractor, source_descriptor
from mag_nav.ml.models import NearFieldCorrector
from mag_nav.ml.source_factory import make_source
from mag_nav.navigation.orientation import RotationMatrix
from mag_nav.noise.gaussian import GaussianNoise
from mag_nav.signal.ab_params import ab_invariants
from mag_nav.sim.scenario import Scenario
from mag_nav.sim.trajectory import line

# --------------------------- НАСТРОЙКИ --------------------------------------
MODEL_DIR = None  # Например: Path("runs/20261025_123456_by_trajectory_mlp"); None = последний run
BEACON_ID = 0
SOURCE_SEED = 42

# Размеры текущего Solenoid. None = параметры фабрики, как при обучении.
BEACON_RADIUS_M = 0.04
BEACON_LENGTH_M = 0.30
BEACON_TURNS = 200
BEACON_CURRENT_A = 0.5

NOISE_RELATIVE_SIGMA = None  # None = как при обучении; например 0.01 = 1%
N_POINTS = 120
SEED = 42
START = np.array([0.50, 0.50, 0.50])
END = np.array([0.045, 0.045, 0.120])
# -----------------------------------------------------------------------------


def find_model_dir():
    if MODEL_DIR is not None:
        path = Path(MODEL_DIR)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        return path
    runs = Path(__file__).resolve().parents[1] / "runs"
    models = [
        p.parent
        for p in runs.rglob("model_metadata.json")
        if (p.parent / "model.pt").exists() or (p.parent / "model.cbm").exists()
    ]
    if not models:
        raise FileNotFoundError("Не нашёл сохранённую модель. Укажи MODEL_DIR в начале скрипта.")
    return max(models, key=lambda p: (p / "model_metadata.json").stat().st_mtime)


def make_beacon():
    # Должны совпадать с seed и BEACON_ID из обучения. Тип источника задаётся
    # одной строкой в mag_nav/ml/source_factory.py.
    source = make_source(BEACON_ID, SOURCE_SEED)
    changes = {}
    if BEACON_RADIUS_M is not None:
        changes["radius"] = BEACON_RADIUS_M
    if BEACON_LENGTH_M is not None:
        changes["length"] = BEACON_LENGTH_M
    if BEACON_TURNS is not None:
        changes["n_turns"] = BEACON_TURNS
    if BEACON_CURRENT_A is not None:
        changes["current"] = BEACON_CURRENT_A
    return replace(source, **changes) if changes else source


def make_feature_frame(result, source, eps):
    """Сформировать те же признаки, что и при обучении."""
    p_true = result.true_positions
    p_dip = result.estimated_positions
    a, b = result.a_meas, result.b_meas
    inv = ab_invariants(a, b)
    r_true = np.linalg.norm(p_true, axis=1)
    r_dip = np.linalg.norm(p_dip, axis=1)
    data = {
        "trajectory_id": ["flight"] * len(p_true),
        "beacon_id": [f"beacon-{BEACON_ID:04d}"] * len(p_true),
        "sample_idx": np.arange(len(p_true)),
        "source_type": [type(source).__name__] * len(p_true),
        "solver_valid": result.valid,
        "solver_reason": result.solution.reasons,
        "solver_r": result.solution.r,
        "solver_uvw_norm": result.solution.uvw_norm,
        "true_x": p_true[:, 0],
        "true_y": p_true[:, 1],
        "true_z": p_true[:, 2],
        "dip_x": p_dip[:, 0],
        "dip_y": p_dip[:, 1],
        "dip_z": p_dip[:, 2],
        "dip_r": r_dip,
        "dip_inv_r": 1 / (r_dip + eps),
        "dip_inv_r2": 1 / (r_dip**2 + eps),
        "r_true": r_true,
        "r_dip": r_dip,
        "delta_x": p_true[:, 0] - p_dip[:, 0],
        "delta_y": p_true[:, 1] - p_dip[:, 1],
        "delta_z": p_true[:, 2] - p_dip[:, 2],
        "S_aa": inv.asq,
        "S_bb": inv.bsq,
        "S_ab": inv.ab,
        "log_S_aa": np.log(np.maximum(inv.asq, eps)),
        "log_S_bb": np.log(np.maximum(inv.bsq, eps)),
        "a_x": a[:, 0],
        "b_x": b[:, 0],
        "a_y": a[:, 1],
        "b_y": b[:, 1],
        "a_z": a[:, 2],
        "b_z": b[:, 2],
    }
    names, values = source_descriptor(source)
    for name, value in zip(names, values):
        data[name] = value
    return pd.DataFrame(data)


def plot_flight(title, true_path, estimate, limits):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*true_path.T, color="black", linewidth=2, label="Истинная траектория")
    ax.plot(*estimate.T, color="red", linewidth=1.5, label="Оценка навигации")
    ax.set(title=title, xlabel="x, м", ylabel="y, м", zlabel="z, м")
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_zlim(limits[2])
    ax.set_box_aspect((1, 1, 1))
    ax.legend()
    fig.tight_layout()


def main():
    run_dir = find_model_dir()
    model = NearFieldCorrector.load(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    generation = config.get("generation", {})
    noise_sigma = (
        generation.get("relative_noise_sigma", 0.003)
        if NOISE_RELATIVE_SIGMA is None
        else NOISE_RELATIVE_SIGMA
    )
    n_harmonic = generation.get("n_harmonic", 32)
    source = make_beacon()
    beacon_csv = run_dir / "beacon_parameters.csv"
    if beacon_csv.exists():
        saved = pd.read_csv(beacon_csv)
        if "source_type" in saved and not saved.empty:
            saved_type = saved.iloc[0]["source_type"]
            if saved_type != type(source).__name__:
                raise ValueError(
                    f"Модель обучена на {saved_type}, а SOURCE_FACTORY создал {type(source).__name__}. "
                    "Используй прежнюю фабрику или обучи модель заново."
                )
    trajectory = line(START, END, N_POINTS) + np.asarray(source.origin).reshape(3)
    rotation = RotationMatrix.from_euler(*np.random.default_rng(SEED).uniform(-np.pi, np.pi, 3))
    noise = GaussianNoise(sigma=noise_sigma, relative=True) if noise_sigma > 0 else None

    result = Scenario(
        source=source,
        trajectory=trajectory,
        noise=noise,
        inject_at="ab" if noise else "none",
        frame_alignment=rotation,
        seed=SEED,
        n_harmonic=n_harmonic,
    ).run()
    true_path = result.true_positions
    baseline = result.estimated_positions.copy()

    # На инференсе фильтруем только по валидности солвера и конечности признаков.
    frame = make_feature_frame(result, source, model.feature_config.eps)
    extractor = FeatureExtractor(model.feature_config)
    columns = extractor.frame_feature_columns
    eligible = frame.loc[frame.solver_valid & np.all(np.isfinite(frame[columns]), axis=1)]
    X, _, _, aligned, feature_names = extractor.transform(frame, eligible)
    if model.feature_names and feature_names != model.feature_names:
        raise ValueError("Признаки не совпали с обучением. Проверь модель и SOURCE_FACTORY.")

    delta = model.predict_delta(X, r_dip=aligned.r_dip.to_numpy())
    corrected = np.full_like(baseline, np.nan)
    indices = aligned.sample_idx.to_numpy(dtype=int)
    corrected[indices] = baseline[indices] + delta

    # Одинаковые масштабы осей в обоих окнах, чтобы сравнение было честным.
    all_paths = np.concatenate([true_path, baseline, corrected])
    all_paths = all_paths[np.all(np.isfinite(all_paths), axis=1)]
    low, high = all_paths.min(axis=0), all_paths.max(axis=0)
    center = (low + high) / 2
    radius = max(np.max(high - low) * 0.55, 1e-3)
    limits = [(c - radius, c + radius) for c in center]

    for title, estimate in [
        ("1. Без ML-корректора", baseline),
        ("2. С ML-корректором", corrected),
    ]:
        error = np.linalg.norm(true_path - estimate, axis=1)
        error = error[np.isfinite(error)]
        if len(error):
            print(
                f"{title}: mean={error.mean():.4f} м, "
                f"median={np.median(error):.4f} м, "
                f"P95={np.percentile(error, 95):.4f} м"
            )
        plot_flight(title, true_path, estimate, limits)

    print(f"Модель: {run_dir} | источник: {type(source).__name__} | шум: {noise_sigma:g}")
    plt.show()


if __name__ == "__main__":
    main()
