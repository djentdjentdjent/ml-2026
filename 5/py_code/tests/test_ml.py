from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mag_nav.ml.config import MLConfig, TrainingConfig
from mag_nav.ml.dataset import FeatureExtractor
from mag_nav.ml.metrics import distance_bin_metrics, evaluate_predictions
from mag_nav.ml.models import NearFieldCorrector
from mag_nav.ml.preprocessing import filter_dataset
from mag_nav.ml.splitting import split_dataset


def _synthetic_frame(n: int = 8) -> pd.DataFrame:
    rows = []
    for i in range(n):
        tx, ty, tz = 0.20 + i * 0.03, 0.25 + i * 0.005, 0.3 + i * 0.01
        dx, dy, dz = tx - 0.01, ty - 0.02, tz - 0.015
        r_t = float(np.linalg.norm([tx, ty, tz]))
        r_d = float(np.linalg.norm([dx, dy, dz]))
        rows.append({
            "trajectory_id": f"traj-{i // 2:02d}",
            "beacon_id": f"beacon-{i // 4:02d}",
            "sample_idx": i % 2,
            "source_type": "Synthetic",
            "solver_valid": True,
            "true_x": tx, "true_y": ty, "true_z": tz,
            "dip_x": dx, "dip_y": dy, "dip_z": dz,
            "dip_r": r_d, "dip_inv_r": 1 / (r_d + 1e-6), "dip_inv_r2": 1 / (r_d**2 + 1e-6),
            "r_true": r_t, "r_dip": r_d,
            "delta_x": tx-dx, "delta_y": ty-dy, "delta_z": tz-dz,
            "S_aa": 1e-10 + i * 1e-12, "S_bb": 2e-10 + i * 1e-12, "S_ab": 1e-11,
            "log_S_aa": np.log(1e-10 + i * 1e-12), "log_S_bb": np.log(2e-10 + i * 1e-12),
            "a_x": 1e-5 + i * 1e-7, "b_x": 2e-5,
            "a_y": 3e-5, "b_y": 4e-5,
            "a_z": 5e-5, "b_z": 6e-5,
            "source_dipole_constant": 1e-7,
            "source_radius": 0.05 + (i // 4) * 0.005,
        })
    return pd.DataFrame(rows)


def test_filter_dataset_detects_octant_and_residual_outliers():
    frame = _synthetic_frame(4)
    frame.loc[0, "dip_y"] = -0.2
    frame.loc[0, "delta_y"] = frame.loc[0, "true_y"] - frame.loc[0, "dip_y"]
    frame.loc[1, "solver_valid"] = False
    frame.loc[1, ["dip_x", "dip_y", "dip_z", "dip_r", "r_dip"]] = np.nan
    filtered, report = filter_dataset(frame, MLConfig(octant_outlier_tau=0.8))
    counts = report.set_index("category")["count"]
    assert counts["octant_mismatch"] >= 1
    assert counts["solver_invalid"] >= 1
    assert len(filtered) < len(frame)
    assert "filter_octant_mismatch" in filtered.columns
    assert np.isfinite(filtered[["dip_x", "dip_y", "dip_z"]].to_numpy()).all()


def test_feature_window_is_current_to_past_and_repeat_first():
    frame = _synthetic_frame(8)
    # Все кадры одной траектории должны получить независимые хронологические окна.
    frame["trajectory_id"] = "traj-00"
    frame["sample_idx"] = np.arange(len(frame))
    config = MLConfig(window_size=3, include_beacon_params=False, octant_outlier_tau=10.0)
    filtered, _ = filter_dataset(frame, config)
    X, Y, weights, aligned, feature_names = FeatureExtractor(config).transform(frame, filtered)
    assert X.shape == (len(frame), 3 * 11)
    assert Y.shape == (len(frame), 3)
    assert weights.shape == (len(frame),)
    assert feature_names[0] == "t0_dip_x"
    # dip_x = true_x - 0.01: offset 0 = current, offset 1 = previous.
    assert X[0, 0] == pytest.approx(frame.iloc[0]["dip_x"])
    assert X[0, 11] == pytest.approx(frame.iloc[0]["dip_x"])
    assert X[0, 22] == pytest.approx(frame.iloc[0]["dip_x"])
    assert X[1, 0] == pytest.approx(frame.iloc[1]["dip_x"])
    assert X[1, 11] == pytest.approx(frame.iloc[0]["dip_x"])


def test_padding_drop_removes_first_window_targets():
    frame = _synthetic_frame(8)
    frame["trajectory_id"] = "traj-00"
    frame["sample_idx"] = np.arange(len(frame))
    config = MLConfig(window_size=3, padding="drop", include_beacon_params=False, octant_outlier_tau=10.0)
    filtered, _ = filter_dataset(frame, config)
    X, _, _, aligned, _ = FeatureExtractor(config).transform(frame, filtered)
    assert len(X) == len(frame) - 2
    assert aligned["sample_idx"].tolist() == list(range(2, len(frame)))


def test_group_splits_have_no_leakage():
    rows = _synthetic_frame(32)
    trajectory = split_dataset(rows, "by_trajectory", 42)
    group_sets = {
        name: set(rows.iloc[idx]["trajectory_id"])
        for name, idx in trajectory.items()
    }
    assert not (group_sets["train"] & group_sets["val"])
    assert not (group_sets["train"] & group_sets["test"])
    assert not (group_sets["val"] & group_sets["test"])

    beacon = split_dataset(rows, "by_beacon", 42)
    beacon_sets = {name: set(rows.iloc[idx]["beacon_id"]) for name, idx in beacon.items()}
    assert not (beacon_sets["train"] & beacon_sets["val"])
    assert not (beacon_sets["train"] & beacon_sets["test"])
    assert not (beacon_sets["val"] & beacon_sets["test"])


def test_metrics_report_baseline_and_corrected():
    true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    dipole = true + np.array([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]])
    corrected = true + np.array([[0.01, 0, 0], [0, 0.02, 0], [0, 0, 0.03]])
    result = evaluate_predictions(true, dipole, corrected).set_index("estimator")
    assert result.loc["Baseline (Dipole)", "mean_m"] == pytest.approx(0.2)
    assert result.loc["Corrected (Dipole + ML)", "mean_m"] == pytest.approx(0.02)
    bins = distance_bin_metrics(true, dipole, corrected, n_bins=3)
    assert len(bins) >= 1
    assert "corrected_p95_m" in bins


def test_mlp_fit_save_load_roundtrip(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(90, 6))
    coef = rng.normal(size=(6, 3)) * 0.05
    Y = X @ coef
    feature_config = MLConfig(include_beacon_params=False)
    train_config = TrainingConfig(model_type="mlp", epochs=4, patience=3, batch_size=16, verbose=False)
    model = NearFieldCorrector(train_config, feature_config, feature_names=[f"f{i}" for i in range(6)])
    model.fit(X[:60], Y[:60], X[60:75], Y[60:75], sample_weight=np.ones(60))
    pred_before = model.predict_delta(X[75:])
    model.save(tmp_path)
    loaded = NearFieldCorrector.load(tmp_path)
    pred_after = loaded.predict_delta(X[75:])
    assert np.allclose(pred_before, pred_after, atol=1e-7)
    assert (tmp_path / "training_history.csv").exists()
