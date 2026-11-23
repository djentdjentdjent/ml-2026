#!/usr/bin/env python3
"""Обучение CatBoost-поправки к дипольному решению и сравнение с базовой линией.

Train и test — независимые датасеты от ``make_dataset.py`` (разные seed),
валидация для early stopping выделяется из train.

Запуск:
    python train_catboost.py --train data/train.npz --test data/test.npz --out results/run1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostRegressor

R_BINS = np.array([0.15, 0.25, 0.4, 0.6, 0.9, 1.5])


def load(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2))) if x.size else float("nan")


def binned_rms(r: np.ndarray, err: np.ndarray) -> list[float]:
    idx = np.digitize(r, R_BINS) - 1
    return [rms(err[idx == i]) for i in range(len(R_BINS) - 1)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True, help="папка для модели, отчёта и графиков")
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tr, te = load(args.train), load(args.test)
    names = [str(s) for s in tr["feature_names"]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(tr["X"]))
    n_val = int(len(perm) * args.val_frac)
    i_val, i_fit = perm[:n_val], perm[n_val:]

    model = CatBoostRegressor(
        loss_function="MultiRMSE",  # одна модель на все 3 компоненты поправки
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.lr,
        random_seed=args.seed,
        od_type="Iter",
        od_wait=200,
        verbose=500,
        allow_writing_files=False,
    )
    model.fit(
        tr["X"][i_fit], tr["y"][i_fit],
        eval_set=(tr["X"][i_val], tr["y"][i_val]),
        use_best_model=True,
    )
    model.save_model(str(out / "model.cbm"))

    # ---- оценка на независимом test ------------------------------------- #
    r_dip = np.linalg.norm(te["p_dip"], axis=1)
    p_ml = te["p_dip"] + r_dip[:, None] * model.predict(te["X"])
    # Честная базовая линия без ML: отражение в 1-й октант. При шуме знак a·b
    # у плоскости y=0 иногда переворачивается, и y_dip < 0 — это выброс,
    # который исправляется тривиально, без обучения.
    p_oct = np.abs(te["p_dip"])
    err_dip = np.linalg.norm(te["p_dip"] - te["p_true"], axis=1)
    err_oct = np.linalg.norm(p_oct - te["p_true"], axis=1)
    err_ml = np.linalg.norm(p_ml - te["p_true"], axis=1)
    r = te["r_true"]

    report = {
        "train_size": int(len(i_fit)),
        "val_size": int(n_val),
        "test_size": int(len(r)),
        "sigma_T": float(te["sigma"]),
        "best_iteration": int(model.get_best_iteration()),
        "rms_dipole_m": rms(err_dip),
        "rms_octant_m": rms(err_oct),
        "rms_catboost_m": rms(err_ml),
        "rms_rel_dipole": rms(err_dip / r),
        "rms_rel_octant": rms(err_oct / r),
        "rms_rel_catboost": rms(err_ml / r),
        "median_dipole_m": float(np.median(err_dip)),
        "median_octant_m": float(np.median(err_oct)),
        "median_catboost_m": float(np.median(err_ml)),
        "p95_dipole_m": float(np.percentile(err_dip, 95)),
        "p95_octant_m": float(np.percentile(err_oct, 95)),
        "p95_catboost_m": float(np.percentile(err_ml, 95)),
        "r_bins_m": R_BINS.tolist(),
        "binned_rms_dipole_m": binned_rms(r, err_dip),
        "binned_rms_octant_m": binned_rms(r, err_oct),
        "binned_rms_catboost_m": binned_rms(r, err_ml),
        "feature_importance": dict(zip(names, map(float, model.get_feature_importance()))),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nTest: {len(r)} точек, sigma = {report['sigma_T']:.1e} Тл")
    print(f"  {'':10s} {'диполь':>10s} {'+октант':>10s} {'CatBoost':>10s}")
    for key, label in (("rms", "RMS, м"), ("median", "медиана, м"), ("p95", "p95, м")):
        print(f"  {label:10s} {report[key + '_dipole_m']:10.3e} {report[key + '_octant_m']:10.3e} "
              f"{report[key + '_catboost_m']:10.3e}")
    print(f"  {'RMS Δp/r':10s} {report['rms_rel_dipole']:10.3e} {report['rms_rel_octant']:10.3e} "
          f"{report['rms_rel_catboost']:10.3e}")
    print("  RMS |Δp| по дальности, м (диполь / +октант / CatBoost):")
    for lo, hi, d, o, m in zip(R_BINS[:-1], R_BINS[1:], report["binned_rms_dipole_m"],
                               report["binned_rms_octant_m"], report["binned_rms_catboost_m"]):
        print(f"    r ∈ [{lo:.2f}, {hi:.2f}):  {d:.3e}  {o:.3e}  {m:.3e}   (CatBoost лучше октанта в {o / m:.1f} раз)")

    # ---- графики --------------------------------------------------------- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    centers = np.sqrt(R_BINS[:-1] * R_BINS[1:])
    ax1.semilogy(centers, report["binned_rms_dipole_m"], "o-", label="дипольное решение")
    ax1.semilogy(centers, report["binned_rms_octant_m"], "^--", label="диполь + отражение в октант")
    ax1.semilogy(centers, report["binned_rms_catboost_m"], "s-", label="диполь + CatBoost")
    ax1.set_xlabel("дальность r, м")
    ax1.set_ylabel("RMS |Δp|, м")
    ax1.set_title(f"Погрешность на test (σ = {report['sigma_T']:.0e} Тл)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    imp = model.get_feature_importance()
    order = np.argsort(imp)
    ax2.barh(np.array(names)[order], imp[order])
    ax2.set_title("Важность признаков")
    ax2.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "errors_vs_range.png", dpi=130)
    print(f"\nСохранено: {out / 'model.cbm'}, {out / 'report.json'}, {out / 'errors_vs_range.png'}")


if __name__ == "__main__":
    main()
