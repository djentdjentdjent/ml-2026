#!/usr/bin/env python3
"""Генерация датасета для ML-поправки к дипольному решению.

Идея
----
Аналитический решатель ``solve_position`` выведен для точечного диполя.
Для соленоида конечных размеров (и при шуме) он даёт смещённую оценку
``p_dip``. Модель учится предсказывать поправку

    target = (p_true - p_dip) / r_dip            (безразмерная, по 3 осям)

только по тому, что реально доступно приёмнику: измеренным ``a``, ``b``
и самой дипольной оценке. Итоговая оценка: ``p = p_dip + r_dip * target``.

Каждая строка датасета — одна точка наблюдения:
  * признаки (см. ``FEATURE_NAMES``) — функции от измерений;
  * ``target`` (3) — нормированная поправка;
  * служебное: ``p_true``, ``p_dip``, ``r_true`` — для оценки качества.

Запуск:
    python make_dataset.py --mag-nav ../py_code --out data/sigma1nT.npz --sigma 1e-9
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FEATURE_NAMES: list[str] = [
    # инварианты, нормированные на S = |a|^2 + |b|^2 (не зависят от дальности)
    "A_S",
    "B_S",
    "C_S",
    "D_S",
    # амплитуда сигнала — несёт дальность и отношение сигнал/шум
    "log10_sqrtS",
    # компоненты a, b, нормированные на sqrt(S)
    "ax_n",
    "ay_n",
    "az_n",
    "bx_n",
    "by_n",
    "bz_n",
    # дипольная оценка
    "log10_r_dip",
    "u_dip",
    "v_dip",
    "w_dip",
    "x_dip",
    "y_dip",
    "z_dip",
]


def features_from_ab(
    a: NDArray[np.float64], b: NDArray[np.float64], b0: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Признаки по измеренным a/b. Та же функция используется на инференсе.

    Returns
    -------
    (X, p_dip, valid) : ``(n, n_features)``, ``(n, 3)``, ``(n,)``
    """
    from mag_nav import solve_position

    sol = solve_position(a, b, b0)
    inv = sol.invariants
    s = inv.asq + inv.bsq
    sqrt_s = np.sqrt(s)
    r_dip = sol.r
    X = np.column_stack(
        [
            inv.asq / s,
            inv.bsq / s,
            inv.ab / s,
            inv.denominator / s,
            np.log10(sqrt_s),
            a / sqrt_s[:, None],
            b / sqrt_s[:, None],
            np.log10(r_dip),
            sol.uvw,
            sol.points,
        ]
    )
    valid = sol.valid & np.isfinite(X).all(axis=1)
    return X, sol.points, valid


def make_source(kind: str):
    from mag_nav import Solenoid

    # Тот же маяк, что в scripts/simple_nav_solen_mc.py
    sol = Solenoid(
        origin=np.zeros(3),
        n_turns=200,
        current=0.5,
        radius=0.04,
        length=0.08,
        n_quad=96,
    )
    return sol if kind == "solenoid" else sol.as_dipole()


def generate(
    n: int,
    *,
    source_kind: str = "solenoid",
    sigma: float = 1e-9,
    r_min: float = 0.15,
    r_max: float = 1.5,
    seed: int = 0,
    chunk: int = 5000,
) -> dict[str, NDArray]:
    """Сгенерировать ``n`` точек (до фильтрации недействительных)."""
    from mag_nav import GaussianNoise, Scenario, SensorConfig, random_first_octant

    source = make_source(source_kind)
    b0 = source.dipole_constant
    rng = np.random.default_rng(seed)
    pts = random_first_octant(n, r_min, r_max, rng=rng)

    if sigma > 0:
        sensor = SensorConfig(
            samples_per_period=64,
            n_periods=4,
            noise=GaussianNoise(sigma=sigma),
            demod_method="lstsq",
            with_dc=True,
        )
        mode = "samples"
    else:
        sensor, mode = None, "none"

    parts: dict[str, list[NDArray]] = {k: [] for k in ("X", "p_true", "p_dip")}
    for i, start in enumerate(range(0, n, chunk)):
        block = pts[start : start + chunk]
        res = Scenario(
            source=source,
            trajectory=block,
            sensor=sensor,
            inject_at=mode,
            seed=int(rng.integers(2**31)),
        ).run()
        X, p_dip, valid = features_from_ab(res.a_meas, res.b_meas, b0)
        parts["X"].append(X[valid])
        parts["p_true"].append(res.true_positions[valid])
        parts["p_dip"].append(p_dip[valid])

    X = np.concatenate(parts["X"])
    p_true = np.concatenate(parts["p_true"])
    p_dip = np.concatenate(parts["p_dip"])
    r_dip = np.linalg.norm(p_dip, axis=1)
    return {
        "X": X,
        "y": (p_true - p_dip) / r_dip[:, None],
        "p_true": p_true,
        "p_dip": p_dip,
        "r_true": np.linalg.norm(p_true, axis=1),
        "feature_names": np.array(FEATURE_NAMES),
        "b0": np.array(b0),
        "sigma": np.array(sigma),
        "n_requested": np.array(n),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True, help="файл .npz")
    ap.add_argument("--n", type=int, default=60000, help="число точек")
    ap.add_argument(
        "--sigma", type=float, default=1e-9, help="СКО шума отсчётов, Тл (0 — без шума)"
    )
    ap.add_argument("--source", choices=["solenoid", "dipole"], default="solenoid")
    ap.add_argument("--r-min", type=float, default=0.15)
    ap.add_argument("--r-max", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.perf_counter()
    data = generate(
        args.n,
        source_kind=args.source,
        sigma=args.sigma,
        r_min=args.r_min,
        r_max=args.r_max,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    kept = len(data["X"])
    print(
        f"{out}: {kept}/{args.n} точек ({kept / args.n:.1%} действительных), "
        f"{len(FEATURE_NAMES)} признаков, {time.perf_counter() - t0:.1f} с"
    )


if __name__ == "__main__":
    main()
