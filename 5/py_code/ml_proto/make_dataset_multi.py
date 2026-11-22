#!/usr/bin/env python3
"""Генерация датасета для разных типов маяков.

Генерирует данные для нескольких конфигураций соленоида,
чтобы модель научилась работать с разными маяками.
"""

import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "py_code"))
sys.path.insert(0, str(Path(__file__).parent))

from mag_nav import Solenoid, GaussianNoise, Scenario, SensorConfig, random_first_octant
from make_dataset import features_from_ab, FEATURE_NAMES

# Разные конфигурации маяков
BEACON_CONFIGS = [
    # (n_turns, current, radius, length, name)
    (200, 0.5, 0.04, 0.08, "small"),      # Маленький соленоид
    (500, 1.0, 0.06, 0.12, "medium"),     # Средний соленоид
    (1000, 2.0, 0.08, 0.16, "large"),     # Большой соленоид
    (100, 0.3, 0.03, 0.05, "tiny"),       # Очень маленький
]


def generate_for_beacon(
    n_turns, current, radius, length, n_points, sigma, seed, r_min=0.15, r_max=1.5
):
    """Генерировать данные для одного типа маяка."""
    source = Solenoid(
        origin=np.zeros(3),
        n_turns=n_turns,
        current=current,
        radius=radius,
        length=length,
        n_quad=96,
    )
    b0 = source.dipole_constant
    
    rng = np.random.default_rng(seed)
    pts = random_first_octant(n_points, r_min, r_max, rng=rng)
    
    sensor = SensorConfig(
        samples_per_period=64, n_periods=4,
        noise=GaussianNoise(sigma=sigma),
        demod_method="lstsq", with_dc=True,
    )
    
    res = Scenario(
        source=source, trajectory=pts, sensor=sensor,
        inject_at="samples", seed=int(rng.integers(2**31)),
    ).run()
    
    X, p_dip, valid = features_from_ab(res.a_meas, res.b_meas, b0)
    
    # Применяем маску valid ко всем массивам
    p_true = res.true_positions[valid]
    p_dip = p_dip[valid]  # ← ЭТУ СТРОКУ НУЖНО ДОБАВИТЬ!
    r_dip = np.linalg.norm(p_dip, axis=1)
    
    # Добавляем параметры маяка как дополнительные признаки
    beacon_params = np.array([
        np.log10(n_turns),
        np.log10(current),
        np.log10(radius),
        np.log10(length),
        np.log10(b0),
    ])
    
    # Повторяем параметры маяка для каждой точки
    beacon_params_expanded = np.tile(beacon_params, (X.shape[0], 1))
    
    # Объединяем признаки
    X_extended = np.hstack([X, beacon_params_expanded])
    
    y = (p_true - p_dip) / r_dip[:, None]
    
    return X_extended, y, beacon_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Выходной файл .npz")
    ap.add_argument("--n-per-beacon", type=int, default=15000, help="Точек на каждый маяк")
    ap.add_argument("--sigma", type=float, default=1e-9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    
    all_X = []
    all_y = []
    all_beacon_params = []
    
    for i, (n_turns, current, radius, length, name) in enumerate(BEACON_CONFIGS):
        print(f"Генерация для маяка '{name}' (n={n_turns}, I={current}A, r={radius}m, l={length}m)...")
        X, y, beacon_params = generate_for_beacon(
            n_turns, current, radius, length,
            args.n_per_beacon, args.sigma, args.seed + i * 1000
        )
        all_X.append(X)
        all_y.append(y)
        all_beacon_params.append(beacon_params)
        print(f"  ✓ {len(X)} точек")
    
    X = np.vstack(all_X)
    y = np.vstack(all_y)
    
    # Сохраняем
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y)
    
    print(f"\n Сохранено: {out}")
    print(f"  Всего точек: {len(X)}")
    print(f"  Признаков: {X.shape[1]} (18 базовых + 5 параметров маяка)")
    print(f"  Типов маяков: {len(BEACON_CONFIGS)}")


if __name__ == "__main__":
    main()