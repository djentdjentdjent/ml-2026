#!/usr/bin/env python3
"""Демонстрация Монте-Карло симуляции навигации по соленоиду.

Аналог simple_nav_solen.py, но с ансамблем реализаций шума.
Траектория: прямолинейный подлёт к маяку по диагонали первого октанта.

Метрики:
  L2   — корень из среднего квадрата отклонения (RMSE), м.
  L∞   — средний по реализациям максимум |отклонения| по любой координате, м.

Запуск:
    python simple_nav_solen_mc.py
"""

import matplotlib.pyplot as plt
import numpy as np

from mag_nav import (
    GaussianNoise,
    SensorConfig,
    Solenoid,
    line,
)
from mag_nav.sim.monte_carlo_sim import MonteCarloSimulator
from mag_nav.sim.visualize import (
    plot_error_histogram,
    plot_monte_carlo,
    plot_monte_carlo_error_vs_range,
)


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Источник: соленоид конечных размеров
    # ------------------------------------------------------------------ #
    source = Solenoid(
        origin=np.array([0.0, 0.0, 0.0]),
        n_turns=200,
        current=0.5,
        radius=0.04,  # 4 см
        length=0.30,  # 8 см
        n_quad=96,
    )
    print(f"Источник: {source}")
    print(f"  B0 = {source.dipole_constant:.6e} Тл·м³")
    print(f"  m  = {source.magnetic_moment:.4f} А·м²")

    # ------------------------------------------------------------------ #
    # 2. Траектория: подлёт к маяку по диагонали
    #    от (1.0, 0.8, 0.6) до (0.15, 0.12, 0.10) — 500 точек
    # ------------------------------------------------------------------ #
    start = np.array([1.0, 0.8, 0.6])
    end = np.array([0.15, 0.12, 0.10])
    n_pts = 500
    trajectory = line(start, end, n_pts)

    r_start = np.linalg.norm(start)
    r_end = np.linalg.norm(end)
    print(f"\nТраектория: {n_pts} точек, r: {r_start:.3f} → {r_end:.3f} м")

    # ------------------------------------------------------------------ #
    # 3. Шум и тракт измерения
    # ------------------------------------------------------------------ #
    sigma_b = 1e-9  # 1 нТл — типичный шум магнитометра
    noise = GaussianNoise(sigma=sigma_b)

    sensor = SensorConfig(
        samples_per_period=64,
        n_periods=4,
        noise=noise,
        demod_method="lstsq",
        with_dc=True,
    )
    print(f"Шум: σ = {sigma_b:.1e} Тл, отсчётов = {sensor.n_samples}")

    # ------------------------------------------------------------------ #
    # 4. Симулятор Монте-Карло
    # ------------------------------------------------------------------ #
    simulator = MonteCarloSimulator(
        source=source,
        trajectory=trajectory,
        sensor=sensor,
        noise=noise,
        inject_at="samples",
        n_harmonic=64,
    )

    n_trials = 200
    print(f"\nЗапуск Монте-Карло: {n_trials} реализаций...")
    mc = simulator.run(n_trials=n_trials, base_seed=42, keep_results=False)

    # ------------------------------------------------------------------ #
    # 5. Результаты
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print(mc.summary())
    print("=" * 60)
    print(f"  L2 (RMSE)        = {mc.l2:.6e} м")
    print(f"  L∞ (mean max)    = {mc.l_inf:.6e} м")
    print(f"  L2 относительная = {mc.l2_relative:.4e}  (доля от средней дальности)")
    print(f"  Доля валидных    = {100 * mc.validity_ratio:.1f} %")

    # Распределение по траектории
    print("\n  L2 по участкам траектории:")
    n_seg = 5
    seg_len = n_pts // n_seg
    for i in range(n_seg):
        sl = slice(i * seg_len, (i + 1) * seg_len)
        seg_l2 = mc.l2_per_point[sl]
        r_seg = np.mean(np.linalg.norm(trajectory[sl], axis=-1))
        print(
            f"    [{i * seg_len:3d}:{(i + 1) * seg_len:3d}]  "
            f"r≈{r_seg:.3f} м,  L2={np.nanmean(seg_l2):.3e} м"
        )

    # ------------------------------------------------------------------ #
    # 6. Визуализация ансамбля (ничего не сохраняется на диск)
    # ------------------------------------------------------------------ #
    plot_monte_carlo(mc)
    plot_error_histogram(mc, bins=80)
    plot_monte_carlo_error_vs_range(
        mc,
        trajectory,
        origin=source.origin,
        relative=True,
    )

    print("\nГотово. Открываю графики...")
    plt.show()


if __name__ == "__main__":
    main()
