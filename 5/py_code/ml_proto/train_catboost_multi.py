#!/usr/bin/env python3
"""Обучение CatBoost для разных типов маяков."""

import argparse
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor

def load(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    
    tr, te = load(args.train), load(args.test)
    
    # Проверяем размеры
    print(f"Train: X.shape = {tr['X'].shape}, y.shape = {tr['y'].shape}")
    print(f"Test: X.shape = {te['X'].shape}, y.shape = {te['y'].shape}")
    
    # Обрезаем до минимального размера, если есть несоответствие
    n_train = min(tr['X'].shape[0], tr['y'].shape[0])
    tr_X = tr['X'][:n_train]
    tr_y = tr['y'][:n_train]
    
    n_test = min(te['X'].shape[0], te['y'].shape[0])
    te_X = te['X'][:n_test]
    te_y = te['y'][:n_test]
    
    print(f"После обрезки: train = {n_train}, test = {n_test}")
    
    # Разделяем данные на train и validation
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_train)
    n_val = int(n_train * 0.15)
    i_val, i_fit = perm[:n_val], perm[n_val:]
    
    # Обучаем модель
    model = CatBoostRegressor(
        loss_function="MultiRMSE",
        iterations=3000,
        depth=8,
        learning_rate=0.05,
        random_seed=42,
        od_type="Iter",
        od_wait=200,
        verbose=500,
    )
    
    print("Обучение модели...")
    model.fit(
        tr_X[i_fit], tr_y[i_fit],
        eval_set=(tr_X[i_val], tr_y[i_val]),
        use_best_model=True,
    )
    
    # Сохраняем модель
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "model_multi.cbm"))
    
    # Оцениваем на тесте
    if te.get("p_dip") is not None and te.get("p_true") is not None:
        p_dip_test = te["p_dip"][:n_test]
        p_true_test = te["p_true"][:n_test]
        r_dip = np.linalg.norm(p_dip_test, axis=1)
        delta = model.predict(te_X)
        p_ml = p_dip_test + r_dip[:, None] * delta
        
        err_dip = np.linalg.norm(p_dip_test - p_true_test, axis=1)
        err_ml = np.linalg.norm(p_ml - p_true_test, axis=1)
        
        rms_dip = np.sqrt(np.mean(err_dip**2))
        rms_ml = np.sqrt(np.mean(err_ml**2))
        
        print(f"\nРезультаты на тесте:")
        print(f"  RMS (Диполь): {rms_dip:.4e} м")
        print(f"  RMS (CatBoost): {rms_ml:.4e} м")
        print(f"  Улучшение: {rms_dip/rms_ml:.2f}x")
    
    print(f"\n✓ Модель сохранена: {out / 'model_multi.cbm'}")
    print(f"  Лучшая итерация: {model.get_best_iteration()}")
    print(f"  Количество признаков: {tr_X.shape[1]}")


if __name__ == "__main__":
    main()