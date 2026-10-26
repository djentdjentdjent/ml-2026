"""Модели компенсации остатка позиции: CatBoost и PyTorch MLP."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import MLConfig, TrainingConfig, available_cpu_count

logger = logging.getLogger(__name__)


def _build_torch_network(input_size: int, hidden_layers: tuple[int, ...], activation: str):
    import torch.nn as nn

    layers: list[Any] = []
    previous = input_size
    activation_class = nn.ReLU if activation == "relu" else nn.GELU
    for width in hidden_layers:
        layers.extend([nn.Linear(previous, int(width)), activation_class()])
        previous = int(width)
    layers.append(nn.Linear(previous, 3))
    return nn.Sequential(*layers)


class NearFieldCorrector:
    """CatBoost MultiRMSE или MLP с единым интерфейсом.

    ``predict_delta`` всегда возвращает поправку в метрах в глобальной СК маяка.
    CPU-параллелизм регулируется ``TrainingConfig.cpu_threads``; значение 0 не
    ограничивает число потоков библиотеки. Для PyTorch поддерживается выбор
    ``torch_device='cpu'|'cuda'`` без изменения остального API.
    """

    def __init__(
        self,
        training_config: TrainingConfig | None = None,
        feature_config: MLConfig | None = None,
        *,
        feature_names: list[str] | None = None,
    ) -> None:
        self.training_config = training_config or TrainingConfig()
        self.feature_config = feature_config or MLConfig()
        self.feature_names = list(feature_names or [])
        self.model: Any = None
        self.scaler_x: Any = None
        self.scaler_y: Any = None
        self.history_: list[dict[str, float]] = []
        self.best_iteration_: int | None = None
        self.input_size_: int | None = None
        self.device_: Any = None

    def fit(
        self,
        X_train: ArrayLike,
        Y_train: ArrayLike,
        X_val: ArrayLike,
        Y_val: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
    ) -> "NearFieldCorrector":
        Xtr = np.asarray(X_train, dtype=np.float64)
        Ytr = np.asarray(Y_train, dtype=np.float64).reshape(-1, 3)
        Xv = np.asarray(X_val, dtype=np.float64)
        Yv = np.asarray(Y_val, dtype=np.float64).reshape(-1, 3)
        if Xtr.ndim != 2 or Xv.ndim != 2 or Xtr.shape[1] != Xv.shape[1]:
            raise ValueError("X_train/X_val должны быть 2D и иметь одинаковое число признаков")
        if len(Xtr) != len(Ytr) or len(Xv) != len(Yv):
            raise ValueError("Размер X и Y не совпадает")
        if not (np.all(np.isfinite(Xtr)) and np.all(np.isfinite(Ytr)) and np.all(np.isfinite(Xv)) and np.all(np.isfinite(Yv))):
            raise ValueError("В обучающие/валидационные массивы попали NaN/Inf")
        weights = np.ones(len(Xtr), dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if weights.shape[0] != len(Xtr) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("sample_weight должен быть конечным, неотрицательным и ненулевым")
        weights = weights / max(float(weights.mean()), 1e-12)
        self.input_size_ = int(Xtr.shape[1])

        if self.training_config.model_type == "catboost":
            self._fit_catboost(Xtr, Ytr, Xv, Yv, weights)
        else:
            self._fit_mlp(Xtr, Ytr, Xv, Yv, weights)
        return self

    def _fit_catboost(self, Xtr, Ytr, Xv, Yv, weights) -> None:
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Для CatBoost установите `pip install -e '.[ml]'`") from exc
        cfg = self.training_config
        params: dict[str, Any] = {
            "loss_function": "MultiRMSE",
            "eval_metric": "MultiRMSE",
            "iterations": cfg.catboost_iterations,
            "learning_rate": cfg.catboost_learning_rate,
            "depth": cfg.catboost_depth,
            "random_seed": cfg.random_seed,
            "allow_writing_files": False,
            "thread_count": cfg.cpu_threads if cfg.cpu_threads > 0 else -1,
            "task_type": cfg.catboost_task_type,
            "verbose": 100 if cfg.verbose else False,
        }
        if cfg.catboost_devices:
            params["devices"] = cfg.catboost_devices
        model = CatBoostRegressor(**params)
        model.fit(
            Xtr,
            Ytr,
            sample_weight=weights,
            eval_set=(Xv, Yv),
            early_stopping_rounds=cfg.catboost_early_stopping_rounds,
            use_best_model=True,
            verbose=100 if cfg.verbose else False,
        )
        self.model = model
        self.best_iteration_ = int(model.get_best_iteration())
        results = model.get_evals_result()
        learn = self._first_metric(results.get("learn", {}))
        validation = self._first_metric(results.get("validation", {}))
        self.history_ = []
        for index in range(max(len(learn), len(validation))):
            train_loss = float(learn[index]) if index < len(learn) else float("nan")
            val_loss = float(validation[index]) if index < len(validation) else float("nan")
            self.history_.append({
                "epoch": float(index + 1),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_rmse": train_loss,
                "val_rmse": val_loss,
                "best_iteration": float(self.best_iteration_ + 1),
            })
        logger.info(
            "CatBoost завершён: task_type=%s, thread_count=%s, best_iteration=%d",
            cfg.catboost_task_type,
            params["thread_count"],
            self.best_iteration_ + 1,
        )

    @staticmethod
    def _first_metric(results: dict[str, list[float]]) -> list[float]:
        if not results:
            return []
        return [float(value) for value in next(iter(results.values()))]

    def _fit_mlp(self, Xtr, Ytr, Xv, Yv, weights) -> None:
        try:
            import torch
            from sklearn.preprocessing import StandardScaler
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Для MLP установите `pip install -e '.[ml]'`") from exc
        cfg = self.training_config
        torch.manual_seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)
        # По умолчанию используем все CPU, доступные процессу. Нет лимита в
        # 4 потока; cpu_threads позволяет явно ограничить их при необходимости.
        torch.set_num_threads(cfg.cpu_threads if cfg.cpu_threads > 0 else available_cpu_count())
        if cfg.torch_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Запрошен torch_device='cuda', но CUDA недоступна в PyTorch")
            self.device_ = torch.device("cuda")
        else:
            self.device_ = torch.device("cpu")

        self.scaler_x = StandardScaler().fit(Xtr)
        self.scaler_y = StandardScaler().fit(Ytr)
        Xtr_scaled = self.scaler_x.transform(Xtr).astype(np.float32)
        Xv_scaled = self.scaler_x.transform(Xv).astype(np.float32)
        Ytr_scaled = self.scaler_y.transform(Ytr).astype(np.float32)
        Yv_scaled = self.scaler_y.transform(Yv).astype(np.float32)
        weights = np.asarray(weights, dtype=np.float32)
        weights /= max(float(weights.mean()), 1e-12)

        dataset = TensorDataset(
            torch.from_numpy(Xtr_scaled), torch.from_numpy(Ytr_scaled), torch.from_numpy(weights)
        )
        generator = torch.Generator().manual_seed(cfg.random_seed)
        loader = DataLoader(
            dataset,
            batch_size=min(cfg.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
            num_workers=cfg.dataloader_num_workers,
            pin_memory=self.device_.type == "cuda",
            persistent_workers=cfg.dataloader_num_workers > 0,
        )
        self.model = _build_torch_network(Xtr.shape[1], tuple(cfg.hidden_layers), cfg.activation).to(self.device_)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        xtr_tensor = torch.from_numpy(Xtr_scaled).to(self.device_)
        xv_tensor = torch.from_numpy(Xv_scaled).to(self.device_)
        best_state = None
        best_val = float("inf")
        wait = 0
        self.history_ = []
        logger.info(
            "PyTorch MLP: device=%s, torch_threads=%d, dataloader_workers=%d",
            self.device_, torch.get_num_threads(), cfg.dataloader_num_workers,
        )

        def rmse_in_meters(x_tensor, original_y, sample_weights=None) -> float:
            self.model.eval()
            with torch.no_grad():
                pred_scaled = self.model(x_tensor).detach().cpu().numpy()
            prediction = self.scaler_y.inverse_transform(pred_scaled)
            squared_error = np.sum((original_y - prediction) ** 2, axis=1)
            if sample_weights is None:
                return float(np.sqrt(np.mean(squared_error)))
            ww = np.asarray(sample_weights, dtype=np.float64)
            return float(np.sqrt(np.sum(ww * squared_error) / max(ww.sum(), 1e-12)))

        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            for batch_x, batch_y, batch_w in loader:
                batch_x = batch_x.to(self.device_, non_blocking=True)
                batch_y = batch_y.to(self.device_, non_blocking=True)
                batch_w = batch_w.to(self.device_, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.model(batch_x)
                per_example = torch.sum((prediction - batch_y) ** 2, dim=1)
                loss = torch.sum(batch_w * per_example) / torch.clamp(batch_w.sum(), min=1e-12)
                loss.backward()
                optimizer.step()

            train_rmse = rmse_in_meters(xtr_tensor, Ytr, weights)
            val_rmse = rmse_in_meters(xv_tensor, Yv)
            self.history_.append({
                "epoch": float(epoch),
                "train_loss": train_rmse,
                "val_loss": val_rmse,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "best_iteration": float(epoch if val_rmse < best_val else (self.best_iteration_ or epoch)),
            })
            if cfg.verbose and (epoch == 1 or epoch % 10 == 0):
                logger.info("MLP epoch %d/%d train_RMSE=%.6g val_RMSE=%.6g", epoch, cfg.epochs, train_rmse, val_rmse)
            if val_rmse < best_val - cfg.min_delta:
                best_val = val_rmse
                wait = 0
                self.best_iteration_ = epoch - 1
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= cfg.patience:
                    if cfg.verbose:
                        logger.info("Early stopping на эпохе %d; лучшая эпоха=%d", epoch, (self.best_iteration_ or 0) + 1)
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        if self.history_:
            for row in self.history_:
                row["best_iteration"] = float((self.best_iteration_ or 0) + 1)

    def predict_target(self, X: ArrayLike) -> NDArray[np.float64]:
        """Предсказать target в единицах обучающей цели."""
        if self.model is None:
            raise RuntimeError("Модель ещё не обучена или не загружена")
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or (self.input_size_ is not None and arr.shape[1] != self.input_size_):
            raise ValueError("X имеет неверную размерность")
        if self.training_config.model_type == "catboost":
            return np.asarray(self.model.predict(arr), dtype=np.float64).reshape(-1, 3)
        import torch

        if self.device_ is None:
            self.device_ = next(self.model.parameters()).device
        scaled = self.scaler_x.transform(arr).astype(np.float32)
        with torch.no_grad():
            pred_scaled = self.model(torch.from_numpy(scaled).to(self.device_)).detach().cpu().numpy()
        return np.asarray(self.scaler_y.inverse_transform(pred_scaled), dtype=np.float64)

    def predict_delta(self, X: ArrayLike, r_dip: ArrayLike | None = None) -> NDArray[np.float64]:
        """Предсказать поправку в метрах в системе координат маяка."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        prediction = self.predict_target(arr)
        if self.feature_config.target_mode == "relative":
            radius = np.asarray(r_dip if r_dip is not None else arr[:, 3], dtype=np.float64).reshape(-1)
            if radius.size != len(arr):
                raise ValueError("r_dip должен содержать одну дальность на каждый пример")
            prediction = prediction * radius[:, None]
        return prediction

    def save(self, directory: str | Path) -> Path:
        """Сохранить обученную модель, историю и метаданные."""
        if self.model is None:
            raise RuntimeError("Нельзя сохранить необученную модель")
        import joblib

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        model_type = self.training_config.model_type
        if model_type == "catboost":
            self.model.save_model(str(path / "model.cbm"))
        else:
            import torch

            state_cpu = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
            torch.save({
                "state_dict": state_cpu,
                "input_size": self.input_size_,
                "hidden_layers": list(self.training_config.hidden_layers),
                "activation": self.training_config.activation,
            }, path / "model.pt")
            joblib.dump({"scaler_x": self.scaler_x, "scaler_y": self.scaler_y}, path / "scalers.joblib")
        if self.history_:
            import pandas as pd

            pd.DataFrame(self.history_).to_csv(path / "training_history.csv", index=False)
        metadata = {
            "model_type": model_type,
            "training_config": self.training_config.to_dict(),
            "feature_config": self.feature_config.to_dict(),
            "feature_names": self.feature_names,
            "input_size": self.input_size_,
            "best_iteration": self.best_iteration_,
        }
        (path / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "NearFieldCorrector":
        """Загрузить модель. Сохранённая на CUDA модель может работать на CPU."""
        import joblib

        path = Path(directory)
        metadata = json.loads((path / "model_metadata.json").read_text(encoding="utf-8"))
        tc = dict(metadata["training_config"])
        if "hidden_layers" in tc:
            tc["hidden_layers"] = tuple(tc["hidden_layers"])
        training_config = TrainingConfig(**tc)
        feature_config = MLConfig(**metadata["feature_config"])
        obj = cls(training_config, feature_config, feature_names=metadata.get("feature_names", []))
        obj.input_size_ = metadata.get("input_size")
        obj.best_iteration_ = metadata.get("best_iteration")
        if metadata["model_type"] == "catboost":
            from catboost import CatBoostRegressor

            obj.model = CatBoostRegressor()
            obj.model.load_model(str(path / "model.cbm"))
        else:
            import torch

            bundle = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
            obj.input_size_ = int(bundle["input_size"])
            obj.model = _build_torch_network(obj.input_size_, tuple(bundle["hidden_layers"]), bundle["activation"])
            obj.model.load_state_dict(bundle["state_dict"])
            obj.device_ = torch.device("cpu")
            if training_config.torch_device == "cuda" and torch.cuda.is_available():
                obj.device_ = torch.device("cuda")
                obj.model.to(obj.device_)
            obj.model.eval()
            scalers = joblib.load(path / "scalers.joblib")
            obj.scaler_x, obj.scaler_y = scalers["scaler_x"], scalers["scaler_y"]
        history_path = path / "training_history.csv"
        if history_path.exists():
            import pandas as pd

            obj.history_ = pd.read_csv(history_path).to_dict(orient="records")
        return obj
