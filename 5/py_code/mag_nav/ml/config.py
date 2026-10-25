"""Конфигурация экспериментов ML для компенсации ошибок ближнего поля."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Literal


def available_cpu_count() -> int:
    """Число CPU, доступных процессу с учётом affinity/cgroup, если возможно."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


@dataclass(frozen=True)
class MLConfig:
    """Параметры подготовки датасета, признаков и группового разбиения.

    ``n_workers=0`` означает автоматически использовать доступные CPU-ядер
    для независимых задач генерации траекторий. Значение ``1`` отключает пул.
    """

    window_size: int = 1
    use_raw_ab: bool = False
    use_invariants: bool = True
    include_beacon_params: bool = True
    target_mode: Literal["meters", "relative"] = "meters"
    padding: Literal["repeat_first", "drop"] = "repeat_first"
    split_mode: Literal["by_trajectory", "by_beacon"] = "by_trajectory"
    random_seed: int = 42
    octant_outlier_tau: float = 0.8
    coordinate_zero_tol_m: float = 1e-4
    eps: float = 1e-6
    sample_weight_alpha: float = 0.0
    n_workers: int = 0

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size должен быть >= 1")
        if self.target_mode not in ("meters", "relative"):
            raise ValueError(f"Неизвестный target_mode: {self.target_mode}")
        if self.padding not in ("repeat_first", "drop"):
            raise ValueError(f"Неизвестный padding: {self.padding}")
        if self.split_mode not in ("by_trajectory", "by_beacon"):
            raise ValueError(f"Неизвестный split_mode: {self.split_mode}")
        if self.octant_outlier_tau <= 0:
            raise ValueError("octant_outlier_tau должен быть > 0")
        if self.eps <= 0:
            raise ValueError("eps должен быть > 0")
        if self.n_workers < 0:
            raise ValueError("n_workers должен быть >= 0; 0 = автоматически")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    """Параметры CatBoost и PyTorch.

    ``cpu_threads=0`` оставляет библиотеке выбор по числу доступных ядер.
    У MLP по умолчанию используется CPU; ``torch_device='cuda'`` доступен
    без переписывания цикла обучения, если PyTorch собран с поддержкой CUDA.
    """

    model_type: Literal["catboost", "mlp"] = "mlp"
    random_seed: int = 42
    learning_rate: float = 1e-3
    epochs: int = 300
    batch_size: int = 256
    patience: int = 30
    min_delta: float = 1e-8
    hidden_layers: tuple[int, ...] = (128, 128, 64)
    activation: Literal["relu", "gelu"] = "relu"
    cpu_threads: int = 0
    dataloader_num_workers: int = 0
    torch_device: Literal["cpu", "cuda"] = "cpu"
    catboost_iterations: int = 1500
    catboost_learning_rate: float = 0.05
    catboost_depth: int = 6
    catboost_early_stopping_rounds: int = 100
    catboost_task_type: Literal["CPU", "GPU"] = "CPU"
    catboost_devices: str | None = None
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.model_type not in ("catboost", "mlp"):
            raise ValueError(f"Неизвестная модель: {self.model_type}")
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size и patience должны быть положительными")
        if not self.hidden_layers or any(h < 1 for h in self.hidden_layers):
            raise ValueError("hidden_layers должен содержать положительные размеры слоёв")
        if self.cpu_threads < 0:
            raise ValueError("cpu_threads должен быть >= 0; 0 = автоматически")
        if self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers должен быть >= 0")
        if self.torch_device not in ("cpu", "cuda"):
            raise ValueError("torch_device должен быть 'cpu' или 'cuda'")
        if self.catboost_task_type not in ("CPU", "GPU"):
            raise ValueError("catboost_task_type должен быть CPU или GPU")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["hidden_layers"] = list(self.hidden_layers)
        return data
