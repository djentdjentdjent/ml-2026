"""Единственная точка выбора физической модели маяка для ML-экспериментов.

Чтобы переключить проект на другой источник, достаточно заменить назначение
``SOURCE_FACTORY`` одной строкой на callable ``(beacon_id, rng) -> MagneticSource``.
Новый класс должен реализовывать контракт из ``mag_nav.source.base.MagneticSource``.
Физика существующего Solenoid не модифицируется.
"""
from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

import numpy as np

from ..source.base import MagneticSource
from ..source.solenoid import Solenoid

SourceFactory = Callable[[int, np.random.Generator], MagneticSource]


def _default_solenoid_factory(beacon_id: int, rng: np.random.Generator) -> MagneticSource:
    """Создаёт конечный вращающийся соленоид с воспроизводимой конфигурацией.

    Параметры варьируются между beacon_id для сценария ``by_beacon``.
    При ``by_trajectory`` один источник создаётся один раз и общий для всех
    траекторий.
    """
    del beacon_id  # ID отвечает за идентичность группы, параметры выбирает rng.
    radius = float(rng.uniform(0.040, 0.065))
    length = float(rng.uniform(0.070, 0.160))
    n_turns = int(rng.integers(80, 161))
    current = float(rng.uniform(0.65, 1.35))
    return Solenoid(
        origin=np.zeros(3, dtype=np.float64),
        n_turns=n_turns,
        current=current,
        radius=radius,
        length=length,
        n_quad=48,
        quadrant_policy="off",
    )


# ЗАМЕНИТЕ ТОЛЬКО ЭТУ СТРОКУ, ЧТОБЫ ВЫБРАТЬ ДРУГУЮ МОДЕЛЬ ИСТОЧНИКА.
SOURCE_FACTORY: SourceFactory | str = _default_solenoid_factory


def make_source(beacon_id: int, seed: int = 42) -> MagneticSource:
    """Создать источник; seed отделяет выбор маяка от других ГСЧ эксперимента."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(beacon_id), 9143]))
    factory = SOURCE_FACTORY
    if isinstance(factory, str):
        module_name, separator, attribute_name = factory.partition(":")
        if not separator or not module_name or not attribute_name:
            raise ValueError("SOURCE_FACTORY в строковом виде должен быть 'module.path:factory_name'")
        factory = getattr(import_module(module_name), attribute_name)
    return factory(int(beacon_id), rng)
