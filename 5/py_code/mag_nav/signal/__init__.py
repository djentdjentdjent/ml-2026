"""Прямая задача сигнала: a/b-параметры и их восстановление из отсчётов."""

from .ab_params import (
    ABInvariants,
    ABParams,
    ab_from_dipole,
    ab_from_source,
    ab_invariants,
    first_harmonic,
    harmonic_residual,
    invariants_from_geometry,
    phase_grid,
    polarization_ellipse,
)
from .recovery import (
    DemodulationResult,
    demodulate,
    demodulate_lstsq,
    demodulation_error,
    estimate_amplitude,
    make_phase_grid,
    synthesize_series,
)

__all__ = [
    "ABParams",
    "ABInvariants",
    "ab_from_dipole",
    "ab_from_source",
    "ab_invariants",
    "invariants_from_geometry",
    "first_harmonic",
    "harmonic_residual",
    "polarization_ellipse",
    "phase_grid",
    "DemodulationResult",
    "make_phase_grid",
    "synthesize_series",
    "demodulate",
    "demodulate_lstsq",
    "estimate_amplitude",
    "demodulation_error",
]