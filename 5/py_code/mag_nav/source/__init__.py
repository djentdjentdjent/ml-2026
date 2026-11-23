"""Модели источников магнитного поля."""

from .base import (
    MagneticSource,
    field_line_tracer,
    orthonormal_frame,
    rotation_to_frame,
    trace_field_line,
)
from .dipole import (
    RotatingDipole,
    dipole_ab_unit_vectors,
    dipole_ab_vectors,
    dipole_field,
    dipole_from_current_loop,
    dipole_invariants,
)
from .solenoid import Solenoid, gauss_legendre, loop_field_cylindrical

__all__ = [
    "MagneticSource",
    "RotatingDipole",
    "Solenoid",
    "dipole_field",
    "dipole_ab_unit_vectors",
    "dipole_ab_vectors",
    "dipole_invariants",
    "dipole_from_current_loop",
    "loop_field_cylindrical",
    "gauss_legendre",
    "orthonormal_frame",
    "rotation_to_frame",
    "trace_field_line",
    "field_line_tracer",
]