"""Символьный аппарат магнитной навигации (SymPy).

Текущее состояние: выполнены этапы 0 (ядро) и 1 (гармоники) плана.
"""

from mag_nav.symbolic.core import (
    CONVENTIONS,
    E_X,
    E_Y,
    E_Z,
    MU0_OVER_4PI,
    OMEGA,
    PHI,
    R,
    SQRT2,
    THETA,
    TIME,
    Tensor,
    X,
    Y,
    Z,
    basis_tensor,
    complex_moment,
    complex_pair,
    demod_phase,
    describe_conventions,
    levi_civita,
    metric,
    spherical_frame_matrix,
    spherical_to_cartesian,
    tensor_from,
    time_phase,
    validate_lm,
)

__all__ = [
    "Tensor", "tensor_from", "basis_tensor",
    "X", "Y", "Z", "R", "THETA", "PHI", "SQRT2",
    "E_X", "E_Y", "E_Z", "metric", "levi_civita", "spherical_frame_matrix",
    "spherical_to_cartesian",
    "OMEGA", "TIME", "MU0_OVER_4PI", "CONVENTIONS", "describe_conventions",
    "time_phase", "demod_phase", "complex_pair", "complex_moment", "validate_lm",
]
