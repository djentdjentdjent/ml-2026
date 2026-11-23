"""mag_nav — симулятор навигации по вращающемуся магнитному маяку.

Порт MATLAB-проекта. Текущая итерация:
  * первый квадрант (x, y, z >= 0);
  * ориентация **не** определяется (рамки приёмника и маяка совпадают);
  * навигация по a/b-параметрам первой гармоники;
  * опциональный шум в отсчётах или в a/b.

Быстрый старт
-------------
>>> import numpy as np
>>> from mag_nav import RotatingDipole, Scenario, GaussianNoise, random_first_octant
>>> src = RotatingDipole(origin=np.zeros(3), moment_amplitude=1.0)
>>> traj = random_first_octant(500, 0.15, 1.0, seed=1)
>>> res = Scenario(src, traj).run()
>>> res.summary()  # doctest: +SKIP
'SimResult: N=500/500 (valid), RMS=... м, ...'
"""

from __future__ import annotations

__version__ = "0.1.0"

from .convention import (
    AXIS_X,
    AXIS_Y,
    AXIS_Z,
    MU_0,
    FieldVec,
    Position,
    QuadrantPolicy,
    dipole_constant,
    direction_cosines,
    enforce_quadrant,
    moment_from_constant,
    norm,
    unit,
)
from .navigation.orientation import IDENTITY, RotationMatrix, solve_orientation
from .navigation.position import (
    PositionSolution,
    error_statistics,
    position_error,
    solve_position,
)
from .noise import (
    AdditiveNoise,
    ClipNoise,
    CompositeNoise,
    ConstantBias,
    GaussianNoise,
    NoiseChain,
    NoiseModel,
    QuantizationNoise,
    SumNoise,
)
from .signal import (
    ABInvariants,
    ABParams,
    DemodulationResult,
    ab_from_dipole,
    ab_from_source,
    ab_invariants,
    demodulate,
    demodulate_lstsq,
    demodulation_error,
    harmonic_residual,
    make_phase_grid,
    polarization_ellipse,
    synthesize_series,
)
from .sim import (
    MonteCarloResult,
    MonteCarloSimulator,
    MonteCarloSummary,
    Scenario,
    SensorConfig,
    SimResult,
    arc,
    boundary_radius_map,
    calibration_sweep,
    grid3d,
    helix,
    line,
    model_discrepancy_map,
    monte_carlo,
    plane_grid,
    radial_rays,
    random_first_octant,
    uniform_moving_observer,
    visualize,
    waypoint_path,
)
from .source import (
    MagneticSource,
    RotatingDipole,
    Solenoid,
    dipole_field,
    dipole_from_current_loop,
    trace_field_line,
)

__all__ = [
    "__version__",
    # convention
    "MU_0",
    "AXIS_X",
    "AXIS_Y",
    "AXIS_Z",
    "Position",
    "FieldVec",
    "QuadrantPolicy",
    "dipole_constant",
    "moment_from_constant",
    "direction_cosines",
    "enforce_quadrant",
    "norm",
    "unit",
    # source
    "MagneticSource",
    "RotatingDipole",
    "Solenoid",
    "dipole_field",
    "dipole_from_current_loop",
    "trace_field_line",
    # signal
    "ABParams",
    "ABInvariants",
    "ab_from_dipole",
    "ab_from_source",
    "ab_invariants",
    "harmonic_residual",
    "polarization_ellipse",
    "DemodulationResult",
    "make_phase_grid",
    "synthesize_series",
    "demodulate",
    "demodulate_lstsq",
    "demodulation_error",
    "MonteCarloSimulator",
    "MonteCarloResult",
    # navigation
    "PositionSolution",
    "solve_position",
    "position_error",
    "error_statistics",
    "RotationMatrix",
    "solve_orientation",
    "IDENTITY",
    # noise
    "NoiseModel",
    "AdditiveNoise",
    "GaussianNoise",
    "ConstantBias",
    "SumNoise",
    "CompositeNoise",
    "NoiseChain",
    "QuantizationNoise",
    "ClipNoise",
    # sim
    "Scenario",
    "SensorConfig",
    "SimResult",
    "MonteCarloSummary",
    "monte_carlo",
    "model_discrepancy_map",
    "boundary_radius_map",
    "calibration_sweep",
    "line",
    "waypoint_path",
    "arc",
    "helix",
    "grid3d",
    "plane_grid",
    "radial_rays",
    "random_first_octant",
    "uniform_moving_observer",
    "visualize",
]
