"""Симуляция: траектории, сценарии, визуализация."""

from . import visualize
from .monte_carlo_sim import MonteCarloResult, MonteCarloSimulator
from .scenario import (
    MonteCarloSummary,
    Scenario,
    SensorConfig,
    SimResult,
    boundary_radius_map,
    calibration_sweep,
    model_discrepancy_map,
    monte_carlo,
)
from .trajectory import (
    arc,
    circle,
    grid3d,
    helix,
    line,
    plane_grid,
    radial_rays,
    random_first_octant,
    uniform_moving_observer,
    waypoint_path,
)

__all__ = [
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
    "circle",
    "helix",
    "grid3d",
    "plane_grid",
    "radial_rays",
    "random_first_octant",
    "uniform_moving_observer",
    "visualize",
    "MonteCarloSimulator",
    "MonteCarloResult",
]
