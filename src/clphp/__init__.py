"""
Continuous-Loop Pulsating Heat Pipe (CLPHP) Simulation & PI-RNN Framework.

A hybrid mathematical modeling and physics-informed recurrent neural network
suite for real-time transient simulation and thermal prediction of spacecraft-grade
pulsating heat pipes.
"""

__version__ = "0.1.0"

from clphp.physics.fluids import FluidProperties, get_fluid
from clphp.physics.geometry import PHPGeometry
from clphp.physics.ode_system import ContinuousLoopPHP
from clphp.simulation.solver import simulate_clphp, SimulationResult

__all__ = [
    "FluidProperties",
    "get_fluid",
    "PHPGeometry",
    "ContinuousLoopPHP",
    "simulate_clphp",
    "SimulationResult",
    "__version__",
]
