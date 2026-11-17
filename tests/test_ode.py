"""
Tests for ContinuousLoopPHP ODE system.
"""

import numpy as np
import pytest
from clphp.physics.ode_system import ContinuousLoopPHP
from clphp.physics.geometry import PHPGeometry
from clphp.physics.fluids import get_fluid


def test_ode_initial_state():
    geo = PHPGeometry(n_turns=5)
    php = ContinuousLoopPHP(geometry=geo, fluid=get_fluid("acetone"))
    
    y0 = php.initial_state_vector()
    assert len(y0) == 6 * 5  # 6 variables * 5 turns
    
    state = php.unpack_state(y0)
    assert len(state.displacements) == 5
    assert len(state.velocities) == 5
    assert len(state.pressures) == 5
    assert len(state.vapor_temps) == 5
    assert len(state.evap_wall_temps) == 5
    assert len(state.cond_wall_temps) == 5


def test_ode_rhs_computation():
    geo = PHPGeometry(n_turns=4)
    php = ContinuousLoopPHP(geometry=geo, fluid=get_fluid("acetone"))
    
    y0 = php.initial_state_vector()
    dydt = php.rhs(t=0.0, y=y0, q_in_total=40.0)
    
    assert len(dydt) == len(y0)
    assert not np.any(np.isnan(dydt))
    assert not np.any(np.isinf(dydt))


def test_initial_state_is_seeded_and_does_not_touch_global_rng():
    import numpy as np
    from clphp.physics.ode_system import ContinuousLoopPHP

    a = ContinuousLoopPHP(seed=7).initial_state_vector()
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    b = ContinuousLoopPHP(seed=7).initial_state_vector()
    c = ContinuousLoopPHP(seed=8).initial_state_vector()
    assert np.array_equal(a, b)          # same seed -> identical initial state
    assert not np.array_equal(a, c)      # different seed -> different perturbation
    assert np.array_equal(before, np.random.get_state()[1])  # global RNG untouched


def test_default_solver_is_radau_with_requested_tolerances():
    import inspect
    from clphp.simulation.solver import simulate_clphp
    from clphp.simulation.dataset_generator import DatasetConfig

    sig = inspect.signature(simulate_clphp).parameters
    assert (sig["solver_method"].default, sig["rtol"].default, sig["atol"].default) == ("Radau", 1e-4, 1e-6)
    cfg = DatasetConfig()
    assert (cfg.solver_method, cfg.rtol, cfg.atol) == ("Radau", 1e-4, 1e-6)
