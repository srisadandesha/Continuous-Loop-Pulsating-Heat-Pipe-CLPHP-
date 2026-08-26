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
