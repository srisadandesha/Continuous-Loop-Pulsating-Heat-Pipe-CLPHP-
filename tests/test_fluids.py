"""
Tests for FluidProperties and thermodynamic queries.
"""

import pytest
import math
from clphp.physics.fluids import FluidProperties, get_fluid, MOLAR_MASSES


def test_fluid_initialization():
    for name in ["acetone", "ammonia", "water", "ethanol", "methanol", "r134a"]:
        fluid = get_fluid(name)
        assert fluid.molar_mass == MOLAR_MASSES[name]
        assert fluid.R_specific > 0


def test_invalid_fluid():
    with pytest.raises(ValueError):
        get_fluid("unobtanium")


def test_acetone_properties():
    fluid = get_fluid("acetone")
    T = 293.15  # 20 C

    state = fluid.get_state(T)
    assert 1e4 <= state.pressure <= 5e4  # Approx 24-25 kPa for acetone at 20C
    assert 700 <= state.rho_l <= 850      # ~790 kg/m3
    assert 0.1 <= state.rho_v <= 2.0
    assert 0.01 <= state.sigma <= 0.04    # ~0.023 N/m
    assert 4e5 <= state.h_fg <= 6e5       # ~5.2e5 J/kg
    assert state.gamma > 1.05


def test_critical_capillary_diameter():
    fluid = get_fluid("acetone")
    d_crit = fluid.critical_diameter(293.15, g=9.81)
    
    # For acetone in 1g, D_crit is approx 3.4 mm
    assert 0.002 <= d_crit <= 0.005

    # Microgravity check
    d_crit_0g = fluid.critical_diameter(293.15, g=0.0)
    assert d_crit_0g > d_crit
