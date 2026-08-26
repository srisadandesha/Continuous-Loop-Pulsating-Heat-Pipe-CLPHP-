"""
Tests for PHPGeometry and capillary limits.
"""

import pytest
import math
from clphp.physics.geometry import PHPGeometry
from clphp.physics.fluids import get_fluid


def test_geometry_initialization():
    geo = PHPGeometry(n_turns=5, d_inner=0.002, d_outer=0.003, fill_ratio=0.50)
    assert geo.n_branches == 10
    assert geo.cross_sectional_area == pytest.approx(math.pi * 0.002**2 / 4.0)
    assert geo.evaporator_wall_mass > 0
    assert geo.condenser_wall_mass > 0
    assert geo.total_internal_volume > 0


def test_geometry_invalid():
    with pytest.raises(ValueError):
        PHPGeometry(n_turns=0)
    with pytest.raises(ValueError):
        PHPGeometry(d_inner=0.003, d_outer=0.002)  # Outer must be > inner
    with pytest.raises(ValueError):
        PHPGeometry(fill_ratio=1.5)


def test_capillary_check():
    geo = PHPGeometry(n_turns=5, d_inner=0.002, d_outer=0.003)
    fluid = get_fluid("acetone")
    
    check = geo.check_capillary_limit(
        fluid_sigma=fluid.surface_tension(293.15),
        fluid_rho_l=fluid.liquid_density(293.15),
        fluid_rho_v=fluid.vapor_density(293.15),
        g=9.81,
    )
    assert check["is_valid"] is True
    assert check["bond_number"] < 4.0
