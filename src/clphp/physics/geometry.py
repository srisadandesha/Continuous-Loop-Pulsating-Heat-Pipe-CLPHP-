"""
Continuous-Loop Pulsating Heat Pipe (CLPHP) Geometric Model.

Defines spatial layout, section lengths, capillary diameter validation,
tube wall physical properties, and multi-turn channel layout.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional


# Common tube material properties [k (W/m-K), rho (kg/m3), cp (J/kg-K)]
TUBE_MATERIALS = {
    "copper": {"k": 390.0, "rho": 8960.0, "cp": 385.0},
    "aluminum": {"k": 205.0, "rho": 2700.0, "cp": 900.0},
    "stainless_steel": {"k": 16.0, "rho": 7900.0, "cp": 500.0},
    "titanium": {"k": 21.9, "rho": 4500.0, "cp": 520.0},
}


@dataclass
class PHPGeometry:
    """
    Geometric and structural configuration of a Continuous-Loop Pulsating Heat Pipe.
    """
    n_turns: int = 5                  # Number of evaporator turns (U-bends)
    d_inner: float = 0.002            # Inner diameter [m] (e.g. 2.0 mm)
    d_outer: float = 0.003            # Outer diameter [m] (e.g. 3.0 mm)
    l_evap: float = 0.05              # Evaporator section length [m] (50 mm)
    l_adiabatic: float = 0.10         # Adiabatic section length [m] (100 mm)
    l_cond: float = 0.05              # Condenser section length [m] (50 mm)
    fill_ratio: float = 0.50          # Volumetric liquid filling ratio (0.3 - 0.8)
    material: str = "copper"          # Wall material ('copper', 'aluminum', etc.)
    inclination_angle: float = 90.0   # Inclination angle in degrees (90 = vertical bottom-heat, 0 = horizontal)

    def __post_init__(self):
        if self.n_turns < 1:
            raise ValueError("Number of turns must be at least 1.")
        if self.d_inner <= 0 or self.d_outer <= self.d_inner:
            raise ValueError("Invalid diameters: d_outer must be strictly greater than d_inner > 0.")
        if not (0.05 <= self.fill_ratio <= 0.95):
            raise ValueError("Fill ratio must be between 0.05 (5%) and 0.95 (95%).")

        mat = self.material.lower().strip()
        if mat in TUBE_MATERIALS:
            props = TUBE_MATERIALS[mat]
            self.k_wall = props["k"]
            self.rho_wall = props["rho"]
            self.cp_wall = props["cp"]
        else:
            self.k_wall = 390.0
            self.rho_wall = 8960.0
            self.cp_wall = 385.0

    @property
    def n_branches(self) -> int:
        """Total number of parallel pipe branches in the loop (2 * n_turns)."""
        return 2 * self.n_turns

    @property
    def branch_length(self) -> float:
        """Single branch length: L_e + 2*L_a + L_c."""
        return self.l_evap + 2.0 * self.l_adiabatic + self.l_cond

    @property
    def total_length(self) -> float:
        """Total axial internal loop length [m] (sum of all branches and U-bends)."""
        u_bend_arc = math.pi * (0.01)  # approx 10 mm bend radius
        return self.n_branches * self.branch_length + self.n_branches * u_bend_arc

    @property
    def cross_sectional_area(self) -> float:
        """Internal fluid channel cross-sectional area A_c [m^2]."""
        return math.pi * (self.d_inner ** 2) / 4.0

    @property
    def inner_perimeter(self) -> float:
        """Internal channel perimeter P_in [m]."""
        return math.pi * self.d_inner

    @property
    def wall_cross_sectional_area(self) -> float:
        """Solid tube wall cross-sectional area A_w [m^2]."""
        return math.pi * (self.d_outer ** 2 - self.d_inner ** 2) / 4.0

    @property
    def total_internal_volume(self) -> float:
        """Total internal fluid volume V_tot [m^3]."""
        return self.cross_sectional_area * self.total_length

    @property
    def evaporator_wall_mass(self) -> float:
        """Total solid wall mass in the evaporator section [kg]."""
        v_wall_e = self.wall_cross_sectional_area * (self.l_evap * self.n_turns)
        return self.rho_wall * v_wall_e

    @property
    def condenser_wall_mass(self) -> float:
        """Total solid wall mass in the condenser section [kg]."""
        v_wall_c = self.wall_cross_sectional_area * (self.l_cond * self.n_turns)
        return self.rho_wall * v_wall_c

    def check_capillary_limit(self, fluid_sigma: float, fluid_rho_l: float, fluid_rho_v: float, g: float = 9.81) -> dict:
        """
        Verify if the inner diameter satisfies the critical capillary diameter for slug/plug formation:
        D_inner <= 2 * sqrt( sigma / (g * (rho_l - rho_v)) )
        Bond Number: Bo = g * (rho_l - rho_v) * D_inner^2 / sigma <= 4.0
        """
        delta_rho = max(1.0, fluid_rho_l - fluid_rho_v)
        
        if g <= 1e-4:
            # Microgravity
            bo = 0.0
            d_crit = 0.02
            is_valid = True
        else:
            bo = (g * delta_rho * (self.d_inner ** 2)) / max(1e-6, fluid_sigma)
            d_crit = 2.0 * math.sqrt(fluid_sigma / (g * delta_rho))
            is_valid = self.d_inner <= d_crit

        return {
            "is_valid": is_valid,
            "d_inner_mm": self.d_inner * 1e3,
            "d_crit_mm": d_crit * 1e3,
            "bond_number": bo,
            "max_allowable_bo": 4.0,
            "margin_pct": ((d_crit - self.d_inner) / d_crit) * 100.0 if d_crit > 0 else 100.0,
        }
