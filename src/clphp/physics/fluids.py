"""
Thermodynamic Fluid Properties Provider for Pulsating Heat Pipe Simulation.

Supports CoolProp high-precision equations of state with built-in analytical
NIST/Antoine fallback correlations for common PHP working fluids (Acetone,
Ammonia, Water, Ethanol, Methanol, R134a).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False


# Fluid molecular weights (kg/mol)
MOLAR_MASSES: dict[str, float] = {
    "acetone": 0.05808,
    "ammonia": 0.01703,
    "water": 0.018015,
    "ethanol": 0.04607,
    "methanol": 0.03204,
    "r134a": 0.10203,
}

UNIVERSAL_GAS_CONSTANT = 8.314462  # J/(mol*K)


@dataclass
class FluidState:
    """Represents a thermodynamic snapshot of a fluid at temperature T and/or pressure P."""
    fluid_name: str
    temperature: float  # [K]
    pressure: float  # [Pa]
    rho_l: float  # Liquid density [kg/m^3]
    rho_v: float  # Vapor density [kg/m^3]
    mu_l: float  # Liquid dynamic viscosity [Pa*s]
    cp_l: float  # Liquid specific heat [J/(kg*K)]
    cp_v: float  # Vapor specific heat [J/(kg*K)]
    gamma: float  # Vapor specific heat ratio (cp/cv)
    sigma: float  # Surface tension [N/m]
    h_fg: float  # Latent heat of vaporization [J/kg]
    R_specific: float  # Specific gas constant [J/(kg*K)]


class FluidProperties:
    """
    Thermodynamic property calculator for pulsating heat pipe fluids.
    """

    def __init__(self, fluid_name: str = "acetone", prefer_coolprop: bool = True):
        self.fluid_name = fluid_name.lower().strip()
        self.use_coolprop = prefer_coolprop and COOLPROP_AVAILABLE
        
        if self.fluid_name not in MOLAR_MASSES:
            valid = list(MOLAR_MASSES.keys())
            raise ValueError(f"Fluid '{self.fluid_name}' is not supported. Supported fluids: {valid}")
        
        self.molar_mass = MOLAR_MASSES[self.fluid_name]
        self.R_specific = UNIVERSAL_GAS_CONSTANT / self.molar_mass

        # CoolProp fluid name mapping
        self._cp_name_map = {
            "acetone": "Acetone",
            "ammonia": "Ammonia",
            "water": "Water",
            "ethanol": "Ethanol",
            "methanol": "Methanol",
            "r134a": "R134a",
        }
        self.cp_name = self._cp_name_map.get(self.fluid_name, "Acetone")

    def sat_pressure(self, T_kelvin: float) -> float:
        """Calculate saturation pressure P_sat [Pa] at temperature T [K]."""
        T = max(200.0, min(T_kelvin, 550.0))
        if self.use_coolprop:
            try:
                return float(CP.PropsSI("P", "T", T, "Q", 1.0, self.cp_name))
            except Exception:
                pass
        return self._analytical_p_sat(T)

    def sat_temperature(self, P_pa: float) -> float:
        """Calculate saturation temperature T_sat [K] at pressure P [Pa]."""
        P = max(100.0, min(P_pa, 5.0e6))
        if self.use_coolprop:
            try:
                return float(CP.PropsSI("T", "P", P, "Q", 1.0, self.cp_name))
            except Exception:
                pass
        return self._analytical_t_sat(P)

    def liquid_density(self, T_kelvin: float) -> float:
        """Liquid density rho_l [kg/m^3]."""
        T = max(200.0, min(T_kelvin, 550.0))
        if self.use_coolprop:
            try:
                return float(CP.PropsSI("D", "T", T, "Q", 0.0, self.cp_name))
            except Exception:
                pass
        return self._analytical_rho_l(T)

    def vapor_density(self, T_kelvin: float, P_pa: Optional[float] = None) -> float:
        """Vapor density rho_v [kg/m^3]."""
        T = max(200.0, min(T_kelvin, 550.0))
        if P_pa is None:
            P_pa = self.sat_pressure(T)
        P = max(100.0, min(P_pa, 5.0e6))

        if self.use_coolprop:
            try:
                return float(CP.PropsSI("D", "T", T, "P", P, self.cp_name))
            except Exception:
                pass
        # Ideal gas law approximation
        return max(1e-4, P / (self.R_specific * T))

    def liquid_viscosity(self, T_kelvin: float) -> float:
        """Liquid dynamic viscosity mu_l [Pa*s]."""
        T = max(200.0, min(T_kelvin, 550.0))
        if self.use_coolprop:
            try:
                return float(CP.PropsSI("V", "T", T, "Q", 0.0, self.cp_name))
            except Exception:
                pass
        return self._analytical_mu_l(T)

    def surface_tension(self, T_kelvin: float) -> float:
        """Surface tension sigma [N/m]."""
        T = max(200.0, min(T_kelvin, 550.0))
        if self.use_coolprop:
            try:
                return float(CP.PropsSI("I", "T", T, "Q", 0.0, self.cp_name))
            except Exception:
                pass
        return self._analytical_sigma(T)

    def latent_heat(self, T_kelvin: float) -> float:
        """Latent heat of vaporization h_fg [J/kg]."""
        T = max(200.0, min(T_kelvin, 550.0))
        if self.use_coolprop:
            try:
                h_v = CP.PropsSI("H", "T", T, "Q", 1.0, self.cp_name)
                h_l = CP.PropsSI("H", "T", T, "Q", 0.0, self.cp_name)
                return float(h_v - h_l)
            except Exception:
                pass
        return self._analytical_h_fg(T)

    def liquid_cp(self, T_kelvin: float) -> float:
        """Liquid specific heat capacity c_p,l [J/(kg*K)]."""
        T = max(200.0, min(T_kelvin, 550.0))
        if self.use_coolprop:
            try:
                return float(CP.PropsSI("C", "T", T, "Q", 0.0, self.cp_name))
            except Exception:
                pass
        return self._analytical_cp_l(T)

    def vapor_cp(self, T_kelvin: float) -> float:
        """Vapor specific heat capacity c_p,v [J/(kg*K)]."""
        T = max(200.0, min(T_kelvin, 550.0))
        if self.use_coolprop:
            try:
                return float(CP.PropsSI("C", "T", T, "Q", 1.0, self.cp_name))
            except Exception:
                pass
        return self._analytical_cp_v(T)

    def specific_heat_ratio(self, T_kelvin: float) -> float:
        """Vapor specific heat ratio gamma = cp / cv."""
        cp = self.vapor_cp(T_kelvin)
        cv = max(100.0, cp - self.R_specific)
        return min(1.67, max(1.05, cp / cv))

    def critical_diameter(self, T_kelvin: float, g: float = 9.81) -> float:
        """
        Calculate maximum allowable internal capillary diameter (Bond criterion Bo <= 4):
        D_crit = 2.0 * sqrt( sigma / (g * (rho_l - rho_v)) )
        In microgravity (g -> 0), returns a default upper physical limit (e.g. 0.02 m).
        """
        if g <= 1e-4:
            return 0.02  # 20 mm in microgravity
        
        sigma = self.surface_tension(T_kelvin)
        rho_l = self.liquid_density(T_kelvin)
        rho_v = self.vapor_density(T_kelvin)
        delta_rho = max(1.0, rho_l - rho_v)
        
        return 2.0 * math.sqrt(sigma / (g * delta_rho))

    def get_state(self, T_kelvin: float, P_pa: Optional[float] = None) -> FluidState:
        """Return a complete FluidState dataclass for the given temperature and pressure."""
        T = max(200.0, min(T_kelvin, 550.0))
        P = self.sat_pressure(T) if P_pa is None else max(100.0, P_pa)
        rho_l = self.liquid_density(T)
        rho_v = self.vapor_density(T, P)
        mu_l = self.liquid_viscosity(T)
        cp_l = self.liquid_cp(T)
        cp_v = self.vapor_cp(T)
        gamma = self.specific_heat_ratio(T)
        sigma = self.surface_tension(T)
        h_fg = self.latent_heat(T)

        return FluidState(
            fluid_name=self.fluid_name,
            temperature=T,
            pressure=P,
            rho_l=rho_l,
            rho_v=rho_v,
            mu_l=mu_l,
            cp_l=cp_l,
            cp_v=cp_v,
            gamma=gamma,
            sigma=sigma,
            h_fg=h_fg,
            R_specific=self.R_specific,
        )

    # -------------------------------------------------------------------------
    # Analytical Correlations (NIST / Antoine / DIPPR Fallbacks)
    # -------------------------------------------------------------------------

    def _analytical_p_sat(self, T: float) -> float:
        """Antoine equation: log10(P_bar) = A - B / (T + C) -> converted to Pa."""
        # Antoine constants (P in bar, T in K)
        constants = {
            "acetone": (4.42448, 1312.253, -32.445),
            "ammonia": (4.86886, 1113.928, -10.409),
            "water": (5.11564, 1687.537, -42.98),
            "ethanol": (5.24677, 1598.673, -46.424),
            "methanol": (5.20409, 1581.341, -33.50),
            "r134a": (4.06734, 888.450, -32.610),
        }
        A, B, C = constants.get(self.fluid_name, constants["acetone"])
        try:
            log10_p_bar = A - (B / (T + C))
            p_bar = 10.0 ** log10_p_bar
            return max(100.0, p_bar * 1.0e5)
        except Exception:
            return 101325.0

    def _analytical_t_sat(self, P: float) -> float:
        """Inverse Antoine equation."""
        constants = {
            "acetone": (4.42448, 1312.253, -32.445),
            "ammonia": (4.86886, 1113.928, -10.409),
            "water": (5.11564, 1687.537, -42.98),
            "ethanol": (5.24677, 1598.673, -46.424),
            "methanol": (5.20409, 1581.341, -33.50),
            "r134a": (4.06734, 888.450, -32.610),
        }
        A, B, C = constants.get(self.fluid_name, constants["acetone"])
        p_bar = max(1e-4, P / 1.0e5)
        try:
            log10_p = math.log10(p_bar)
            return max(200.0, (B / (A - log10_p)) - C)
        except Exception:
            return 329.0

    def _analytical_rho_l(self, T: float) -> float:
        # Approximate temperature-dependent liquid density
        ref_densities = {
            "acetone": (791.0, 329.0, 1.25),
            "ammonia": (682.0, 240.0, 1.50),
            "water": (997.0, 298.0, 0.45),
            "ethanol": (789.0, 298.0, 0.85),
            "methanol": (792.0, 298.0, 0.95),
            "r134a": (1206.0, 298.0, 3.2),
        }
        rho_ref, T_ref, beta = ref_densities.get(self.fluid_name, (791.0, 329.0, 1.25))
        return max(400.0, rho_ref - beta * (T - T_ref))

    def _analytical_mu_l(self, T: float) -> float:
        # Dynamic viscosity Andrade formula: mu = A * exp(B / T)
        andrade = {
            "acetone": (1.1e-5, 850.0),
            "ammonia": (5.0e-6, 750.0),
            "water": (2.414e-5, 570.0),
            "ethanol": (4.5e-6, 1200.0),
            "methanol": (6.0e-6, 950.0),
            "r134a": (8.0e-6, 650.0),
        }
        A, B = andrade.get(self.fluid_name, (1.1e-5, 850.0))
        return max(5e-5, A * math.exp(B / max(100.0, T)))

    def _analytical_sigma(self, T: float) -> float:
        # Surface tension correlation: sigma = sigma_0 * (1 - T/Tc)^n
        crit_data = {
            "acetone": (0.0237, 508.2, 1.2),
            "ammonia": (0.0234, 405.4, 1.22),
            "water": (0.0728, 647.1, 1.15),
            "ethanol": (0.0223, 513.9, 1.18),
            "methanol": (0.0226, 512.6, 1.18),
            "r134a": (0.0080, 374.2, 1.25),
        }
        sigma_0, T_c, n = crit_data.get(self.fluid_name, (0.0237, 508.2, 1.2))
        tau = max(0.01, 1.0 - (T / T_c))
        return max(1e-4, sigma_0 * (tau ** n))

    def _analytical_h_fg(self, T: float) -> float:
        # Latent heat Watson correlation
        watson = {
            "acetone": (5.18e5, 508.2, 0.38),
            "ammonia": (1.37e6, 405.4, 0.38),
            "water": (2.26e6, 647.1, 0.38),
            "ethanol": (8.46e5, 513.9, 0.38),
            "methanol": (1.10e6, 512.6, 0.38),
            "r134a": (2.17e5, 374.2, 0.38),
        }
        h_0, T_c, n = watson.get(self.fluid_name, (5.18e5, 508.2, 0.38))
        tau = max(0.01, 1.0 - (T / T_c))
        return max(1e4, h_0 * (tau ** n))

    def _analytical_cp_l(self, T: float) -> float:
        cp_ref = {
            "acetone": 2180.0,
            "ammonia": 4750.0,
            "water": 4184.0,
            "ethanol": 2440.0,
            "methanol": 2530.0,
            "r134a": 1420.0,
        }
        return cp_ref.get(self.fluid_name, 2180.0)

    def _analytical_cp_v(self, T: float) -> float:
        cp_v_ref = {
            "acetone": 1450.0,
            "ammonia": 2180.0,
            "water": 1900.0,
            "ethanol": 1580.0,
            "methanol": 1500.0,
            "r134a": 1020.0,
        }
        return cp_v_ref.get(self.fluid_name, 1450.0)


def get_fluid(name: str = "acetone") -> FluidProperties:
    """Helper factory function to create a FluidProperties instance."""
    return FluidProperties(fluid_name=name)
