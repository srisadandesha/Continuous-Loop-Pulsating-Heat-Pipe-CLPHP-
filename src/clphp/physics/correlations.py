"""
Empirical and Semi-Analytical Correlations for Pulsating Heat Pipe Dynamics.

Includes:
- Oscillating Flow Darcy Friction Factor (Poiseuille, Churchill, Womersley correction)
- Dynamic Contact Angle Hysteresis Capillary Force
- Hertz-Knudsen-Schrage Phase-Change Mass Transfer Rate
- Two-Phase and Slug Convective Heat Transfer Coefficients (Nusselt numbers)
"""

from __future__ import annotations
import math
from typing import Tuple


def reynolds_number(rho: float, velocity: float, d_inner: float, mu: float) -> float:
    """Calculate Reynolds number Re = rho * |u| * D / mu."""
    return max(1e-6, (rho * abs(velocity) * d_inner) / max(1e-7, mu))


def womersley_number(d_inner: float, frequency_hz: float, rho: float, mu: float) -> float:
    """
    Calculate Womersley number Wo = (D / 2) * sqrt(omega * rho / mu).
    Characterizes the ratio of transient inertial forces to viscous forces.
    """
    omega = 2.0 * math.pi * max(0.01, frequency_hz)
    return (d_inner / 2.0) * math.sqrt((omega * rho) / max(1e-7, mu))


def friction_factor(
    re: float,
    relative_roughness: float = 1e-4,
    womersley: float = 0.0,
) -> float:
    """
    Calculate Darcy-Weisbach friction factor with oscillating flow correction.
    
    Uses Churchill (1977) continuous correlation across laminar, transition,
    and turbulent regimes, combined with Womersley amplification for oscillatory flow.
    """
    re = max(1e-3, re)
    
    # Churchill correlation components
    a = (2.457 * math.log(1.0 / ((7.0 / re) ** 0.9 + 0.27 * relative_roughness))) ** 16
    b = (37530.0 / re) ** 16
    f_churchill = 8.0 * ((8.0 / re) ** 12 + 1.0 / ((a + b) ** 1.5)) ** (1.0 / 12.0)

    # Steady friction factor
    f_steady = max(0.005, min(f_churchill, 2.0))

    # Oscillatory flow amplification factor (Sexl-Womersley effect)
    if womersley > 0.5:
        # High-frequency boundary layer thinning enhances apparent shear stress
        f_amp = math.sqrt(1.0 + (womersley / 4.0) ** 2)
        return f_steady * f_amp
    
    return f_steady


def slug_friction_force(
    velocity: float,
    slug_length: float,
    d_inner: float,
    area_c: float,
    rho_l: float,
    mu_l: float,
    frequency_hz: float = 2.0,
) -> float:
    """
    Calculate viscous shear friction force resisting liquid slug motion:
    F_fric = 0.5 * f_osc * rho_l * u * |u| * A_c * (L_slug / D_inner)
    """
    if abs(velocity) < 1e-6 or slug_length <= 1e-6:
        return 0.0

    re = reynolds_number(rho_l, velocity, d_inner, mu_l)
    wo = womersley_number(d_inner, frequency_hz, rho_l, mu_l)
    f = friction_factor(re, womersley=wo)

    return 0.5 * f * rho_l * velocity * abs(velocity) * area_c * (slug_length / d_inner)


def dynamic_capillary_force(
    velocity: float,
    d_inner: float,
    sigma: float,
    theta_receding_deg: float = 30.0,
    theta_advancing_deg: float = 60.0,
) -> float:
    """
    Capillary resistance force due to contact angle hysteresis:
    F_cap = pi * D_inner * sigma * (cos(theta_r) - cos(theta_a)) * sign(velocity)
    """
    if abs(velocity) < 1e-6:
        return 0.0

    rad_r = math.radians(theta_receding_deg)
    rad_a = math.radians(theta_advancing_deg)
    delta_cos = math.cos(rad_r) - math.cos(rad_a)
    
    sign_u = 1.0 if velocity > 0 else -1.0
    return math.pi * d_inner * sigma * delta_cos * sign_u


def hertz_knudsen_mass_transfer(
    t_interface: float,
    p_interface_sat: float,
    t_vapor: float,
    p_vapor: float,
    molar_mass: float,
    area_contact: float,
    accommodation_coeff: float = 0.05,
) -> float:
    """
    Phase-change mass flow rate via the modified Hertz-Knudsen-Schrage relation:
    m_dot = C_acc * (2 * sigma_acc / (2 - sigma_acc)) * sqrt(M / (2*pi*R_u)) * ( P_sat(T_w)/sqrt(T_w) - P_v/sqrt(T_v) ) * Area
    
    Returns:
      m_dot > 0: Net Evaporation [kg/s]
      m_dot < 0: Net Condensation [kg/s]
    """
    r_univ = 8.314462  # J/(mol*K)
    c_coeff = (2.0 * accommodation_coeff) / (2.0 - accommodation_coeff)
    k_factor = math.sqrt(molar_mass / (2.0 * math.pi * r_univ))

    t_int = max(100.0, t_interface)
    t_vap = max(100.0, t_vapor)

    flux_term = (p_interface_sat / math.sqrt(t_int)) - (p_vapor / math.sqrt(t_vap))
    m_dot = c_coeff * k_factor * flux_term * max(0.0, area_contact)
    
    # Clip extreme fluxes to maintain numerical stability in stiff ODE regions
    max_mass_rate = 0.05  # kg/s limit per plug
    return max(-max_mass_rate, min(max_mass_rate, m_dot))


def convective_heat_transfer_coeff(
    velocity: float,
    d_inner: float,
    rho_l: float,
    mu_l: float,
    cp_l: float,
    k_l: float = 0.18,  # Default thermal conductivity of liquid [W/(m*K)]
) -> float:
    """
    Calculate convective heat transfer coefficient h_c [W/(m^2*K)] for liquid slug.
    Uses Sieder-Tate / Gnielinski Nusselt number correlation.
    """
    re = reynolds_number(rho_l, velocity, d_inner, mu_l)
    pr = max(0.5, (mu_l * cp_l) / max(1e-4, k_l))

    if re < 2300.0:
        # Laminar flow with entry region enhancement
        nu = 4.36 + (0.0668 * (d_inner / 0.05) * re * pr) / (1.0 + 0.04 * ((d_inner / 0.05) * re * pr) ** (2.0 / 3.0))
    else:
        # Turbulent flow (Gnielinski)
        f_turb = (0.790 * math.log(re) - 1.64) ** (-2.0)
        nu = ((f_turb / 8.0) * (re - 1000.0) * pr) / (1.0 + 12.7 * math.sqrt(f_turb / 8.0) * (pr ** (2.0 / 3.0) - 1.0))
    
    nu = max(3.66, min(500.0, nu))
    return (nu * k_l) / d_inner
