"""
Coupled Lumped-Parameter ODE System for Continuous-Loop Pulsating Heat Pipes (CLPHP).

Formulates the dynamic system of differential equations:
- Liquid slug momentum balance (pressure diff, oscillating friction, capillary force, gravity)
- Vapor plug thermodynamics (isentropic/polytropic compression, Hertz-Knudsen phase change)
- Evaporator and Condenser wall thermal capacitance energy balances
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

from clphp.physics.fluids import FluidProperties, get_fluid
from clphp.physics.geometry import PHPGeometry
from clphp.physics.correlations import (
    slug_friction_force,
    dynamic_capillary_force,
    hertz_knudsen_mass_transfer,
    convective_heat_transfer_coeff,
)


@dataclass
class CLPHPState:
    """Convenience structure representing unpacked state vector at time t."""
    displacements: np.ndarray    # [m] (N,)
    velocities: np.ndarray       # [m/s] (N,)
    pressures: np.ndarray        # [Pa] (N,)
    vapor_temps: np.ndarray      # [K] (N,)
    evap_wall_temps: np.ndarray  # [K] (N,)
    cond_wall_temps: np.ndarray  # [K] (N,)


class ContinuousLoopPHP:
    """
    Mathematical ODE formulation of a Continuous-Loop Pulsating Heat Pipe.
    """

    def __init__(
        self,
        geometry: Optional[PHPGeometry] = None,
        fluid: Optional[FluidProperties] = None,
        g_accel: float = 9.81,
        t_sink_k: float = 293.15,
        t_init_k: float = 293.15,
    ):
        self.geo = geometry or PHPGeometry()
        self.fluid = fluid or get_fluid("acetone")
        self.g = g_accel
        self.t_sink = t_sink_k
        self.t_init = t_init_k
        self.n = self.geo.n_turns

        # Internal dimensions
        self.A_c = self.geo.cross_sectional_area
        self.D_i = self.geo.d_inner
        self.P_in = self.geo.inner_perimeter

        # Compute initial slug and plug partitioning based on fill ratio
        self.total_vol = self.geo.total_internal_volume
        self.v_liquid_tot = self.total_vol * self.geo.fill_ratio
        self.v_vapor_tot = self.total_vol * (1.0 - self.geo.fill_ratio)

        # Baseline per slug/plug lengths and volumes
        self.l_slug_0 = (self.v_liquid_tot / self.A_c) / self.n
        self.v_plug_0 = self.v_vapor_tot / self.n
        self.l_plug_0 = self.v_plug_0 / self.A_c

        # Fluid properties at initial reference temperature
        self.ref_state = self.fluid.get_state(self.t_init)
        self.slug_mass_0 = self.ref_state.rho_l * self.A_c * self.l_slug_0

        # Wall node thermal masses (evaporator & condenser per turn)
        self.m_w_e = max(1e-4, self.geo.evaporator_wall_mass / self.n)
        self.m_w_c = max(1e-4, self.geo.condenser_wall_mass / self.n)
        self.cp_w = self.geo.cp_wall

        # Wall heat transfer contact areas
        self.area_evap_turn = self.P_in * self.geo.l_evap
        self.area_cond_turn = self.P_in * self.geo.l_cond

    def initial_state_vector(self) -> np.ndarray:
        """
        Generate initial state vector y0:
        [x_0..x_N-1, u_0..u_N-1, P_0..P_N-1, Tv_0..Tv_N-1, Twe_0..Twe_N-1, Twc_0..Twc_N-1]
        Length: 6 * N
        """
        np.random.seed(42)
        x0 = np.zeros(self.n)
        u0 = 1e-4 * (np.sin(np.linspace(0, 2 * np.pi, self.n, endpoint=False)) + 0.05 * np.random.randn(self.n))
        p0 = np.full(self.n, self.ref_state.pressure)
        tv0 = np.full(self.n, self.t_init)
        twe0 = np.full(self.n, self.t_init + 0.2)
        twc0 = np.full(self.n, self.t_sink)

        return np.concatenate([x0, u0, p0, tv0, twe0, twc0])

    def unpack_state(self, y: np.ndarray) -> CLPHPState:
        """Unpack flat 1D state vector into named structured arrays."""
        n = self.n
        return CLPHPState(
            displacements=y[0:n],
            velocities=y[n:2*n],
            pressures=y[2*n:3*n],
            vapor_temps=y[3*n:4*n],
            evap_wall_temps=y[4*n:5*n],
            cond_wall_temps=y[5*n:6*n],
        )

    def rhs(self, t: float, y: np.ndarray, q_in_total: float = 50.0) -> np.ndarray:
        """
        Evaluate time derivatives dy/dt for the coupled system at time t.
        """
        n = self.n
        x = y[0:n]
        u = y[n:2*n]
        p = np.clip(y[2*n:3*n], 50.0, 5e6)
        tv = np.clip(y[3*n:4*n], 180.0, 550.0)
        twe = np.clip(y[4*n:5*n], 180.0, 550.0)
        twc = np.clip(y[5*n:6*n], 180.0, 550.0)

        # 1. Kinematics and Volumes
        dx_dt = u.copy()

        x_next = np.roll(x, -1)
        u_next = np.roll(u, -1)
        
        vol_v = self.v_plug_0 + self.A_c * (x_next - x)
        min_vol = 0.10 * self.v_plug_0
        max_vol = 2.0 * self.v_plug_0
        vol_v_safe = np.clip(vol_v, min_vol, max_vol)

        dvol_dt = self.A_c * (u_next - u)

        # 2. Phase-Change Mass Transfer Rates (Hertz-Knudsen)
        m_dot_evap = np.zeros(n)
        m_dot_cond = np.zeros(n)

        t_mean_v = float(np.mean(tv))
        h_fg = self.fluid.latent_heat(t_mean_v)
        gamma = self.fluid.specific_heat_ratio(t_mean_v)
        rho_l = self.fluid.liquid_density(t_mean_v)
        mu_l = self.fluid.liquid_viscosity(t_mean_v)
        sigma = self.fluid.surface_tension(t_mean_v)
        cp_l = self.fluid.liquid_cp(t_mean_v)

        for i in range(n):
            p_sat_we = self.fluid.sat_pressure(twe[i])
            p_sat_wc = self.fluid.sat_pressure(twc[i])

            m_dot_evap[i] = max(
                0.0,
                hertz_knudsen_mass_transfer(
                    t_interface=twe[i],
                    p_interface_sat=p_sat_we,
                    t_vapor=tv[i],
                    p_vapor=p[i],
                    molar_mass=self.fluid.molar_mass,
                    area_contact=self.area_evap_turn * 0.5,
                )
            )

            m_dot_cond[i] = min(
                0.0,
                hertz_knudsen_mass_transfer(
                    t_interface=twc[i],
                    p_interface_sat=p_sat_wc,
                    t_vapor=tv[i],
                    p_vapor=p[i],
                    molar_mass=self.fluid.molar_mass,
                    area_contact=self.area_cond_turn * 0.5,
                )
            )

        m_dot_net = m_dot_evap + m_dot_cond

        # 3. Liquid Slug Dynamics
        p_prev = np.roll(p, 1)
        delta_p = p_prev - p

        du_dt = np.zeros(n)
        inclination_rad = math.radians(self.geo.inclination_angle)

        for i in range(n):
            slug_mass = max(1e-5, self.slug_mass_0)

            f_press = delta_p[i] * self.A_c
            f_fric = slug_friction_force(
                velocity=u[i],
                slug_length=self.l_slug_0,
                d_inner=self.D_i,
                area_c=self.A_c,
                rho_l=rho_l,
                mu_l=mu_l,
            )
            f_cap = dynamic_capillary_force(
                velocity=u[i],
                d_inner=self.D_i,
                sigma=sigma,
            )
            branch_angle_factor = math.sin((2.0 * math.pi * i) / n)
            f_grav = slug_mass * self.g * math.sin(inclination_rad) * branch_angle_factor

            # Soft restoring force near volume boundaries
            f_barrier = 0.0
            if vol_v[i] < min_vol:
                f_barrier += 1e4 * (min_vol - vol_v[i]) * self.A_c
            prev_idx = (i - 1) % n
            if vol_v[prev_idx] < min_vol:
                f_barrier -= 1e4 * (min_vol - vol_v[prev_idx]) * self.A_c

            net_force = f_press - f_fric - f_cap - f_grav + f_barrier
            du_dt[i] = net_force / slug_mass

        # 4. Vapor Plug Thermodynamics
        dp_dt = np.zeros(n)
        dtv_dt = np.zeros(n)

        for i in range(n):
            v_i = vol_v_safe[i]
            dv_dt_i = dvol_dt[i]
            p_i = p[i]
            t_i = tv[i]

            m_v_i = max(1e-7, (p_i * v_i) / (self.fluid.R_specific * t_i))

            h_v = 60.0
            q_wall_v = (
                h_v * self.area_evap_turn * 0.2 * (twe[i] - t_i) +
                h_v * self.area_cond_turn * 0.2 * (twc[i] - t_i)
            )

            term_work = -(gamma * p_i / v_i) * dv_dt_i
            term_mass_heat = ((gamma - 1.0) / v_i) * (m_dot_net[i] * h_fg + q_wall_v)
            dp_dt[i] = term_work + term_mass_heat

            dtv_dt[i] = t_i * (
                (dp_dt[i] / p_i) + (dv_dt_i / v_i) - (m_dot_net[i] / m_v_i)
            )

        # 5. Wall Thermal Nodes
        dtwe_dt = np.zeros(n)
        dtwc_dt = np.zeros(n)
        q_in_per_turn = q_in_total / n
        h_sink = 400.0

        for i in range(n):
            h_slug_e = convective_heat_transfer_coeff(
                velocity=u[i],
                d_inner=self.D_i,
                rho_l=rho_l,
                mu_l=mu_l,
                cp_l=cp_l,
            )
            q_conv_e = h_slug_e * self.area_evap_turn * 0.5 * (twe[i] - tv[i])
            q_evap_latent = m_dot_evap[i] * h_fg

            dtwe_dt[i] = (q_in_per_turn - q_conv_e - q_evap_latent) / (self.m_w_e * self.cp_w)

            h_slug_c = convective_heat_transfer_coeff(
                velocity=u[i],
                d_inner=self.D_i,
                rho_l=rho_l,
                mu_l=mu_l,
                cp_l=cp_l,
            )
            q_sink_cooling = h_sink * self.area_cond_turn * (twc[i] - self.t_sink)
            q_conv_c = h_slug_c * self.area_cond_turn * 0.5 * (tv[i] - twc[i])
            q_cond_latent = abs(m_dot_cond[i]) * h_fg

            dtwc_dt[i] = (-q_sink_cooling + q_conv_c + q_cond_latent) / (self.m_w_c * self.cp_w)

        dydt = np.concatenate([dx_dt, du_dt, dp_dt, dtv_dt, dtwe_dt, dtwc_dt])
        return dydt
