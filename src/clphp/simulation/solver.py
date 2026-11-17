"""
Numerical ODE Solver for Continuous-Loop Pulsating Heat Pipe Simulations.

Integrates the coupled non-linear multi-turn ODE system using adaptive
numerical solvers (RK45, Radau, BDF, LSODA) and packages comprehensive time-series results.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, Optional, Union
import numpy as np
from scipy.integrate import solve_ivp

from clphp.physics.ode_system import ContinuousLoopPHP
from clphp.physics.fluids import FluidProperties, get_fluid
from clphp.physics.geometry import PHPGeometry


@dataclass
class SimulationResult:
    """Encapsulates the complete time-series and diagnostic results of a CLPHP simulation."""
    time: np.ndarray             # [s] (T,)
    displacements: np.ndarray    # [m] (T, N)
    velocities: np.ndarray       # [m/s] (T, N)
    pressures: np.ndarray        # [Pa] (T, N)
    vapor_temps: np.ndarray      # [K] (T, N)
    evap_wall_temps: np.ndarray  # [K] (T, N)
    cond_wall_temps: np.ndarray  # [K] (T, N)
    q_in: np.ndarray             # [W] (T,)
    
    # Summary derived metrics
    t_evap_mean: np.ndarray      # [K] (T,)
    t_cond_mean: np.ndarray      # [K] (T,)
    thermal_resistance: np.ndarray  # [K/W] (T,)
    oscillation_velocity_rms: float  # [m/s]
    oscillation_frequency_hz: float  # [Hz]
    steady_state_r_th: float     # [K/W]
    elapsed_solve_time_s: float  # Real execution time [s]
    success: bool
    status_message: str


def simulate_clphp(
    php: Optional[ContinuousLoopPHP] = None,
    geometry: Optional[PHPGeometry] = None,
    fluid_name: str = "acetone",
    q_in: Union[float, Callable[[float], float]] = 50.0,
    t_span: tuple[float, float] = (0.0, 20.0),
    t_eval_step: float = 0.05,
    solver_method: str = "RK45",
    t_sink_k: float = 293.15,
    t_init_k: float = 293.15,
    g_accel: float = 9.81,
    rtol: float = 1e-2,
    atol: float = 1e-3,
    seed: int = 42,
) -> SimulationResult:
    """
    Run high-fidelity transient numerical simulation of a CLPHP.

    Parameters:
        php: Pre-configured ContinuousLoopPHP model (optional)
        geometry: PHPGeometry instance (optional)
        fluid_name: Working fluid name ('acetone', 'ammonia', 'water', etc.)
        q_in: Heat load [W], either constant float or time-dependent callable Q(t)
        t_span: (t_start, t_end) [s]
        t_eval_step: Output sampling interval [s]
        solver_method: Scipy ODE integrator ('RK45', 'Radau', 'BDF', 'LSODA')
        t_sink_k: Heat sink temperature [K]
        t_init_k: Initial loop temperature [K]
        g_accel: Acceleration due to gravity [m/s^2] (0.0 for microgravity)
        rtol: Relative error tolerance
        atol: Absolute error tolerance
        seed: Seed for the tiny initial velocity perturbation (reproducibility)

    Returns:
        SimulationResult containing state trajectories and thermal metrics.
    """
    start_clock = time.perf_counter()

    if php is None:
        geo = geometry or PHPGeometry()
        fluid = get_fluid(fluid_name)
        php = ContinuousLoopPHP(
            geometry=geo,
            fluid=fluid,
            g_accel=g_accel,
            t_sink_k=t_sink_k,
            t_init_k=t_init_k,
            seed=seed,
        )

    n = php.n
    y0 = php.initial_state_vector()
    t_eval = np.arange(t_span[0], t_span[1] + t_eval_step, t_eval_step)

    # Dynamic or constant heat input wrapper
    if callable(q_in):
        q_func = q_in
    else:
        q_func = lambda t: float(q_in)

    def ode_func(t, y):
        q_current = q_func(t)
        return php.rhs(t, y, q_in_total=q_current)

    try:
        sol = solve_ivp(
            fun=ode_func,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method=solver_method,
            rtol=rtol,
            atol=atol,
        )
        success = sol.success
        msg = sol.message
        t_arr = sol.t
        y_arr = sol.y
    except Exception as e:
        try:
            sol = solve_ivp(
                fun=ode_func,
                t_span=t_span,
                y0=y0,
                t_eval=t_eval,
                method="RK23",
                rtol=rtol,
                atol=atol,
            )
            success = sol.success
            msg = f"Fallback RK23: {sol.message}"
            t_arr = sol.t
            y_arr = sol.y
        except Exception as e2:
            success = False
            msg = f"Integration error: {str(e2)}"
            t_arr = t_eval
            y_arr = np.repeat(y0[:, np.newaxis], len(t_eval), axis=1)

    elapsed_time = time.perf_counter() - start_clock

    # Unpack state history matrices: shape (T, N)
    y_t = y_arr.T
    displacements = y_t[:, 0:n]
    velocities = y_t[:, n:2*n]
    pressures = y_t[:, 2*n:3*n]
    vapor_temps = y_t[:, 3*n:4*n]
    evap_wall_temps = y_t[:, 4*n:5*n]
    cond_wall_temps = y_t[:, 5*n:6*n]

    q_arr = np.array([q_func(t) for t in t_arr])

    # Mean temperatures across turns
    t_evap_mean = np.mean(evap_wall_temps, axis=1)
    t_cond_mean = np.mean(cond_wall_temps, axis=1)

    # Thermal resistance R_th = (T_e,mean - T_c,mean) / Q_in
    safe_q = np.where(q_arr > 1.0, q_arr, 1.0)
    delta_t = np.maximum(0.01, t_evap_mean - t_cond_mean)
    thermal_resistance = delta_t / safe_q

    # Oscillation analysis
    steady_start_idx = int(len(t_arr) * 0.4)
    steady_velocities = velocities[steady_start_idx:, :]
    v_rms = float(np.sqrt(np.mean(steady_velocities ** 2)))

    # Compute FFT dominant frequency
    try:
        dt = float(t_arr[1] - t_arr[0]) if len(t_arr) > 1 else t_eval_step
        sig = steady_velocities[:, 0] - np.mean(steady_velocities[:, 0])
        fft_vals = np.abs(np.fft.rfft(sig))
        fft_freqs = np.fft.rfftfreq(len(sig), d=dt)
        if len(fft_vals) > 1:
            dom_idx = np.argmax(fft_vals[1:]) + 1
            dom_freq = float(fft_freqs[dom_idx])
        else:
            dom_freq = 0.0
    except Exception:
        dom_freq = 0.0

    steady_r_th = float(np.mean(thermal_resistance[steady_start_idx:]))

    return SimulationResult(
        time=t_arr,
        displacements=displacements,
        velocities=velocities,
        pressures=pressures,
        vapor_temps=vapor_temps,
        evap_wall_temps=evap_wall_temps,
        cond_wall_temps=cond_wall_temps,
        q_in=q_arr,
        t_evap_mean=t_evap_mean,
        t_cond_mean=t_cond_mean,
        thermal_resistance=thermal_resistance,
        oscillation_velocity_rms=v_rms,
        oscillation_frequency_hz=dom_freq,
        steady_state_r_th=steady_r_th,
        elapsed_solve_time_s=elapsed_time,
        success=success,
        status_message=msg,
    )
