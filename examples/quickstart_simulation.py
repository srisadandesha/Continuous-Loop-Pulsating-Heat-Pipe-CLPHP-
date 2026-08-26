"""
Quickstart Example: Transient Simulation of a 5-Turn Continuous-Loop PHP.

Simulates an Acetone-filled CLPHP under a dynamic heat step (30W -> 80W) and
generates a comprehensive diagnostic plot.
"""

import os
from clphp.physics.geometry import PHPGeometry
from clphp.physics.fluids import get_fluid
from clphp.simulation.solver import simulate_clphp
from clphp.analysis.plotting import plot_simulation_dashboard


def main():
    print("=== CLPHP Quickstart Transient Simulation ===")

    # 1. Define 5-turn copper geometry with 2 mm ID
    geo = PHPGeometry(
        n_turns=3,
        d_inner=0.002,         # 2.0 mm
        d_outer=0.003,         # 3.0 mm
        l_evap=0.05,           # 50 mm
        l_adiabatic=0.10,      # 100 mm
        l_cond=0.05,           # 50 mm
        fill_ratio=0.50,       # 50% liquid fill
        material="copper",
        inclination_angle=90.0 # Vertical bottom-heat
    )

    fluid = get_fluid("acetone")

    # Verify capillary limit (Bond number criterion Bo <= 4)
    cap_check = geo.check_capillary_limit(
        fluid_sigma=fluid.surface_tension(293.15),
        fluid_rho_l=fluid.liquid_density(293.15),
        fluid_rho_v=fluid.vapor_density(293.15),
        g=9.81,
    )
    print(f"Capillary check: Inner D = {cap_check['d_inner_mm']:.2f} mm | Critical D = {cap_check['d_crit_mm']:.2f} mm (Bo = {cap_check['bond_number']:.2f})")
    assert cap_check["is_valid"], "Capillary limit violated! Slug-plug flow cannot form."

    # 2. Define a time-dependent heat step profile Q(t): 30W for t < 5s, then 80W
    def dynamic_q_in(t: float) -> float:
        if t < 5.0:
            return 30.0
        return 80.0

    print("Running numerical ODE solver (0 to 10s)...")
    res = simulate_clphp(
        geometry=geo,
        fluid_name="acetone",
        q_in=dynamic_q_in,
        t_span=(0.0, 10.0),
        t_eval_step=0.05,
        t_sink_k=293.15,
        g_accel=9.81,
        solver_method="RK23",
    )

    print(f"Simulation completed in {res.elapsed_solve_time_s:.2f} seconds.")
    print(f"Steady-State Thermal Resistance: {res.steady_state_r_th:.4f} K/W")
    print(f"Dominant Oscillation Frequency: {res.oscillation_frequency_hz:.2f} Hz")
    print(f"RMS Fluid Velocity: {res.oscillation_velocity_rms * 1e3:.1f} mm/s")

    os.makedirs("output", exist_ok=True)
    out_img = "output/quickstart_dashboard.png"
    plot_simulation_dashboard(res, title="Acetone CLPHP Step Response (30W -> 80W)", save_path=out_img)
    print(f"Result dashboard plotted to: {out_img}")


if __name__ == "__main__":
    main()
