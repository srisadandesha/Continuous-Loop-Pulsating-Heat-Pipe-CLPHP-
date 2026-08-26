"""
Spacecraft Thermal Management Demo: Orbital Eclipse & Microgravity Transient.

Simulates a spacecraft payload with periodic thermal dissipation pulses
under microgravity (g = 0 m/s^2) vs launch acceleration (g = 9.81 m/s^2).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from clphp.physics.geometry import PHPGeometry
from clphp.physics.fluids import get_fluid
from clphp.simulation.solver import simulate_clphp


def main():
    print("=== Spacecraft Thermal Management: Orbital Eclipse Simulation ===")
    os.makedirs("output", exist_ok=True)

    # 1. Spacecraft-grade Ammonia/Acetone CLPHP (microgravity)
    geo = PHPGeometry(
        n_turns=4,
        d_inner=0.0018,        # 1.8 mm ID
        d_outer=0.0025,        # 2.5 mm OD (lightweight aerospace tube)
        l_evap=0.04,
        l_adiabatic=0.10,
        l_cond=0.04,
        fill_ratio=0.50,
        material="aluminum",   # Aerospace aluminum
        inclination_angle=0.0  # Zero-g neutral orientation
    )

    # Dynamic orbit payload profile: 25W standby -> 80W radar burst -> 35W cooling
    def orbit_payload_q(t: float) -> float:
        if t < 5.0:
            return 25.0  # Standby / eclipse
        elif t < 12.0:
            return 80.0  # High-rate payload burst
        else:
            return 35.0  # Post-burst cooling

    print("Simulating Case A: Microgravity flight condition (g = 0.0 m/s^2)...")
    res_zero_g = simulate_clphp(
        geometry=geo,
        fluid_name="acetone",
        q_in=orbit_payload_q,
        t_span=(0.0, 15.0),
        t_eval_step=0.05,
        t_sink_k=273.15,  # Space radiator sink temp (0 °C)
        g_accel=0.0,
    )
    print(f"Case A completed in {res_zero_g.elapsed_solve_time_s:.2f} s.")

    print("Simulating Case B: Powered flight acceleration (g = 9.81 m/s^2)...")
    res_high_g = simulate_clphp(
        geometry=geo,
        fluid_name="acetone",
        q_in=orbit_payload_q,
        t_span=(0.0, 15.0),
        t_eval_step=0.05,
        t_sink_k=273.15,
        g_accel=9.81,
    )
    print(f"Case B completed in {res_high_g.elapsed_solve_time_s:.2f} s.")

    # Generate Comparison Plot
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    t = res_zero_g.time

    # Panel 1: Evaporator Temperatures
    ax1 = axes[0]
    ax1.plot(t, res_zero_g.t_evap_mean - 273.15, "r-", lw=2, label="Microgravity ($g = 0$ m/s²)")
    ax1.plot(t, res_high_g.t_evap_mean - 273.15, "b--", lw=1.8, label="Ground / Powered Flight ($g = 9.81$ m/s²)")
    ax1.axhline(65.0, color="k", linestyle=":", label="Maximum Electronic Junction Margin (65°C)")
    ax1.set_ylabel("Evaporator Temp (°C)")
    ax1.set_title("Spacecraft CLPHP Transient Response: Orbital Power Burst & Gravity Variations", fontweight="bold")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    # Panel 2: Heat Load & Thermal Resistance
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    ax2.plot(t, res_zero_g.thermal_resistance, "g-", lw=1.8, label=r"Thermal Resistance $R_{th}$ ($g=0$)")
    ax2.plot(t, res_high_g.thermal_resistance, "c--", lw=1.8, label=r"Thermal Resistance $R_{th}$ ($g=9.81$)")
    ax2_twin.plot(t, res_zero_g.q_in, "k:", alpha=0.7, lw=2, label=r"Payload Heat Dissipation $Q_{in}(t)$")

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel(r"Thermal Resistance $R_{th}$ (K/W)")
    ax2_twin.set_ylabel("Heat Input (W)")
    ax2.grid(True)

    lines_a, labels_a = ax2.get_legend_handles_labels()
    lines_b, labels_b = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_a + lines_b, labels_a + labels_b, loc="upper right")

    plt.tight_layout()
    out_file = "output/spacecraft_eclipse_transient.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Spacecraft transient figure saved to: {out_file}")


if __name__ == "__main__":
    main()
