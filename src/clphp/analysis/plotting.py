"""
Publication-Quality Plotting and Diagnostic Visualizations for CLPHP Systems.

Generates:
- 4-Panel Transient Simulation Dashboard (T(t), u(t), Phase Space, FFT Spectrum)
- P-V Dynamic Thermodynamic Indicator Diagrams
- Ground Truth ODE vs. PI-RNN Neural Surrogate Comparison
- Training History Curves
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from clphp.simulation.solver import SimulationResult
from clphp.analysis.metrics import compute_fft_spectrum


# Set scientific styling
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "lines.linewidth": 1.5,
    "grid.alpha": 0.3,
})


def plot_simulation_dashboard(
    res: SimulationResult,
    title: str = "Continuous-Loop Pulsating Heat Pipe Transient Response",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Generate a 4-panel comprehensive diagnostic dashboard.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontweight="bold", y=0.98)

    t = res.time
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.05

    # Panel 1: Transient Temperatures & Heat Input
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()

    ax1.plot(t, res.t_evap_mean - 273.15, "r-", label=r"Evaporator $T_{e,avg}$ (°C)")
    ax1.plot(t, res.t_cond_mean - 273.15, "b-", label=r"Condenser $T_{c,avg}$ (°C)")
    ax1_twin.plot(t, res.q_in, "k--", alpha=0.6, label=r"$Q_{in}$ (W)")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Temperature (°C)")
    ax1_twin.set_ylabel("Heat Input (W)", color="k")
    ax1.set_title("(a) Evaporator & Condenser Thermal Response")
    ax1.grid(True)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Panel 2: Liquid Slug Oscillatory Velocities
    ax2 = axes[0, 1]
    n_slugs = res.velocities.shape[1]
    for i in range(min(4, n_slugs)):
        ax2.plot(t, res.velocities[:, i], alpha=0.8, label=f"Slug {i+1}")

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Axial Velocity (m/s)")
    ax2.set_title(f"(b) Liquid Slug Velocities (RMS: {res.oscillation_velocity_rms*1e3:.1f} mm/s)")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    # Panel 3: Phase Space Portrait (u vs x)
    ax3 = axes[1, 0]
    mid_idx = int(len(t) * 0.2)
    slug_idx = 0
    x_raw = res.displacements[mid_idx:, slug_idx]
    x_plot = (x_raw - x_raw[0]) * 1e3  # relative mm
    u_plot = res.velocities[mid_idx:, slug_idx]

    ax3.plot(x_plot, u_plot, color="darkviolet", alpha=0.85, lw=1.2)
    ax3.set_xlabel("Relative Displacement Δx (mm)")
    ax3.set_ylabel("Velocity u (m/s)")
    ax3.set_title("(c) Phase Space Orbit (Limit Cycle Trajectory)")
    ax3.grid(True)

    # Panel 4: FFT Spectral Power Spectrum
    ax4 = axes[1, 1]
    steady_v = res.velocities[mid_idx:, 0]
    freqs, amps, dom_freq, peak_amp = compute_fft_spectrum(steady_v, dt)

    mask = (freqs >= 0.1) & (freqs <= 15.0)
    ax4.plot(freqs[mask], amps[mask], "g-", lw=1.6)
    if dom_freq > 0:
        ax4.axvline(dom_freq, color="r", linestyle=":", label=f"Peak: {dom_freq:.2f} Hz")

    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Amplitude Spectrum")
    ax4.set_title(f"(d) FFT Frequency Spectrum (Dominant: {res.oscillation_frequency_hz:.2f} Hz)")
    ax4.grid(True)
    ax4.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_surrogate_comparison(
    time_arr: np.ndarray,
    ground_truth: np.ndarray,
    surrogate_preds: np.ndarray,
    q_in_arr: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Compare Ground Truth ODE trajectories with PI-RNN Surrogate forecasts.
    
    Parameters:
        time_arr: Time array [s]
        ground_truth: (T, 3) -> [T_e, T_c, R_th]
        surrogate_preds: (T, 3) -> [T_e, T_c, R_th]
    """
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("Ground Truth ODE vs. Physics-Informed RNN Surrogate", fontweight="bold", y=0.98)

    # 1. Evaporator Temp
    ax1 = axes[0]
    ax1.plot(time_arr, ground_truth[:, 0] - 273.15, "k-", lw=2, label="Ground Truth ODE")
    ax1.plot(time_arr, surrogate_preds[:, 0] - 273.15, "r--", lw=1.8, label="PI-RNN Prediction")
    ax1.set_ylabel("Evaporator T (°C)")
    ax1.grid(True)
    ax1.legend(loc="upper left")
    ax1.set_title("(a) Evaporator Temperature Profile")

    # 2. Condenser Temp
    ax2 = axes[1]
    ax2.plot(time_arr, ground_truth[:, 1] - 273.15, "k-", lw=2, label="Ground Truth ODE")
    ax2.plot(time_arr, surrogate_preds[:, 1] - 273.15, "b--", lw=1.8, label="PI-RNN Prediction")
    ax2.set_ylabel("Condenser T (°C)")
    ax2.grid(True)
    ax2.legend(loc="upper left")
    ax2.set_title("(b) Condenser Temperature Profile")

    # 3. Thermal Resistance & Residual Error
    ax3 = axes[2]
    ax3.plot(time_arr, ground_truth[:, 2], "k-", lw=2, label=r"Ground Truth $R_{th}$")
    ax3.plot(time_arr, surrogate_preds[:, 2], "m--", lw=1.8, label=r"PI-RNN $R_{th}$")
    ax3.set_ylabel(r"$R_{th}$ (K/W)")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True)
    ax3.legend(loc="upper right")
    ax3.set_title(r"(c) Instantaneous Thermal Resistance $R_{th} = \Delta T / Q_{in}$")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_training_history(
    history: Dict[str, list],
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot training and validation loss curves over epochs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax.semilogy(epochs, history["train_loss"], "b-", label="Train Loss (Total PINN)")
    ax.semilogy(epochs, history["val_loss"], "r--", label="Val Loss (Total PINN)")
    if "train_data_loss" in history:
        ax.semilogy(epochs, history["train_data_loss"], "g:", label="Train Data MSE")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Log Scale)")
    ax.set_title("Physics-Informed Recurrent Neural Network Training Convergence")
    ax.grid(True, which="both", linestyle="--")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history saved to: {save_path}")

    if show:
        plt.show()

    return fig
