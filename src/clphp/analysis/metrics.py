"""
Thermal, Fluid Dynamic, and Machine Learning Evaluation Metrics for CLPHPs.

Calculates:
- Effective Thermal Resistance (R_th)
- Fast Fourier Transform (FFT) Frequency Spectrum & Spectral Power
- Startup Delay Time Detection
- Surrogate vs Ground Truth Accuracy (RMSE, MAE, R^2, Inference Speedup)
"""

from __future__ import annotations
import math
from typing import Dict, Tuple, Optional
import numpy as np


def compute_thermal_resistance(
    t_evap_mean: np.ndarray,
    t_cond_mean: np.ndarray,
    q_in: np.ndarray,
    steady_fraction: float = 0.4,
) -> Dict[str, float]:
    """
    Calculate transient and steady-state thermal resistance R_th = (T_e - T_c) / Q_in [K/W].
    """
    safe_q = np.maximum(1.0, q_in)
    delta_t = np.maximum(0.01, t_evap_mean - t_cond_mean)
    r_th_transient = delta_t / safe_q

    start_idx = int(len(r_th_transient) * (1.0 - steady_fraction))
    r_th_steady = float(np.mean(r_th_transient[start_idx:]))
    r_th_std = float(np.std(r_th_transient[start_idx:]))

    return {
        "r_th_steady_mean": r_th_steady,
        "r_th_steady_std": r_th_std,
        "delta_t_mean": float(np.mean(delta_t[start_idx:])),
        "q_in_mean": float(np.mean(safe_q[start_idx:])),
    }


def compute_fft_spectrum(
    signal: np.ndarray,
    sample_interval_s: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute single-sided FFT amplitude spectrum and extract dominant frequency.

    Returns:
        freqs: Array of frequency bins [Hz]
        amplitudes: Array of FFT magnitude amplitudes
        dominant_freq_hz: Frequency corresponding to peak spectral power
        peak_power: Peak amplitude magnitude
    """
    n = len(signal)
    if n < 4:
        return np.array([0.0]), np.array([0.0]), 0.0, 0.0

    # Remove DC bias
    sig_detrend = signal - np.mean(signal)
    
    # Apply Hanning window to suppress spectral leakage
    window = np.hanning(n)
    sig_windowed = sig_detrend * window

    fft_vals = np.fft.rfft(sig_windowed)
    freqs = np.fft.rfftfreq(n, d=sample_interval_s)
    amplitudes = (2.0 / n) * np.abs(fft_vals)

    if len(amplitudes) > 1:
        peak_idx = np.argmax(amplitudes[1:]) + 1
        dom_freq = float(freqs[peak_idx])
        peak_power = float(amplitudes[peak_idx])
    else:
        dom_freq = 0.0
        peak_power = 0.0

    return freqs, amplitudes, dom_freq, peak_power


def detect_startup_time(
    time_arr: np.ndarray,
    velocities: np.ndarray,
    velocity_threshold: float = 0.005,
    consecutive_samples: int = 5,
) -> float:
    """
    Detect the time instant when pulsating oscillations spontaneously start.
    """
    # Calculate RMS velocity across turns at each time step
    v_rms_t = np.sqrt(np.mean(velocities ** 2, axis=1))
    
    count = 0
    for idx, v in enumerate(v_rms_t):
        if v >= velocity_threshold:
            count += 1
            if count >= consecutive_samples:
                start_idx = max(0, idx - consecutive_samples + 1)
                return float(time_arr[start_idx])
        else:
            count = 0

    return float(time_arr[-1])  # Did not oscillate / sluggish


def evaluate_surrogate_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute statistical regression metrics between ground truth and surrogate predictions.
    
    Parameters:
        y_true: Ground truth array of shape (N, ...)
        y_pred: Predicted array of shape (N, ...)
    """
    y_t = y_true.flatten()
    y_p = y_pred.flatten()

    diff = y_p - y_t
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    # R^2 score
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    ss_res = np.sum(diff ** 2)
    r2 = float(1.0 - (ss_res / max(1e-8, ss_tot)))

    # Mean Absolute Percentage Error
    non_zero_mask = np.abs(y_t) > 1e-4
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs(diff[non_zero_mask] / y_t[non_zero_mask])) * 100.0)
    else:
        mape = 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2,
        "mape_pct": mape,
    }
