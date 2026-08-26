"""
Thermal analysis and metrics module for Pulsating Heat Pipes.
"""

from clphp.analysis.metrics import (
    compute_thermal_resistance,
    compute_fft_spectrum,
    detect_startup_time,
    evaluate_surrogate_accuracy,
)

__all__ = [
    "compute_thermal_resistance",
    "compute_fft_spectrum",
    "detect_startup_time",
    "evaluate_surrogate_accuracy",
]
