"""
Held-out evaluation of CLPHP surrogates.

All metrics are computed on whole simulation runs that were never used for
training or validation, per output channel and in physical units
(T_e [K], T_c [K], R_th [K/W]) -- never pooled across channels.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch

from clphp.simulation.dataset_generator import create_sliding_windows

CHANNELS = ("T_e", "T_c", "R_th")
UNITS = {"T_e": "K", "T_c": "K", "R_th": "K/W"}

# Feature column indices (see dataset_generator): Q_in, T_sink, fill, g, time, T_e, T_c
COL_Q, COL_TE, COL_TC = 0, 5, 6


def per_channel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """R^2, RMSE, MAE for each channel of arrays shaped (..., 3), in physical units."""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1, len(CHANNELS))
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1, len(CHANNELS))
    out: Dict[str, Dict[str, float]] = {}
    for i, ch in enumerate(CHANNELS):
        d = yp[:, i] - yt[:, i]
        ss_tot = float(np.sum((yt[:, i] - yt[:, i].mean()) ** 2))
        out[ch] = {
            "r2": float(1.0 - np.sum(d ** 2) / ss_tot) if ss_tot > 1e-12 else float("nan"),
            "rmse": float(np.sqrt(np.mean(d ** 2))),
            "mae": float(np.mean(np.abs(d))),
            "unit": UNITS[ch],
        }
    return out


def denorm_targets(y_norm: np.ndarray, norm: Dict[str, np.ndarray]) -> np.ndarray:
    return y_norm * norm["targ_std"] + norm["targ_mean"]


def denorm_features(x_norm: np.ndarray, norm: Dict[str, np.ndarray]) -> np.ndarray:
    return x_norm * norm["feat_std"] + norm["feat_mean"]


@torch.no_grad()
def predict_windows(model, X: np.ndarray, norm: Dict[str, np.ndarray], batch_size: int = 1024) -> np.ndarray:
    """Surrogate predictions for normalized windows, returned in physical units (N, H, 3)."""
    model.eval()
    dev = next(model.parameters()).device
    outs = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=dev)
        outs.append(model(xb).cpu().numpy())
    return denorm_targets(np.concatenate(outs, axis=0), norm)


def evaluate_one_step(
    model,
    runs: List[Dict[str, Any]],
    norm: Dict[str, np.ndarray],
    seq_len: int,
    pred_horizon: int,
) -> Dict[str, Any]:
    """
    One-step-ahead evaluation: every window of every held-out run (stride 1) is
    fed *true* history, and the first predicted step is compared with truth.
    """
    X, y, = create_sliding_windows(runs, seq_len, pred_horizon, stride=1, norm_params=norm)[:2]
    y_true = denorm_targets(y, norm)
    y_pred = predict_windows(model, X, norm)
    return {
        "n_windows": int(len(X)),
        "surrogate": per_channel_metrics(y_true[:, 0], y_pred[:, 0]),
    }
