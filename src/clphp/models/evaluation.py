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


def physical_r_th(t_e, t_c, q_in):
    """R_th = (T_e - T_c) / Q_in, with the same clamps the simulator uses."""
    return np.maximum(0.01, t_e - t_c) / np.maximum(1.0, q_in)


def persistence_forecast(X_norm: np.ndarray, norm: Dict[str, np.ndarray], pred_horizon: int) -> np.ndarray:
    """
    Persistence baseline: every future value equals the last observed value.

    T_e and T_c are taken from the last input row; R_th is the value implied by
    those last observed temperatures. Returns physical units, shape (N, H, 3).
    """
    last = denorm_features(X_norm[:, -1, :], norm)
    t_e, t_c = last[:, COL_TE], last[:, COL_TC]
    base = np.stack([t_e, t_c, physical_r_th(t_e, t_c, last[:, COL_Q])], axis=-1)
    return np.repeat(base[:, None, :], pred_horizon, axis=1)


def format_comparison(title: str, results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    """Pretty table of surrogate vs persistence for each channel."""
    lines = [title]
    for ch in CHANNELS:
        for who in ("surrogate", "persistence"):
            m = results[who][ch]
            lines.append(
                f"   {ch:5s} {who:11s} R2={m['r2']:8.4f}  RMSE={m['rmse']:.4g} {m['unit']}  MAE={m['mae']:.4g} {m['unit']}"
            )
    return "\n".join(lines)


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
    y_base = persistence_forecast(X, norm, pred_horizon)
    return {
        "n_windows": int(len(X)),
        "surrogate": per_channel_metrics(y_true[:, 0], y_pred[:, 0]),
        "persistence": per_channel_metrics(y_true[:, 0], y_base[:, 0]),
        # Whole forecast horizon (teacher-forced), for reference
        "surrogate_horizon": per_channel_metrics(y_true, y_pred),
        "persistence_horizon": per_channel_metrics(y_true, y_base),
    }


@torch.no_grad()
def rollout_batch(
    model,
    feats_true: np.ndarray,
    norm: Dict[str, np.ndarray],
    seq_len: int,
    pred_horizon: int,
) -> np.ndarray:
    """
    Multi-step autoregressive rollout, a batch of equal-length runs at once.

    The model sees the first ``seq_len`` TRUE rows of each run, then predicts
    ``pred_horizon`` steps; those predictions (T_e, T_c) are written back into
    the input window and the loop repeats until the end of the run. Exogenous
    inputs (Q_in, T_sink, fill ratio, g, time) are the known schedule and are
    taken from the run; no ground-truth T_e / T_c is used after the seed window.

    Parameters:
        feats_true: physical-unit features, shape (B, T, F)
    Returns:
        physical predictions (B, T - seq_len, 3) = [T_e, T_c, R_th]
    """
    model.eval()
    dev = next(model.parameters()).device
    B, T, _ = feats_true.shape
    n_total = T - seq_len
    if n_total <= 0:
        raise ValueError("Runs must be longer than seq_len")
    preds = np.empty((B, n_total, len(CHANNELS)))
    window = feats_true[:, :seq_len].copy()
    pos = 0
    while pos < n_total:
        x = torch.as_tensor((window - norm["feat_mean"]) / norm["feat_std"], dtype=torch.float32, device=dev)
        out = denorm_targets(model(x).cpu().numpy(), norm)  # (B, H, 3)
        k = min(pred_horizon, n_total - pos)
        preds[:, pos:pos + k] = out[:, :k]
        new = feats_true[:, seq_len + pos: seq_len + pos + k].copy()
        new[:, :, COL_TE] = out[:, :k, 0]  # feed predictions back
        new[:, :, COL_TC] = out[:, :k, 1]
        window = np.concatenate([window, new], axis=1)[:, -seq_len:]
        pos += k
    return preds


def persistence_rollout(feats_true: np.ndarray, seq_len: int) -> np.ndarray:
    """Hold the last observed seed-window values for the entire rollout (B, T-seq_len, 3)."""
    last = feats_true[:, seq_len - 1]
    base = np.stack([last[:, COL_TE], last[:, COL_TC],
                     physical_r_th(last[:, COL_TE], last[:, COL_TC], last[:, COL_Q])], axis=-1)
    return np.repeat(base[:, None, :], feats_true.shape[1] - seq_len, axis=1)


def evaluate_rollout(
    model,
    runs: List[Dict[str, Any]],
    norm: Dict[str, np.ndarray],
    seq_len: int,
    pred_horizon: int,
    horizons: Sequence[Optional[int]] = (20, 100, None),
) -> Dict[str, Any]:
    """
    Autoregressive rollout on whole held-out runs vs. the persistence baseline.

    ``horizons`` are numbers of rolled-out steps over which errors are scored
    (``None`` = the whole remaining run). Metrics are per channel, physical units.
    """
    by_len: Dict[int, List[Dict[str, Any]]] = {}
    for r in runs:
        by_len.setdefault(len(r["features"]), []).append(r)

    truth, sur, base = [], [], []
    for T, group in by_len.items():
        feats = np.stack([r["features"] for r in group])
        truth.append(np.stack([r["targets"][seq_len:] for r in group]))
        sur.append(rollout_batch(model, feats, norm, seq_len, pred_horizon))
        base.append(persistence_rollout(feats, seq_len))

    out: Dict[str, Any] = {"n_runs": len(runs), "horizons": {}}
    n_avail = min(t.shape[1] for t in truth)
    # A horizon at or beyond the available rollout length is the same as 'full': skip duplicates
    horizons = [h for h in horizons if h is None or h < n_avail]
    if None not in horizons:
        horizons.append(None)
    for h in horizons:
        label = "full" if h is None else str(h)
        parts = [(t[:, :h], s[:, :h], b[:, :h]) for t, s, b in zip(truth, sur, base)]
        yt = np.concatenate([p[0].reshape(-1, 3) for p in parts])
        ys = np.concatenate([p[1].reshape(-1, 3) for p in parts])
        yb = np.concatenate([p[2].reshape(-1, 3) for p in parts])
        out["horizons"][label] = {
            "steps": int(min(t.shape[1] if h is None else min(h, t.shape[1]) for t in truth)),
            "surrogate": per_channel_metrics(yt, ys),
            "persistence": per_channel_metrics(yt, yb),
        }
    return out


def report_test_runs(
    model,
    splits: Dict[str, List[Dict[str, Any]]],
    norm: Dict[str, np.ndarray],
    seq_len: int,
    pred_horizon: int,
    rollout_horizons: Sequence[Optional[int]] = (20, 100, None),
) -> str:
    """Human-readable held-out evaluation of every ``test_*`` split (one-step + rollout vs persistence)."""
    lines = []
    for name in sorted(k for k in splits if k.startswith("test_") and splits[k]):
        qs = sorted({r["q_in"] for r in splits[name]})
        one = evaluate_one_step(model, splits[name], norm, seq_len, pred_horizon)
        lines.append(f"-> {name}: {len(splits[name])} held-out runs (Q_in={qs} W), {one['n_windows']} windows")
        lines.append(format_comparison("   one-step ahead (true history in, step +1 out):", one))
        roll = evaluate_rollout(model, splits[name], norm, seq_len, pred_horizon, horizons=rollout_horizons)
        for label, h in roll["horizons"].items():
            lines.append(format_comparison(
                f"   autoregressive rollout, {h['steps']} steps (predictions fed back):", h))
    return "\n".join(lines)
