"""
Physics-term ablation: train with and without the physics penalties, compare
held-out test metrics.

Both variants share the same data, split, architecture, optimizer, epochs and
seeds; the only difference is whether the physics loss weights are on. Each
variant is trained once per seed so the report can show mean +/- std over seeds
(a single seed cannot distinguish an effect from noise).
"""

from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from clphp.models.evaluation import (
    CHANNELS, evaluate_one_step, evaluate_rollout,
)
from clphp.models.trainer import train_surrogate

VARIANTS = {"with_physics": True, "data_only": False}


def run_physics_ablation(
    windows: Dict[str, Tuple[np.ndarray, np.ndarray]],
    splits: Dict[str, List[Dict[str, Any]]],
    norm_params: Dict[str, np.ndarray],
    seq_len: int,
    pred_horizon: int,
    seeds: Sequence[int] = (0, 1, 2),
    test_names: Sequence[str] = ("test_interp", "test_extrap"),
    rollout_horizons: Sequence = (20, None),
    **train_kwargs: Any,
) -> Dict[str, Any]:
    """
    Returns ``results[variant][test_name]`` -> {"one_step": {ch: {metric: [per-seed values]}},
    "rollout": {horizon_label: {ch: {metric: [...]}}}}.
    """
    X_tr, y_tr = windows["train"]
    X_va, y_va = windows["val"]
    results: Dict[str, Any] = {v: {n: {"one_step": {}, "rollout": {}} for n in test_names} for v in VARIANTS}

    def _push(store: Dict, metrics: Dict[str, Dict[str, float]]):
        for ch in CHANNELS:
            for k in ("r2", "rmse", "mae"):
                store.setdefault(ch, {}).setdefault(k, []).append(metrics[ch][k])

    for variant, use_physics in VARIANTS.items():
        for seed in seeds:
            model, _ = train_surrogate(
                X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va, norm_params=norm_params,
                use_physics=use_physics, seed=seed, verbose=False, **train_kwargs,
            )
            for name in test_names:
                one = evaluate_one_step(model, splits[name], norm_params, seq_len, pred_horizon)
                _push(results[variant][name]["one_step"], one["surrogate"])
                roll = evaluate_rollout(model, splits[name], norm_params, seq_len, pred_horizon,
                                        horizons=rollout_horizons)
                for label, h in roll["horizons"].items():
                    _push(results[variant][name]["rollout"].setdefault(label, {}), h["surrogate"])
    return results


def _ms(values: List[float]) -> str:
    v = np.asarray(values, dtype=float)
    return f"{np.nanmean(v):.4g} +/- {np.nanstd(v):.2g}"


def format_ablation(results: Dict[str, Any]) -> str:
    """Markdown table: mean +/- std over seeds, RMSE in physical units and R2, per channel."""
    lines = ["| test set | evaluation | channel | variant | RMSE (mean +/- std) | R2 (mean +/- std) |",
             "|---|---|---|---|---|---|"]
    units = {"T_e": "K", "T_c": "K", "R_th": "K/W"}
    names = next(iter(results.values())).keys()
    for name in names:
        evals = [("one-step", lambda r, n=name: r[n]["one_step"])]
        for label in results["with_physics"][name]["rollout"]:
            evals.append((f"rollout {label}", lambda r, n=name, l=label: r[n]["rollout"][l]))
        for ev_name, getter in evals:
            for ch in CHANNELS:
                for variant in VARIANTS:
                    m = getter(results[variant])[ch]
                    lines.append(f"| {name} | {ev_name} | {ch} | {variant} | "
                                 f"{_ms(m['rmse'])} {units[ch]} | {_ms(m['r2'])} |")
    return "\n".join(lines)
