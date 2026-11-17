"""
Wall-clock benchmark: ODE solver vs. surrogate rollout.

A like-for-like comparison: for the same held-out test runs and the same
simulated duration, on the same machine, we time

  * ``simulate_clphp`` (scipy ``solve_ivp`` on the lumped ODE system), and
  * the surrogate's autoregressive rollout, with batch size 1 (one run at a
    time) and batched (all runs stacked into one tensor), reported separately.

The surrogate is told the true first ``seq_len`` samples of each run (its seed
window), so it rolls out ``T - seq_len`` steps; the solver integrates the whole
``T`` steps from t=0. Both durations are reported so the comparison is explicit.

``real_time_factor`` = simulated seconds per wall-clock second. It is NOT a
speedup over the solver; the speedup is ``solver_wall / surrogate_wall`` for the
same runs.
"""

from __future__ import annotations
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from clphp.models.evaluation import rollout_batch
from clphp.physics.geometry import PHPGeometry
from clphp.simulation.solver import simulate_clphp


def _median_time(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def time_solver(run: Dict[str, Any], sim: Dict[str, Any]) -> float:
    """Wall-clock seconds for solve_ivp on one run (same settings used to generate the data)."""
    geo = PHPGeometry(n_turns=sim["n_turns"], d_inner=sim["d_inner"], fill_ratio=run["fill_ratio"])
    t0 = time.perf_counter()
    simulate_clphp(
        geometry=geo,
        fluid_name=run.get("fluid", sim.get("fluid", "acetone")),
        q_in=run["q_in"],
        t_span=(0.0, sim["t_sim_end"]),
        t_eval_step=sim["t_step"],
        t_sink_k=run["t_sink"],
        g_accel=run["g_accel"],
        solver_method=sim["solver_method"],
        rtol=sim["rtol"],
        atol=sim["atol"],
    )
    return time.perf_counter() - t0


def benchmark_speed(
    model,
    runs: List[Dict[str, Any]],
    norm: Dict[str, np.ndarray],
    sim: Dict[str, Any],
    seq_len: int,
    pred_horizon: int,
    n_solver_runs: int = 2,
    repeats: int = 5,
) -> Dict[str, Any]:
    """
    Benchmark solver vs surrogate on ``runs`` (all must have the same length).

    ``sim`` carries the generation settings: n_turns, d_inner, t_sim_end, t_step,
    solver_method, rtol, atol (and optionally fluid).
    Returns per-run wall times, speedups and real-time factors.
    """
    model = model.to("cpu").eval()  # same machine, same device class as the solver
    feats = np.stack([r["features"] for r in runs])
    n_runs, T, _ = feats.shape
    sim_duration = sim["t_sim_end"]
    rolled_duration = (T - seq_len) * sim["t_step"]

    solver_times = [time_solver(r, sim) for r in runs[:n_solver_runs]]
    solver_per_run = float(np.mean(solver_times))

    rollout_batch(model, feats[:1], norm, seq_len, pred_horizon)  # warm-up
    t_bs1 = _median_time(lambda: rollout_batch(model, feats[:1], norm, seq_len, pred_horizon), repeats)
    t_batch = _median_time(lambda: rollout_batch(model, feats, norm, seq_len, pred_horizon), repeats)

    # Single forward pass latency (what the old "1000 / latency" figure measured)
    x1 = torch.zeros(1, seq_len, feats.shape[-1])
    with torch.no_grad():
        model(x1)
        t_fwd = _median_time(lambda: model(x1), max(repeats, 20))

    return {
        "n_runs": n_runs,
        "simulated_duration_s": sim_duration,
        "surrogate_rolled_duration_s": rolled_duration,
        "solver": {
            "method": sim["solver_method"], "rtol": sim["rtol"], "atol": sim["atol"],
            "n_timed_runs": len(solver_times),
            "wall_s_per_run": solver_per_run,
            "real_time_factor": sim_duration / solver_per_run,
        },
        "surrogate_batch1": {
            "wall_s_per_run": t_bs1,
            "speedup_vs_solver": solver_per_run / t_bs1,
            "real_time_factor": rolled_duration / t_bs1,
        },
        "surrogate_batched": {
            "batch_size": n_runs,
            "wall_s_total": t_batch,
            "wall_s_per_run": t_batch / n_runs,
            "speedup_vs_solver": solver_per_run / (t_batch / n_runs),
            "real_time_factor": rolled_duration * n_runs / t_batch,
        },
        "single_forward": {
            "latency_ms": t_fwd * 1e3,
            # The former headline "1000 / latency_ms": forward passes per second.
            "real_time_factor_legacy": 1000.0 / (t_fwd * 1e3),
        },
    }


def format_benchmark(b: Dict[str, Any]) -> str:
    s, b1, bn, f = b["solver"], b["surrogate_batch1"], b["surrogate_batched"], b["single_forward"]
    return "\n".join([
        f"   Solver ({s['method']}, rtol={s['rtol']:g}, atol={s['atol']:g}): "
        f"{s['wall_s_per_run']:.2f} s/run for {b['simulated_duration_s']:g} s simulated "
        f"(real-time factor {s['real_time_factor']:.2f}x, {s['n_timed_runs']} runs timed)",
        f"   Surrogate rollout, batch=1:  {b1['wall_s_per_run'] * 1e3:.2f} ms/run -> speedup {b1['speedup_vs_solver']:.0f}x "
        f"(real-time factor {b1['real_time_factor']:.0f}x)",
        f"   Surrogate rollout, batched ({bn['batch_size']} runs): {bn['wall_s_per_run'] * 1e3:.2f} ms/run -> "
        f"speedup {bn['speedup_vs_solver']:.0f}x (real-time factor {bn['real_time_factor']:.0f}x)",
        f"   Single forward pass: {f['latency_ms']:.3f} ms  =>  legacy 'real-time factor' 1000/latency = "
        f"{f['real_time_factor_legacy']:.0f} (inference rate, not a speedup over the solver)",
    ])
