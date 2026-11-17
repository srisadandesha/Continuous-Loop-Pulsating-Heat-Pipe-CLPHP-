"""
End-to-end, honest evaluation of the CLPHP LSTM surrogate.

1. Generate (or load) a parametric ODE dataset (fixed seed, Radau rtol=1e-4/atol=1e-6).
2. Split WHOLE RUNS by Q_in into train / val / test-interp / test-extrap.
3. Fit normalization on the train runs only.
4. Train the surrogate (with physics loss), evaluate on held-out test runs:
   one-step and autoregressive rollout, per channel, physical units, next to
   a persistence baseline.
5. Benchmark solve_ivp vs surrogate rollout (batch=1 and batched).
6. Ablation: with vs without physics terms, several seeds.

Usage:  python examples/run_experiment.py --profile full --jobs 12
Outputs are written to results/ (text, markdown tables and JSON).
"""

import argparse
import json
import os
import time

import numpy as np

from clphp.analysis.benchmark import benchmark_speed, format_benchmark
from clphp.models.ablation import format_ablation, run_physics_ablation
from clphp.models.evaluation import report_test_runs
from clphp.models.trainer import train_surrogate
from clphp.simulation.dataset_generator import (
    DatasetConfig,
    build_split_windows,
    generate_parametric_dataset,
    runs_from_npz,
)

PROFILES = {
    # Fast smoke profile (minutes): small grid, short runs.
    "quick": dict(
        q_in_values=None, fill_ratios=[0.50], g_accels=[9.81], sink_temps=[293.15],
        t_sim_end=2.0, t_step=0.05, n_turns=3,
    ),
    # Enlarged grid (see README for the runtime it needs).
    "full": dict(
        q_in_values=None, fill_ratios=[0.40, 0.50, 0.60], g_accels=[0.0, 9.81], sink_temps=[288.15, 298.15],
        t_sim_end=4.0, t_step=0.05, n_turns=3,
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES), default="quick")
    ap.add_argument("--dataset", default=None, help="Reuse/save .npz dataset at this path")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel simulation workers")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = DatasetConfig(**PROFILES[args.profile])
    t0 = time.perf_counter()
    if args.dataset and os.path.exists(args.dataset):
        runs = runs_from_npz(args.dataset)
        print(f"Loaded {len(runs)} runs from {args.dataset}")
    else:
        runs = generate_parametric_dataset(cfg, output_path=args.dataset, n_jobs=args.jobs)["runs"]
    t_data = time.perf_counter() - t0

    splits, windows, norm = build_split_windows(runs, seq_len=args.seq_len, pred_horizon=args.horizon, stride=2)
    split_summary = {k: sorted({r["q_in"] for r in v}) for k, v in splits.items()}
    print("Run-level split (Q_in, W):", split_summary)
    print({k: len(v[0]) for k, v in windows.items()}, "windows")

    # --- main model -------------------------------------------------------
    X_tr, y_tr = windows["train"]
    X_va, y_va = windows["val"]
    model, hist = train_surrogate(
        X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va, norm_params=norm,
        epochs=args.epochs, hidden_dim=args.hidden, seed=args.seeds[0], verbose=True,
    )
    report = report_test_runs(model, splits, norm, args.seq_len, args.horizon)
    print("\nHeld-out test runs:\n" + report)

    # --- benchmark ----------------------------------------------------------
    test_runs = splits["test_interp"] + splits["test_extrap"]
    bench = benchmark_speed(model, test_runs, norm, sim=cfg.sim_settings(),
                            seq_len=args.seq_len, pred_horizon=args.horizon, n_solver_runs=2)
    print("\nBenchmark:\n" + format_benchmark(bench))

    # --- ablation -------------------------------------------------------------
    abl = run_physics_ablation(
        windows, splits, norm, args.seq_len, args.horizon, seeds=args.seeds,
        rollout_horizons=(20, None), epochs=args.epochs, hidden_dim=args.hidden,
    )
    table = format_ablation(abl)
    print("\nPhysics ablation (mean +/- std over seeds):\n" + table)

    with open(os.path.join(args.out, "test_report.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(args.out, "benchmark.txt"), "w") as f:
        f.write(format_benchmark(bench) + "\n")
    with open(os.path.join(args.out, "ablation.md"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({
            "profile": args.profile, "config": cfg.sim_settings(), "n_runs": len(runs),
            "split_q_in": split_summary, "dataset_generation_s": t_data,
            "benchmark": bench, "ablation": abl, "seeds": args.seeds,
        }, f, indent=2, default=float)
    print(f"\nWrote results to {args.out}/")


if __name__ == "__main__":
    main()
