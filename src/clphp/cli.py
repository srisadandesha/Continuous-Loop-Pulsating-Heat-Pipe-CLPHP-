"""
Command-Line Interface (CLI) for the Continuous-Loop Pulsating Heat Pipe (CLPHP) Framework.
"""

from __future__ import annotations
import argparse
import sys
import os
import numpy as np

from clphp.physics.geometry import PHPGeometry
from clphp.physics.fluids import get_fluid
from clphp.simulation.solver import simulate_clphp
from clphp.simulation.dataset_generator import (
    generate_parametric_dataset,
    build_split_windows,
    runs_from_npz,
    DatasetConfig,
)
from clphp.models.trainer import train_surrogate, SurrogateTrainer
from clphp.analysis.plotting import (
    plot_simulation_dashboard,
    plot_surrogate_comparison,
    plot_training_history,
)
from clphp.models.evaluation import evaluate_one_step, evaluate_rollout, format_comparison
from clphp.analysis.benchmark import benchmark_speed, format_benchmark


def cmd_simulate(args: argparse.Namespace):
    """Execute a single high-fidelity ODE simulation."""
    print(f"\n--- Running CLPHP Simulation ---")
    print(f"Fluid: {args.fluid} | Heat Input: {args.qin} W | Turns: {args.turns} | Time: {args.time} s | g: {args.g} m/s^2")

    geo = PHPGeometry(
        n_turns=args.turns,
        d_inner=args.d_inner * 1e-3,
        fill_ratio=args.fill_ratio,
        inclination_angle=args.inclination,
    )

    res = simulate_clphp(
        geometry=geo,
        fluid_name=args.fluid,
        q_in=args.qin,
        t_span=(0.0, args.time),
        t_eval_step=args.dt,
        t_sink_k=args.tsink + 273.15,
        g_accel=args.g,
    )

    print("\n--- Simulation Summary ---")
    print(f"Status: {'Success' if res.success else 'Failed'} ({res.status_message})")
    print(f"Elapsed Solver Time: {res.elapsed_solve_time_s:.2f} s")
    print(f"Steady Evaporator Temp: {np.mean(res.t_evap_mean[-50:]) - 273.15:.2f} °C")
    print(f"Steady Condenser Temp:  {np.mean(res.t_cond_mean[-50:]) - 273.15:.2f} °C")
    print(f"Steady Thermal Resistance: {res.steady_state_r_th:.4f} K/W")
    print(f"Dominant Oscillation Frequency: {res.oscillation_frequency_hz:.2f} Hz")
    print(f"Oscillation RMS Velocity: {res.oscillation_velocity_rms * 1e3:.2f} mm/s")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        plot_simulation_dashboard(res, save_path=args.output)


def cmd_generate_data(args: argparse.Namespace):
    """Generate synthetic dataset across parametric grid."""
    print("\n--- Generating CLPHP Parametric Dataset ---")
    q_vals = [float(x) for x in args.qin_list.split(",")]
    fr_vals = [float(x) for x in args.fr_list.split(",")]
    g_vals = [float(x) for x in args.g_list.split(",")]
    fluids = [x.strip() for x in args.fluids.split(",")]

    cfg = DatasetConfig(
        q_in_values=q_vals,
        fill_ratios=fr_vals,
        g_accels=g_vals,
        fluids=fluids,
        t_sim_end=args.time,
        t_step=args.dt,
        n_turns=args.turns,
    )

    dataset = generate_parametric_dataset(
        config=cfg,
        output_path=args.output,
        verbose=True,
    )
    print(f"Successfully generated {dataset['n_runs']} simulation runs.")


def cmd_train(args: argparse.Namespace):
    """Train PI-RNN surrogate on dataset."""
    print("\n--- Training Physics-Informed RNN Surrogate ---")
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file '{args.dataset}' does not exist.")
        sys.exit(1)

    raw_runs = runs_from_npz(args.dataset)
    splits, windows, norm_params = build_split_windows(
        raw_runs,
        seq_len=args.seq_len,
        pred_horizon=args.horizon,
        stride=args.stride,
    )
    X_tr, y_tr = windows["train"]
    X_va, y_va = windows["val"]
    print("Run-level split (whole runs, never windows): " + ", ".join(
        f"{k}={len(v)} runs" for k, v in splits.items()))
    print(f"Train windows: {len(X_tr)} | Val windows: {len(X_va)}")

    model, history = train_surrogate(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        norm_params=norm_params,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        rnn_type=args.rnn_type,
        lr=args.lr,
        save_path=args.save_model,
        verbose=True,
    )

    if args.plot_loss:
        os.makedirs(os.path.dirname(os.path.abspath(args.plot_loss)), exist_ok=True)
        plot_training_history(history, save_path=args.plot_loss)


def cmd_demo(args: argparse.Namespace):
    """Execute end-to-end interactive demonstration."""
    print("=================================================================")
    print("  CLPHP Framework: End-to-End Physics Simulation & PI-RNN Demo   ")
    print("=================================================================\n")

    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Physics Simulation
    print("Step 1: Running High-Fidelity Multi-Turn CLPHP ODE Simulation...")
    geo = PHPGeometry(n_turns=3, d_inner=0.002, fill_ratio=0.50)
    res = simulate_clphp(geometry=geo, fluid_name="acetone", q_in=60.0, t_span=(0.0, 5.0), t_eval_step=0.1, solver_method="RK23")
    
    dash_path = os.path.join(out_dir, "simulation_dashboard.png")
    plot_simulation_dashboard(res, save_path=dash_path)
    print(f"-> Simulation completed. Dashboard saved to {dash_path}")

    # 2. Quick Dataset Generation
    print("\nStep 2: Generating Compact Parametric Dataset for Training...")
    cfg = DatasetConfig(
        q_in_values=[20.0, 30.0, 45.0, 60.0, 75.0, 90.0, 110.0],
        fill_ratios=[0.50],
        g_accels=[0.0, 9.81],
        fluids=["acetone"],
        t_sim_end=4.0,
        t_step=0.1,
        n_turns=3,
        solver_method="RK23",
    )
    dataset = generate_parametric_dataset(config=cfg, verbose=False)
    splits, windows, norm_params = build_split_windows(
        dataset["runs"], seq_len=10, pred_horizon=3,
        train_q=[20.0, 30.0, 60.0, 90.0], val_q=[45.0],
        test_q={"interp": [75.0], "extrap": [110.0]},
    )
    X, y = windows["train"]
    X_val, y_val = windows["val"]
    print(f"-> Created {len(X)} train / {len(X_val)} validation windows from whole-run splits.")

    # 3. Train Physics-Informed LSTM
    print("\nStep 3: Training Physics-Informed LSTM Neural Surrogate...")
    model_path = os.path.join(out_dir, "pinn_surrogate.pt")
    loss_plot_path = os.path.join(out_dir, "training_loss.png")
    model, history = train_surrogate(
        X_train=X,
        y_train=y,
        X_val=X_val,
        y_val=y_val,
        norm_params=norm_params,
        epochs=20,
        batch_size=16,
        hidden_dim=32,
        num_layers=2,
        rnn_type="lstm",
        save_path=model_path,
        verbose=False,
    )
    plot_training_history(history, save_path=loss_plot_path)
    print(f"-> Model trained and saved to {model_path}")

    # 4. Evaluate on held-out test RUNS (never seen in training/validation)
    print("\nStep 4: Evaluating on held-out test runs (per channel, physical units)...")
    for name in ("test_interp", "test_extrap"):
        res_one = evaluate_one_step(model, splits[name], norm_params, seq_len=10, pred_horizon=3)
        print(f"-> {name} ({len(splits[name])} runs, {res_one['n_windows']} windows), one-step ahead:")
        print(format_comparison("   (surrogate vs persistence baseline)", res_one))
        res_roll = evaluate_rollout(model, splits[name], norm_params, seq_len=10, pred_horizon=3, horizons=(10, None))
        for label, hres in res_roll["horizons"].items():
            print(format_comparison(f"   autoregressive rollout, first {label} steps (predictions fed back):", hres))

    # 5. Honest speed benchmark: solve_ivp vs surrogate rollout on the same test runs
    print("\nStep 5: Wall-clock benchmark (same runs, same duration, same machine)...")
    bench = benchmark_speed(
        model, splits["test_interp"] + splits["test_extrap"], norm_params,
        sim=cfg.sim_settings(), seq_len=10, pred_horizon=3, n_solver_runs=2,
    )
    print(format_benchmark(bench))

    print("\n=================================================================")
    print(f"  Demo Successfully Finished! All artifacts generated in: {out_dir}/")
    print("=================================================================\n")


def main():
    parser = argparse.ArgumentParser(
        prog="clphp",
        description="Continuous-Loop Pulsating Heat Pipe (CLPHP) Hybrid Simulation & PI-RNN Framework",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: simulate
    p_sim = subparsers.add_parser("simulate", help="Run transient ODE simulation")
    p_sim.add_argument("--fluid", type=str, default="acetone", help="Working fluid name")
    p_sim.add_argument("--qin", type=float, default=50.0, help="Heat input power [W]")
    p_sim.add_argument("--turns", type=int, default=5, help="Number of turns")
    p_sim.add_argument("--d-inner", type=float, default=2.0, help="Inner diameter [mm]")
    p_sim.add_argument("--fill-ratio", type=float, default=0.50, help="Fill ratio (0.1 - 0.9)")
    p_sim.add_argument("--time", type=float, default=30.0, help="Simulation duration [s]")
    p_sim.add_argument("--dt", type=float, default=0.02, help="Output time step [s]")
    p_sim.add_argument("--tsink", type=float, default=20.0, help="Sink temp [°C]")
    p_sim.add_argument("--g", type=float, default=9.81, help="Gravity accel [m/s^2]")
    p_sim.add_argument("--inclination", type=float, default=90.0, help="Inclination angle [deg]")
    p_sim.add_argument("-o", "--output", type=str, default="output/simulation_dashboard.png", help="Output plot path")

    # Command: generate-data
    p_gen = subparsers.add_parser("generate-data", help="Generate parametric dataset")
    p_gen.add_argument("--qin-list", type=str, default="15,30,50,75,100,130", help="Comma-separated Q_in values")
    p_gen.add_argument("--fr-list", type=str, default="0.40,0.50,0.60", help="Comma-separated fill ratios")
    p_gen.add_argument("--g-list", type=str, default="0.0,9.81", help="Comma-separated gravity accelerations")
    p_gen.add_argument("--fluids", type=str, default="acetone", help="Comma-separated fluids")
    p_gen.add_argument("--time", type=float, default=20.0, help="Duration per run [s]")
    p_gen.add_argument("--dt", type=float, default=0.05, help="Time step [s]")
    p_gen.add_argument("--turns", type=int, default=5, help="Number of turns")
    p_gen.add_argument("-o", "--output", type=str, default="data/clphp_dataset.npz", help="Dataset save path")

    # Command: train
    p_train = subparsers.add_parser("train", help="Train PI-RNN surrogate model")
    p_train.add_argument("--dataset", type=str, default="data/clphp_dataset.npz", help="Path to .npz dataset")
    p_train.add_argument("--rnn-type", type=str, default="lstm", choices=["lstm", "gru"], help="Recurrent backbone")
    p_train.add_argument("--epochs", type=int, default=40, help="Training epochs")
    p_train.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p_train.add_argument("--hidden-dim", type=int, default=64, help="Hidden state size")
    p_train.add_argument("--layers", type=int, default=2, help="Number of recurrent layers")
    p_train.add_argument("--seq-len", type=int, default=20, help="History lookback sequence length")
    p_train.add_argument("--horizon", type=int, default=5, help="Future prediction horizon")
    p_train.add_argument("--stride", type=int, default=2, help="Sliding window stride")
    p_train.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p_train.add_argument("--save-model", type=str, default="models/pinn_surrogate.pt", help="Model checkpoint path")
    p_train.add_argument("--plot-loss", type=str, default="output/training_loss.png", help="Loss curve save path")

    # Command: demo
    subparsers.add_parser("demo", help="Run quick full end-to-end demo")

    args = parser.parse_args()

    if args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "generate-data":
        cmd_generate_data(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
