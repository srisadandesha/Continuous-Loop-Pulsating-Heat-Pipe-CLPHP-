"""
Example: Train and Validate a Physics-Informed LSTM Surrogate for CLPHP.

Generates a training grid, trains a PI-LSTM with thermodynamic loss constraints,
and evaluates prediction accuracy on an unseen test condition.
"""

import os
import numpy as np
import torch

from clphp.simulation.dataset_generator import generate_parametric_dataset, build_split_windows, DatasetConfig
from clphp.models.trainer import train_surrogate
from clphp.analysis.plotting import plot_training_history, plot_surrogate_comparison
from clphp.models.evaluation import report_test_runs, rollout_batch


def main():
    print("=== Training Physics-Informed Neural Network Surrogate ===")
    os.makedirs("output", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 1. Generate Parametric Training Dataset
    # NOTE: quick demo grid with loose RK23 tolerances (fast, but NOT converged; see README).
    cfg = DatasetConfig(
        q_in_values=[20.0, 30.0, 45.0, 60.0, 70.0, 90.0, 110.0],
        fill_ratios=[0.50],
        g_accels=[9.81],
        fluids=["acetone"],
        t_sim_end=4.0,
        t_step=0.1,
        n_turns=3,
        solver_method="RK23", rtol=1e-2, atol=1e-3,
    )
    dataset_file = "data/surrogate_demo_dataset.npz"
    dataset = generate_parametric_dataset(config=cfg, output_path=dataset_file, verbose=True)

    # 2. Windowing
    seq_len = 10
    pred_horizon = 3
    splits, windows, norm_params = build_split_windows(
        dataset["runs"], seq_len=seq_len, pred_horizon=pred_horizon,
        train_q=[20.0, 30.0, 60.0, 90.0], val_q=[45.0],
        test_q={"interp": [70.0], "extrap": [110.0]},
    )
    X, y = windows["train"]
    X_val, y_val = windows["val"]
    print(f"Dataset generated: {len(X)} train / {len(X_val)} val windows (split by whole run). X: {X.shape}, y: {y.shape}")

    # 3. Train PI-LSTM
    model_path = "output/pinn_lstm_model.pt"
    model, history = train_surrogate(
        X_train=X,
        y_train=y,
        X_val=X_val,
        y_val=y_val,
        norm_params=norm_params,
        epochs=25,
        batch_size=16,
        hidden_dim=48,
        num_layers=2,
        rnn_type="lstm",
        lr=1.5e-3,
        save_path=model_path,
        verbose=True,
    )
    plot_training_history(history, save_path="output/pinn_training_curve.png")

    # 4. Held-out test RUNS (never seen in training or validation): interpolation + extrapolation
    print("\n--- Held-out test runs (per channel, physical units; surrogate vs persistence) ---")
    print(report_test_runs(model, splits, norm_params, seq_len, pred_horizon, rollout_horizons=(10, None)))

    # 5. Plot an autoregressive rollout on the first interpolation test run
    run = splits["test_interp"][0]
    feats = run["features"][np.newaxis]
    preds = rollout_batch(model.to("cpu"), feats, norm_params, seq_len, pred_horizon)[0]
    gt = run["targets"][seq_len:]
    t_arr = run["features"][seq_len:, 4]
    comp_img = "output/surrogate_test_comparison.png"
    plot_surrogate_comparison(t_arr, gt, preds, save_path=comp_img)
    print(f"Rollout comparison figure saved to {comp_img}")


if __name__ == "__main__":
    main()
