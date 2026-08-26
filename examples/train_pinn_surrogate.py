"""
Example: Train and Validate a Physics-Informed LSTM Surrogate for CLPHP.

Generates a training grid, trains a PI-LSTM with thermodynamic loss constraints,
and evaluates prediction accuracy on an unseen test condition.
"""

import os
import numpy as np
import torch

from clphp.simulation.dataset_generator import generate_parametric_dataset, create_sliding_windows, DatasetConfig
from clphp.models.trainer import train_surrogate
from clphp.analysis.plotting import plot_training_history, plot_surrogate_comparison
from clphp.analysis.metrics import evaluate_surrogate_accuracy
from clphp.simulation.solver import simulate_clphp
from clphp.physics.geometry import PHPGeometry


def main():
    print("=== Training Physics-Informed Neural Network Surrogate ===")
    os.makedirs("output", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 1. Generate Parametric Training Dataset
    cfg = DatasetConfig(
        q_in_values=[30.0, 60.0, 90.0],
        fill_ratios=[0.50],
        g_accels=[9.81],
        fluids=["acetone"],
        t_sim_end=4.0,
        t_step=0.1,
        n_turns=3,
    )
    dataset_file = "data/surrogate_demo_dataset.npz"
    dataset = generate_parametric_dataset(config=cfg, output_path=dataset_file, verbose=True)

    # 2. Windowing
    seq_len = 10
    pred_horizon = 3
    X, y, norm_params = create_sliding_windows(dataset["runs"], seq_len=seq_len, pred_horizon=pred_horizon)
    print(f"Dataset generated: {len(X)} sequence samples. X: {X.shape}, y: {y.shape}")

    # 3. Train PI-LSTM
    model_path = "output/pinn_lstm_model.pt"
    model, history = train_surrogate(
        X=X,
        y=y,
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

    # 4. Unseen Test Case Validation (Q_in = 70W, unseen in training grid)
    print("\n--- Testing on Unseen Load: Q_in = 70 W (Unseen Condition) ---")
    test_geo = PHPGeometry(n_turns=3, d_inner=0.002, fill_ratio=0.50)
    test_res = simulate_clphp(geometry=test_geo, fluid_name="acetone", q_in=70.0, t_span=(0.0, 4.0), t_eval_step=0.1, solver_method="RK23")

    # Construct features for test simulation
    t_arr = test_res.time
    t_e = test_res.t_evap_mean
    t_c = test_res.t_cond_mean
    p_mean = np.mean(test_res.pressures, axis=1)
    v_rms = np.sqrt(np.mean(test_res.velocities ** 2, axis=1))

    raw_test_features = np.column_stack([
        np.full_like(t_arr, 70.0),
        np.full_like(t_arr, 293.15),
        np.full_like(t_arr, 0.50),
        np.full_like(t_arr, 9.81),
        t_arr,
        t_e,
        t_c,
        p_mean,
        v_rms,
    ])
    raw_test_targets = np.column_stack([t_e, t_c, test_res.thermal_resistance])

    # Run autoregressive sliding-window inference
    feat_mean, feat_std = norm_params["feat_mean"], norm_params["feat_std"]
    targ_mean, targ_std = norm_params["targ_mean"], norm_params["targ_std"]

    norm_test_features = (raw_test_features - feat_mean) / feat_std
    
    preds_list = []
    times_tested = []
    device = next(model.parameters()).device
    for i in range(len(t_arr) - seq_len - pred_horizon + 1):
        x_window = torch.tensor(norm_test_features[i:i+seq_len][np.newaxis, :, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            out_norm = model(x_window).cpu().numpy()[0]  # (pred_horizon, 3)
        # Denormalize
        out_denorm = out_norm * targ_std + targ_mean
        preds_list.append(out_denorm[0])  # Take 1-step ahead prediction
        times_tested.append(t_arr[i + seq_len])

    preds_array = np.array(preds_list)
    gt_array = raw_test_targets[seq_len:seq_len + len(preds_array)]
    t_tested_arr = np.array(times_tested)

    metrics = evaluate_surrogate_accuracy(gt_array, preds_array)
    print(f"Test Evaluation Metrics on Unseen 70W Case:")
    print(f"  RMSE (All channels): {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R^2 Score: {metrics['r2_score']:.4f}")

    comp_img = "output/surrogate_test_comparison.png"
    plot_surrogate_comparison(t_tested_arr, gt_array, preds_array, save_path=comp_img)
    print(f"Comparison figure saved to {comp_img}")


if __name__ == "__main__":
    main()
