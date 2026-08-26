"""
Parametric Dataset Generator for CLPHP Machine Learning Surrogates.

Runs automated sweeps across thermal heat loads, fill ratios, orientations,
and gravity environments to generate structured time-series datasets for
recurrent neural network and PINN training.
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm

from clphp.physics.geometry import PHPGeometry
from clphp.physics.fluids import get_fluid
from clphp.physics.ode_system import ContinuousLoopPHP
from clphp.simulation.solver import simulate_clphp, SimulationResult


@dataclass
class DatasetConfig:
    """Configuration for parametric dataset generation."""
    q_in_values: List[float] = None
    fill_ratios: List[float] = None
    sink_temps: List[float] = None
    g_accels: List[float] = None
    fluids: List[str] = None
    t_sim_end: float = 20.0
    t_step: float = 0.05
    n_turns: int = 5
    d_inner: float = 0.002

    def __post_init__(self):
        if self.q_in_values is None:
            self.q_in_values = [15.0, 30.0, 50.0, 75.0, 100.0, 130.0]
        if self.fill_ratios is None:
            self.fill_ratios = [0.40, 0.50, 0.60]
        if self.sink_temps is None:
            self.sink_temps = [293.15]
        if self.g_accels is None:
            self.g_accels = [0.0, 9.81]  # Microgravity & 1g ground
        if self.fluids is None:
            self.fluids = ["acetone"]


def generate_parametric_dataset(
    config: Optional[DatasetConfig] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Generate synthetic time-series dataset from ODE solver across parameter grid.
    
    Returns a dictionary of raw sequences and metadata, and optionally saves to .npz.
    """
    cfg = config or DatasetConfig()
    runs_data = []

    total_runs = (
        len(cfg.fluids)
        * len(cfg.fill_ratios)
        * len(cfg.g_accels)
        * len(cfg.sink_temps)
        * len(cfg.q_in_values)
    )

    if verbose:
        print(f"Generating CLPHP dataset across {total_runs} parametric conditions...")

    pbar = tqdm(total=total_runs, disable=not verbose, desc="CLPHP Simulations")

    run_id = 0
    for fluid_name in cfg.fluids:
        for fr in cfg.fill_ratios:
            geo = PHPGeometry(
                n_turns=cfg.n_turns,
                d_inner=cfg.d_inner,
                fill_ratio=fr,
            )
            for g in cfg.g_accels:
                for t_sink in cfg.sink_temps:
                    for q in cfg.q_in_values:
                        res = simulate_clphp(
                            geometry=geo,
                            fluid_name=fluid_name,
                            q_in=q,
                            t_span=(0.0, cfg.t_sim_end),
                            t_eval_step=cfg.t_step,
                            t_sink_k=t_sink,
                            g_accel=g,
                            solver_method="RK45",
                        )

                        # Extract time series
                        t_arr = res.time
                        t_e = res.t_evap_mean
                        t_c = res.t_cond_mean
                        p_mean = np.mean(res.pressures, axis=1)
                        v_rms_t = np.sqrt(np.mean(res.velocities ** 2, axis=1))
                        r_th = res.thermal_resistance

                        # Features: [Q_in, T_sink, FillRatio, g, Time, T_e, T_c, P_mean, V_rms]
                        features_matrix = np.column_stack([
                            np.full_like(t_arr, q),
                            np.full_like(t_arr, t_sink),
                            np.full_like(t_arr, fr),
                            np.full_like(t_arr, g),
                            t_arr,
                            t_e,
                            t_c,
                            p_mean,
                            v_rms_t,
                        ])

                        # Targets: [T_e, T_c, R_th]
                        targets_matrix = np.column_stack([
                            t_e,
                            t_c,
                            r_th,
                        ])

                        runs_data.append({
                            "run_id": run_id,
                            "fluid": fluid_name,
                            "q_in": q,
                            "fill_ratio": fr,
                            "g_accel": g,
                            "t_sink": t_sink,
                            "features": features_matrix,
                            "targets": targets_matrix,
                            "steady_r_th": res.steady_state_r_th,
                            "dom_freq_hz": res.oscillation_frequency_hz,
                        })

                        run_id += 1
                        pbar.update(1)

    pbar.close()

    # Consolidate into dataset arrays
    all_features = [r["features"] for r in runs_data]
    all_targets = [r["targets"] for r in runs_data]

    dataset = {
        "runs": runs_data,
        "n_runs": len(runs_data),
        "feature_names": ["q_in", "t_sink", "fill_ratio", "g_accel", "time", "t_e", "t_c", "p_mean", "v_rms"],
        "target_names": ["t_e", "t_c", "r_th"],
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.savez_compressed(
            output_path,
            features=np.array(all_features, dtype=object),
            targets=np.array(all_targets, dtype=object),
            feature_names=dataset["feature_names"],
            target_names=dataset["target_names"],
        )
        if verbose:
            print(f"Dataset successfully saved to: {output_path}")

    return dataset


def create_sliding_windows(
    runs_data: List[Dict[str, Any]],
    seq_len: int = 20,
    pred_horizon: int = 5,
    stride: int = 2,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[float, float]]]:
    """
    Transform raw simulation sequences into supervised (X, y) sliding window tensors.

    Parameters:
        runs_data: List of run dictionary items containing 'features' and 'targets'
        seq_len: Past history window size (lookback)
        pred_horizon: Future forecast steps
        stride: Step size between consecutive sliding windows

    Returns:
        X: Tensor of shape (N_samples, seq_len, n_features)
        y: Tensor of shape (N_samples, pred_horizon, n_targets)
        norm_params: Dict containing mean and std for normalization
    """
    x_list, y_list = [], []

    # Concatenate all features to compute global normalization stats
    all_feat = np.concatenate([r["features"] for r in runs_data], axis=0)
    all_targ = np.concatenate([r["targets"] for r in runs_data], axis=0)

    feat_mean = np.mean(all_feat, axis=0)
    feat_std = np.std(all_feat, axis=0) + 1e-8
    targ_mean = np.mean(all_targ, axis=0)
    targ_std = np.std(all_targ, axis=0) + 1e-8

    norm_params = {
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "targ_mean": targ_mean,
        "targ_std": targ_std,
    }

    for r in runs_data:
        f_norm = (r["features"] - feat_mean) / feat_std
        t_norm = (r["targets"] - targ_mean) / targ_std
        total_steps = len(f_norm)

        for start in range(0, total_steps - seq_len - pred_horizon + 1, stride):
            end = start + seq_len
            x_window = f_norm[start:end]
            y_window = t_norm[end:end + pred_horizon]
            x_list.append(x_window)
            y_list.append(y_window)

    X = np.array(x_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, y, norm_params
