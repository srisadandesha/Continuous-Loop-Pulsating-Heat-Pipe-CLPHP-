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
from typing import List, Optional, Sequence, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm

from clphp.physics.geometry import PHPGeometry
from clphp.physics.fluids import get_fluid
from clphp.physics.ode_system import ContinuousLoopPHP
from clphp.simulation.solver import simulate_clphp, SimulationResult


# ---------------------------------------------------------------------------
# Run-level split definition.  Whole simulation runs (identified by their Q_in)
# are assigned to exactly one split, so no window of a test run is ever seen
# in training.  75 W is interpolation (between 60 and 100 W); 160 W is
# extrapolation (above the 130 W training maximum).
# ---------------------------------------------------------------------------
DEFAULT_TRAIN_Q: Tuple[float, ...] = (15.0, 20.0, 30.0, 50.0, 60.0, 100.0, 130.0)
DEFAULT_VAL_Q: Tuple[float, ...] = (40.0, 115.0)
DEFAULT_TEST_Q: Dict[str, Tuple[float, ...]] = {
    "interp": (75.0,),
    "extrap": (160.0,),
}


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
    solver_method: str = "Radau"
    rtol: float = 1e-4
    atol: float = 1e-6
    seed: int = 42  # seeds the initial velocity perturbation of every run

    def sim_settings(self) -> Dict[str, Any]:
        """Generation settings as a dict (used to re-time the solver in benchmarks)."""
        return {
            "n_turns": self.n_turns, "d_inner": self.d_inner, "t_sim_end": self.t_sim_end,
            "t_step": self.t_step, "solver_method": self.solver_method,
            "rtol": self.rtol, "atol": self.atol,
        }

    def __post_init__(self):
        if self.q_in_values is None:
            self.q_in_values = sorted(
                set(DEFAULT_TRAIN_Q) | set(DEFAULT_VAL_Q) | {q for v in DEFAULT_TEST_Q.values() for q in v}
            )
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
                            solver_method=cfg.solver_method,
                            rtol=cfg.rtol,
                            atol=cfg.atol,
                            seed=cfg.seed,
                        )

                        # Extract time series
                        t_arr = res.time
                        t_e = res.t_evap_mean
                        t_c = res.t_cond_mean
                        p_mean = np.mean(res.pressures, axis=1)
                        v_rms_t = np.sqrt(np.mean(res.velocities ** 2, axis=1))
                        r_th = res.thermal_resistance

                        # Features: [Q_in, T_sink, FillRatio, g, Time, T_e, T_c]
                        # Exogenous: cols 0-4 (known in advance). States: cols 5-6 (T_e, T_c).
                        # P_mean / V_rms are NOT inputs: the surrogate does not predict them,
                        # so they could not be fed back during autoregressive rollout.
                        features_matrix = np.column_stack([
                            np.full_like(t_arr, q),
                            np.full_like(t_arr, t_sink),
                            np.full_like(t_arr, fr),
                            np.full_like(t_arr, g),
                            t_arr,
                            t_e,
                            t_c,
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
                            "aux": {"p_mean": p_mean, "v_rms": v_rms_t},
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
        "feature_names": ["q_in", "t_sink", "fill_ratio", "g_accel", "time", "t_e", "t_c"],
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


def fit_norm_params(train_runs: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Fit normalization statistics on *training runs only*."""
    if not train_runs:
        raise ValueError("Cannot fit normalization statistics on zero runs")
    all_feat = np.concatenate([r["features"] for r in train_runs], axis=0)
    all_targ = np.concatenate([r["targets"] for r in train_runs], axis=0)
    return {
        "feat_mean": np.mean(all_feat, axis=0),
        "feat_std": np.std(all_feat, axis=0) + 1e-8,
        "targ_mean": np.mean(all_targ, axis=0),
        "targ_std": np.std(all_targ, axis=0) + 1e-8,
    }


def create_sliding_windows(
    runs_data: List[Dict[str, Any]],
    seq_len: int = 20,
    pred_horizon: int = 5,
    stride: int = 2,
    norm_params: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Transform raw simulation sequences into supervised (X, y) sliding window tensors.

    Windows never span two runs.  Call this *after* assigning runs to splits.

    Parameters:
        runs_data: List of run dictionaries containing 'features' and 'targets'
        seq_len: Past history window size (lookback)
        pred_horizon: Future forecast steps
        stride: Step size between consecutive sliding windows
        norm_params: Normalization statistics to apply.  Pass the statistics
            fitted on the *training* runs when windowing validation/test runs.
            If None, statistics are fitted on ``runs_data`` itself (only
            correct when ``runs_data`` are the training runs).

    Returns:
        X: Tensor of shape (N_samples, seq_len, n_features)
        y: Tensor of shape (N_samples, pred_horizon, n_targets)
        norm_params: The statistics used for normalization
    """
    x_list, y_list = [], []

    if norm_params is None:
        norm_params = fit_norm_params(runs_data)
    feat_mean, feat_std = norm_params["feat_mean"], norm_params["feat_std"]
    targ_mean, targ_std = norm_params["targ_mean"], norm_params["targ_std"]

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

    n_f, n_t = len(feat_mean), len(targ_mean)
    X = np.array(x_list, dtype=np.float32).reshape(-1, seq_len, n_f)
    y = np.array(y_list, dtype=np.float32).reshape(-1, pred_horizon, n_t)

    return X, y, norm_params


def split_runs_by_q_in(
    runs_data: List[Dict[str, Any]],
    train_q: Sequence[float] = DEFAULT_TRAIN_Q,
    val_q: Sequence[float] = DEFAULT_VAL_Q,
    test_q: Optional[Dict[str, Sequence[float]]] = None,
    tol: float = 1e-6,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Assign whole simulation runs to train / val / test splits by their Q_in value.

    Every run (all fill ratios, gravity levels, sink temperatures sharing a
    Q_in) goes to exactly one split; windows are only cut *after* this
    assignment.  ``test_q`` maps a test-set name (e.g. ``"interp"``,
    ``"extrap"``) to its Q_in values; the result has keys ``"train"``,
    ``"val"`` and ``"test_<name>"``.

    Raises ValueError if the Q_in sets overlap or a run's Q_in is in no split.
    """
    test_q = DEFAULT_TEST_Q if test_q is None else test_q
    assignment: Dict[str, Sequence[float]] = {"train": train_q, "val": val_q}
    for name, qs in test_q.items():
        assignment[f"test_{name}"] = qs

    names = list(assignment)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            clash = [x for x in assignment[a] for y in assignment[b] if abs(x - y) <= tol]
            if clash:
                raise ValueError(f"Q_in values {clash} appear in both '{a}' and '{b}' splits")

    splits: Dict[str, List[Dict[str, Any]]] = {n: [] for n in names}
    for r in runs_data:
        q = float(r["q_in"])
        owner = [n for n in names if any(abs(q - x) <= tol for x in assignment[n])]
        if not owner:
            raise ValueError(f"Run with Q_in={q} W is not assigned to any split")
        splits[owner[0]].append(r)
    return splits


def runs_from_npz(path: str) -> List[Dict[str, Any]]:
    """Load runs saved by generate_parametric_dataset, recovering per-run metadata."""
    data = np.load(path, allow_pickle=True)
    runs = []
    for i in range(len(data["features"])):
        f = data["features"][i]
        runs.append({
            "run_id": i,
            "q_in": float(f[0, 0]),
            "t_sink": float(f[0, 1]),
            "fill_ratio": float(f[0, 2]),
            "g_accel": float(f[0, 3]),
            "features": f,
            "targets": data["targets"][i],
        })
    return runs


def build_split_windows(
    runs_data: List[Dict[str, Any]],
    seq_len: int = 20,
    pred_horizon: int = 5,
    stride: int = 2,
    train_q: Sequence[float] = DEFAULT_TRAIN_Q,
    val_q: Sequence[float] = DEFAULT_VAL_Q,
    test_q: Optional[Dict[str, Sequence[float]]] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, np.ndarray]]:
    """
    Split runs by Q_in first, then cut windows per split.

    Returns (run_splits, windows, norm_params) where ``windows[name] = (X, y)``
    for every split that contains runs.  Normalization statistics are fitted
    on the training windows' runs and applied to validation and test runs.
    """
    splits = split_runs_by_q_in(runs_data, train_q, val_q, test_q)
    if not splits["train"] or not splits["val"]:
        raise ValueError("Both the train and val splits must contain at least one run")
    norm = fit_norm_params(splits["train"])  # train runs only; reused for val/test
    X_tr, y_tr, _ = create_sliding_windows(splits["train"], seq_len, pred_horizon, stride, norm_params=norm)
    windows = {"train": (X_tr, y_tr)}
    for name, runs in splits.items():
        if name == "train" or not runs:
            continue
        X, y, _ = create_sliding_windows(runs, seq_len, pred_horizon, stride, norm_params=norm)
        windows[name] = (X, y)
    return splits, windows, norm
