"""
End-to-End Pipeline Integration Test.
"""

import numpy as np
import pytest
from clphp.physics.geometry import PHPGeometry
from clphp.simulation.solver import simulate_clphp
from clphp.simulation.dataset_generator import generate_parametric_dataset, create_sliding_windows, build_split_windows, DatasetConfig
from clphp.models.trainer import train_surrogate
from clphp.analysis.metrics import compute_thermal_resistance, compute_fft_spectrum


def test_simulation_pipeline():
    geo = PHPGeometry(n_turns=3, d_inner=0.002, fill_ratio=0.50)
    res = simulate_clphp(geometry=geo, fluid_name="acetone", q_in=40.0, t_span=(0.0, 0.5), t_eval_step=0.05)
    
    assert res.success is True
    assert len(res.time) == len(res.t_evap_mean)
    assert res.steady_state_r_th > 0


def test_training_pipeline():
    cfg = DatasetConfig(
        q_in_values=[20.0, 30.0, 40.0, 60.0],
        fill_ratios=[0.50],
        g_accels=[9.81],
        fluids=["acetone"],
        t_sim_end=0.5,
        t_step=0.05,
        n_turns=3,
        solver_method="RK23", rtol=1e-2, atol=1e-3,  # plumbing test: speed over accuracy
    )
    dataset = generate_parametric_dataset(config=cfg, verbose=False)
    splits, windows, norm_params = build_split_windows(
        dataset["runs"], seq_len=5, pred_horizon=2, stride=1,
        train_q=[20.0, 60.0], val_q=[30.0], test_q={"interp": [40.0]},
    )
    X_tr, y_tr = windows["train"]
    X_va, y_va = windows["val"]
    assert len(X_tr) > 0 and len(X_va) > 0

    model, history = train_surrogate(
        X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va,
        norm_params=norm_params,
        epochs=3,
        batch_size=8,
        hidden_dim=16,
        num_layers=1,
        verbose=False,
    )

    assert len(history["train_loss"]) == 3


def test_dataset_generation_is_reproducible_and_parallel_matches_serial():
    cfg = DatasetConfig(
        q_in_values=[30.0, 60.0], fill_ratios=[0.50], g_accels=[9.81], fluids=["acetone"],
        t_sim_end=0.4, t_step=0.05, n_turns=3, solver_method="RK23", rtol=1e-2, atol=1e-3, seed=7,
    )
    a = generate_parametric_dataset(config=cfg, verbose=False, n_jobs=1)["runs"]
    b = generate_parametric_dataset(config=cfg, verbose=False, n_jobs=2)["runs"]
    c = generate_parametric_dataset(config=cfg, verbose=False, n_jobs=1)["runs"]
    assert [r["run_id"] for r in a] == [0, 1]
    for x, y, z in zip(a, b, c):
        assert np.array_equal(x["features"], y["features"]) and np.array_equal(x["targets"], y["targets"])
        assert np.array_equal(x["features"], z["features"])


def test_default_config_is_enlarged_and_contains_all_split_q_values():
    from clphp.simulation.dataset_generator import DEFAULT_TRAIN_Q, DEFAULT_VAL_Q, DEFAULT_TEST_Q
    cfg = DatasetConfig()
    needed = set(DEFAULT_TRAIN_Q) | set(DEFAULT_VAL_Q) | {q for v in DEFAULT_TEST_Q.values() for q in v}
    assert needed <= set(cfg.q_in_values)
    assert len(cfg.q_in_values) >= 11 and len(cfg.fill_ratios) >= 5 and cfg.seed == 42
