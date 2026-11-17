"""Tests for the solver-vs-surrogate wall-clock benchmark."""

from clphp.analysis.benchmark import benchmark_speed, format_benchmark
from clphp.models.rnn_surrogate import PIRNNSurrogate
from clphp.simulation.dataset_generator import (
    DatasetConfig, generate_parametric_dataset, fit_norm_params,
)


def test_benchmark_reports_solver_batch1_and_batched_separately():
    cfg = DatasetConfig(
        q_in_values=[30.0, 60.0], fill_ratios=[0.5], g_accels=[9.81],
        t_sim_end=0.5, t_step=0.05, n_turns=3,
        solver_method="RK23", rtol=1e-2, atol=1e-3,  # plumbing test: speed over accuracy
    )
    runs = generate_parametric_dataset(config=cfg, verbose=False)["runs"]
    norm = fit_norm_params(runs)
    model = PIRNNSurrogate(input_dim=7, hidden_dim=8, num_layers=1, output_dim=3, pred_horizon=3)

    b = benchmark_speed(model, runs, norm, cfg.sim_settings(), seq_len=5, pred_horizon=3,
                        n_solver_runs=1, repeats=2)

    assert b["solver"]["wall_s_per_run"] > 0
    assert b["surrogate_batch1"]["wall_s_per_run"] > 0
    assert b["surrogate_batched"]["batch_size"] == 2
    # speedup is solver time over surrogate time for the same run, not 1000/latency
    assert abs(b["surrogate_batch1"]["speedup_vs_solver"]
               - b["solver"]["wall_s_per_run"] / b["surrogate_batch1"]["wall_s_per_run"]) < 1e-9
    assert "real_time_factor_legacy" in b["single_forward"]
    text = format_benchmark(b)
    assert "batch=1" in text and "batched" in text and "legacy 'real-time factor'" in text
