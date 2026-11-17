"""Tests for held-out evaluation helpers."""

import numpy as np
import torch

from clphp.models.evaluation import per_channel_metrics, evaluate_one_step, persistence_forecast, rollout_batch, evaluate_rollout, CHANNELS
from clphp.models.rnn_surrogate import PIRNNSurrogate
from tests.test_split import _fake_runs
from clphp.simulation.dataset_generator import fit_norm_params


def test_per_channel_metrics_are_not_pooled():
    rng = np.random.default_rng(0)
    y = np.stack([300 + 10 * rng.random(50), 290 + 5 * rng.random(50), 0.1 + 0.01 * rng.random(50)], axis=-1)
    pred = y.copy()
    pred[:, 0] += 2.0  # constant 2 K bias on T_e only
    m = per_channel_metrics(y, pred)
    assert set(m) == set(CHANNELS)
    assert np.isclose(m["T_e"]["rmse"], 2.0) and np.isclose(m["T_e"]["mae"], 2.0)
    assert m["T_c"]["rmse"] == 0.0 and m["R_th"]["rmse"] == 0.0
    assert m["T_c"]["r2"] == 1.0
    assert m["T_e"]["unit"] == "K" and m["R_th"]["unit"] == "K/W"


def test_evaluate_one_step_runs_on_held_out_runs():
    runs = _fake_runs([30.0, 60.0])
    norm = fit_norm_params(runs[:2])
    model = PIRNNSurrogate(input_dim=9, hidden_dim=8, num_layers=1, output_dim=3, pred_horizon=2)
    res = evaluate_one_step(model, runs[2:], norm, seq_len=5, pred_horizon=2)
    assert res["n_windows"] == 2 * (30 - 5 - 2 + 1)
    assert set(res["surrogate"]) == set(CHANNELS)


def test_persistence_repeats_last_observed_value():
    runs = _fake_runs([30.0])
    norm = fit_norm_params(runs)
    from clphp.simulation.dataset_generator import create_sliding_windows
    X, y, _ = create_sliding_windows(runs[:1], 5, 3, 1, norm_params=norm)
    base = persistence_forecast(X, norm, 3)
    last = X[:, -1, :] * norm["feat_std"] + norm["feat_mean"]
    assert base.shape == (len(X), 3, 3)
    for h in range(3):
        assert np.allclose(base[:, h, 0], last[:, 5]) and np.allclose(base[:, h, 1], last[:, 6])
    assert np.allclose(base[:, 0, 2], np.maximum(0.01, last[:, 5] - last[:, 6]) / 30.0)


def test_one_step_result_contains_persistence():
    runs = _fake_runs([30.0, 60.0])
    norm = fit_norm_params(runs[:2])
    model = PIRNNSurrogate(input_dim=9, hidden_dim=8, num_layers=1, output_dim=3, pred_horizon=2)
    res = evaluate_one_step(model, runs[2:], norm, seq_len=5, pred_horizon=2)
    assert set(res["persistence"]) == set(CHANNELS)


class _ConstModel(torch.nn.Module):
    """Predicts a fixed normalized value, to make rollout bookkeeping checkable."""
    def __init__(self, value, horizon=3):
        super().__init__()
        self.value, self.horizon = value, horizon
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.seen = []

    def forward(self, x):
        self.seen.append(x.detach().clone())
        return torch.full((x.shape[0], self.horizon, 3), self.value) + 0 * self.p


def test_rollout_feeds_predictions_back_and_uses_no_future_truth():
    runs = _fake_runs([30.0, 60.0], fill_ratios=(0.5,), steps=20)
    norm = fit_norm_params(runs)
    model = _ConstModel(0.0, horizon=3)  # predicts the train mean for every channel
    feats = np.stack([r["features"] for r in runs])
    preds = rollout_batch(model, feats, norm, seq_len=5, pred_horizon=3)
    assert preds.shape == (2, 15, 3)
    assert np.allclose(preds[:, :, 0], norm["targ_mean"][0])
    # Second call's input window must contain the model's own predictions in T_e/T_c columns
    second = model.seen[1].numpy() * norm["feat_std"] + norm["feat_mean"]
    assert np.allclose(second[:, -3:, 5], norm["targ_mean"][0], atol=1e-3)
    # ... and true exogenous time for those rows
    assert np.allclose(second[:, -3:, 4], feats[:, 5:8, 4], atol=1e-3)


def test_evaluate_rollout_reports_horizons_and_baseline():
    runs = _fake_runs([30.0, 60.0], fill_ratios=(0.5,), steps=20)
    norm = fit_norm_params(runs)
    model = PIRNNSurrogate(input_dim=9, hidden_dim=8, num_layers=1, output_dim=3, pred_horizon=3)
    res = evaluate_rollout(model, runs, norm, seq_len=5, pred_horizon=3, horizons=(4, None))
    assert set(res["horizons"]) == {"4", "full"}
    assert res["horizons"]["full"]["steps"] == 15
    assert set(res["horizons"]["4"]["persistence"]) == set(CHANNELS)
