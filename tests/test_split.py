"""Tests for run-level train/val/test splitting (no window-level leakage)."""

import numpy as np
import pytest

from clphp.simulation.dataset_generator import (
    build_split_windows,
    split_runs_by_q_in,
    DEFAULT_TRAIN_Q,
    DEFAULT_VAL_Q,
    DEFAULT_TEST_Q,
)

ALL_Q = sorted(set(DEFAULT_TRAIN_Q) | set(DEFAULT_VAL_Q) | {q for v in DEFAULT_TEST_Q.values() for q in v})


def _fake_runs(q_values, fill_ratios=(0.4, 0.6), steps=30):
    """Synthetic runs; feature column 0 holds Q_in so windows can be traced to a run."""
    runs, rid = [], 0
    for fr in fill_ratios:
        for q in q_values:
            t = np.arange(steps, dtype=float)
            feats = np.column_stack([
                np.full(steps, q), np.full(steps, 293.15), np.full(steps, fr),
                np.zeros(steps), t, 293.15 + q * 0.1 * t / steps, np.full(steps, 293.15),
                np.full(steps, 1e5), np.full(steps, 0.01),
            ])
            targ = np.column_stack([feats[:, 5], feats[:, 6], (feats[:, 5] - feats[:, 6]) / q])
            runs.append({"run_id": rid, "q_in": q, "fill_ratio": fr, "features": feats, "targets": targ})
            rid += 1
    return runs


def test_every_run_assigned_to_exactly_one_split():
    runs = _fake_runs(ALL_Q)
    splits = split_runs_by_q_in(runs)
    ids = [r["run_id"] for rs in splits.values() for r in rs]
    assert sorted(ids) == sorted(r["run_id"] for r in runs)
    assert len(ids) == len(set(ids))


def test_held_out_q_values_never_in_train():
    splits = split_runs_by_q_in(_fake_runs(ALL_Q))
    train_q = {r["q_in"] for r in splits["train"]}
    assert train_q == set(DEFAULT_TRAIN_Q)
    assert {r["q_in"] for r in splits["test_interp"]} == {75.0}
    assert {r["q_in"] for r in splits["test_extrap"]} == {160.0}
    assert 75.0 not in train_q and 160.0 not in train_q
    # extrapolation case really lies outside the training range
    assert max(r["q_in"] for r in splits["test_extrap"]) > max(train_q)
    # all fill ratios of a held-out Q_in are held out together
    assert len(splits["test_interp"]) == 2


def test_overlapping_q_sets_rejected():
    with pytest.raises(ValueError, match="both"):
        split_runs_by_q_in(_fake_runs([30.0, 75.0]), train_q=[30.0, 75.0], val_q=[], test_q={"a": [75.0]})


def test_unassigned_run_rejected():
    with pytest.raises(ValueError, match="not assigned"):
        split_runs_by_q_in(_fake_runs([30.0, 999.0]), train_q=[30.0], val_q=[], test_q={})


def test_windows_do_not_cross_splits():
    runs = _fake_runs(ALL_Q)
    splits, windows, norm = build_split_windows(runs, seq_len=5, pred_horizon=2, stride=1)
    allowed = {
        "train": set(DEFAULT_TRAIN_Q),
        "val": set(DEFAULT_VAL_Q),
        "test_interp": {75.0},
        "test_extrap": {160.0},
    }
    for name, (X, y) in windows.items():
        # De-normalise column 0 (Q_in) and confirm each window holds a single, allowed run
        q = X[:, :, 0] * norm["feat_std"][0] + norm["feat_mean"][0]
        assert np.allclose(q, q[:, :1], atol=1e-3), f"{name}: window spans several runs"
        assert set(np.round(q[:, 0]).astype(float)) <= allowed[name]
    # windows per run = steps - seq_len - horizon + 1 = 24
    assert len(windows["test_interp"][0]) == 24 * 2


def test_shuffling_cannot_happen_before_split():
    """Window counts per split are fixed by the runs, independent of any RNG."""
    runs = _fake_runs(ALL_Q)
    _, w1, _ = build_split_windows(runs, seq_len=5, pred_horizon=2, stride=1)
    np.random.seed(123)
    _, w2, _ = build_split_windows(runs, seq_len=5, pred_horizon=2, stride=1)
    for k in w1:
        assert np.array_equal(w1[k][0], w2[k][0])
