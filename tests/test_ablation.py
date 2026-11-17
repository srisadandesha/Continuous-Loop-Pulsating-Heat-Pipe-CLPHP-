"""Tests for training reproducibility and the physics ablation harness."""

import numpy as np
import torch

from clphp.models.ablation import run_physics_ablation, format_ablation, VARIANTS
from clphp.models.trainer import train_surrogate
from clphp.simulation.dataset_generator import build_split_windows
from tests.test_split import _fake_runs, ALL_Q


def _data():
    runs = _fake_runs(ALL_Q, fill_ratios=(0.5,), steps=25)
    return build_split_windows(runs, seq_len=5, pred_horizon=2, stride=1)


def test_same_seed_gives_identical_weights():
    splits, w, norm = _data()
    kw = dict(X_train=w["train"][0], y_train=w["train"][1], X_val=w["val"][0], y_val=w["val"][1],
              norm_params=norm, epochs=2, batch_size=16, hidden_dim=8, num_layers=1, verbose=False, device="cpu")
    m1, _ = train_surrogate(seed=3, **kw)
    m2, _ = train_surrogate(seed=3, **kw)
    m3, _ = train_surrogate(seed=4, **kw)
    p1 = torch.cat([p.flatten() for p in m1.parameters()])
    p2 = torch.cat([p.flatten() for p in m2.parameters()])
    p3 = torch.cat([p.flatten() for p in m3.parameters()])
    assert torch.equal(p1, p2)
    assert not torch.equal(p1, p3)


def test_ablation_trains_both_variants_per_seed_and_evaluates_on_test_runs():
    splits, w, norm = _data()
    res = run_physics_ablation(
        w, splits, norm, seq_len=5, pred_horizon=2, seeds=(0, 1), rollout_horizons=(5,),
        epochs=2, batch_size=16, hidden_dim=8, num_layers=1, device="cpu",
    )
    assert set(res) == set(VARIANTS)
    for variant in VARIANTS:
        one = res[variant]["test_interp"]["one_step"]
        assert len(one["T_e"]["rmse"]) == 2  # one value per seed
        assert "5" in res[variant]["test_extrap"]["rollout"]
    table = format_ablation(res)
    assert "with_physics" in table and "data_only" in table and "R_th" in table
