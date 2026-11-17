"""
Tests for PyTorch PIRNNSurrogate and PhysicsInformedLoss.
"""

import torch
import numpy as np
from clphp.models.rnn_surrogate import PIRNNSurrogate
from clphp.models.pinn_loss import PhysicsInformedLoss


def test_lstm_surrogate_forward():
    model = PIRNNSurrogate(input_dim=9, hidden_dim=32, num_layers=2, output_dim=3, pred_horizon=5, rnn_type="lstm")
    
    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 9)
    out = model(x)
    
    assert out.shape == (batch_size, 5, 3)


def test_gru_surrogate_forward():
    model = PIRNNSurrogate(input_dim=9, hidden_dim=32, num_layers=2, output_dim=3, pred_horizon=5, rnn_type="gru")
    
    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 9)
    out = model(x)
    
    assert out.shape == (batch_size, 5, 3)


def test_physics_informed_loss():
    loss_fn = PhysicsInformedLoss()
    
    pred = torch.tensor([[[2.0, 1.0, 1.0]]], requires_grad=True)  # Te > Tc, Rth > 0 (Physics consistent)
    target = torch.tensor([[[2.1, 1.05, 0.95]]])
    
    loss, loss_dict = loss_fn(pred, target)
    assert loss.item() > 0
    assert "loss_data" in loss_dict
    assert "loss_temp_direction" in loss_dict
    
    # Backpropagation check
    loss.backward()
    assert pred.grad is not None


def _norm():
    return {
        "feat_mean": np.array([50.0, 293.0, 0.5, 9.8, 5.0, 330.0, 300.0]),
        "feat_std": np.array([20.0, 1.0, 0.1, 1.0, 3.0, 10.0, 5.0]),
        "targ_mean": np.array([330.0, 300.0, 0.5]),
        "targ_std": np.array([10.0, 5.0, 0.2]),
    }


def _phys_to_norm(arr, n):
    return (arr - n["targ_mean"]) / n["targ_std"]


def _x_with_q(q, n, seq=3):
    x = torch.zeros(1, seq, 7)
    x[:, :, 0] = (q - n["feat_mean"][0]) / n["feat_std"][0]
    return x


def test_pinn_loss_zero_for_physically_consistent_perfect_prediction():
    n = _norm()
    q, t_e, t_c = 50.0, 340.0, 300.0
    r = (t_e - t_c) / q
    target = torch.tensor(_phys_to_norm(np.array([[[t_e, t_c, r]] * 2]), n), dtype=torch.float32)
    loss, d = PhysicsInformedLoss(norm_params=n)(target.clone(), target, _x_with_q(q, n))
    assert d["loss_resistance_consistency"] < 1e-9
    assert d["loss_temp_direction"] == 0.0 and d["loss_positivity"] == 0.0


def test_pinn_resistance_term_uses_q_in_from_features():
    n = _norm()
    pred = torch.tensor(_phys_to_norm(np.array([[[340.0, 300.0, 0.8]]]), n), dtype=torch.float32)
    losses = []
    for q in (50.0, 100.0):
        _, d = PhysicsInformedLoss(norm_params=n)(pred, pred, _x_with_q(q, n))
        losses.append(d["loss_resistance_consistency"])
    # 40 K / 50 W = 0.8 K/W -> consistent at Q=50 only
    assert losses[0] < 1e-9 < losses[1]


def test_pinn_positivity_enforced_at_physical_zero_not_normalized_zero():
    n = _norm()
    neg = torch.tensor(_phys_to_norm(np.array([[[340.0, 300.0, -0.01]]]), n), dtype=torch.float32)
    pos = torch.tensor(_phys_to_norm(np.array([[[340.0, 300.0, 0.01]]]), n), dtype=torch.float32)
    f = PhysicsInformedLoss(norm_params=n)
    assert f(neg, neg)[1]["loss_positivity"] > 0
    assert f(pos, pos)[1]["loss_positivity"] == 0


def test_pinn_direction_penalty_and_data_only_switch():
    n = _norm()
    bad = torch.tensor(_phys_to_norm(np.array([[[300.0, 330.0, 0.2]]]), n), dtype=torch.float32)  # T_c > T_e
    l_full, d_full = PhysicsInformedLoss(norm_params=n)(bad, bad)
    l_off, d_off = PhysicsInformedLoss.data_only(norm_params=n)(bad, bad)
    assert d_full["loss_temp_direction"] > 0
    assert l_full.item() > 0 and l_off.item() == 0.0
