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
