"""
Recurrent Neural Network (LSTM / GRU) Surrogate Models for CLPHP Dynamics.

Provides sequence-to-sequence neural architectures designed for real-time
transient temperature and thermal resistance predictions.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Literal, Tuple, Optional


class PIRNNSurrogate(nn.Module):
    """
    Physics-Informed Recurrent Neural Network (LSTM / GRU) Surrogate.

    Maps past time-series trajectories of thermal inputs and states to future
    evaporator/condenser temperatures and thermal resistance.
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 3,
        pred_horizon: int = 5,
        rnn_type: Literal["lstm", "gru"] = "lstm",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.pred_horizon = pred_horizon
        self.rnn_type = rnn_type.lower()

        # Input feature projection layer
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Recurrent core
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}. Choose 'lstm' or 'gru'.")

        # Multi-step forecast decoder head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_horizon * output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            out: Forecast tensor of shape (batch_size, pred_horizon, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project inputs
        h_proj = self.input_proj(x)  # (batch_size, seq_len, hidden_dim)

        # Recurrent encoding
        if self.rnn_type == "lstm":
            rnn_out, (hn, cn) = self.rnn(h_proj)
            last_hidden = hn[-1]  # (batch_size, hidden_dim)
        else:
            rnn_out, hn = self.rnn(h_proj)
            last_hidden = hn[-1]

        # Forecast future horizon
        flat_pred = self.head(last_hidden)  # (batch_size, pred_horizon * output_dim)
        out = flat_pred.view(batch_size, self.pred_horizon, self.output_dim)
        return out
