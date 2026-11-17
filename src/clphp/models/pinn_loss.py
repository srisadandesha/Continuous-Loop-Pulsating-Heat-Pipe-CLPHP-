"""
Physics-Informed Loss Functions for Pulsating Heat Pipe Surrogates.

Embeds thermodynamic constraints into gradient descent. All physics terms are
evaluated in PHYSICAL units: predictions are de-normalized inside the loss,
R_th = (T_e - T_c) / Q_in is rebuilt from the predicted temperatures and the
input power Q_in read from the input features, and R_th >= 0 is enforced at
physical zero (not at an offset in normalized space).
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

Q_COL = 0  # column of Q_in in the input feature vector


def _as_buffer(x, default: float) -> torch.Tensor:
    if x is None:
        return torch.full((3,), default, dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32)


class PhysicsInformedLoss(nn.Module):
    """
    Composite loss: normalized-space data MSE + physics penalties in physical units.

    Targets/predictions are ``(batch, horizon, 3)`` = [T_e [K], T_c [K], R_th [K/W]]
    (normalized with ``norm_params['targ_*']``). ``x`` is the normalized input
    window ``(batch, seq_len, n_features)``; Q_in of the *last observed step* is
    used for the whole forecast horizon (Q_in is held constant over a window).
    With ``norm_params=None`` the tensors are assumed to already be physical.
    """

    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_thermal_res: float = 0.1,
        lambda_temp_direction: float = 0.05,
        lambda_positivity: float = 0.05,
        norm_params: Optional[Dict[str, Tuple[float, float]]] = None,
        q_col: int = Q_COL,
    ):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_thermal_res = lambda_thermal_res
        self.lambda_temp_direction = lambda_temp_direction
        self.lambda_positivity = lambda_positivity
        self.q_col = q_col
        self.mse_fn = nn.MSELoss()
        self.norm_params = norm_params

        p = norm_params or {}
        self.register_buffer("targ_mean", _as_buffer(p.get("targ_mean"), 0.0))
        self.register_buffer("targ_std", _as_buffer(p.get("targ_std"), 1.0))
        feat_mean, feat_std = p.get("feat_mean"), p.get("feat_std")
        self.register_buffer("q_mean", torch.tensor(0.0 if feat_mean is None else float(feat_mean[q_col])))
        self.register_buffer("q_std", torch.tensor(1.0 if feat_std is None else float(feat_std[q_col])))

    @classmethod
    def data_only(cls, norm_params: Optional[Dict] = None) -> "PhysicsInformedLoss":
        """Same loss with every physics weight zero (for ablations); terms are still logged."""
        return cls(lambda_thermal_res=0.0, lambda_temp_direction=0.0, lambda_positivity=0.0,
                   norm_params=norm_params)

    def forward(
        self,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Supervised loss in normalized space (all channels on equal footing)
        l_data = self.mse_fn(pred_norm, target_norm)

        # De-normalize inside the loss -> physical K, K, K/W
        pred = pred_norm * self.targ_std + self.targ_mean
        t_e, t_c, r_th = pred[..., 0], pred[..., 1], pred[..., 2]
        t_scale = self.targ_std[0]  # make penalties dimensionless: divide by data scale
        r_scale = self.targ_std[2]

        # 2. Second law: heat flows evaporator -> condenser, so T_e >= T_c [K]
        l_temp_dir = torch.mean((torch.relu(t_c - t_e) / t_scale) ** 2)

        # 3. R_th >= 0 at physical zero [K/W]
        l_pos = torch.mean((torch.relu(-r_th) / r_scale) ** 2)

        # 4. Definition: R_th = (T_e - T_c) / Q_in, with Q_in from the input features
        if x is not None:
            q_in = x[:, -1, self.q_col] * self.q_std + self.q_mean  # [W], last observed step
            q_safe = torch.clamp(q_in, min=1.0)[:, None]
            r_from_t = torch.clamp(t_e - t_c, min=0.01) / q_safe  # same clamps as the simulator
            l_res = torch.mean(((r_th - r_from_t) / r_scale) ** 2)
        else:
            l_res = torch.zeros((), device=pred.device)

        total_loss = (
            self.lambda_data * l_data
            + self.lambda_temp_direction * l_temp_dir
            + self.lambda_positivity * l_pos
            + self.lambda_thermal_res * l_res
        )

        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_data": float(l_data.item()),
            "loss_temp_direction": float(l_temp_dir.item()),
            "loss_positivity": float(l_pos.item()),
            "loss_resistance_consistency": float(l_res.item()),
        }
        return total_loss, loss_dict
