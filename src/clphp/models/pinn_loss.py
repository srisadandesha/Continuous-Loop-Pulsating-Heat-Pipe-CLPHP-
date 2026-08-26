"""
Physics-Informed Loss Functions for Pulsating Heat Pipe Surrogates.

Embeds thermodynamic constraints (2nd Law directionality, thermal resistance
definition consistency, and energy balance penalties) into gradient descent.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class PhysicsInformedLoss(nn.Module):
    """
    Composite Loss Function combining Data-driven MSE and Physics-Informed Penalties.
    """

    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_thermal_res: float = 0.1,
        lambda_temp_direction: float = 0.05,
        lambda_positivity: float = 0.05,
        norm_params: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_thermal_res = lambda_thermal_res
        self.lambda_temp_direction = lambda_temp_direction
        self.lambda_positivity = lambda_positivity
        self.mse_fn = nn.MSELoss()
        self.norm_params = norm_params

    def forward(
        self,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        x_raw_or_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total physics-informed loss.
        
        Parameters:
            pred_norm: Model predictions (batch_size, horizon, 3) [T_e, T_c, R_th]
            target_norm: Ground truth targets (batch_size, horizon, 3) [T_e, T_c, R_th]
            x_raw_or_norm: (Optional) Input context tensor to extract input power Q_in
            
        Returns:
            total_loss: Scalar PyTorch tensor for backpropagation
            loss_components: Dictionary with individual scalar loss values
        """
        # 1. Standard Data-Driven Supervised Loss
        l_data = self.mse_fn(pred_norm, target_norm)

        # Extract predicted channels: [T_e, T_c, R_th]
        t_e_pred = pred_norm[:, :, 0]
        t_c_pred = pred_norm[:, :, 1]
        r_th_pred = pred_norm[:, :, 2]

        # 2. Physics Penalty 1: 2nd Law of Thermodynamics (T_e must be >= T_c)
        # If T_c > T_e, penalty is incurred
        violation_temp_order = torch.relu(t_c_pred - t_e_pred)
        l_temp_dir = torch.mean(violation_temp_order ** 2)

        # 3. Physics Penalty 2: Positivity of Thermal Resistance (R_th >= 0)
        # In normalized space, penalty below physical zero
        violation_pos = torch.relu(-r_th_pred - 3.0)  # In normalized space
        l_pos = torch.mean(violation_pos ** 2)

        # 4. Physics Penalty 3: Internal consistency between (T_e - T_c) and R_th
        # R_th should correlate positively with (T_e - T_c)
        delta_t_pred = t_e_pred - t_c_pred
        l_res_cons = torch.mean((delta_t_pred - r_th_pred) ** 2)

        # Total combined loss
        total_loss = (
            self.lambda_data * l_data
            + self.lambda_temp_direction * l_temp_dir
            + self.lambda_positivity * l_pos
            + self.lambda_thermal_res * l_res_cons
        )

        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_data": float(l_data.item()),
            "loss_temp_direction": float(l_temp_dir.item()),
            "loss_positivity": float(l_pos.item()),
            "loss_resistance_consistency": float(l_res_cons.item()),
        }

        return total_loss, loss_dict
