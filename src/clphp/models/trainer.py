"""
Training and Evaluation Pipeline for CLPHP Physics-Informed Recurrent Neural Networks.

Handles data splitting, mini-batch training, physics loss backpropagation,
validation tracking, learning rate decay, and model serialization.
"""

from __future__ import annotations
import os
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from clphp.models.rnn_surrogate import PIRNNSurrogate
from clphp.models.pinn_loss import PhysicsInformedLoss


class SurrogateTrainer:
    """
    Trainer for Physics-Informed Recurrent Neural Networks.
    """

    def __init__(
        self,
        model: PIRNNSurrogate,
        loss_fn: Optional[PhysicsInformedLoss] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.loss_fn = (loss_fn or PhysicsInformedLoss()).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-5)
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "train_data_loss": [],
            "val_data_loss": [],
            "learning_rate": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run one training epoch over mini-batches."""
        self.model.train()
        total_loss = 0.0
        data_loss_sum = 0.0
        n_batches = len(dataloader)

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(x_batch)
            loss, loss_dict = self.loss_fn(preds, y_batch, x_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            total_loss += loss_dict["loss_total"]
            data_loss_sum += loss_dict["loss_data"]

        return {
            "loss_total": total_loss / max(1, n_batches),
            "loss_data": data_loss_sum / max(1, n_batches),
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation or test dataset."""
        self.model.eval()
        total_loss = 0.0
        data_loss_sum = 0.0
        n_batches = len(dataloader)

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                preds = self.model(x_batch)
                loss, loss_dict = self.loss_fn(preds, y_batch, x_batch)

                total_loss += loss_dict["loss_total"]
                data_loss_sum += loss_dict["loss_data"]

        return {
            "val_loss": total_loss / max(1, n_batches),
            "val_data_loss": data_loss_sum / max(1, n_batches),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """Train model across multiple epochs with validation tracking."""
        start_time = time.perf_counter()
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.scheduler.step()

            self.history["train_loss"].append(train_metrics["loss_total"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["train_data_loss"].append(train_metrics["loss_data"])
            self.history["val_data_loss"].append(val_metrics["val_data_loss"])
            self.history["learning_rate"].append(current_lr)

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs or epoch == 1):
                print(
                    f"Epoch {epoch:03d}/{epochs:03d} | "
                    f"Train Loss: {train_metrics['loss_total']:.6f} | "
                    f"Val Loss: {val_metrics['val_loss']:.6f} | "
                    f"LR: {current_lr:.2e}"
                )

        elapsed = time.perf_counter() - start_time
        if verbose:
            print(f"Training completed in {elapsed:.2f}s. Best Val Loss: {best_val_loss:.6f}")

        return self.history

    def save_checkpoint(self, file_path: str, norm_params: Optional[Dict[str, Any]] = None):
        """Save model weights and metadata to file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "input_dim": self.model.input_dim,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
                "output_dim": self.model.output_dim,
                "pred_horizon": self.model.pred_horizon,
                "rnn_type": self.model.rnn_type,
            },
            "norm_params": norm_params,
        }
        torch.save(checkpoint, file_path)

    @classmethod
    def load_checkpoint(cls, file_path: str, device: Optional[str] = None) -> Tuple[PIRNNSurrogate, Optional[Dict[str, Any]]]:
        """Load trained surrogate model from checkpoint."""
        dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(file_path, map_location=dev, weights_only=False)
        cfg = checkpoint["model_config"]
        
        model = PIRNNSurrogate(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            output_dim=cfg["output_dim"],
            pred_horizon=cfg["pred_horizon"],
            rnn_type=cfg["rnn_type"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(dev)
        model.eval()
        
        return model, checkpoint.get("norm_params")


def train_surrogate(
    X: np.ndarray,
    y: np.ndarray,
    norm_params: Dict[str, Any],
    epochs: int = 40,
    batch_size: int = 32,
    hidden_dim: int = 64,
    num_layers: int = 2,
    rnn_type: str = "lstm",
    lr: float = 1e-3,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[PIRNNSurrogate, Dict[str, list]]:
    """
    High-level convenience function to prepare data loaders, instantiate model, and train.
    """
    # 80/20 train/val split
    n_samples = len(X)
    n_train = int(n_samples * 0.8)
    indices = np.random.RandomState(42).permutation(n_samples)

    train_idx, val_idx = indices[:n_train], indices[n_train:]

    x_train_t = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train_t = torch.tensor(y[train_idx], dtype=torch.float32)
    x_val_t = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val_t = torch.tensor(y[val_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    input_dim = X.shape[-1]
    pred_horizon = y.shape[1]
    output_dim = y.shape[-1]

    model = PIRNNSurrogate(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        pred_horizon=pred_horizon,
        rnn_type=rnn_type,
    )

    loss_fn = PhysicsInformedLoss(norm_params=norm_params)
    trainer = SurrogateTrainer(model=model, loss_fn=loss_fn, lr=lr)
    history = trainer.fit(train_loader, val_loader, epochs=epochs, verbose=verbose)

    if save_path:
        trainer.save_checkpoint(save_path, norm_params=norm_params)
        if verbose:
            print(f"Model saved to {save_path}")

    return model, history
