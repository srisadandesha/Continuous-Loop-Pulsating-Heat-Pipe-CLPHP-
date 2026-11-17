"""
Training and Evaluation Pipeline for CLPHP Physics-Informed Recurrent Neural Networks.

Handles data splitting, mini-batch training, physics loss backpropagation,
validation tracking, learning rate decay, and model serialization.
"""

from __future__ import annotations
import os
import time
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from clphp.models.rnn_surrogate import PIRNNSurrogate
from clphp.models.pinn_loss import PhysicsInformedLoss


def get_default_device() -> torch.device:
    """Detect and return the best available compute device (CUDA, DirectML, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            self.device = get_default_device()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

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
        best_state = None

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

            # Select on the data-only validation loss so runs with different physics
            # weights (ablations) are compared with the same criterion.
            if val_metrics["val_data_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_data_loss"]
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs or epoch == 1):
                print(
                    f"Epoch {epoch:03d}/{epochs:03d} | "
                    f"Train Loss: {train_metrics['loss_total']:.6f} | "
                    f"Val Loss: {val_metrics['val_loss']:.6f} | "
                    f"LR: {current_lr:.2e}"
                )

        # Restore the weights with the best loss on the held-out *validation runs*
        if best_state is not None:
            self.model.load_state_dict(best_state)

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
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    norm_params: Dict[str, Any],
    epochs: int = 40,
    batch_size: int = 32,
    hidden_dim: int = 64,
    num_layers: int = 2,
    rnn_type: str = "lstm",
    lr: float = 1e-3,
    device: Optional[Union[str, torch.device]] = None,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[PIRNNSurrogate, Dict[str, list]]:
    """
    Prepare data loaders, instantiate the model and train.

    The caller supplies windows that were already cut from *separate* train
    and validation runs (see ``split_runs_by_q_in``).  No random window-level
    split happens here: shuffling windows from the same run into both train
    and validation sets leaks information between them.
    """
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("train_surrogate needs non-empty train and validation windows")

    x_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[-1]
    pred_horizon = y_train.shape[1]
    output_dim = y_train.shape[-1]

    model = PIRNNSurrogate(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        pred_horizon=pred_horizon,
        rnn_type=rnn_type,
    )

    loss_fn = PhysicsInformedLoss(norm_params=norm_params)
    trainer = SurrogateTrainer(model=model, loss_fn=loss_fn, lr=lr, device=device)
    history = trainer.fit(train_loader, val_loader, epochs=epochs, verbose=verbose)

    if save_path:
        trainer.save_checkpoint(save_path, norm_params=norm_params)
        if verbose:
            print(f"Model saved to {save_path}")

    return model, history
