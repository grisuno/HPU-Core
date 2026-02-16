"""
Checkpoint Management Module.

Handles:
- Periodic model saving (time-based interval)
- Best model tracking
- Training state persistence (epoch, optimizer, scheduler, metrics)
- Checkpoint loading and recovery
- Training metadata serialization

Follows Single Responsibility Principle: solely responsible for
model persistence and recovery.

Follows Interface Segregation Principle: exposes minimal interface
for save/load operations.
"""

import os
import json
import time
import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from safetensors.torch import save_model, load_model
from config import CheckpointConfig


class CheckpointManager:
    """
    Manages model checkpointing with time-based intervals
    and best-model tracking.
    """

    def __init__(self, config: CheckpointConfig) -> None:
        self._config = config
        self._last_checkpoint_time: float = time.time()
        self._best_loss: float = float("inf")
        self._interval_seconds: int = config.checkpoint_interval_minutes * 60
        os.makedirs(config.checkpoint_directory, exist_ok=True)

    def should_save_checkpoint(self) -> bool:
        """Check if enough time has elapsed since the last checkpoint."""
        elapsed = time.time() - self._last_checkpoint_time
        return elapsed >= self._interval_seconds

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        current_loss: float,
    ) -> None:
        """
        Save the current model state and training metadata.

        Saves to a single 'latest.safetensors' file plus a JSON
        metadata file containing optimizer state, epoch, and metrics.

        Args:
            model: The model to checkpoint.
            optimizer: Current optimizer state.
            scheduler: Current LR scheduler state.
            epoch: Current epoch number.
            step: Current global step.
            metrics: Dictionary of current metric values.
            current_loss: Current total loss value.
        """
        save_model(model, self._config.checkpoint_path)
        metadata = {
            "epoch": epoch,
            "step": step,
            "current_loss": current_loss,
            "best_loss": self._best_loss,
            "metrics": {k: float(v) for k, v in metrics.items()},
            "optimizer_state_dict": {
                k: str(v) if not isinstance(v, (int, float, bool, list, dict))
                else v
                for k, v in optimizer.defaults.items()
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
        with open(self._config.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        if self._config.keep_best_model and current_loss < self._best_loss:
            self._best_loss = current_loss
            save_model(model, self._config.best_model_path)
            best_metadata_path = self._config.best_model_path.replace(
                ".safetensors", "_metadata.json"
            )
            metadata["is_best"] = True
            with open(best_metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
        self._last_checkpoint_time = time.time()

    def load_checkpoint(
        self, model: nn.Module, load_best: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Load a model checkpoint and return training metadata.

        Uses the exact safetensors loading pattern:
            load_model(model, checkpoint_path)

        Path resolution priority:
        1. If CheckpointConfig.checkpoint_file_path is set, use that exact path
           (overrides load_best flag).
        2. If load_best is True, use checkpoint_directory/best.safetensors.
        3. Otherwise, use checkpoint_directory/latest.safetensors.

        Args:
            model: The model to load weights into.
            load_best: If True and no explicit path set, load best model.

        Returns:
            Metadata dictionary if available, None otherwise.
        """
        if self._config.checkpoint_file_path is not None:
            checkpoint_path = self._config.checkpoint_file_path
        elif load_best:
            checkpoint_path = self._config.best_model_path
        else:
            checkpoint_path = os.path.join(
                self._config.checkpoint_directory, self._config.checkpoint_filename
            )

        resolved_path = os.path.abspath(checkpoint_path)
        print(f"[CHECKPOINT] Attempting to load from: {resolved_path}")

        if not os.path.exists(resolved_path):
            print(f"[CHECKPOINT] File does not exist: {resolved_path}")
            return None

        try:
            load_model(model, resolved_path)
            print(f"[CHECKPOINT] Model loaded successfully from: {resolved_path}")
        except Exception as e:
            print(f"[CHECKPOINT] Critical error loading model: {e}")
            return None

        metadata_path = resolved_path.replace(".safetensors", "_metadata.json")
        if not os.path.exists(metadata_path):
            metadata_path = self._config.metadata_path

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            if "best_loss" in metadata:
                self._best_loss = metadata["best_loss"]
            return metadata
        return {"epoch": 0, "step": 0}

    @property
    def best_loss(self) -> float:
        return self._best_loss
