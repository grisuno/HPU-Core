"""
Hamiltonian Audio Training Module.

Orchestrates the complete training pipeline:
- Data preparation (audio -> spectrogram patches)
- Training loop with all Hamiltonian loss terms
- Metric tracking and progress bar display
- Periodic checkpointing
- Learning rate scheduling with warmup
- Early stopping
- Validation

Follows Single Responsibility Principle: orchestration only.
All computation is delegated to specialized modules.

Follows Dependency Inversion Principle: depends on abstractions
(config, model, loss computer, metrics tracker, checkpoint manager)
rather than concrete implementations.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from typing import Optional, Tuple

from config import HamiltonianAudioConfig
from model import HamiltonianNeuralNetwork
from audio_io import AudioProcessor
from losses import HamiltonianLossComputer
from metrics import HamiltonianMetricsTracker
from checkpoint_manager import CheckpointManager


class AudioSpectrogramDatasetBuilder:
    """
    Builds a TensorDataset of spectrogram patches from an audio file.

    Segments the full mel spectrogram into overlapping patches
    of size (n_mels, matrix_size_width) for training.
    """

    def __init__(self, config: HamiltonianAudioConfig) -> None:
        self._config = config

    def build_dataset(
        self, mel_spectrogram: torch.Tensor
    ) -> TensorDataset:
        """
        Segment a mel spectrogram into training patches.

        Args:
            mel_spectrogram: Full spectrogram [1, 1, n_mels, time_frames].

        Returns:
            TensorDataset of (input_patch, target_patch) pairs.
        """
        spec = mel_spectrogram.squeeze(0)
        _, n_mels, total_frames = spec.shape
        patch_width = self._config.model.matrix_size_width
        stride = patch_width // 2
        patches = []
        start = 0
        while start + patch_width <= total_frames:
            patch = spec[:, :, start : start + patch_width]
            patches.append(patch)
            start += stride
        if len(patches) == 0:
            padded = torch.zeros(1, n_mels, patch_width, device=spec.device)
            padded[:, :, :total_frames] = spec[:, :, :total_frames]
            patches.append(padded)
        patches_tensor = torch.stack(patches, dim=0)
        return TensorDataset(patches_tensor, patches_tensor.clone())


class HamiltonianAudioTrainer:
    """
    Complete training pipeline for the Hamiltonian Audio Network.

    Manages:
    - Model initialization and checkpoint recovery
    - Optimizer and scheduler configuration
    - Training and validation loops
    - Full metric reporting at every step
    - Time-based checkpointing
    - Early stopping
    """

    def __init__(self, config: HamiltonianAudioConfig) -> None:
        self._config = config
        self._device = torch.device(config.device)
        config.validate_all()
        config.ensure_directories()
        self._model = HamiltonianNeuralNetwork(config.model).to(self._device)
        self._loss_computer = HamiltonianLossComputer(config.training)
        self._metrics_tracker = HamiltonianMetricsTracker(config.metrics)
        self._checkpoint_manager = CheckpointManager(config.checkpoint)
        self._audio_processor = AudioProcessor(config.audio, config.device)
        self._dataset_builder = AudioSpectrogramDatasetBuilder(config)
        self._optimizer = optim.AdamW(
            self._model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self._scheduler = optim.lr_scheduler.StepLR(
            self._optimizer,
            step_size=config.training.lr_scheduler_step_size,
            gamma=config.training.lr_scheduler_gamma,
        )
        self._warmup_scheduler = optim.lr_scheduler.LinearLR(
            self._optimizer,
            start_factor=config.training.warmup_start_factor,
            total_iters=config.training.warmup_epochs,
        )
        self._combined_scheduler = optim.lr_scheduler.SequentialLR(
            self._optimizer,
            schedulers=[self._warmup_scheduler, self._scheduler],
            milestones=[config.training.warmup_epochs],
        )
        self._start_epoch: int = 0
        self._global_step: int = 0
        self._best_val_loss: float = float("inf")
        self._patience_counter: int = 0
        self._attempt_checkpoint_recovery()

    def _attempt_checkpoint_recovery(self) -> None:
        """Load existing checkpoint if available."""
        metadata = self._checkpoint_manager.load_checkpoint(self._model)
        if metadata is not None:
            self._start_epoch = metadata.get("epoch", 0)
            self._global_step = metadata.get("step", 0)
            print(
                f"[CHECKPOINT] Recovered from epoch {self._start_epoch}, "
                f"step {self._global_step}, "
                f"best_loss={self._checkpoint_manager.best_loss:.6f}"
            )
        else:
            print("[CHECKPOINT] No existing checkpoint found. Training from scratch.")
        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(
            p.numel() for p in self._model.parameters() if p.requires_grad
        )
        print(
            f"[MODEL] Total parameters: {total_params:,} | "
            f"Trainable: {trainable_params:,}"
        )

    def train(self, audio_file_path: str) -> None:
        """
        Execute the full training pipeline on an audio file.

        Args:
            audio_file_path: Path to the input audio file.
        """
        print(f"[AUDIO] Loading: {audio_file_path}")
        waveform, sample_rate = self._audio_processor.load_audio(audio_file_path)
        print(
            f"[AUDIO] Loaded: {waveform.shape[1]} samples at {sample_rate} Hz "
            f"({waveform.shape[1] / sample_rate:.2f} seconds)"
        )
        db_min, db_max = self._audio_processor.get_spectrogram_db_range(waveform)
        mel_spec = self._audio_processor.waveform_to_mel_spectrogram(waveform)
        print(
            f"[SPECTROGRAM] Shape: {mel_spec.shape} | "
            f"dB range: [{db_min:.2f}, {db_max:.2f}]"
        )
        dataset = self._dataset_builder.build_dataset(mel_spec)
        total_size = len(dataset)
        val_size = max(1, int(total_size * self._config.training.validation_split_ratio))
        train_size = total_size - val_size
        generator = torch.Generator().manual_seed(self._config.training.random_seed)
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self._config.training.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self._config.training.pin_memory and self._device.type == "cuda",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self._config.training.batch_size,
            shuffle=False,
            num_workers=0,
        )
        print(
            f"[DATASET] Total patches: {total_size} | "
            f"Train: {train_size} | Validation: {val_size}"
        )
        print(
            f"[TRAINING] Starting epoch {self._start_epoch + 1} / "
            f"{self._config.training.num_epochs}"
        )
        print(
            f"[TRAINING] Checkpoint interval: "
            f"{self._config.checkpoint.checkpoint_interval_minutes} minutes"
        )
        for epoch in range(self._start_epoch, self._config.training.num_epochs):
            train_metrics = self._train_one_epoch(train_loader, epoch)
            val_metrics = self._validate(val_loader, epoch)
            self._combined_scheduler.step()
            current_lr = self._optimizer.param_groups[0]["lr"]
            self._metrics_tracker.record_learning_rate(current_lr)
            val_loss = val_metrics.get("total_loss", float("inf"))
            if val_loss < self._best_val_loss - self._config.training.early_stopping_min_delta:
                self._best_val_loss = val_loss
                self._patience_counter = 0
            else:
                self._patience_counter += 1
            print(
                f"[EPOCH {epoch + 1}/{self._config.training.num_epochs}] "
                f"Train Loss: {train_metrics.get('total_loss', 0.0):.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"Patience: {self._patience_counter}/{self._config.training.early_stopping_patience} | "
                f"Best: {self._best_val_loss:.6f}"
            )
            if self._checkpoint_manager.should_save_checkpoint():
                all_metrics = self._metrics_tracker.get_current_metrics()
                self._checkpoint_manager.save_checkpoint(
                    model=self._model,
                    optimizer=self._optimizer,
                    scheduler=self._combined_scheduler,
                    epoch=epoch + 1,
                    step=self._global_step,
                    metrics=all_metrics,
                    current_loss=val_loss,
                )
                print(
                    f"[CHECKPOINT] Saved at epoch {epoch + 1}, step {self._global_step}"
                )
            if self._patience_counter >= self._config.training.early_stopping_patience:
                print(
                    f"[EARLY STOPPING] No improvement for "
                    f"{self._config.training.early_stopping_patience} epochs. Stopping."
                )
                break
        all_metrics = self._metrics_tracker.get_current_metrics()
        self._checkpoint_manager.save_checkpoint(
            model=self._model,
            optimizer=self._optimizer,
            scheduler=self._combined_scheduler,
            epoch=self._config.training.num_epochs,
            step=self._global_step,
            metrics=all_metrics,
            current_loss=self._best_val_loss,
        )
        print("[TRAINING] Complete. Final checkpoint saved.")

    def _train_one_epoch(
        self, train_loader: DataLoader, epoch: int
    ) -> dict:
        """Execute one training epoch with full metric tracking."""
        self._model.train()
        epoch_metrics = {}
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}",
            unit="batch",
            leave=True,
            ncols=200,
        )
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            self._optimizer.zero_grad()
            predictions, intermediates = self._model.forward_with_intermediates(inputs)
            total_loss, loss_components = self._loss_computer.compute_total_loss(
                predictions, targets, intermediates, self._model
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                self._config.training.gradient_clip_max_norm,
            )
            self._optimizer.step()
            self._global_step += 1
            self._metrics_tracker.increment_step()
            for name, value in loss_components.items():
                self._metrics_tracker.record_loss_component(name, value)
            grad_norm = self._metrics_tracker.record_gradient_norm(
                self._model.parameters()
            )
            param_norm = self._metrics_tracker.record_parameter_norm(
                self._model.parameters()
            )
            if len(intermediates) >= 2:
                q = intermediates[0].detach()
                p = (intermediates[1] - intermediates[0]).detach()
                h_energy = self._metrics_tracker.compute_hamiltonian_energy(q, p)
                self._metrics_tracker.compute_phase_space_volume(q, p)
                if len(intermediates) >= 3:
                    dq = (intermediates[1] - intermediates[0]).detach()
                    dp = (intermediates[2] - intermediates[1]).detach()
                    self._metrics_tracker.compute_symplectic_form(q, p, dq, dp)
            with torch.no_grad():
                pred_fft = torch.fft.rfft2(predictions)
                target_fft = torch.fft.rfft2(targets)
                self._metrics_tracker.compute_spectral_entropy(torch.abs(pred_fft))
                self._metrics_tracker.compute_reconstruction_snr(targets, predictions)
                self._metrics_tracker.compute_spectral_convergence(
                    torch.abs(target_fft), torch.abs(pred_fft)
                )
                phase_pred = torch.angle(pred_fft)
                phase_target = torch.angle(target_fft)
                self._metrics_tracker.compute_phase_coherence(phase_target, phase_pred)
            current = self._metrics_tracker.get_current_metrics()
            epoch_metrics = current
            progress_bar.set_postfix(
                loss=f"{current.get('total_loss', 0.0):.4e}",
                recon=f"{current.get('reconstruction_loss', 0.0):.4e}",
                energy=f"{current.get('energy_conservation_loss', 0.0):.4e}",
                sympl=f"{current.get('symplectic_loss', 0.0):.4e}",
                spec_conv=f"{current.get('spectral_convergence', 0.0):.4f}",
                snr=f"{current.get('reconstruction_snr', 0.0):.2f}dB",
                phase_coh=f"{current.get('phase_coherence', 0.0):.4f}",
                grad=f"{grad_norm:.4f}",
                h_energy=f"{current.get('hamiltonian_energy', 0.0):.4e}",
                action=f"{current.get('action_minimization_loss', 0.0):.4e}",
                liouv=f"{current.get('liouville_loss', 0.0):.4e}",
                ham_cstr=f"{current.get('hamiltonian_constraint_loss', 0.0):.4e}",
            )
            if self._checkpoint_manager.should_save_checkpoint():
                all_metrics = self._metrics_tracker.get_current_metrics()
                self._checkpoint_manager.save_checkpoint(
                    model=self._model,
                    optimizer=self._optimizer,
                    scheduler=self._combined_scheduler,
                    epoch=epoch + 1,
                    step=self._global_step,
                    metrics=all_metrics,
                    current_loss=current.get("total_loss", float("inf")),
                )
                progress_bar.write(
                    f"  [CHECKPOINT] Saved at step {self._global_step}"
                )
        return epoch_metrics

    def _validate(
        self, val_loader: DataLoader, epoch: int
    ) -> dict:
        """Run validation pass and return metrics."""
        self._model.eval()
        total_loss_accum = 0.0
        component_accum = {}
        num_batches = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                predictions, intermediates = self._model.forward_with_intermediates(
                    inputs
                )
                total_loss, loss_components = self._loss_computer.compute_total_loss(
                    predictions, targets, intermediates, self._model
                )
                total_loss_accum += total_loss.item()
                for name, value in loss_components.items():
                    component_accum[name] = component_accum.get(name, 0.0) + value
                num_batches += 1
        if num_batches > 0:
            avg_metrics = {
                name: value / num_batches
                for name, value in component_accum.items()
            }
            avg_metrics["total_loss"] = total_loss_accum / num_batches
        else:
            avg_metrics = {"total_loss": float("inf")}
        return avg_metrics

    @property
    def model(self) -> HamiltonianNeuralNetwork:
        return self._model

    @property
    def audio_processor(self) -> AudioProcessor:
        return self._audio_processor
