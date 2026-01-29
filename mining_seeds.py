#!/usr/bin/env python3
"""
app.py

Author: Gris Iscomeback
Email: grisiscomeback@gmail.com
Date of creation: 2026
License: AGPL v3

Description:  
Hamiltonian Grokking - Mining HPU Crystals
Sequential seed mining for deterministic reproducibility
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import deque
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.linalg import eigh
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA


@dataclass
class Config:
    # Core dimensions
    GRID_SIZE: int = 16
    HIDDEN_DIM: int = 32
    NUM_SPECTRAL_LAYERS: int = 2
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.005
    WEIGHT_DECAY: float = 1e-4
    EPOCHS: int = 5000
    CHECKPOINT_INTERVAL_MINUTES: int = 5
    MAX_CHECKPOINTS: int = 10
    TARGET_ACCURACY: float = 0.90
    TIME_STEPS: int = 2
    DT: float = 0.01
    TRAIN_RATIO: float = 0.7
    NUM_SAMPLES: int = 200
    
    # Metrics and analysis
    ENTROPY_BINS: int = 50
    PCA_COMPONENTS: int = 2
    KDE_BANDWIDTH: str = 'scott'
    MIN_VARIANCE_THRESHOLD: float = 1e-8
    ENTROPY_EPS: float = 1e-10
    HBAR: float = 1e-6
    POYNTING_THRESHOLD: float = 1.0
    ENERGY_FLOW_SCALE: float = 0.1
    DISCRETIZATION_MARGIN: float = 0.1
    TARGET_SLOTS: int = 7
    
    # Device
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_SEED: int = 42
    LOG_LEVEL: str = 'INFO'
    RESULTS_DIR: str = 'boltzmann_results'
    
    # Mining parameters
    MINING_MAX_ATTEMPTS: int = 1000
    MINING_START_SEED: int = 1
    MINING_GLASS_PATIENCE_EPOCHS: int = 50
    MINING_TARGET_LC: float = 0.01
    MINING_TARGET_SP: float = 0.01
    MINING_TARGET_KAPPA: float = 1.01
    MINING_TARGET_DELTA: float = 0.001
    MINING_TARGET_TEMP: float = 1e-10
    MINING_TARGET_CV: float = 1e-10


def set_seed(seed: int = Config.RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if Config.DEVICE == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_logger(name: str, level: str = Config.LOG_LEVEL) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class IAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        pass


class IMetricsCalculator(ABC):
    @abstractmethod
    def compute(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        pass


class HamiltonianOperator:
    """True Hamiltonian operator H = -nabla^2 on torus."""
    
    def __init__(self, grid_size: int = Config.GRID_SIZE):
        self.grid_size = grid_size
        self._precompute_spectral_operators()
    
    def _precompute_spectral_operators(self):
        kx = torch.fft.fftfreq(self.grid_size, d=1.0) * 2 * np.pi
        ky = torch.fft.fftfreq(self.grid_size, d=1.0) * 2 * np.pi
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        
        self.laplacian_spectrum = -(KX**2 + KY**2).float()
    
    def apply(self, field: torch.Tensor) -> torch.Tensor:
        field_fft = torch.fft.fft2(field)
        laplacian_fft = field_fft * self.laplacian_spectrum
        return torch.fft.ifft2(laplacian_fft).real
    
    def time_evolution(self, field: torch.Tensor, dt: float = Config.DT) -> torch.Tensor:
        hamiltonian_action = self.apply(field)
        evolved = field + hamiltonian_action * dt
        return evolved / (torch.norm(evolved) + 1e-8) * torch.norm(field)


class FastDataset(Dataset):
    """Fast dataset for Hamiltonian operator learning."""
    
    def __init__(
        self,
        num_samples: int = Config.NUM_SAMPLES,
        grid_size: int = Config.GRID_SIZE,
        time_steps: int = Config.TIME_STEPS,
        dt: float = Config.DT,
        seed: int = Config.RANDOM_SEED,
        train_ratio: float = Config.TRAIN_RATIO
    ):
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.time_steps = time_steps
        self.dt = dt
        self.train_ratio = train_ratio
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.hamiltonian = HamiltonianOperator(grid_size)
        
        # Generate samples
        self.initial_fields = []
        self.target_fields = []
        
        for i in range(num_samples):
            field = torch.randn(grid_size, grid_size)
            field = field / (torch.norm(field) + 1e-8)
            
            evolved = field.clone()
            for _ in range(time_steps):
                evolved = self.hamiltonian.time_evolution(evolved, dt)
            
            self.initial_fields.append(field)
            self.target_fields.append(evolved)
        
        self.initial_fields = torch.stack(self.initial_fields)
        self.target_fields = torch.stack(self.target_fields)
        
        split_idx = int(num_samples * train_ratio)
        self.train_fields = self.initial_fields[:split_idx]
        self.train_targets = self.target_fields[:split_idx]
        self.val_fields = self.initial_fields[split_idx:]
        self.val_targets = self.target_fields[split_idx:]
    
    def __len__(self):
        return len(self.train_fields)
    
    def __getitem__(self, idx):
        return self.train_fields[idx], self.train_targets[idx]
    
    def get_val_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.val_fields, self.val_targets


class SpectralLayer(nn.Module):
    """Spectral layer with correct complex multiplication."""
    
    def __init__(self, channels: int, grid_size: int):
        super().__init__()
        self.channels = channels
        self.grid_size = grid_size
        
        # CORRECT: BOTH kernel_real AND kernel_imag
        self.kernel_real = nn.Parameter(
            torch.randn(channels, channels, grid_size // 2 + 1, grid_size) * 0.1
        )
        self.kernel_imag = nn.Parameter(
            torch.randn(channels, channels, grid_size // 2 + 1, grid_size) * 0.1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fft = torch.fft.rfft2(x)
        batch, channels, freq_h, freq_w = x_fft.shape
        
        # Use both kernels
        kernel_real = self.kernel_real.mean(dim=0)
        kernel_imag = self.kernel_imag.mean(dim=0)
        
        kernel_real_exp = kernel_real.unsqueeze(0).unsqueeze(0).squeeze(0)
        kernel_imag_exp = kernel_imag.unsqueeze(0).unsqueeze(0).squeeze(0)
        
        # Interpolate to match
        kernel_real_interp = F.interpolate(
            kernel_real_exp,
            size=(freq_h, freq_w),
            mode='bilinear',
            align_corners=False
        )
        kernel_imag_interp = F.interpolate(
            kernel_imag_exp,
            size=(freq_h, freq_w),
            mode='bilinear',
            align_corners=False
        )
        
        # CORRECT: Full complex multiplication
        # (a + ib)(c + id) = (ac - bd) + i(ad + bc)
        real_part = x_fft.real * kernel_real_interp - x_fft.imag * kernel_imag_interp
        imag_part = x_fft.real * kernel_imag_interp + x_fft.imag * kernel_real_interp
        
        output_fft = torch.complex(real_part, imag_part)
        output = torch.fft.irfft2(output_fft, s=(self.grid_size, self.grid_size))
        
        return output


class SimpleHamiltonianNet(nn.Module):
    """Compact network for Hamiltonian operator learning."""
    
    def __init__(
        self,
        grid_size: int = Config.GRID_SIZE,
        hidden_dim: int = Config.HIDDEN_DIM,
        num_spectral_layers: int = Config.NUM_SPECTRAL_LAYERS
    ):
        super().__init__()
        self.grid_size = grid_size
        
        # Initial projection
        self.input_proj = nn.Conv2d(1, hidden_dim, kernel_size=1)
        
        # Spectral layers
        self.spectral_layers = nn.ModuleList([
            SpectralLayer(hidden_dim, grid_size)
            for _ in range(num_spectral_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_dim, 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = F.gelu(self.input_proj(x))
        
        for spectral_layer in self.spectral_layers:
            x = F.gelu(spectral_layer(x))
        
        return self.output_proj(x).squeeze(1)


class LocalComplexityAnalyzer:
    @staticmethod
    def compute_local_complexity(weights: torch.Tensor, epsilon: float = 1e-6) -> float:
        """Compute Local Complexity (LC) metric for weight matrix."""
        if weights.numel() == 0:
            return 0.0
        
        w = weights.flatten()
        w = w / (torch.norm(w) + epsilon)
        w_expanded = w.unsqueeze(0)
        similarities = F.cosine_similarity(w_expanded, w_expanded.unsqueeze(1), dim=2)
        mask = ~torch.eye(similarities.size(0), device=similarities.device, dtype=torch.bool)
        avg_similarity = (similarities.abs() * mask).sum() / mask.sum()
        lc = 1.0 - avg_similarity.item()
        return max(0.0, min(1.0, lc))


class SuperpositionAnalyzer:
    @staticmethod
    def compute_superposition(weights: torch.Tensor) -> float:
        """Compute Superposition (SP) metric for weight matrix."""
        if weights.size(0) < 2:
            return 0.0
        
        if weights.dim() > 2:
            weights = weights.reshape(weights.size(0), -1)
        
        if weights.size(0) < 2:
            return 0.0
        
        correlation_matrix = torch.corrcoef(weights)
        
        if correlation_matrix.numel() == 0:
            return 0.0
        
        correlation_matrix = correlation_matrix.nan_to_num(nan=0.0)
        
        n = correlation_matrix.size(0)
        mask = ~torch.eye(n, device=correlation_matrix.device, dtype=torch.bool)
        
        if mask.sum() == 0:
            return 0.0
        
        avg_correlation = (correlation_matrix.abs() * mask).sum() / mask.sum()
        return avg_correlation.item()


class CrystallographyMetrics:
    @staticmethod
    def compute_kappa(model: nn.Module, dataloader, num_batches: int = 1) -> float:
        model.eval()
        grad_norms = []
        
        for i, (batch_x, batch_y) in enumerate(dataloader):
            if i >= num_batches:
                break
            batch_x, batch_y = batch_x.to(Config.DEVICE), batch_y.to(Config.DEVICE)
            
            model.zero_grad()
            outputs = model(batch_x)
            loss = nn.MSELoss()(outputs, batch_y)
            try:
                grad = torch.autograd.grad(loss, model.parameters(), create_graph=False)
                grad_norms.append(torch.cat([g.flatten()[:500] for g in grad if g.nelement() > 0]))  # Limit size
            except Exception:
                continue
        
        if len(grad_norms) < 2:
            return 1.0  # Return well-conditioned value for safety
        
        try:
            grads_tensor = torch.stack([g[:500] for g in grad_norms])  # Further limit size
            if grads_tensor.size(0) < 2 or grads_tensor.size(1) < 2:
                return 1.0
            
            cov_matrix = torch.cov(grads_tensor.T)
            eigenvalues = torch.linalg.eigvals(cov_matrix).real
            eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter small values
            if len(eigenvalues) < 2:
                return 1.0
            return (eigenvalues.max() / eigenvalues.min()).item()
        except Exception:
            return 1.0
    
    @staticmethod
    def compute_discretization_margin(coeffs: Dict[str, torch.Tensor]) -> float:
        margins = []
        for tensor in coeffs.values():
            if tensor.numel() > 0:
                margin = (tensor - tensor.round()).abs().max().item()
                margins.append(margin)
        return max(margins) if margins else 0.0
    
    @staticmethod
    def compute_alpha_purity(coeffs: Dict[str, torch.Tensor]) -> float:
        delta = CrystallographyMetrics.compute_discretization_margin(coeffs)
        if delta < 1e-10:
            return 20.0
        return -np.log(delta + 1e-15)
    
    @staticmethod
    def compute_kappa_quantum(coeffs: Dict[str, torch.Tensor], hbar: float = Config.HBAR) -> float:
        flat_params = torch.cat([c.flatten()[:1000] for c in coeffs.values()])  # Limit size
        n = flat_params.numel()
        if n < 2:
            return 1.0
        
        params_centered = flat_params - flat_params.mean()
        cov_matrix = torch.outer(params_centered, params_centered) / n
        cov_matrix = cov_matrix + hbar * torch.eye(n, device=flat_params.device)
        try:
            eigenvals = torch.linalg.eigvalsh(cov_matrix)
            eigenvals = eigenvals[eigenvals > hbar]
            return (eigenvals.max() / eigenvals.min()).item() if len(eigenvals) > 0 else 1.0
        except Exception:
            return 1.0
    
    @staticmethod
    def compute_poynting_vector(coeffs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        E = torch.cat([c.flatten()[:1000] for c in coeffs.values()])  # Limit size
        H_magnitude = torch.norm(torch.matmul(E.unsqueeze(1)[:50, :50], E.unsqueeze(0)[:50, :50]) - torch.eye(min(50, E.size(0)), device=E.device))
        poynting_magnitude = torch.norm(E) * H_magnitude * Config.ENERGY_FLOW_SCALE
        energy_distribution = {
            'magnitude': torch.norm(E).item()
        }
        return {
            'poynting_magnitude': poynting_magnitude.item(),
            'energy_distribution': energy_distribution,
            'is_radiating': poynting_magnitude.item() > Config.POYNTING_THRESHOLD,
            'field_orthogonality': H_magnitude.item()
        }
    
    @staticmethod
    def compute_all_metrics(model: nn.Module, dataloader) -> Dict[str, Any]:
        coeffs = {}
        for name, param in model.named_parameters():
            coeffs[name] = param.data.clone()
        
        metrics = {
            'kappa': CrystallographyMetrics.compute_kappa(model, dataloader),
            'delta': CrystallographyMetrics.compute_discretization_margin(coeffs),
            'alpha': CrystallographyMetrics.compute_alpha_purity(coeffs),
            'kappa_q': CrystallographyMetrics.compute_kappa_quantum(coeffs),
            'poynting': CrystallographyMetrics.compute_poynting_vector(coeffs)
        }
        metrics['purity_index'] = 1.0 - metrics['delta']
        metrics['is_crystal'] = metrics['alpha'] > 7.0
        metrics['energy_flow'] = metrics['poynting']['poynting_magnitude']
        return metrics


class ThermodynamicMetrics:
    @staticmethod
    def compute_effective_temperature(gradient_buffer: List[torch.Tensor], learning_rate: float) -> float:
        if len(gradient_buffer) < 2:
            return 0.0
        # Limit buffer size to prevent memory issues
        limited_gradients = [g.flatten()[:500] for g in gradient_buffer[-10:]]
        if not limited_gradients:
            return 0.0
        grads = torch.stack(limited_gradients)
        second_moment = torch.mean(torch.norm(grads, dim=1)**2)
        first_moment_sq = torch.norm(torch.mean(grads, dim=0))**2
        variance = second_moment - first_moment_sq
        return float((learning_rate / 2.0) * variance)
    
    @staticmethod
    def compute_specific_heat(loss_history: List[float], temp_history: List[float], cv_threshold: float = 1.0) -> Tuple[float, bool]:
        if len(loss_history) < 2 or len(temp_history) < 2:
            return 0.0, False
        u_var = torch.tensor(loss_history[-50:]).var()  # Limit history
        t_mean = torch.tensor(temp_history[-50:]).mean()  # Limit history
        cv = u_var / (t_mean**2 + 1e-10)
        is_latent_crystallization = cv > cv_threshold
        return float(cv), is_latent_crystization


class SpectroscopyMetrics:
    @staticmethod
    def compute_weight_diffraction(coeffs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        W = torch.cat([c.flatten()[:1000] for c in coeffs.values()])  # Limit size
        W_reshaped = W.reshape(-1, 1)
        fft_spectrum = torch.fft.fft(W_reshaped.squeeze())
        power_spectrum = torch.abs(fft_spectrum)**2
        peaks = []
        threshold = torch.mean(power_spectrum) + 2 * torch.std(power_spectrum)
        for i, power in enumerate(power_spectrum):
            if power > threshold and len(peaks) < 10:  # Limit number of peaks
                peaks.append({'frequency': i, 'intensity': float(power)})
        is_crystalline = len(peaks) > 0 and len(peaks) < len(power_spectrum) // 2
        return {
            'power_spectrum': power_spectrum.cpu().numpy().tolist()[:100],  # Limit spectrum
            'bragg_peaks': peaks,
            'is_crystalline_structure': is_crystalline,
            'spectral_entropy': float(SpectroscopyMetrics._compute_spectral_entropy(power_spectrum))
        }
    
    @staticmethod
    def _compute_spectral_entropy(power_spectrum: torch.Tensor) -> float:
        ps_normalized = power_spectrum / (torch.sum(power_spectrum) + 1e-10)
        ps_normalized = ps_normalized[ps_normalized > 1e-10]
        if len(ps_normalized) == 0:
            return 0.0
        entropy = -torch.sum(ps_normalized * torch.log(ps_normalized + 1e-10))
        return float(entropy)


class CheckpointManager:
    def __init__(self, interval_minutes: int = Config.CHECKPOINT_INTERVAL_MINUTES, max_checkpoints: int = Config.MAX_CHECKPOINTS):
        self.interval_minutes = interval_minutes
        self.max_checkpoints = max_checkpoints
        self.last_checkpoint_time = time.time()
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_files = []
    
    def should_save_checkpoint(self) -> bool:
        current_time = time.time()
        elapsed_minutes = (current_time - self.last_checkpoint_time) / 60
        return elapsed_minutes >= self.interval_minutes
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, epoch: int, metrics: Dict[str, Any]):
        # Create simplified checkpoint to avoid pickling issues
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': {
                'GRID_SIZE': Config.GRID_SIZE,
                'HIDDEN_DIM': Config.HIDDEN_DIM,
                'NUM_SPECTRAL_LAYERS': Config.NUM_SPECTRAL_LAYERS,
                'BATCH_SIZE': Config.BATCH_SIZE,
                'LEARNING_RATE': Config.LEARNING_RATE,
                'WEIGHT_DECAY': Config.WEIGHT_DECAY,
                'EPOCHS': Config.EPOCHS,
                'CHECKPOINT_INTERVAL_MINUTES': Config.CHECKPOINT_INTERVAL_MINUTES,
                'MAX_CHECKPOINTS': Config.MAX_CHECKPOINTS,
                'TARGET_ACCURACY': Config.TARGET_ACCURACY,
                'TIME_STEPS': Config.TIME_STEPS,
                'DT': Config.DT,
                'TRAIN_RATIO': Config.TRAIN_RATIO,
                'NUM_SAMPLES': Config.NUM_SAMPLES,
                'ENTROPY_BINS': Config.ENTROPY_BINS,
                'PCA_COMPONENTS': Config.PCA_COMPONENTS,
                'KDE_BANDWIDTH': Config.KDE_BANDWIDTH,
                'MIN_VARIANCE_THRESHOLD': Config.MIN_VARIANCE_THRESHOLD,
                'ENTROPY_EPS': Config.ENTROPY_EPS,
                'HBAR': Config.HBAR,
                'POYNTING_THRESHOLD': Config.POYNTING_THRESHOLD,
                'ENERGY_FLOW_SCALE': Config.ENERGY_FLOW_SCALE,
                'DISCRETIZATION_MARGIN': Config.DISCRETIZATION_MARGIN,
                'TARGET_SLOTS': Config.TARGET_SLOTS,
                'DEVICE': Config.DEVICE,
                'RANDOM_SEED': Config.RANDOM_SEED,
                'LOG_LEVEL': Config.LOG_LEVEL,
                'RESULTS_DIR': Config.RESULTS_DIR
            }
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}_{timestamp}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        self.checkpoint_files.append(checkpoint_path)
        
        # Keep only max_checkpoints most recent files
        if len(self.checkpoint_files) > self.max_checkpoints:
            oldest_file = self.checkpoint_files.pop(0)
            if os.path.exists(oldest_file):
                os.remove(oldest_file)
        
        # Update latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, "latest.pth")
        torch.save(checkpoint, latest_path)
        
        self.last_checkpoint_time = time.time()
        return checkpoint_path


class TrainingMonitor:
    def __init__(self):
        self.metrics_history = {
            'epoch': [],
            'loss': [],
            'val_loss': [],
            'val_acc': [],
            'lc': [],
            'sp': [],
            'alpha': [],
            'kappa': [],
            'delta': [],
            'temperature': [],
            'specific_heat': [],
            'poynting_magnitude': []
        }
        self.gradient_buffer = deque(maxlen=50)
        self.loss_history = deque(maxlen=100)
        self.temp_history = deque(maxlen=100)
        self.cv_history = deque(maxlen=100)
    
    def update_metrics(self, epoch: int, loss: float, val_loss: float, val_acc: float, 
                      lc: float, sp: float, alpha: float, kappa: float, delta: float,
                      temperature: float, specific_heat: float, poynting_magnitude: float):
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['loss'].append(loss)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['val_acc'].append(val_acc)
        self.metrics_history['lc'].append(lc)
        self.metrics_history['sp'].append(sp)
        self.metrics_history['alpha'].append(alpha)
        self.metrics_history['kappa'].append(kappa)
        self.metrics_history['delta'].append(delta)
        self.metrics_history['temperature'].append(temperature)
        self.metrics_history['specific_heat'].append(specific_heat)
        self.metrics_history['poynting_magnitude'].append(poynting_magnitude)


class GlassStopper:
    def __init__(self, patience_epochs: int = Config.MINING_GLASS_PATIENCE_EPOCHS):
        self.patience_epochs = patience_epochs
        self.metrics_buffer = deque(maxlen=patience_epochs)
    
    def should_stop(self, epoch: int, lc: float, sp: float, kappa: float, 
                   delta: float, temp: float, cv: float) -> bool:
        """Check if the system is in glass state and should stop mining."""
        # Store current metrics
        self.metrics_buffer.append({
            'epoch': epoch,
            'lc': lc,
            'sp': sp,
            'kappa': kappa,
            'delta': delta,
            'temp': temp,
            'cv': cv
        })
        
        # Check if we've exceeded patience epochs
        if epoch > self.patience_epochs:
            # Get the metrics from the patience period
            recent_metrics = list(self.metrics_buffer)[-self.patience_epochs:]
            
            # Calculate averages for the patience period
            avg_lc = np.mean([m['lc'] for m in recent_metrics])
            avg_sp = np.mean([m['sp'] for m in recent_metrics])
            avg_kappa = np.mean([m['kappa'] for m in recent_metrics])
            avg_delta = np.mean([m['delta'] for m in recent_metrics])
            avg_temp = np.mean([m['temp'] for m in recent_metrics])
            avg_cv = np.mean([m['cv'] for m in recent_metrics])
            
            # Check if metrics indicate glass state
            is_glass = (
                avg_lc > Config.MINING_TARGET_LC or  # Not approaching 0
                avg_sp > Config.MINING_TARGET_SP or  # Not approaching 0
                avg_kappa > Config.MINING_TARGET_KAPPA or  # Not approaching 1
                avg_delta > Config.MINING_TARGET_DELTA or  # Not approaching 0
                avg_temp > Config.MINING_TARGET_TEMP or  # Not approaching 0
                avg_cv > Config.MINING_TARGET_CV  # Not approaching 0
            )
            
            return is_glass
        
        # Check if we haven't made progress in the first 50 epochs
        if epoch <= self.patience_epochs:
            # If after 50 epochs we don't see improvement toward targets, stop
            if epoch == self.patience_epochs:
                if (lc > Config.MINING_TARGET_LC or 
                    sp > Config.MINING_TARGET_SP or 
                    kappa > Config.MINING_TARGET_KAPPA or 
                    delta > Config.MINING_TARGET_DELTA or 
                    temp > Config.MINING_TARGET_TEMP or 
                    cv > Config.MINING_TARGET_CV):
                    return True
        
        return False


def train_with_early_glass_stop(
    model: nn.Module,
    optimizer: optim.Optimizer,
    seed: int,
    epochs: int = 500
) -> bool:
    """Train model with early stopping for glass detection."""
    device = Config.DEVICE
    dataset = FastDataset(num_samples=Config.NUM_SAMPLES, grid_size=Config.GRID_SIZE, time_steps=Config.TIME_STEPS, seed=seed)
    train_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=Config.LEARNING_RATE * 0.01,
        max_lr=Config.LEARNING_RATE * 2.0,
        step_size_up=50,
        mode='triangular2',
        cycle_momentum=False
    )
    
    monitor = TrainingMonitor()
    lc_analyzer = LocalComplexityAnalyzer()
    sp_analyzer = SuperpositionAnalyzer()
    glass_stopper = GlassStopper(patience_epochs=Config.MINING_GLASS_PATIENCE_EPOCHS)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Add noise every 25 epochs to escape local minima
            if epoch % 25 == 0:
                for param in model.parameters():
                    if param.grad is not None:
                        noise = torch.randn_like(param.grad) * 0.01
                        param.grad.add_(noise)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_x, val_y = dataset.get_val_batch()
        val_x = val_x.to(device)
        val_y = val_y.to(device)
        
        with torch.no_grad():
            val_outputs = model(val_x)
            val_loss = criterion(val_outputs, val_y)
            
            mse_per_sample = ((val_outputs - val_y) ** 2).mean(dim=(1, 2))
            val_acc = (mse_per_sample < 0.05).float().mean().item()
        
        # Compute metrics
        lc_values = []
        sp_values = []
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                w = param[:min(param.size(0), 256), :min(param.size(1), 256)]
                lc = lc_analyzer.compute_local_complexity(w)
                sp = sp_analyzer.compute_superposition(w)
                lc_values.append(lc)
                sp_values.append(sp)
        
        lc = np.mean(lc_values) if lc_values else 0.0
        sp = np.mean(sp_values) if sp_values else 0.0
        
        # Additional metrics
        crystal_metrics = CrystallographyMetrics.compute_all_metrics(model, train_loader)
        alpha = crystal_metrics.get('alpha', 0.0)
        kappa = crystal_metrics.get('kappa', 1.0)
        delta = crystal_metrics.get('delta', 1.0)
        
        # Thermodynamic metrics
        temp = ThermodynamicMetrics.compute_effective_temperature(monitor.gradient_buffer, Config.LEARNING_RATE)
        cv, _ = ThermodynamicMetrics.compute_specific_heat(monitor.loss_history, monitor.temp_history)
        
        # Update monitor
        monitor.update_metrics(
            epoch=epoch,
            loss=total_loss/total_samples,
            val_loss=val_loss.item(),
            val_acc=val_acc,
            lc=lc,
            sp=sp,
            alpha=alpha,
            kappa=kappa,
            delta=delta,
            temperature=temp,
            specific_heat=cv,
            poynting_magnitude=crystal_metrics.get('energy_flow', 0.0)
        )
        
        # Check for glass state
        if glass_stopper.should_stop(epoch, lc, sp, kappa, delta, temp, cv):
            print(f"  [-] Glass detected at epoch {epoch}, stopping seed {seed}")
            return False  # Not a crystal seed
        
        # Check for crystal formation
        if (lc < Config.MINING_TARGET_LC and 
            sp < Config.MINING_TARGET_SP and 
            kappa < Config.MINING_TARGET_KAPPA and 
            delta < Config.MINING_TARGET_DELTA and 
            temp < Config.MINING_TARGET_TEMP and 
            cv < Config.MINING_TARGET_CV):
            print(f"  💎 Crystal found at epoch {epoch}, seed {seed}")
            return True  # Crystal seed found!
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:>3}: Loss={total_loss/total_samples:.6f}, "
                  f"ValAcc={val_acc:.4f}, LC={lc:.4f}, SP={sp:.4f}, "
                  f"δ={delta:.4f}, T_eff={temp:.2e}, C_v={cv:.2e}")
    
    return False  # Didn't find crystal in allowed epochs


def seed_miner(total_attempts: int = Config.MINING_MAX_ATTEMPTS):
    """Mine for crystal seeds by trying sequential seeds."""
    logger = setup_logger("SeedMiner")
    logger.info(f"Starting seed mining with {total_attempts} attempts")
    
    for i in range(Config.MINING_START_SEED, Config.MINING_START_SEED + total_attempts):
        current_seed = i
        print(f"\n[*] MINING SEED {current_seed} ({i-Config.MINING_START_SEED+1}/{total_attempts})")
        
        # Set seed for this attempt
        set_seed(current_seed)
        
        # Create fresh model and optimizer
        model = SimpleHamiltonianNet(
            grid_size=Config.GRID_SIZE,
            hidden_dim=Config.HIDDEN_DIM,
            num_spectral_layers=Config.NUM_SPECTRAL_LAYERS
        ).to(Config.DEVICE)
        
        optimizer = optim.SGD(
            model.parameters(), 
            lr=Config.LEARNING_RATE, 
            weight_decay=Config.WEIGHT_DECAY, 
            momentum=0.9
        )
        
        # Train with early glass detection
        success = train_with_early_glass_stop(model, optimizer, current_seed)
        
        if success:
            print(f"💎 CRYSTAL FOUND! Seed {current_seed} crystallized successfully.")
            # Save the crystallized model
            os.makedirs("crystal_seeds", exist_ok=True)
            crystal_path = f"crystal_seeds/crystal_seed_{current_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            torch.save({
                'seed': current_seed,
                'model_state_dict': model.state_dict(),
                'metrics': {
                    'lc': 0.0,  # Placeholder - would need to recompute
                    'sp': 0.0,
                    'kappa': 1.0,
                    'delta': 0.0,
                    'temperature': 0.0,
                    'specific_heat': 0.0
                }
            }, crystal_path)
            logger.info(f"Crystal saved to {crystal_path}")
            return True
    
    print(f"[-] No crystals found after {total_attempts} attempts")
    return False


def main():
    parser = argparse.ArgumentParser(description='Hamiltonian Grokking with Sequential Seed Mining')
    parser.add_argument('--mode', choices=['train', 'mine'], default='mine', 
                       help='Run mode: train or mine for crystals')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, help='Number of training epochs')
    parser.add_argument('--grid_size', type=int, default=Config.GRID_SIZE, help='Grid size for Hamiltonian operator')
    parser.add_argument('--hidden_dim', type=int, default=Config.HIDDEN_DIM, help='Hidden dimension size')
    parser.add_argument('--num_spectral_layers', type=int, default=Config.NUM_SPECTRAL_LAYERS, help='Number of spectral layers')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint for analysis')
    args = parser.parse_args()
    
    if args.mode == 'mine':
        seed_miner(total_attempts=Config.MINING_MAX_ATTEMPTS)
    else:
        # Original training mode
        set_seed(Config.RANDOM_SEED)
        torch.manual_seed(Config.RANDOM_SEED)
        np.random.seed(Config.RANDOM_SEED)
        
        if args.checkpoint_path:
            # Run analysis on provided checkpoint
            from pathlib import Path
            class BoltzmannAnalysisProgram:
                def __init__(self, checkpoint_path: str, results_dir: str = Config.RESULTS_DIR):
                    self.checkpoint_path = checkpoint_path
                    self.results_dir = os.path.join(results_dir, "analysis")
                    os.makedirs(self.results_dir, exist_ok=True)
                    self.logger = setup_logger("BoltzmannAnalysisProgram")
                
                def load_and_analyze_checkpoint(self) -> Dict[str, Any]:
                    self.logger.info(f"Loading checkpoint: {self.checkpoint_path}")
                    
                    checkpoint = torch.load(self.checkpoint_path, map_location=Config.DEVICE, weights_only=False)
                    model = SimpleHamiltonianNet(
                        grid_size=Config.GRID_SIZE,
                        hidden_dim=Config.HIDDEN_DIM,
                        num_spectral_layers=Config.NUM_SPECTRAL_LAYERS
                    ).to(Config.DEVICE)
                    
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    # Create dummy dataset for metrics calculation
                    dummy_dataset = FastDataset(num_samples=10)
                    def dataloader():
                        for i in range(1):  # Reduced for efficiency
                            yield dummy_dataset[i]
                    
                    # Calculate all metrics
                    crystal_metrics = CrystallographyMetrics.compute_all_metrics(model, dataloader())
                    spectroscopy_metrics = SpectroscopyMetrics.compute_weight_diffraction({name: param.data for name, param in model.named_parameters()})
                    
                    results = {
                        'checkpoint_path': self.checkpoint_path,
                        'crystallographic_metrics': crystal_metrics,
                        'spectroscopy_metrics': spectroscopy_metrics,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Save results
                    results_path = os.path.join(self.results_dir, f"analysis_{os.path.basename(self.checkpoint_path)}.json")
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    
                    self.logger.info(f"Analysis completed. Results saved to: {results_path}")
                    return results
            
            analyzer = BoltzmannAnalysisProgram(args.checkpoint_path)
            results = analyzer.load_and_analyze_checkpoint()
            print(f"Analysis completed for checkpoint: {args.checkpoint_path}")
            print(json.dumps(results, indent=2, default=str))
        else:
            # Run training
            device = torch.device(Config.DEVICE)
            print(f"Device: {device}")
            
            # Create dataset
            dataset = FastDataset(num_samples=Config.NUM_SAMPLES, grid_size=args.grid_size, time_steps=Config.TIME_STEPS)
            train_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
            
            # Create model
            model = SimpleHamiltonianNet(
                grid_size=args.grid_size,
                hidden_dim=args.hidden_dim,
                num_spectral_layers=args.num_spectral_layers
            ).to(device)
            
            # Use SGD with momentum for better exploration of the loss landscape
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY, momentum=0.9)
            # Add scheduler to gradually reduce learning rate
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.01)
            criterion = nn.MSELoss()
            
            # Initialize components
            checkpoint_manager = CheckpointManager()
            monitor = TrainingMonitor()
            lc_analyzer = LocalComplexityAnalyzer()
            sp_analyzer = SuperpositionAnalyzer()
            
            # Training loop
            print(f"\nTraining for {args.epochs} epochs...")
            start_time = time.time()
            
            for epoch in range(1, args.epochs + 1):
                model.train()
                total_loss = 0.0
                total_samples = 0
                
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # Gradient clipping to avoid exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    total_loss += loss.item() * batch_x.size(0)
                    total_samples += batch_x.size(0)
                
                # Step the scheduler
                scheduler.step()
                
                # Validation
                model.eval()
                val_x, val_y = dataset.get_val_batch()
                val_x = val_x.to(device)
                val_y = val_y.to(device)
                
                with torch.no_grad():
                    val_outputs = model(val_x)
                    val_loss = criterion(val_outputs, val_y)
                    
                    mse_per_sample = ((val_outputs - val_y) ** 2).mean(dim=(1, 2))
                    val_acc = (mse_per_sample < 0.05).float().mean().item()
                
                # Compute comprehensive metrics
                lc_values = []
                sp_values = []
                for name, param in model.named_parameters():
                    if 'weight' in name and param.dim() >= 2:
                        w = param[:min(param.size(0), 256), :min(param.size(1), 256)]
                        lc = lc_analyzer.compute_local_complexity(w)
                        sp = sp_analyzer.compute_superposition(w)
                        lc_values.append(lc)
                        sp_values.append(sp)
                
                lc = np.mean(lc_values) if lc_values else 0.0
                sp = np.mean(sp_values) if sp_values else 0.0
                
                # Additional metrics
                crystal_metrics = CrystallographyMetrics.compute_all_metrics(model, train_loader)
                alpha = crystal_metrics.get('alpha', 0.0)
                kappa = crystal_metrics.get('kappa', 1.0)
                delta = crystal_metrics.get('delta', 1.0)
                poynting_magnitude = crystal_metrics.get('energy_flow', 0.0)
                
                # Thermodynamic metrics
                temp = ThermodynamicMetrics.compute_effective_temperature(monitor.gradient_buffer, args.lr)
                cv, _ = ThermodynamicMetrics.compute_specific_heat(monitor.loss_history, monitor.temp_history)
                
                # Update monitor
                monitor.update_metrics(
                    epoch=epoch,
                    loss=total_loss/total_samples,
                    val_loss=val_loss.item(),
                    val_acc=val_acc,
                    lc=lc,
                    sp=sp,
                    alpha=alpha,
                    kappa=kappa,
                    delta=delta,
                    temperature=temp,
                    specific_heat=cv,
                    poynting_magnitude=poynting_magnitude
                )
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:>4}: Loss={total_loss/total_samples:.6f}, "
                          f"ValLoss={val_loss.item():.6f}, ValAcc={val_acc:.4f}, "
                          f"LC={lc:.4f}, SP={sp:.4f}, α={alpha:.2f}, κ={kappa:.2f}, "
                          f"δ={delta:.4f}, T_eff={temp:.2e}, C_v={cv:.2e}")
                
                # Checkpoint saving
                if checkpoint_manager.should_save_checkpoint():
                    metrics_snapshot = {
                        'val_loss': val_loss.item(),
                        'val_acc': val_acc,
                        'lc': lc,
                        'sp': sp,
                        'alpha': alpha,
                        'kappa': kappa,
                        'delta': delta,
                        'temperature': temp,
                        'specific_heat': cv,
                        'poynting_magnitude': poynting_magnitude
                    }
                    checkpoint_path = checkpoint_manager.save_checkpoint(model, optimizer, epoch, metrics_snapshot)
                    print(f"Checkpoint saved: {checkpoint_path}")
            
            elapsed = time.time() - start_time
            print(f"\nTraining completed in {elapsed:.1f} seconds")
            
            # Save final checkpoint
            os.makedirs("weights", exist_ok=True)
            final_metrics = {
                'val_loss': val_loss.item(),
                'val_acc': val_acc,
                'lc': lc,
                'sp': sp,
                'alpha': alpha,
                'kappa': kappa,
                'delta': delta,
                'temperature': temp,
                'specific_heat': cv,
                'poynting_magnitude': poynting_magnitude
            }
            checkpoint = {
                'epoch': args.epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': {
                    'grid_size': args.grid_size,
                    'hidden_dim': args.hidden_dim,
                    'num_spectral_layers': args.num_spectral_layers,
                    'model_type': 'fast_hamiltonian'
                },
                'final_metrics': final_metrics
            }
            torch.save(checkpoint, "weights/model_checkpoint.pth")
            print(f"Final checkpoint saved to weights/model_checkpoint.pth")
            
            print(f"\nFinal Results:")
            print(f"  Validation Loss: {final_metrics['val_loss']:.6f}")
            print(f"  Validation Accuracy: {final_metrics['val_acc']:.4f}")
            print(f"  Local Complexity: {final_metrics['lc']:.4f}")
            print(f"  Superposition: {final_metrics['sp']:.4f}")
            print(f"  Alpha (Purity): {final_metrics['alpha']:.2f}")
            print(f"  Kappa (Condition): {final_metrics['kappa']:.2f}")
            print(f"  Delta (Discretization): {final_metrics['delta']:.4f}")
            print(f"  Effective Temperature: {final_metrics['temperature']:.2e}")
            print(f"  Specific Heat: {final_metrics['specific_heat']:.2e}")
            print(f"  Poynting Magnitude: {final_metrics['poynting_magnitude']:.2e}")


if __name__ == "__main__":
    main()
