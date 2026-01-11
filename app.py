#!/usr/bin/env python3
# _*_ coding: utf8 _*_
"""
app.py

Autor: Gris Iscomeback
Correo electrónico: grisiscomeback[at]gmail[dot]com
Fecha de creación: xx/xx/xxxx
Licencia: GPL v3

Descripción:  

Hamiltonian Grokking - Training 
Validation of Theorem 1.1 implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
from typing import Dict, Tuple
import argparse


class SimpleConfig:
    def __init__(
        self,
        grid_size: int = 16,
        hidden_dim: int = 32,
        num_spectral_layers: int = 2,
        target_accuracy: float = 0.90,
        learning_rate: float = 0.005
    ):
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.num_spectral_layers = num_spectral_layers
        self.target_accuracy = target_accuracy
        self.learning_rate = learning_rate


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


class HamiltonianOperator:
    """True Hamiltonian operator H = -nabla^2 on torus."""
    
    def __init__(self, grid_size: int = 16):
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
    
    def time_evolution(self, field: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        hamiltonian_action = self.apply(field)
        evolved = field + hamiltonian_action * dt
        return evolved / (torch.norm(evolved) + 1e-8) * torch.norm(field)


class FastDataset(Dataset):
    """Fast dataset for Hamiltonian operator learning."""
    
    def __init__(
        self,
        num_samples: int = 200,
        grid_size: int = 16,
        time_steps: int = 2,
        dt: float = 0.01,
        seed: int = 42,
        train_ratio: float = 0.7
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
        grid_size: int = 16,
        hidden_dim: int = 32,
        num_spectral_layers: int = 2
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


def train_model(
    grid_size: int = 16,
    epochs: int = 50,
    hidden_dim: int = 32,
    num_spectral_layers: int = 2,
    lr: float = 0.005
) -> Dict:
    """Train the Hamiltonian operator model."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create dataset
    dataset = FastDataset(num_samples=200, grid_size=grid_size, time_steps=2)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = SimpleHamiltonianNet(
        grid_size=grid_size,
        hidden_dim=hidden_dim,
        num_spectral_layers=num_spectral_layers
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()
    
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
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)
        
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
                lc = compute_local_complexity(w)
                sp = compute_superposition(w)
                lc_values.append(lc)
                sp_values.append(sp)
        
        lc = np.mean(lc_values) if lc_values else 0.0
        sp = np.mean(sp_values) if sp_values else 0.0
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:>3}: Loss={total_loss/total_samples:.4f}, "
                  f"ValLoss={val_loss.item():.4f}, ValAcc={val_acc:.4f}, LC={lc:.4f}, SP={sp:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f} seconds")
    
    # Save checkpoint
    os.makedirs("weights", exist_ok=True)
    checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'config': {
            'grid_size': grid_size,
            'hidden_dim': hidden_dim,
            'num_spectral_layers': num_spectral_layers,
            'model_type': 'fast'
        },
        'final_val_loss': val_loss.item(),
        'final_val_acc': val_acc,
        'final_lc': lc,
        'final_sp': sp
    }
    torch.save(checkpoint, "weights/model_checkpoint.pth")
    print(f"Checkpoint saved to weights/model_checkpoint.pth")
    
    return {
        'val_loss': val_loss.item(),
        'val_acc': val_acc,
        'lc': lc,
        'sp': sp
    }


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    parser = argparse.ArgumentParser(description='Fast Hamiltonian Grokking')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--grid_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_spectral_layers', type=int, default=2)
    args = parser.parse_args()
    
    results = train_model(
        grid_size=args.grid_size,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_spectral_layers=args.num_spectral_layers
    )
    
    print(f"\nFinal Results:")
    print(f"  Validation Loss: {results['val_loss']:.6f}")
    print(f"  Validation Accuracy: {results['val_acc']:.4f}")
    print(f"  Local Complexity: {results['lc']:.4f}")
    print(f"  Superposition: {results['sp']:.4f}")


if __name__ == "__main__":
    main()
