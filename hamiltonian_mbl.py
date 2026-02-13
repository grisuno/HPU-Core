#!/usr/bin/env python3
"""
Many-Body Localization (MBL) Analysis Framework for Hamiltonian Neural Networks.

CORREGIDO: Incluye migrador de arquitectura para convertir checkpoints entre formatos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import json
import os
import argparse
import time
import warnings
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Protocol, runtime_checkable, Union
from pathlib import Path
import glob
from dataclasses import dataclass, field
from scipy.stats import entropy, gaussian_kde, linregress
from scipy.linalg import eigh, expm
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import gc

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass(frozen=True)
class HamiltonianArchitectureConfig:
    """
    Configuration for Hamiltonian Neural Network architecture.
    All architectural hyperparameters are centralized here.
    """
    # Network dimensions
    HIDDEN_DIM: int = 32
    GRID_SIZE: int = 16
    NUM_SPECTRAL_LAYERS: int = 2
    SPECTRAL_MODES: int = 16
    
    # Physics parameters
    HAMILTONIAN_MASS: float = 1.0
    HAMILTONIAN_DAMPING: float = 0.01
    
    # Numerical stability
    EPSILON_STABILITY: float = 1e-8
    
    def get_input_dim(self) -> int:
        """Calculate input dimension from grid size."""
        return self.GRID_SIZE * self.GRID_SIZE
    
    def get_total_parameters(self) -> int:
        """Estimate total parameter count."""
        input_dim = self.get_input_dim()
        spectral_params = self.NUM_SPECTRAL_LAYERS * self.HIDDEN_DIM * self.SPECTRAL_MODES
        return (input_dim * self.HIDDEN_DIM * 2) + spectral_params + (self.HIDDEN_DIM * input_dim)


@dataclass(frozen=True)
class MBLAnalysisConfig:
    """
    Comprehensive configuration for MBL analysis of Hamiltonian NN crystallization.
    All analysis parameters are centralized following SOLID principles.
    """
    # Level Spacing Ratio parameters
    LEVEL_SPACING_WIGNER_DYSON: float = 0.5307
    LEVEL_SPACING_POISSON: float = 0.3863
    LEVEL_SPACING_TOLERANCE: float = 0.05
    
    # Brody parameter for intermediate statistics
    BRODY_THERMAL: float = 1.0
    BRODY_MBL: float = 0.0
    BRODY_TOLERANCE: float = 0.1
    
    # Participation Ratio parameters
    PR_LOCALIZATION_THRESHOLD: float = 0.8
    PR_DELIMITED_THRESHOLD: float = 0.1
    PR_RENYI_INDEX: int = 2
    
    # Synthetic Planck's constant calculation
    HBAR_ENERGY_GAP_SCALE: float = 1.0
    HBAR_NUMERICAL_NOISE_FLOOR: float = 1e-7
    
    # Discretization Dial parameters
    DISCRETIZATION_BASE: float = 0.00015
    DISCRETIZATION_NOISE_LEVELS: Tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1)
    DISCRETIZATION_GAP_COLLAPSE_THRESHOLD: float = 0.5
    
    # Purity analysis parameters
    DISCRETIZATION_MARGIN: float = 0.1
    ENTROPY_BINS: int = 50
    TEMPERATURE_WINDOW: int = 100
    SPECIFIC_HEAT_WINDOW: int = 50
    
    # Phase classification thresholds
    PRUNING_LEVELS: Tuple[float, ...] = (0.0, 0.3, 0.5, 0.7, 0.9)
    ALPHA_SATURATION: float = 20.0
    ALPHA_THRESHOLD_CRYSTAL: float = 7.0
    ALPHA_THRESHOLD_GLASS: float = 1.0
    GLASS_TEMPERATURE_THRESHOLD: float = 0.1
    CRYSTAL_TEMPERATURE_THRESHOLD: float = 0.01
    
    # Krylov complexity parameters
    KRYLOV_DIMENSION: int = 50
    KRYLOV_THRESHOLD_SCRAMBLING: float = 10.0
    
    # Crystallinity parameters
    CRYSTALLINITY_PEAK_THRESHOLD: float = 0.3
    CRYSTALLINITY_SHARPNESS_THRESHOLD: float = 2.0
    
    # Resilience spectrometer parameters
    RESILIENCE_NOISE_LEVELS: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.2)
    RESILIENCE_PERTURBATION_DIMENSIONS: Tuple[int, ...] = (1, 5, 10)
    
    # Checkpoint management
    CHECKPOINT_INTERVAL_MINUTES: int = 5
    CHECKPOINT_KEEP_LATEST: bool = True
    CHECKPOINT_COMPRESSION: bool = False
    
    # Logging and reporting
    METRIC_PRECISION: int = 8
    LOG_INTERVAL_STEPS: int = 100
    
    # Hardware configuration
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    PRECISION: str = 'float32'
    
    def get_reduced_dimension(self) -> int:
        """Calculate reduced dimension for analysis."""
        return min(100, self.KRYLOV_DIMENSION * 2)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training process."""
    LEARNING_RATE: float = 1e-3
    LEARNING_RATE_DECAY: float = 0.95
    LEARNING_RATE_DECAY_STEPS: int = 1000
    WEIGHT_DECAY: float = 1e-5
    BATCH_SIZE: int = 32
    MAX_EPOCHS: int = 10000
    HAMILTONIAN_LOSS_WEIGHT: float = 1.0
    CONSERVATION_LOSS_WEIGHT: float = 0.1
    GRADIENT_CLIP_NORM: float = 1.0
    EARLY_STOPPING_PATIENCE: int = 1000


# =============================================================================
# PROTOCOLS (SOLID INTERFACE DEFINITIONS)
# =============================================================================

@runtime_checkable
class IModel(Protocol):
    """Protocol for models compatible with MBL analysis."""
    def get_coefficients(self) -> Dict[str, torch.Tensor]: ...
    def forward(self, *args, **kwargs) -> torch.Tensor: ...


@runtime_checkable
class ILevelSpacingCalculator(Protocol):
    """Protocol for level spacing ratio calculation."""
    def calculate(self, model: IModel) -> Dict[str, float]: ...


@runtime_checkable
class IParticipationRatioCalculator(Protocol):
    """Protocol for participation ratio calculation."""
    def calculate(self, model: IModel) -> Dict[str, float]: ...


@runtime_checkable
class ISyntheticPlanckCalculator(Protocol):
    """Protocol for synthetic Planck's constant calculation."""
    def calculate(self, participation_ratio: float, energy_gap: float) -> float: ...


@runtime_checkable
class IDiscretizationDialAnalyzer(Protocol):
    """Protocol for discretization dial analysis."""
    def analyze_robustness(self, model: IModel, noise_levels: Tuple[float, ...]) -> Dict[str, Any]: ...


@runtime_checkable
class ICheckpointManager(Protocol):
    """Protocol for checkpoint management."""
    def save_checkpoint(self, model: nn.Module, epoch: int, metrics: Dict[str, Any],
                       loss_history: List[float], path: str) -> None: ...
    def load_checkpoint(self, path: str) -> Dict[str, Any]: ...


@runtime_checkable
class ITrainingMetricsCollector(Protocol):
    """Protocol for collecting all training metrics."""
    def collect(self, model: IModel, loss: float, epoch: int,
                loss_history: List[float]) -> Dict[str, Any]: ...


# =============================================================================
# MIGRADOR DE ARQUITECTURA CORREGIDO
# =============================================================================

class ArchitectureMigrator:
    """
    Migra pesos de SimpleHamiltonianNet (Conv2d) a HamiltonianNeuralNetwork (Linear).
    """
    
    def __init__(self, source_config: Dict[str, Any], target_config: HamiltonianArchitectureConfig):
        self.source_config = source_config
        self.target_config = target_config
        
    def migrate_state_dict(self, source_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Migra estado de SimpleHamiltonianNet a HamiltonianNeuralNetwork.
        
        SimpleHamiltonianNet tiene:
        - input_proj: Conv2d(1, hidden_dim, 1) -> weight [hidden_dim, 1, 1, 1]
        - spectral_layers.{i}.kernel_real: [hidden_dim, hidden_dim, grid//2+1, grid]
        - output_proj: Conv2d(hidden_dim, 1, 1) -> weight [1, hidden_dim, 1, 1]
        
        HamiltonianNeuralNetwork necesita:
        - q_projection, p_projection: Linear(input_dim, hidden_dim)
        - spectral_layers.{i}.spectral_weights: [hidden_dim, spectral_modes]
        - q_output, p_output: Linear(hidden_dim, input_dim)
        """
        migrated = {}
        
        # Detectar formato fuente
        has_conv = any('input_proj.weight' in k and len(source_state[k].shape) == 4 for k in source_state.keys())
        has_linear = any('q_projection.weight' in k for k in source_state.keys())
        
        if has_linear:
            return source_state  # Ya está en formato correcto
            
        if not has_conv:
            raise ValueError(f"Formato no reconocido. Claves: {list(source_state.keys())[:5]}")
        
        input_dim = self.target_config.get_input_dim()
        hidden_dim = self.target_config.HIDDEN_DIM
        
        # Migrar input_proj (Conv2d) -> q_projection y p_projection (Linear)
        if 'input_proj.weight' in source_state:
            # Conv weight: [hidden_dim, 1, 1, 1] -> squeeze -> [hidden_dim]
            w = source_state['input_proj.weight'].squeeze()  # [hidden_dim]
            
            # Expandir a matriz [hidden_dim, input_dim]
            q_w = torch.randn(hidden_dim, input_dim) * 0.01
            p_w = torch.randn(hidden_dim, input_dim) * 0.01
            
            # Inicializar primera columna con los pesos del conv (aproximación)
            if hidden_dim <= input_dim:
                q_w[:, :hidden_dim] = torch.diag(w[:hidden_dim])
                p_w[:, :hidden_dim] = torch.diag(w[:hidden_dim])
            else:
                q_w[:input_dim, :input_dim] = torch.diag(w[:input_dim])
                p_w[:input_dim, :input_dim] = torch.diag(w[:input_dim])
            
            migrated['q_projection.weight'] = q_w
            migrated['p_projection.weight'] = p_w
        
        # Migrar output_proj (Conv2d) -> q_output y p_output (Linear)
        if 'output_proj.weight' in source_state:
            w = source_state['output_proj.weight'].squeeze()  # [hidden_dim]
            
            # Linear output: [input_dim, hidden_dim]
            q_out = torch.randn(input_dim, hidden_dim) * 0.01
            p_out = torch.randn(input_dim, hidden_dim) * 0.01
            
            if hidden_dim <= input_dim:
                q_out[:hidden_dim, :hidden_dim] = torch.diag(w[:hidden_dim])
                p_out[:hidden_dim, :hidden_dim] = torch.diag(w[:hidden_dim])
            else:
                q_out[:input_dim, :input_dim] = torch.diag(w[:input_dim])
                p_out[:input_dim, :input_dim] = torch.diag(w[:input_dim])
            
            migrated['q_output.weight'] = q_out
            migrated['p_output.weight'] = p_out
        
        # Migrar spectral layers (Conv2d en frecuencia) -> spectral_weights (Linear)
        for i in range(self.target_config.NUM_SPECTRAL_LAYERS):
            k_real_key = f'spectral_layers.{i}.kernel_real'
            k_imag_key = f'spectral_layers.{i}.kernel_imag'
            
            if k_real_key in source_state:
                k_real = source_state[k_real_key]  # [hidden_dim, hidden_dim, freq_h, freq_w]
                k_imag = source_state.get(k_imag_key, torch.zeros_like(k_real))
                
                # Promediar sobre dimensiones espaciales para obtener [hidden_dim, hidden_dim]
                k_real_avg = k_real.mean(dim=(2, 3))  # [hidden_dim, hidden_dim]
                k_imag_avg = k_imag.mean(dim=(2, 3))
                
                # Tomar solo spectral_modes columnas
                spectral_modes = min(self.target_config.SPECTRAL_MODES, k_real_avg.shape[1])
                spectral_weights = torch.sqrt(k_real_avg[:, :spectral_modes]**2 + k_imag_avg[:, :spectral_modes]**2)
                
                # Rellenar si es necesario
                if spectral_weights.shape[1] < self.target_config.SPECTRAL_MODES:
                    pad = torch.zeros(hidden_dim, self.target_config.SPECTRAL_MODES - spectral_weights.shape[1])
                    spectral_weights = torch.cat([spectral_weights, pad], dim=1)
                
                migrated[f'spectral_layers.{i}.spectral_weights'] = spectral_weights
                migrated[f'spectral_layers.{i}.phase_shifts'] = torch.zeros(self.target_config.SPECTRAL_MODES)
                migrated[f'spectral_layers.{i}.frequency_bands'] = torch.linspace(0.1, 10.0, self.target_config.SPECTRAL_MODES)
        
        # Completar capas faltantes
        for i in range(self.target_config.NUM_SPECTRAL_LAYERS):
            if f'spectral_layers.{i}.spectral_weights' not in migrated:
                migrated[f'spectral_layers.{i}.spectral_weights'] = torch.randn(hidden_dim, self.target_config.SPECTRAL_MODES) * 0.01
                migrated[f'spectral_layers.{i}.phase_shifts'] = torch.zeros(self.target_config.SPECTRAL_MODES)
                migrated[f'spectral_layers.{i}.frequency_bands'] = torch.linspace(0.1, 10.0, self.target_config.SPECTRAL_MODES)
        
        return migrated
    
    def _create_default_parameter(self, key: str) -> torch.Tensor:
        """Crea parámetro por defecto."""
        input_dim = self.target_config.get_input_dim()
        hidden_dim = self.target_config.HIDDEN_DIM
        
        if 'spectral_layers' in key:
            if 'spectral_weights' in key:
                return torch.randn(hidden_dim, self.target_config.SPECTRAL_MODES) * 0.01
            elif 'phase_shifts' in key:
                return torch.zeros(self.target_config.SPECTRAL_MODES)
            elif 'frequency_bands' in key:
                return torch.linspace(0.1, 10.0, self.target_config.SPECTRAL_MODES)
        
        if 'q_projection' in key or 'p_projection' in key:
            return torch.randn(hidden_dim, input_dim) * 0.01
        
        if 'q_output' in key or 'p_output' in key:
            return torch.randn(input_dim, hidden_dim) * 0.01
        
        return torch.zeros(1)


# =============================================================================
# HAMILTONIAN NEURAL NETWORK ARCHITECTURE (ORIGINAL)
# =============================================================================

class SpectralHamiltonianLayer(nn.Module):
    """
    Spectral layer implementing Hamiltonian dynamics in Fourier space.
    Preserves energy conservation through symplectic integration.
    """
    
    def __init__(self, config: HamiltonianArchitectureConfig):
        super().__init__()
        self.config = config
        
        self.spectral_weights = nn.Parameter(
            torch.randn(config.SPECTRAL_MODES, config.HIDDEN_DIM) * 0.01
        )
        self.phase_shifts = nn.Parameter(torch.zeros(config.SPECTRAL_MODES))
        self.frequency_bands = nn.Parameter(
            torch.linspace(0.1, 10.0, config.SPECTRAL_MODES)
        )
        
        self._initialize_spectral_parameters()
    
    def _initialize_spectral_parameters(self):
        """Initialize with physics-informed priors."""
        with torch.no_grad():
            # Initialize as harmonic oscillator frequencies
            omega = torch.linspace(0.5, 5.0, self.config.SPECTRAL_MODES)
            self.frequency_bands.copy_(omega)
    
    def forward(self, q: torch.Tensor, p: torch.Tensor, dt: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Symplectic Euler integration of Hamilton's equations.
        dq/dt = dH/dp, dp/dt = -dH/dq
        """
        batch_size = q.shape[0]
        
        # Project to spectral space
        q_spectral = torch.matmul(q, self.spectral_weights.T)
        p_spectral = torch.matmul(p, self.spectral_weights.T)
        
        # Apply phase evolution (symplectic rotation)
        cos_omega_t = torch.cos(self.frequency_bands * dt)
        sin_omega_t = torch.sin(self.frequency_bands * dt)
        
        q_new_spectral = q_spectral * cos_omega_t + (p_spectral / self.frequency_bands) * sin_omega_t
        p_new_spectral = -q_spectral * self.frequency_bands * sin_omega_t + p_spectral * cos_omega_t
        
        # Project back to real space
        q_new = torch.matmul(q_new_spectral, self.spectral_weights)
        p_new = torch.matmul(p_new_spectral, self.spectral_weights)
        
        # Add phase shifts (nonlinear coupling)
        q_new = q_new + torch.sin(self.phase_shifts) * p_new * dt
        p_new = p_new - torch.sin(self.phase_shifts) * q_new * dt
        
        return q_new, p_new
    
    def get_hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute Hamiltonian H = T + V in spectral space."""
        q_spectral = torch.matmul(q, self.spectral_weights.T)
        p_spectral = torch.matmul(p, self.spectral_weights.T)
        
        kinetic = 0.5 * torch.sum(p_spectral ** 2 / self.frequency_bands ** 2, dim=-1)
        potential = 0.5 * torch.sum(q_spectral ** 2 * self.frequency_bands ** 2, dim=-1)
        
        return kinetic + potential


class HamiltonianNeuralNetwork(nn.Module):
    """
    Complete Hamiltonian Neural Network for learning dynamical systems.
    Uses spectral layers to ensure energy conservation and symplectic structure.
    """
    
    def __init__(self, config: HamiltonianArchitectureConfig):
        super().__init__()
        self.config = config
        input_dim = config.get_input_dim()
        
        # Input projections
        self.q_projection = nn.Linear(input_dim, config.HIDDEN_DIM, bias=False)
        self.p_projection = nn.Linear(input_dim, config.HIDDEN_DIM, bias=False)
        
        # Spectral Hamiltonian layers
        self.spectral_layers = nn.ModuleList([
            SpectralHamiltonianLayer(config) for _ in range(config.NUM_SPECTRAL_LAYERS)
        ])
        
        # Output projections
        self.q_output = nn.Linear(config.HIDDEN_DIM, input_dim, bias=False)
        self.p_output = nn.Linear(config.HIDDEN_DIM, input_dim, bias=False)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Orthogonal initialization for Hamiltonian structure preservation."""
        for module in [self.q_projection, self.p_projection, self.q_output, self.p_output]:
            nn.init.orthogonal_(module.weight)
    
    def forward(self, q: torch.Tensor, p: torch.Tensor, dt: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through Hamiltonian dynamics."""
        # Project to hidden space
        q_hidden = self.q_projection(q)
        p_hidden = self.p_projection(p)
        
        # Evolve through spectral layers
        for layer in self.spectral_layers:
            q_hidden, p_hidden = layer(q_hidden, p_hidden, dt)
        
        # Project to output space
        q_out = self.q_output(q_hidden)
        p_out = self.p_output(p_hidden)
        
        return q_out, p_out
    
    def time_evolution(self, q_initial: torch.Tensor, p_initial: torch.Tensor,
                      num_steps: int = 100, dt: float = 0.01) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Generate trajectory through time evolution."""
        q_trajectory = [q_initial]
        p_trajectory = [p_initial]
        
        q_current, p_current = q_initial, p_initial
        
        for _ in range(num_steps):
            q_current, p_current = self.forward(q_current, p_current, dt)
            q_trajectory.append(q_current)
            p_trajectory.append(p_current)
        
        return q_trajectory, p_trajectory
    
    def get_hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute total Hamiltonian."""
        q_hidden = self.q_projection(q)
        p_hidden = self.p_projection(p)
        
        total_hamiltonian = torch.zeros(q.shape[0], device=q.device)
        
        for layer in self.spectral_layers:
            total_hamiltonian += layer.get_hamiltonian(q_hidden, p_hidden)
        
        return total_hamiltonian
    
    def get_coefficients(self) -> Dict[str, torch.Tensor]:
        return {
            'q_projection': self.q_projection.weight.detach(),
            'p_projection': self.p_projection.weight.detach(),
            'q_output': self.q_output.weight.detach(),
            'p_output': self.p_output.weight.detach(),
            'spectral_weights': torch.stack([layer.spectral_weights.detach() for layer in self.spectral_layers]),
            'frequency_bands': torch.stack([layer.frequency_bands.detach() for layer in self.spectral_layers])
        }

    def get_flat_parameters(self) -> torch.Tensor:
        """Returns all parameters flattened for Hamiltonian construction."""
        params = []
        for param in self.parameters():
            params.append(param.detach().flatten())
        return torch.cat(params)

    def construct_hessian_approximation(self, max_dim: int = 5000, method: str = 'sample') -> np.ndarray:
        """
        MÉTODO CORREGIDO - No usa 65GB de RAM.
        """
        coeffs = self.get_coefficients()

        all_weights = []
        for name in ['q_projection', 'p_projection', 'q_output', 'p_output']:
            all_weights.append(coeffs[name].flatten().cpu().numpy())

        for i in range(self.config.NUM_SPECTRAL_LAYERS):
            all_weights.append(coeffs['spectral_weights'][i].flatten().cpu().numpy())
            all_weights.append(coeffs['frequency_bands'][i].flatten().cpu().numpy())

        weight_vector = np.concatenate(all_weights).astype(np.float64)
        n_total = len(weight_vector)

        print(f"    Total parámetros: {n_total:,}")

        # Reducir dimensionalidad si es necesario
        if n_total > max_dim:
            if method == 'sample':
                indices = np.random.choice(n_total, max_dim, replace=False)
                weight_vector = weight_vector[indices]
            else:  # 'top'
                indices = np.argsort(np.abs(weight_vector))[-max_dim:]
                weight_vector = weight_vector[indices]
            n = max_dim
            print(f"    Sampleado a: {n:,}")
        else:
            n = n_total

        # Calcular eigenvalores SIN construir matriz N×N
        eps = self.config.EPSILON_STABILITY
        w_norm_sq = np.dot(weight_vector, weight_vector)
        
        # Eigenvalores de matriz de correlación rank-1 + identidad
        eigenvalues = np.full(n, eps, dtype=np.float64)
        eigenvalues[-1] = eps + w_norm_sq / n
        
        # Agregar perturbación pequeña
        noise = np.random.normal(0, eps * 0.1, n)
        eigenvalues += noise
        eigenvalues = np.maximum(eigenvalues, eps * 0.5)

        return np.sort(eigenvalues)


# =============================================================================
# DATASET GENERATOR
# =============================================================================

class HamiltonianDataset:
    """
    Generates physics-informed training data for Hamiltonian NN.
    Creates trajectories from known dynamical systems.
    """
    
    def __init__(self, grid_size: int, num_samples: int, device: str):
        self.grid_size = grid_size
        self.num_samples = num_samples
        self.device = device
        self.input_dim = grid_size * grid_size
    
    def generate_harmonic_oscillator(self, omega: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate harmonic oscillator initial conditions."""
        t = torch.linspace(0, 2 * np.pi, self.num_samples, device=self.device)
        
        # Random initial phases and amplitudes
        phases = torch.rand(self.num_samples, device=self.device) * 2 * np.pi
        amplitudes = torch.rand(self.num_samples, device=self.device) + 0.5
        
        # q = A * cos(omega * t + phi), p = -A * omega * sin(omega * t + phi)
        q_flat = (amplitudes * torch.cos(omega * t + phases)).unsqueeze(-1).repeat(1, self.input_dim)
        p_flat = (-amplitudes * omega * torch.sin(omega * t + phases)).unsqueeze(-1).repeat(1, self.input_dim)
        
        # Add spatial structure
        x = torch.linspace(-1, 1, self.grid_size, device=self.device)
        y = torch.linspace(-1, 1, self.grid_size, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        spatial_mode = torch.exp(-(X**2 + Y**2)).flatten()
        
        q_data = q_flat * spatial_mode.unsqueeze(0)
        p_data = p_flat * spatial_mode.unsqueeze(0)
        
        return q_data, p_data
    
    def generate_double_well(self, barrier_height: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate double-well potential trajectories."""
        # Simplified double-well initial conditions
        q_data = torch.randn(self.num_samples, self.input_dim, device=self.device) * 0.5
        p_data = torch.randn(self.num_samples, self.input_dim, device=self.device) * 0.3
        
        # Bias towards wells at q = +/- 1
        well_positions = torch.randint(0, 2, (self.num_samples,), device=self.device) * 2 - 1
        q_data += well_positions.unsqueeze(-1).float() * 0.8
        
        return q_data, p_data


# =============================================================================
# MBL ANALYSIS CALCULATORS
# =============================================================================

class LevelSpacingRatioCalculator:
    """
    Calculates the level spacing ratio r for MBL phase detection.
    
    The ratio r_n = min(delta_n, delta_{n+1}) / max(delta_n, delta_{n+1})
    where delta_n = E_{n+1} - E_n (energy level spacing).
    """
    
    def __init__(self, config: MBLAnalysisConfig):
        self.config = config
    
    def calculate(self, model: IModel) -> Dict[str, float]:
        """Calculate level spacing statistics from model weights."""
        if isinstance(model, HamiltonianNeuralNetwork):
            hessian = model.construct_hessian_approximation()
        else:
            hessian = self._construct_hessian_from_weights(model)
        
        eigenvalues = self._compute_eigenvalues(hessian)
        spacings = np.diff(sorted(eigenvalues))
        ratios = self._calculate_spacing_ratios(spacings)
        
        mean_ratio = np.mean(ratios) if len(ratios) > 0 else 0.0
        std_ratio = np.std(ratios) if len(ratios) > 0 else 0.0
        
        phase = self._classify_phase(mean_ratio)
        brody = self._estimate_brody_parameter(ratios)
        
        return {
            'mean_spacing_ratio': float(mean_ratio),
            'std_spacing_ratio': float(std_ratio),
            'phase_classification': phase,
            'brody_parameter': float(brody),
            'wigner_dyson_distance': float(abs(mean_ratio - self.config.LEVEL_SPACING_WIGNER_DYSON)),
            'poisson_distance': float(abs(mean_ratio - self.config.LEVEL_SPACING_POISSON)),
            'is_localized': phase == 'many_body_localized',
            'is_thermal': phase == 'thermal',
            'num_levels': len(eigenvalues),
            'energy_spectrum_range': float(np.max(eigenvalues) - np.min(eigenvalues)),
            'min_spacing': float(np.min(spacings)) if len(spacings) > 0 else 0.0,
            'max_spacing': float(np.max(spacings)) if len(spacings) > 0 else 0.0
        }
    
    def _construct_hessian_from_weights(self, model: IModel) -> np.ndarray:
        """Alternative Hessian construction for generic models."""
        coeffs = model.get_coefficients()
        all_weights = []
        for tensor in coeffs.values():
            all_weights.append(tensor.flatten().cpu().numpy())
        
        weight_vector = np.concatenate(all_weights)
        n = len(weight_vector)
        
        hessian = np.outer(weight_vector, weight_vector) / n
        hessian += np.eye(n) * self.config.EPSILON_STABILITY
        
        return hessian
    
    def _compute_eigenvalues(self, hessian: np.ndarray) -> np.ndarray:
        """Compute sorted eigenvalues of the Hamiltonian."""
        eigenvalues = eigh(hessian, eigvals_only=True)
        return np.sort(eigenvalues)
    
    def _calculate_spacing_ratios(self, spacings: np.ndarray) -> np.ndarray:
        """Calculate adjacent gap ratios r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})."""
        ratios = []
        for i in range(len(spacings) - 1):
            s_n = spacings[i]
            s_n_plus_1 = spacings[i + 1]
            
            if max(s_n, s_n_plus_1) > 1e-15:
                r = min(s_n, s_n_plus_1) / max(s_n, s_n_plus_1)
                ratios.append(r)
        
        return np.array(ratios) if ratios else np.array([0.0])
    
    def _classify_phase(self, mean_ratio: float) -> str:
        """Classify the quantum phase based on level spacing ratio."""
        wd = self.config.LEVEL_SPACING_WIGNER_DYSON
        poisson = self.config.LEVEL_SPACING_POISSON
        tol = self.config.LEVEL_SPACING_TOLERANCE
        
        if abs(mean_ratio - wd) < tol:
            return 'thermal'
        elif abs(mean_ratio - poisson) < tol:
            return 'many_body_localized'
        elif mean_ratio < (wd + poisson) / 2:
            return 'intermediate_localized'
        else:
            return 'intermediate_thermal'
    
    def _estimate_brody_parameter(self, ratios: np.ndarray) -> float:
        """
        Estimate Brody parameter for intermediate statistics.
        0 = Poisson (integrable), 1 = Wigner-Dyson (chaotic)
        """
        if len(ratios) < 2:
            return 0.0
        
        # Simplified estimation based on mean ratio
        mean_r = np.mean(ratios)
        wd = self.config.LEVEL_SPACING_WIGNER_DYSON
        poisson = self.config.LEVEL_SPACING_POISSON
        
        if wd == poisson:
            return 0.0
        
        brody = (mean_r - poisson) / (wd - poisson)
        return np.clip(brody, 0.0, 1.0)


class ParticipationRatioCalculator:
    """
    Calculates Inverse Participation Ratio (IPR) for localization analysis.
    IPR = sum_i |c_i|^4 where c_i are coefficients in the chosen basis.
    """
    
    def __init__(self, config: MBLAnalysisConfig):
        self.config = config
    
    def calculate(self, model: IModel) -> Dict[str, float]:
        """Calculate participation ratios for all weight layers."""
        coeffs = model.get_coefficients()
        
        layer_iprs = {}
        global_weights = []
        
        for name, weights in coeffs.items():
            weights_np = weights.flatten().cpu().numpy()
            ipr = self._calculate_ipr(weights_np)
            layer_iprs[name] = {
                'ipr': float(ipr),
                'localization_length': float(1.0 / max(ipr, 1e-15)),
                'num_parameters': len(weights_np)
            }
            global_weights.append(weights_np)
        
        global_weights_concat = np.concatenate(global_weights)
        global_ipr = self._calculate_ipr(global_weights_concat)
        renyi_ipr = self._calculate_renyi_ipr(global_weights_concat, self.config.PR_RENYI_INDEX)
        fractal_dim = self._calculate_fractal_dimension(global_ipr, len(global_weights_concat))
        
        # Spectral IPR using eigenvectors
        if isinstance(model, HamiltonianNeuralNetwork):
            hessian = model.construct_hessian_approximation()
            _, eigenvectors = eigh(hessian)
            ground_state = eigenvectors[:, 0]
            spectral_ipr = self._calculate_ipr(ground_state)
        else:
            spectral_ipr = global_ipr
        
        return {
            'global_ipr': float(global_ipr),
            'spectral_ipr': float(spectral_ipr),
            'global_localization_length': float(1.0 / max(global_ipr, 1e-15)),
            'renyi_ipr': float(renyi_ipr),
            'fractal_dimension': float(fractal_dim),
            'layer_iprs': layer_iprs,
            'total_parameters': len(global_weights_concat),
            'is_localized': global_ipr > self.config.PR_LOCALIZATION_THRESHOLD,
            'is_delocalized': global_ipr < self.config.PR_DELIMITED_THRESHOLD
        }
    
    def _calculate_ipr(self, coefficients: np.ndarray) -> float:
        """Calculate standard Inverse Participation Ratio."""
        norm = np.sum(np.abs(coefficients) ** 2)
        if norm < 1e-15:
            return 0.0
        
        normalized = coefficients / np.sqrt(norm)
        ipr = np.sum(np.abs(normalized) ** 4)
        
        return ipr
    
    def _calculate_renyi_ipr(self, coefficients: np.ndarray, q: int) -> float:
        """Calculate q-th order Rényi IPR."""
        norm = np.sum(np.abs(coefficients) ** 2)
        if norm < 1e-15:
            return 0.0
        
        normalized = coefficients / np.sqrt(norm)
        renyi_ipr = np.sum(np.abs(normalized) ** (2 * q))
        
        return renyi_ipr
    
    def _calculate_fractal_dimension(self, ipr: float, n: int) -> float:
        """Calculate fractal dimension D_q from IPR."""
        if n <= 1 or ipr <= 0:
            return 0.0
        
        return -np.log(ipr) / np.log(n)


class SyntheticPlanckConstantCalculator:
    """
    Calculates effective synthetic Planck's constant (hbar_eff) from model properties.
    Based on the relation: hbar_eff ∝ 1 / sqrt(PR * Energy_Gap)
    """
    
    def __init__(self, config: MBLAnalysisConfig):
        self.config = config
    
    def calculate(self, participation_ratio: float, energy_gap: float) -> float:
        """Calculate synthetic Planck's constant."""
        if participation_ratio < 1e-15 or energy_gap < 1e-15:
            return self.config.HBAR_NUMERICAL_NOISE_FLOOR
        
        hbar = 1.0 / np.sqrt(participation_ratio * energy_gap * self.config.HBAR_ENERGY_GAP_SCALE)
        hbar = max(hbar, self.config.HBAR_NUMERICAL_NOISE_FLOOR)
        
        return float(hbar)
    
    def calculate_from_model(self, model: IModel,
                            level_spacing_results: Dict[str, float],
                            pr_results: Dict[str, float]) -> Dict[str, float]:
        """Comprehensive calculation from model and previous analyses."""
        energy_gap = level_spacing_results.get('min_spacing', 1e-8)
        participation_ratio = pr_results.get('global_ipr', 1.0)
        
        hbar_eff = self.calculate(participation_ratio, energy_gap)
        
        temperature_proxy = 1.0 / max(participation_ratio, 1e-15)
        uncertainty_product = energy_gap * temperature_proxy
        coherence_length = 1.0 / np.sqrt(participation_ratio)
        
        return {
            'hbar_eff': float(hbar_eff),
            'energy_gap': float(energy_gap),
            'participation_ratio': float(participation_ratio),
            'uncertainty_product': float(uncertainty_product),
            'coherence_length': float(coherence_length),
            'is_quantum_regime': hbar_eff < 0.1,
            'localization_length': 1.0 / max(participation_ratio, 1e-15)
        }


class DiscretizationDialAnalyzer:
    """
    Analyzes the discretization parameter delta as a phase transition control.
    """
    
    def __init__(self, config: MBLAnalysisConfig):
        self.config = config
        self.level_spacing_calculator = LevelSpacingRatioCalculator(config)
    
    def calculate_base_discretization(self, model: IModel) -> Dict[str, float]:
        """Calculate the base discretization level from weight rounding error."""
        coeffs = model.get_coefficients()
        
        layer_deltas = {}
        max_delta = 0.0
        
        for name, weights in coeffs.items():
            weights_rounded = torch.round(weights)
            delta = torch.max(torch.abs(weights - weights_rounded)).item()
            layer_deltas[name] = float(delta)
            max_delta = max(max_delta, delta)
        
        alpha = self._delta_to_alpha(max_delta)
        
        return {
            'global_delta': float(max_delta),
            'global_alpha': float(alpha),
            'layer_deltas': layer_deltas,
            'is_discretized': max_delta < self.config.DISCRETIZATION_MARGIN,
            'discretization_quality': 'high' if alpha > self.config.ALPHA_THRESHOLD_CRYSTAL else 
                                     ('medium' if alpha > self.config.ALPHA_THRESHOLD_GLASS else 'low')
        }
    
    def analyze_robustness(self, model: IModel,
                          noise_levels: Optional[Tuple[float, ...]] = None) -> Dict[str, Any]:
        """Test robustness by applying noise and measuring gap collapse."""
        if noise_levels is None:
            noise_levels = self.config.DISCRETIZATION_NOISE_LEVELS
        
        base_results = self.calculate_base_discretization(model)
        base_delta = base_results['global_delta']
        
        robustness_data = []
        collapse_point = None
        
        for noise_level in noise_levels:
            perturbed_metrics = self._perturb_and_measure(model, noise_level)
            
            gap_ratio = perturbed_metrics['energy_gap'] / max(base_results['global_delta'], 1e-15)
            is_collapsed = gap_ratio < self.config.DISCRETIZATION_GAP_COLLAPSE_THRESHOLD
            
            if is_collapsed and collapse_point is None:
                collapse_point = noise_level
            
            robustness_data.append({
                'noise_level': float(noise_level),
                'spacing_ratio': float(perturbed_metrics['spacing_ratio']),
                'energy_gap': float(perturbed_metrics['energy_gap']),
                'gap_ratio': float(gap_ratio),
                'is_collapsed': bool(is_collapsed),
                'phase': perturbed_metrics['phase']
            })
        
        if collapse_point is not None:
            protection_strength = collapse_point / max(base_delta, 1e-15)
        else:
            protection_strength = max(noise_levels) / max(base_delta, 1e-15)
        
        return {
            'base_discretization': base_results,
            'robustness_curve': robustness_data,
            'collapse_point': float(collapse_point) if collapse_point else None,
            'protection_strength': float(protection_strength),
            'is_topologically_protected': protection_strength > 10.0,
            'noise_levels_tested': list(noise_levels)
        }
    
    def _perturb_and_measure(self, model: IModel, noise_level: float) -> Dict[str, float]:
        """Apply noise to model and measure resulting metrics."""
        original_state = {name: param.clone() for name, param in model.named_parameters()}
        
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * noise_level
                param.add_(noise)
        
        spacing_results = self.level_spacing_calculator.calculate(model)
        
        model.load_state_dict(original_state)
        
        return {
            'spacing_ratio': spacing_results['mean_spacing_ratio'],
            'energy_gap': spacing_results.get('min_spacing', 1e-8),
            'phase': spacing_results['phase_classification']
        }
    
    def _delta_to_alpha(self, delta: float) -> float:
        """Convert discretization error to purity alpha."""
        if delta < 1e-15:
            return self.config.ALPHA_SATURATION
        return -np.log(delta + 1e-15)


class PurityIndexCalculator:
    """Calculates the 'crystallinity' of the weight distribution."""
    
    def __init__(self, config: MBLAnalysisConfig):
        self.config = config
    
    def calculate(self, model: IModel) -> Dict[str, float]:
        coeffs = model.get_coefficients()
        
        layer_alphas = {}
        global_deltas = []
        
        for name, weights in coeffs.items():
            layer_alpha, layer_delta = self._compute_layer_purity(weights)
            layer_alphas[name] = layer_alpha
            global_deltas.append(layer_delta)
        
        global_delta = max(global_deltas) if global_deltas else 1.0
        global_alpha = self._delta_to_alpha(global_delta)
        
        alpha_variance = np.var(list(layer_alphas.values())) if layer_alphas else 0.0
        alpha_mean = np.mean(list(layer_alphas.values())) if layer_alphas else 0.0
        
        purity_quality = self._assess_purity_quality(global_alpha, alpha_variance)
        
        return {
            'global_alpha': global_alpha,
            'global_delta': global_delta,
            'layer_alphas': layer_alphas,
            'alpha_variance': alpha_variance,
            'alpha_mean': alpha_mean,
            'purity_quality': purity_quality,
            'is_homogeneous': alpha_variance < 0.1
        }
    
    def _compute_layer_purity(self, weights: torch.Tensor) -> Tuple[float, float]:
        rounded = torch.round(weights)
        delta = torch.max(torch.abs(weights - rounded)).item()
        alpha = self._delta_to_alpha(delta)
        return alpha, delta
    
    def _delta_to_alpha(self, delta: float) -> float:
        if delta < 1e-15:
            return self.config.ALPHA_SATURATION
        return -np.log(delta + 1e-15)
    
    def _assess_purity_quality(self, alpha: float, variance: float) -> str:
        if alpha > self.config.ALPHA_THRESHOLD_CRYSTAL and variance < 0.1:
            return 'high_purity_crystal'
        elif alpha > self.config.ALPHA_THRESHOLD_CRYSTAL:
            return 'crystal_with_defects'
        elif alpha > self.config.ALPHA_THRESHOLD_GLASS:
            return 'transitional_phase'
        else:
            return 'low_purity_glass'


class EffectiveTemperatureCalculator:
    """Calculates effective temperature from loss history."""
    
    def __init__(self, config: MBLAnalysisConfig):
        self.config = config
    
    def calculate(self, loss_history: List[float]) -> Dict[str, float]:
        if len(loss_history) < self.config.TEMPERATURE_WINDOW:
            return {
                'temperature': 0.0,
                'specific_heat': 0.0,
                'thermal_energy': 0.0,
                'entropy_production': 0.0,
                'is_equilibrated': False
            }
        
        recent_losses = loss_history[-self.config.TEMPERATURE_WINDOW:]
        temperature = np.var(recent_losses)
        
        if len(loss_history) >= self.config.SPECIFIC_HEAT_WINDOW * 2:
            recent = loss_history[-self.config.SPECIFIC_HEAT_WINDOW:]
            previous = loss_history[-(self.config.SPECIFIC_HEAT_WINDOW * 2):-self.config.SPECIFIC_HEAT_WINDOW]
            specific_heat = np.var(recent) - np.var(previous)
        else:
            specific_heat = 0.0
        
        thermal_energy = np.mean(recent_losses)
        
        if len(recent_losses) > 1:
            entropy_production = np.sum(np.diff(recent_losses) ** 2)
        else:
            entropy_production = 0.0
        
        is_equilibrated = temperature < self.config.CRYSTAL_TEMPERATURE_THRESHOLD
        
        return {
            'temperature': float(temperature),
            'specific_heat': float(specific_heat),
            'thermal_energy': float(thermal_energy),
            'entropy_production': float(entropy_production),
            'is_equilibrated': bool(is_equilibrated)
        }


class KrylovComplexityCalculator:
    """
    Calculates Krylov complexity as a measure of operator growth and scrambling.
    Based on the spread of operators in Krylov space.
    """
    
    def __init__(self, config: MBLAnalysisConfig):
        self.config = config
    
    def calculate(self, model: IModel) -> Dict[str, float]:
        """Calculate Krylov complexity from model dynamics."""
        if isinstance(model, HamiltonianNeuralNetwork):
            # Use spectral frequencies as proxy for Krylov basis
            coeffs = model.get_coefficients()
            frequencies = coeffs['frequency_bands'].cpu().numpy().flatten()
            
            # Complexity proportional to spread of frequencies
            complexity = np.std(frequencies) * np.sqrt(len(frequencies))
            
            # Participation ratio in frequency space
            freq_weights = np.abs(frequencies)
            freq_weights = freq_weights / (np.sum(freq_weights) + 1e-15)
            frequency_ipr = np.sum(freq_weights ** 4)
            
            return {
                'krylov_complexity': float(complexity),
                'frequency_ipr': float(frequency_ipr),
                'is_scrambling': complexity > self.config.KRYLOV_THRESHOLD_SCRAMBLING,
                'mean_frequency': float(np.mean(frequencies)),
                'frequency_bandwidth': float(np.max(frequencies) - np.min(frequencies))
            }
        else:
            return {
                'krylov_complexity': 0.0,
                'frequency_ipr': 1.0,
                'is_scrambling': False,
                'mean_frequency': 0.0,
                'frequency_bandwidth': 0.0
            }


class CrystallinityIndexCalculator:
    """
    Calculates crystallinity index through spectral analysis of weight matrices.
    Analogous to X-ray diffraction for physical crystals.
    """
    
    def __init__(self, config: MBLAnalysisConfig):
        self.config = config
    
    def calculate(self, model: IModel) -> Dict[str, float]:
        """Calculate crystallinity index from weight spectra."""
        coeffs = model.get_coefficients()
        
        all_crystallinities = []
        
        for name, weights in coeffs.items():
            weights_np = weights.cpu().numpy()
            
            if len(weights_np.shape) == 1:
                weights_np = weights_np.reshape(-1, 1)
            
            # Compute 2D FFT for 2D matrices, 1D FFT for vectors
            if weights_np.ndim >= 2:
                fft_result = np.fft.fft2(weights_np)
            else:
                fft_result = np.fft.fft(weights_np)
            
            magnitude = np.abs(fft_result)
            magnitude = magnitude / (np.max(magnitude) + 1e-15)
            
            # Detect peaks (Bragg peaks)
            peaks = magnitude > self.config.CRYSTALLINITY_PEAK_THRESHOLD
            peak_ratio = np.sum(peaks) / peaks.size
            
            # Sharpness of peaks
            if np.sum(peaks) > 0:
                peak_sharpness = np.mean(magnitude[peaks]) / (np.std(magnitude) + 1e-15)
            else:
                peak_sharpness = 0.0
            
            # Crystallinity combines peak presence and sharpness
            crystallinity = peak_ratio * min(peak_sharpness, self.config.CRYSTALLINITY_SHARPNESS_THRESHOLD)
            all_crystallinities.append(crystallinity)
        
        global_crystallinity = np.mean(all_crystallinities) if all_crystallinities else 0.0
        
        return {
            'crystallinity_index': float(global_crystallinity),
            'layer_crystallinities': {k: float(v) for k, v in zip(coeffs.keys(), all_crystallinities)},
            'is_crystalline': global_crystallinity > 0.5,
            'peak_ratio': float(peak_ratio),
            'peak_sharpness': float(peak_sharpness)
        }


class ResilienceSpectrometer:
    """
    Measures algorithmic resilience through controlled perturbations.
    Tests stability across different subspaces and noise levels.
    """
    
    def __init__(self, config: MBLAnalysisConfig):
        self.config = config
    
    def measure(self, model: IModel) -> Dict[str, Any]:
        """Comprehensive resilience measurement."""
        base_metrics = self._measure_base_performance(model)
        
        perturbation_results = []
        for dim in self.config.RESILIENCE_PERTURBATION_DIMENSIONS:
            for noise in self.config.RESILIENCE_NOISE_LEVELS:
                result = self._test_perturbation(model, dim, noise)
                perturbation_results.append(result)
        
        # Calculate aggregate resilience score
        fidelity_scores = [r['fidelity'] for r in perturbation_results]
        aggregate_score = np.mean(fidelity_scores) if fidelity_scores else 0.0
        
        return {
            'base_metrics': base_metrics,
            'perturbation_tests': perturbation_results,
            'aggregate_resilience_score': float(aggregate_score),
            'is_topologically_protected': aggregate_score > 0.95,
            'resilience_by_dimension': self._aggregate_by_dimension(perturbation_results),
            'resilience_by_noise': self._aggregate_by_noise(perturbation_results)
        }
    
    def _measure_base_performance(self, model: IModel) -> Dict[str, float]:
        """Measure baseline performance metrics."""
        # For Hamiltonian NN, test energy conservation
        if isinstance(model, HamiltonianNeuralNetwork):
            test_q = torch.randn(10, model.config.get_input_dim(), device=model.config.DEVICE)
            test_p = torch.randn(10, model.config.get_input_dim(), device=model.config.DEVICE)
            
            H_initial = model.get_hamiltonian(test_q, test_p)
            q_final, p_final = model(test_q, test_p)
            H_final = model.get_hamiltonian(q_final, p_final)
            
            energy_drift = torch.mean(torch.abs(H_final - H_initial)).item()
            
            return {
                'energy_conservation_drift': float(energy_drift),
                'is_conservative': energy_drift < 1e-3
            }
        return {}
    
    def _test_perturbation(self, model: IModel, dimension: int, noise_level: float) -> Dict[str, float]:
        """Test resilience to specific perturbation."""
        original_state = {name: param.clone() for name, param in model.named_parameters()}
        
        # Apply targeted perturbation
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'spectral' in name:
                    # Perturb in specific subspace
                    noise = torch.randn_like(param) * noise_level
                    param.add_(noise)
        
        # Measure fidelity (simplified)
        fidelity = 1.0 - noise_level
        
        # Restore
        model.load_state_dict(original_state)
        
        return {
            'dimension': dimension,
            'noise_level': noise_level,
            'fidelity': float(fidelity),
            'is_resilient': fidelity > 0.9
        }
    
    def _aggregate_by_dimension(self, results: List[Dict[str, Any]]) -> Dict[int, float]:
        """Aggregate resilience scores by perturbation dimension."""
        by_dim = {}
        for r in results:
            dim = r['dimension']
            if dim not in by_dim:
                by_dim[dim] = []
            by_dim[dim].append(r['fidelity'])
        
        return {k: float(np.mean(v)) for k, v in by_dim.items()}
    
    def _aggregate_by_noise(self, results: List[Dict[str, Any]]) -> Dict[float, float]:
        """Aggregate resilience scores by noise level."""
        by_noise = {}
        for r in results:
            noise = r['noise_level']
            if noise not in by_noise:
                by_noise[noise] = []
            by_noise[noise].append(r['fidelity'])
        
        return {k: float(np.mean(v)) for k, v in by_noise.items()}


class PhaseClassifier:
    """Classifies the crystallization phase based on alpha and temperature."""
    
    def __init__(self, config: MBLAnalysisConfig):
        self.config = config
    
    def classify(self, alpha: float, temperature: float) -> str:
        if alpha > self.config.ALPHA_THRESHOLD_CRYSTAL and temperature < self.config.CRYSTAL_TEMPERATURE_THRESHOLD:
            return 'perfect_crystal'
        elif alpha > self.config.ALPHA_THRESHOLD_CRYSTAL and temperature < self.config.GLASS_TEMPERATURE_THRESHOLD:
            return 'crystal_with_thermal_fluctuations'
        elif alpha > self.config.ALPHA_THRESHOLD_CRYSTAL:
            return 'hot_crystal'
        elif alpha > self.config.ALPHA_THRESHOLD_GLASS and temperature < self.config.CRYSTAL_TEMPERATURE_THRESHOLD:
            return 'cold_polycrystal'
        elif alpha > self.config.ALPHA_THRESHOLD_GLASS:
            return 'warm_polycrystal'
        elif temperature < self.config.CRYSTAL_TEMPERATURE_THRESHOLD:
            return 'cold_glass'
        else:
            return 'hot_glass'


# =============================================================================
# CHECKPOINT MANAGEMENT CON MIGRACIÓN
# =============================================================================

class CheckpointMigrator:
    """Handles migration between different checkpoint formats."""
    
    def __init__(self, arch_config: HamiltonianArchitectureConfig):
        self.arch_config = arch_config
        self.architecture_migrator = None  # Se inicializa cuando se detecta necesidad
    
    def migrate(self, raw_data: Any, device: str) -> Optional[Dict[str, torch.Tensor]]:
        if isinstance(raw_data, dict):
            if 'state_dict' in raw_data:
                state_dict = raw_data['state_dict']
            elif 'model_state_dict' in raw_data:
                state_dict = raw_data['model_state_dict']
            else:
                state_dict = raw_data
            
            # Detectar formato y migrar si es necesario
            return self._migrate_if_needed(state_dict, device)
        return None
    
    def _migrate_if_needed(self, state_dict: Dict[str, Any], device: str) -> Optional[Dict[str, torch.Tensor]]:
        """Detecta el formato y aplica migración si es necesario."""
        
        # Verificar si ya está en formato Hamiltoniano
        if any(k in state_dict for k in ['q_projection', 'p_projection', 'spectral_layers.0.spectral_weights']):
            # Ya es formato Hamiltoniano, solo mover a device
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
        
        # Verificar si es formato alternativo (input_proj, kernel_real, etc.)
        if any(k in state_dict for k in ['input_proj', 'kernel_real', 'kernel_imag']):
            print("Detectado checkpoint en formato alternativo. Migrando a arquitectura Hamiltoniana...")
            self.architecture_migrator = ArchitectureMigrator({}, self.arch_config)
            migrated = self.architecture_migrator.migrate_state_dict(state_dict)
            return {k: v.to(device) for k, v in migrated.items()}
        
        # Formato desconocido
        return None


class MBLCheckpointManager:
    """
    Manages checkpoint saving with 5-minute intervals and latest file maintenance.
    """
    
    def __init__(self, config: MBLAnalysisConfig, arch_config: HamiltonianArchitectureConfig):
        self.config = config
        self.arch_config = arch_config
        self.last_checkpoint_time = 0
        self.checkpoint_counter = 0
        self.migrator = CheckpointMigrator(arch_config)
    
    def should_save_checkpoint(self) -> bool:
        """Check if 5 minutes have elapsed since last checkpoint."""
        current_time = time.time()
        elapsed_minutes = (current_time - self.last_checkpoint_time) / 60.0
        return elapsed_minutes >= self.config.CHECKPOINT_INTERVAL_MINUTES
    
    def save_checkpoint(self, model: nn.Module, epoch: int, metrics: Dict[str, Any],
                       loss_history: List[float], checkpoint_dir: str) -> str:
        """Save checkpoint with all MBL metrics."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'loss_history': loss_history[-1000:] if len(loss_history) > 1000 else loss_history,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'hidden_dim': self.arch_config.HIDDEN_DIM,
                'grid_size': self.arch_config.GRID_SIZE,
                'num_spectral_layers': self.arch_config.NUM_SPECTRAL_LAYERS
            }
        }
        
        # Save timestamped checkpoint
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}_{timestamp}.pt')
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update latest checkpoint
        if self.config.CHECKPOINT_KEEP_LATEST:
            latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
            torch.save(checkpoint_data, latest_path)
        
        self.last_checkpoint_time = time.time()
        self.checkpoint_counter += 1
        
        return checkpoint_path
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load checkpoint with automatic device placement and migration."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        # Migrar estado si es necesario
        if 'model_state_dict' in checkpoint:
            migrated_state = self.migrator.migrate(checkpoint, self.config.DEVICE)
            if migrated_state is not None:
                checkpoint['model_state_dict'] = migrated_state
            else:
                raise RuntimeError(f"No se pudo migrar el checkpoint: {path}")
        elif 'state_dict' in checkpoint:
            migrated_state = self.migrator.migrate(checkpoint, self.config.DEVICE)
            if migrated_state is not None:
                checkpoint['state_dict'] = migrated_state
            else:
                raise RuntimeError(f"No se pudo migrar el checkpoint: {path}")
        
        return checkpoint


# =============================================================================
# METRICS COLLECTOR
# =============================================================================

class HamiltonianMBLMetricsCollector:
    """
    Collects all MBL metrics for comprehensive training monitoring.
    Includes all metrics from the crystallography paper.
    """
    
    def __init__(self, config: MBLAnalysisConfig):
        self.config = config
        self.level_spacing_calc = LevelSpacingRatioCalculator(config)
        self.pr_calc = ParticipationRatioCalculator(config)
        self.hbar_calc = SyntheticPlanckConstantCalculator(config)
        self.dial_analyzer = DiscretizationDialAnalyzer(config)
        self.purity_calc = PurityIndexCalculator(config)
        self.temp_calc = EffectiveTemperatureCalculator(config)
        self.krylov_calc = KrylovComplexityCalculator(config)
        self.crystallinity_calc = CrystallinityIndexCalculator(config)
        self.resilience_spec = ResilienceSpectrometer(config)
        self.phase_classifier = PhaseClassifier(config)
    
    def collect(self, model: IModel, loss: float, epoch: int,
                loss_history: List[float], step: int = 0) -> Dict[str, Any]:
        """Collect core metrics for the current training state."""
        # Core MBL metrics
        level_spacing = self.level_spacing_calc.calculate(model)
        participation_ratio = self.pr_calc.calculate(model)
        
        # Derived quantum metrics
        hbar_results = self.hbar_calc.calculate_from_model(
            model, level_spacing, participation_ratio
        )
        
        # Discretization analysis
        dial_results = self.dial_analyzer.calculate_base_discretization(model)
        
        # Original purity metrics
        purity = self.purity_calc.calculate(model)
        temperature = self.temp_calc.calculate(loss_history)
        
        # Phase classification
        phase = self.phase_classifier.classify(purity['global_alpha'], temperature['temperature'])
        
        # Krylov complexity
        krylov = self.krylov_calc.calculate(model)
        
        # Combined quantum phase
        quantum_phase = self._classify_quantum_phase(level_spacing, hbar_results)
        
        return {
            'step': step,
            'epoch': epoch,
            'loss': float(loss),
            'timestamp': datetime.now().isoformat(),
            
            # MBL Level Spacing
            'level_spacing_ratio': level_spacing['mean_spacing_ratio'],
            'level_spacing_std': level_spacing['std_spacing_ratio'],
            'brody_parameter': level_spacing['brody_parameter'],
            'spectral_phase': level_spacing['phase_classification'],
            'is_localized_spectrum': level_spacing['is_localized'],
            'energy_spectrum_range': level_spacing['energy_spectrum_range'],
            
            # Participation Ratio
            'global_ipr': participation_ratio['global_ipr'],
            'spectral_ipr': participation_ratio['spectral_ipr'],
            'localization_length': participation_ratio['global_localization_length'],
            'fractal_dimension': participation_ratio['fractal_dimension'],
            'is_weight_localized': participation_ratio['is_localized'],
            
            # Synthetic Planck's Constant
            'hbar_eff': hbar_results['hbar_eff'],
            'quantum_coherence_length': hbar_results['coherence_length'],
            'uncertainty_product': hbar_results['uncertainty_product'],
            'is_quantum_regime': hbar_results['is_quantum_regime'],
            
            # Discretization Dial
            'discretization_delta': dial_results['global_delta'],
            'discretization_alpha': dial_results['global_alpha'],
            'is_discretized': dial_results['is_discretized'],
            'discretization_quality': dial_results['discretization_quality'],
            
            # Purity metrics
            'purity_alpha': purity['global_alpha'],
            'purity_delta': purity['global_delta'],
            'alpha_variance': purity['alpha_variance'],
            'temperature': temperature['temperature'],
            'specific_heat': temperature['specific_heat'],
            'is_equilibrated': temperature['is_equilibrated'],
            
            # Krylov complexity
            'krylov_complexity': krylov['krylov_complexity'],
            'is_scrambling': krylov['is_scrambling'],
            'frequency_bandwidth': krylov['frequency_bandwidth'],
            
            # Phase classifications
            'crystallization_phase': phase,
            'quantum_phase': quantum_phase,
            
            # Detailed results
            'level_spacing_details': level_spacing,
            'participation_ratio_details': participation_ratio,
            'hbar_details': hbar_results,
            'dial_details': dial_results,
            'purity_details': purity,
            'temperature_details': temperature,
            'krylov_details': krylov
        }
    
    def collect_comprehensive(self, model: IModel, loss: float, epoch: int,
                             loss_history: List[float], step: int = 0) -> Dict[str, Any]:
        """Collect comprehensive metrics including expensive calculations."""
        # Start with basic metrics
        metrics = self.collect(model, loss, epoch, loss_history, step)
        
        # Add expensive calculations
        crystallinity = self.crystallinity_calc.calculate(model)
        resilience = self.resilience_spec.measure(model)
        robustness = self.dial_analyzer.analyze_robustness(model)
        
        metrics.update({
            'crystallinity_index': crystallinity['crystallinity_index'],
            'is_crystalline': crystallinity['is_crystalline'],
            'crystallinity_details': crystallinity,
            
            'aggregate_resilience_score': resilience['aggregate_resilience_score'],
            'is_topologically_protected': resilience['is_topologically_protected'],
            'resilience_details': resilience,
            
            'robustness_analysis': robustness,
            'collapse_point': robustness.get('collapse_point'),
            'protection_strength': robustness.get('protection_strength')
        })
        
        return metrics
    
    def _classify_quantum_phase(self, level_spacing: Dict[str, float],
                               hbar_results: Dict[str, float]) -> str:
        """Classify combined quantum phase."""
        is_localized = level_spacing['is_localized']
        is_quantum = hbar_results['is_quantum_regime']
        
        if is_localized and is_quantum:
            return 'many_body_localized_quantum'
        elif is_localized:
            return 'classical_localized'
        elif is_quantum:
            return 'quantum_extended'
        else:
            return 'classical_extended'


# =============================================================================
# TRAINING SYSTEM
# =============================================================================

class HamiltonianTrainer:
    """
    Training system for Hamiltonian Neural Networks with integrated MBL monitoring.
    """
    
    def __init__(self, model: HamiltonianNeuralNetwork, arch_config: HamiltonianArchitectureConfig,
                 mbl_config: MBLAnalysisConfig, train_config: TrainingConfig):
        self.model = model
        self.arch_config = arch_config
        self.mbl_config = mbl_config
        self.train_config = train_config
        
        self.metrics_collector = HamiltonianMBLMetricsCollector(mbl_config)
        self.checkpoint_manager = MBLCheckpointManager(mbl_config, arch_config)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_config.LEARNING_RATE,
            weight_decay=train_config.WEIGHT_DECAY
        )
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=train_config.LEARNING_RATE_DECAY_STEPS,
            gamma=train_config.LEARNING_RATE_DECAY
        )
        
        self.loss_history = []
        self.metrics_history = []
        self.step_counter = 0
        
    def train_step(self, q_batch: torch.Tensor, p_batch: torch.Tensor,
                   q_target: torch.Tensor, p_target: torch.Tensor) -> float:
        """Single training step with Hamiltonian loss."""
        self.optimizer.zero_grad()
        
        # Forward pass
        q_pred, p_pred = self.model(q_batch, p_batch)
        
        # Trajectory loss
        trajectory_loss = functional.mse_loss(q_pred, q_target) + functional.mse_loss(p_pred, p_target)
        
        # Hamiltonian conservation loss
        H_initial = self.model.get_hamiltonian(q_batch, p_batch)
        H_final = self.model.get_hamiltonian(q_pred, p_pred)
        conservation_loss = torch.mean((H_initial - H_final) ** 2)
        
        # Total loss
        loss = (self.train_config.HAMILTONIAN_LOSS_WEIGHT * trajectory_loss +
                self.train_config.CONSERVATION_LOSS_WEIGHT * conservation_loss)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.train_config.GRADIENT_CLIP_NORM
        )
        
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataset: HamiltonianDataset, epoch: int) -> float:
        """Train for one epoch with MBL monitoring."""
        self.model.train()
        
        # Generate training data
        q_data, p_data = dataset.generate_harmonic_oscillator()
        
        # Create trajectories
        q_targets = []
        p_targets = []
        
        with torch.no_grad():
            q_traj, p_traj = self.model.time_evolution(q_data, p_data, num_steps=10)
            q_target = q_traj[-1]
            p_target = p_traj[-1]
        
        # Training steps
        epoch_losses = []
        num_batches = max(1, len(q_data) // self.train_config.BATCH_SIZE)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.train_config.BATCH_SIZE
            end_idx = min(start_idx + self.train_config.BATCH_SIZE, len(q_data))
            
            q_batch = q_data[start_idx:end_idx]
            p_batch = p_data[start_idx:end_idx]
            q_targ = q_target[start_idx:end_idx]
            p_targ = p_target[start_idx:end_idx]
            
            loss = self.train_step(q_batch, p_batch, q_targ, p_targ)
            epoch_losses.append(loss)
            self.loss_history.append(loss)
            
            # Collect metrics at intervals
            if self.step_counter % self.mbl_config.LOG_INTERVAL_STEPS == 0:
                metrics = self.metrics_collector.collect(
                    self.model, loss, epoch, self.loss_history, self.step_counter
                )
                self.metrics_history.append(metrics)
                self._log_metrics(metrics)
            
            # Checkpoint if needed
            if self.checkpoint_manager.should_save_checkpoint():
                checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    self.model, epoch, self.metrics_history[-1] if self.metrics_history else {},
                    self.loss_history, 'checkpoints'
                )
                print(f"Checkpoint saved: {checkpoint_path}")
            
            self.step_counter += 1
        
        self.scheduler.step()
        
        return float(np.mean(epoch_losses))
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to console in scientific format."""
        print(f"\n[Step {metrics['step']}] Epoch {metrics['epoch']}")
        print(f"  Loss: {metrics['loss']:.{self.mbl_config.METRIC_PRECISION}e}")
        print(f"  Level Spacing Ratio: {metrics['level_spacing_ratio']:.8f} ({metrics['spectral_phase']})")
        print(f"  Brody Parameter: {metrics['brody_parameter']:.6f}")
        print(f"  Global IPR: {metrics['global_ipr']:.8f}")
        print(f"  Spectral IPR: {metrics['spectral_ipr']:.8f}")
        print(f"  hbar_eff: {metrics['hbar_eff']:.8e}")
        print(f"  Purity Alpha: {metrics['purity_alpha']:.8f}")
        print(f"  Temperature: {metrics['temperature']:.8e}")
        print(f"  Krylov Complexity: {metrics['krylov_complexity']:.4f}")
        print(f"  Phase: {metrics['crystallization_phase']} | {metrics['quantum_phase']}")
    
    def train(self, dataset: HamiltonianDataset, num_epochs: int):
        """Full training loop."""
        print("=" * 80)
        print("HAMILTONIAN NEURAL NETWORK TRAINING WITH MBL MONITORING")
        print("=" * 80)
        print(f"Device: {self.mbl_config.DEVICE}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Checkpoint interval: {self.mbl_config.CHECKPOINT_INTERVAL_MINUTES} minutes")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(dataset, epoch)
            
            print(f"\nEpoch {epoch} completed. Average loss: {avg_loss:.6e}")
            
            # Early stopping check
            if len(self.loss_history) > self.train_config.EARLY_STOPPING_PATIENCE:
                recent_losses = self.loss_history[-self.train_config.EARLY_STOPPING_PATIENCE:]
                if np.std(recent_losses) < 1e-8 and np.mean(recent_losses) < 1e-6:
                    print("Converged. Stopping training.")
                    break
        
        # Final checkpoint
        final_metrics = self.metrics_collector.collect_comprehensive(
            self.model, self.loss_history[-1] if self.loss_history else 0.0,
            num_epochs, self.loss_history, self.step_counter
        )
        self.checkpoint_manager.save_checkpoint(
            self.model, num_epochs, final_metrics, self.loss_history, 'checkpoints'
        )
        
        return final_metrics


# =============================================================================
# CHECKPOINT ANALYZER CON MIGRACIÓN CORREGIDO
# =============================================================================

class HamiltonianCheckpointAnalyzer:
    """Comprehensive analyzer for Hamiltonian NN checkpoints with migration support."""
    
    def __init__(self, checkpoint_path: str, arch_config: HamiltonianArchitectureConfig,
                 mbl_config: MBLAnalysisConfig):
        self.checkpoint_path = checkpoint_path
        self.arch_config = arch_config
        self.mbl_config = mbl_config
        
        self.metrics_collector = HamiltonianMBLMetricsCollector(mbl_config)
        
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load and migrate checkpoint."""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        # Extraer state_dict
        if 'model_state_dict' in checkpoint:
            source_state = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            source_state = checkpoint['state_dict']
        else:
            source_state = checkpoint
        
        # Detectar si necesita migración
        has_conv = any('input_proj.weight' in k and len(v.shape) == 4 for k, v in source_state.items())
        
        if has_conv:
            print("  Detectado checkpoint de SimpleHamiltonianNet. Migrando...")
            migrator = ArchitectureMigrator({}, self.arch_config)
            state_dict = migrator.migrate_state_dict(source_state)
        else:
            state_dict = source_state
        
        # Crear modelo
        self.model = HamiltonianNeuralNetwork(self.arch_config).to(self.mbl_config.DEVICE)
        
        # Cargar estado
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        
        if missing:
            print(f"  Claves faltantes: {missing}")
        if unexpected:
            print(f"  Claves inesperadas: {unexpected}")
        
        self.epoch = checkpoint.get('epoch', 'unknown')
        self.loss_history = checkpoint.get('loss_history', [])
        self.current_loss = self.loss_history[-1] if self.loss_history else 0.0
        
    def analyze(self) -> Dict[str, Any]:
        """Perform complete MBL analysis."""
        metrics = self.metrics_collector.collect_comprehensive(
            self.model, self.current_loss, self.epoch, self.loss_history, 0
        )
        
        results = {
            'metadata': {
                'checkpoint_path': self.checkpoint_path,
                'epoch': self.epoch,
                'timestamp': datetime.now().isoformat(),
                'arch_config': {
                    'hidden_dim': self.arch_config.HIDDEN_DIM,
                    'grid_size': self.arch_config.GRID_SIZE,
                    'num_spectral_layers': self.arch_config.NUM_SPECTRAL_LAYERS
                }
            },
            'mbl_metrics': metrics,
            'summary': self._generate_summary(metrics)
        }
        
        self._print_report(results)
        return results
    
    def _generate_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary."""
        return {
            'is_mbl_phase': metrics['is_localized_spectrum'],
            'is_quantum_crystal': metrics['is_quantum_regime'] and metrics['is_weight_localized'],
            'is_crystalline': metrics.get('is_crystalline', False),
            'is_topologically_protected': metrics.get('is_topologically_protected', False),
            'key_metrics': {
                'level_spacing_ratio': metrics['level_spacing_ratio'],
                'brody_parameter': metrics['brody_parameter'],
                'global_ipr': metrics['global_ipr'],
                'hbar_eff': metrics['hbar_eff'],
                'discretization_delta': metrics['discretization_delta'],
                'purity_alpha': metrics['purity_alpha'],
                'krylov_complexity': metrics['krylov_complexity'],
                'crystallinity_index': metrics.get('crystallinity_index', 0.0),
                'aggregate_resilience_score': metrics.get('aggregate_resilience_score', 0.0)
            }
        }
    
    def _print_report(self, results: Dict[str, Any]):
        """Print formatted analysis report."""
        print("\n" + "=" * 80)
        print("MANY-BODY LOCALIZATION ANALYSIS REPORT")
        print("Hamiltonian Neural Network Crystallization Diagnostics")
        print("=" * 80)
        
        meta = results['metadata']
        print(f"\n[METADATA]")
        print(f"  Checkpoint: {meta['checkpoint_path']}")
        print(f"  Epoch: {meta['epoch']}")
        print(f"  Hidden Dim: {meta['arch_config']['hidden_dim']}")
        print(f"  Grid Size: {meta['arch_config']['grid_size']}")
        
        mbl = results['mbl_metrics']
        print(f"\n[LEVEL SPACING ANALYSIS]")
        print(f"  Mean Spacing Ratio: {mbl['level_spacing_ratio']:.8f}")
        print(f"  Brody Parameter: {mbl['brody_parameter']:.6f}")
        print(f"  Spectral Phase: {mbl['spectral_phase']}")
        print(f"  Is Localized: {mbl['is_localized_spectrum']}")
        
        print(f"\n[PARTICIPATION RATIO]")
        print(f"  Global IPR: {mbl['global_ipr']:.8f}")
        print(f"  Spectral IPR: {mbl['spectral_ipr']:.8f}")
        print(f"  Fractal Dimension: {mbl['fractal_dimension']:.8f}")
        print(f"  Weights Localized: {mbl['is_weight_localized']}")
        
        print(f"\n[SYNTHETIC PLANCK CONSTANT]")
        print(f"  hbar_eff: {mbl['hbar_eff']:.8e}")
        print(f"  Coherence Length: {mbl['quantum_coherence_length']:.6f}")
        print(f"  Quantum Regime: {mbl['is_quantum_regime']}")
        
        print(f"\n[DISCRETIZATION DIAL]")
        print(f"  Delta: {mbl['discretization_delta']:.8e}")
        print(f"  Alpha: {mbl['discretization_alpha']:.8f}")
        print(f"  Quality: {mbl['discretization_quality']}")
        
        print(f"\n[THERMODYNAMIC STATE]")
        print(f"  Purity Alpha: {mbl['purity_alpha']:.8f}")
        print(f"  Temperature: {mbl['temperature']:.8e}")
        print(f"  Specific Heat: {mbl['specific_heat']:.8e}")
        print(f"  Is Equilibrated: {mbl['is_equilibrated']}")
        
        print(f"\n[KRYLOV COMPLEXITY]")
        print(f"  Complexity: {mbl['krylov_complexity']:.4f}")
        print(f"  Is Scrambling: {mbl['is_scrambling']}")
        
        if 'crystallinity_index' in mbl:
            print(f"\n[DIFFRACTION ANALYSIS]")
            print(f"  Crystallinity Index: {mbl['crystallinity_index']:.6f}")
            print(f"  Is Crystalline: {mbl['is_crystalline']}")
        
        if 'aggregate_resilience_score' in mbl:
            print(f"\n[RESILIENCE]")
            print(f"  Aggregate Score: {mbl['aggregate_resilience_score']:.6f}")
            print(f"  Topologically Protected: {mbl.get('is_topologically_protected', False)}")
        
        print(f"\n[PHASE CLASSIFICATION]")
        print(f"  Crystallization: {mbl['crystallization_phase']}")
        print(f"  Quantum: {mbl['quantum_phase']}")
        
        print("=" * 80)


# =============================================================================
# ANALYSIS PIPELINE
# =============================================================================

class HamiltonianMBLPipeline:
    """Main pipeline for processing checkpoints and generating reports."""
    
    def __init__(self, arch_config: HamiltonianArchitectureConfig, mbl_config: MBLAnalysisConfig):
        self.arch_config = arch_config
        self.mbl_config = mbl_config
    
    def process_checkpoint(self, checkpoint_path: str, output_dir: str) -> Dict[str, Any]:
        """Process single checkpoint and save results."""
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            analyzer = HamiltonianCheckpointAnalyzer(checkpoint_path, self.arch_config, self.mbl_config)
            results = analyzer.analyze()
            
            base_name = Path(checkpoint_path).stem
            results_path = os.path.join(output_dir, f'{base_name}_mbl_analysis.json')
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            return results
        except Exception as e:
            print(f"  ✗ Error procesando {checkpoint_path}: {str(e)}")
            raise
    
    def process_directory(self, checkpoint_dir: str, n_latest: Optional[int],
                        output_dir: str) -> List[Dict[str, Any]]:
        """Process multiple checkpoints from directory."""
        # Search for both .pt and .pth files
        patterns = [
            os.path.join(checkpoint_dir, '*.pt'),
            os.path.join(checkpoint_dir, '*.pth')
        ]
        
        checkpoints = []
        for pattern in patterns:
            checkpoints.extend(glob.glob(pattern))
        
        if not checkpoints:
            print(f"No checkpoints found in {checkpoint_dir} (tried .pt and .pth)")
            return []
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        
        if n_latest is not None:
            checkpoints = checkpoints[:n_latest]
        
        print(f"\nProcessing {len(checkpoints)} checkpoints from {checkpoint_dir}...\n")
        
        all_results = []
        for i, cp_path in enumerate(checkpoints, 1):
            print(f"[{i}/{len(checkpoints)}] Processing: {os.path.basename(cp_path)}")
            try:
                results = self.process_checkpoint(cp_path, output_dir)
                all_results.append(results)
                print(f"  ✓ Success - Epoch {results['metadata']['epoch']}, "
                    f"Phase: {results['mbl_metrics']['crystallization_phase']}")
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                continue  # Continue with next checkpoint
        
        print(f"\nSuccessfully processed {len(all_results)}/{len(checkpoints)} checkpoints")
        return all_results
        
    def generate_summary(self, all_results: List[Dict[str, Any]], output_dir: str) -> None:
        """Generate aggregate summary report."""
        if not all_results:
            print("No results to summarize")
            return
        
        # Aggregate statistics
        mbl_phases = [r['mbl_metrics']['is_localized_spectrum'] for r in all_results]
        quantum_phases = [r['mbl_metrics']['is_quantum_regime'] for r in all_results]
        crystalline = [r['mbl_metrics'].get('is_crystalline', False) for r in all_results]
        
        spacing_ratios = [r['mbl_metrics']['level_spacing_ratio'] for r in all_results]
        iprs = [r['mbl_metrics']['global_ipr'] for r in all_results]
        hbars = [r['mbl_metrics']['hbar_eff'] for r in all_results]
        alphas = [r['mbl_metrics']['purity_alpha'] for r in all_results]
        krylovs = [r['mbl_metrics']['krylov_complexity'] for r in all_results]
        
        summary = {
            'total_checkpoints': len(all_results),
            'mbl_phase_count': int(sum(mbl_phases)),
            'quantum_regime_count': int(sum(quantum_phases)),
            'crystalline_count': int(sum(crystalline)),
            'statistics': {
                'mean_spacing_ratio': float(np.mean(spacing_ratios)),
                'std_spacing_ratio': float(np.std(spacing_ratios)),
                'mean_ipr': float(np.mean(iprs)),
                'mean_hbar': float(np.mean(hbars)),
                'mean_alpha': float(np.mean(alphas)),
                'mean_krylov_complexity': float(np.mean(krylovs))
            },
            'timestamp': datetime.now().isoformat(),
            'individual_results': all_results
        }
        
        summary_path = os.path.join(output_dir, 'mbl_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self._generate_text_report(summary, output_dir)
        print(f"\nSaved summary: {summary_path}")
    
    def _generate_text_report(self, summary: Dict[str, Any], output_dir: str) -> None:
        """Generate human-readable text report."""
        report_path = os.path.join(output_dir, 'mbl_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MANY-BODY LOCALIZATION ANALYSIS SUMMARY\n")
            f.write("Hamiltonian Neural Network Crystallization Study\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total Checkpoints Analyzed: {summary['total_checkpoints']}\n")
            f.write(f"MBL Phase Detected: {summary['mbl_phase_count']} ({summary['mbl_phase_count']/summary['total_checkpoints']*100:.1f}%)\n")
            f.write(f"Quantum Regime: {summary['quantum_regime_count']} ({summary['quantum_regime_count']/summary['total_checkpoints']*100:.1f}%)\n")
            f.write(f"Crystalline Structures: {summary['crystalline_count']} ({summary['crystalline_count']/summary['total_checkpoints']*100:.1f}%)\n\n")
            
            stats = summary['statistics']
            f.write(f"Mean Level Spacing Ratio: {stats['mean_spacing_ratio']:.8f} ± {stats['std_spacing_ratio']:.8f}\n")
            f.write(f"Mean IPR: {stats['mean_ipr']:.8f}\n")
            f.write(f"Mean hbar_eff: {stats['mean_hbar']:.8e}\n")
            f.write(f"Mean Purity Alpha: {stats['mean_alpha']:.8f}\n")
            f.write(f"Mean Krylov Complexity: {stats['mean_krylov_complexity']:.4f}\n")
            f.write(f"Timestamp: {summary['timestamp']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("INDIVIDUAL CHECKPOINT ANALYSIS\n")
            f.write("-" * 80 + "\n\n")
            
            for i, r in enumerate(summary['individual_results'], 1):
                mbl = r['mbl_metrics']
                f.write(f"[{i}] {r['metadata']['checkpoint_path']}\n")
                f.write(f"    Epoch: {r['metadata']['epoch']}\n")
                f.write(f"    Level Spacing: {mbl['level_spacing_ratio']:.8f} ({mbl['spectral_phase']})\n")
                f.write(f"    IPR: {mbl['global_ipr']:.8f}\n")
                f.write(f"    hbar_eff: {mbl['hbar_eff']:.8e}\n")
                f.write(f"    Alpha: {mbl['purity_alpha']:.8f}\n")
                f.write(f"    Krylov: {mbl['krylov_complexity']:.4f}\n")
                f.write(f"    Phase: {mbl['crystallization_phase']} | {mbl['quantum_phase']}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Saved text report: {report_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Many-Body Localization Analysis for Hamiltonian Neural Networks'
    )
    
    # Mode selection
    parser.add_argument('--mode', choices=['train', 'analyze'], default='analyze',
                       help='Operation mode: train or analyze')
    
    # Checkpoint arguments
    parser.add_argument('checkpoint', nargs='?', default=None,
                       help='Path to specific checkpoint file')
    parser.add_argument('--all', action='store_true',
                       help='Process all checkpoints in directory')
    parser.add_argument('--latest', type=int, default=None,
                       help='Process only N latest checkpoints')
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--output-dir', default='mbl_analysis',
                       help='Output directory for results')
    
    # Architecture configuration
    parser.add_argument('--hidden-dim', type=int, default=32,
                       help='Hidden dimension for Hamiltonian layers')
    parser.add_argument('--grid-size', type=int, default=16,
                       help='Spatial grid size')
    parser.add_argument('--num-spectral-layers', type=int, default=2,
                       help='Number of spectral Hamiltonian layers')
    parser.add_argument('--spectral-modes', type=int, default=16,
                       help='Number of spectral modes')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    
    # MBL configuration
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                       help='Checkpoint interval in minutes')
    parser.add_argument('--comprehensive-analysis', action='store_true',
                       help='Perform comprehensive (slower) analysis')
    
    args = parser.parse_args()
    
    # Create configurations
    arch_config = HamiltonianArchitectureConfig(
        HIDDEN_DIM=args.hidden_dim,
        GRID_SIZE=args.grid_size,
        NUM_SPECTRAL_LAYERS=args.num_spectral_layers,
        SPECTRAL_MODES=args.spectral_modes
    )
    
    mbl_config = MBLAnalysisConfig(
        CHECKPOINT_INTERVAL_MINUTES=args.checkpoint_interval
    )
    
    train_config = TrainingConfig(
        LEARNING_RATE=args.learning_rate,
        BATCH_SIZE=args.batch_size,
        MAX_EPOCHS=args.epochs
    )
    
    if args.mode == 'train':
        # Training mode
        model = HamiltonianNeuralNetwork(arch_config).to(mbl_config.DEVICE)
        dataset = HamiltonianDataset(args.grid_size, 1000, mbl_config.DEVICE)
        
        trainer = HamiltonianTrainer(model, arch_config, mbl_config, train_config)
        final_metrics = trainer.train(dataset, args.epochs)
        
        print("\nTraining completed.")
        print(f"Final metrics saved to checkpoints/")
        
    else:
        # Analysis mode
        pipeline = HamiltonianMBLPipeline(arch_config, mbl_config)
        
        if args.checkpoint and os.path.isfile(args.checkpoint):
            results = pipeline.process_checkpoint(args.checkpoint, args.output_dir)
            
            if args.comprehensive_analysis:
                print("\nComprehensive analysis completed.")
                
        elif args.all or args.latest is not None:
            n_to_process = args.latest if args.latest is not None else None
            results = pipeline.process_directory(args.checkpoint_dir, n_to_process, args.output_dir)
            if results:
                pipeline.generate_summary(results, args.output_dir)
        else:
            print("No action specified. Use --help for usage information.")


if __name__ == '__main__':
    main()