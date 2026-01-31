#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import glob
from dataclasses import dataclass
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D

from experiment2 import Config, HamiltonianNeuralNetwork, HamiltonianDataset


@dataclass
class DiracConfig:
    EPSILON_0: float = 8.854187817e-12
    CHARGE_UNIT: float = 1.602176634e-19
    DELTA_THRESHOLD: float = 1e-6
    GAUSSIAN_SIGMA: float = 0.1
    FLUX_INTEGRATION_SAMPLES: int = 1000
    FIELD_DECAY_THRESHOLD: float = 1e-10
    SPATIAL_RESOLUTION: int = 100
    COLORMAP: str = 'viridis'
    FIGURE_DPI: int = 150
    SAVE_FORMAT: str = 'png'


class DiracDeltaAnalyzer:
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.checkpoint_path = checkpoint_path
        
        self.checkpoint = torch.load(
            checkpoint_path, 
            map_location=self.device, 
            weights_only=False
        )
        
        self.model = HamiltonianNeuralNetwork(
            grid_size=Config.GRID_SIZE,
            hidden_dim=Config.HIDDEN_DIM,
            num_spectral_layers=Config.NUM_SPECTRAL_LAYERS
        ).to(self.device)
        
        if 'model_state_dict' in self.checkpoint:
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(self.checkpoint)
        
        self.epoch = self.checkpoint.get('epoch', 'unknown')
        self.metrics = self.checkpoint.get('metrics', {})
        
    def extract_charge_distribution(self) -> torch.Tensor:
        
        weights = []
        for name, param in self.model.named_parameters():
            if param.numel() > 0:
                weights.append(param.data.flatten())
        
        all_weights = torch.cat(weights)
        
        charge_density = all_weights - all_weights.mean()
        
        return charge_density
    
    def compute_dirac_delta_approximation(
        self, 
        charge_density: torch.Tensor
    ) -> Dict[str, Any]:
        
        rounded = torch.round(charge_density)
        delta_deviation = torch.abs(charge_density - rounded)
        
        point_charges = charge_density[delta_deviation < DiracConfig.DELTA_THRESHOLD]
        point_positions = torch.where(delta_deviation < DiracConfig.DELTA_THRESHOLD)[0]
        
        gaussian_weights = torch.exp(
            -delta_deviation**2 / (2 * DiracConfig.GAUSSIAN_SIGMA**2)
        )
        
        discrete_mass = torch.sum(torch.abs(point_charges)).item()
        continuous_mass = torch.sum(
            torch.abs(charge_density) * (1 - gaussian_weights)
        ).item()
        total_mass = discrete_mass + continuous_mass
        
        delta_strength = discrete_mass / total_mass if total_mass > 0 else 0
        
        return {
            'point_charges': point_charges.cpu().numpy(),
            'point_positions': point_positions.cpu().numpy(),
            'num_point_charges': len(point_charges),
            'gaussian_weights': gaussian_weights.cpu().numpy(),
            'delta_strength': delta_strength,
            'discrete_mass': discrete_mass,
            'continuous_mass': continuous_mass,
            'total_mass': total_mass,
            'charge_density_full': charge_density.cpu().numpy()
        }
    
    def compute_electric_field(
        self, 
        dirac_data: Dict[str, Any],
        eval_points: Optional[np.ndarray] = None
    ) -> np.ndarray:
        
        point_charges = dirac_data['point_charges']
        point_positions = dirac_data['point_positions']
        charge_density = dirac_data['charge_density_full']
        
        if eval_points is None:
            eval_points = np.linspace(
                0, 
                len(charge_density) - 1, 
                DiracConfig.SPATIAL_RESOLUTION
            )
        
        electric_field = np.zeros_like(eval_points)
        
        for charge, pos in zip(point_charges, point_positions):
            r = eval_points - pos
            r_safe = np.where(np.abs(r) < 1e-10, 1e-10, r)
            
            field_contribution = (
                charge / (4 * np.pi * DiracConfig.EPSILON_0 * r_safe**2)
            ) * np.sign(r_safe)
            
            electric_field += field_contribution
        
        gaussian_weights = dirac_data['gaussian_weights']
        for i, eval_pt in enumerate(eval_points):
            idx = int(np.clip(eval_pt, 0, len(charge_density) - 1))
            if gaussian_weights[idx] < 0.5:
                r = eval_pt - idx
                r_safe = r if np.abs(r) > 1e-10 else 1e-10
                
                smoothed_charge = charge_density[idx] * (1 - gaussian_weights[idx])
                field_contribution = (
                    smoothed_charge / (4 * np.pi * DiracConfig.EPSILON_0 * r_safe**2)
                ) * np.sign(r_safe)
                
                electric_field[i] += field_contribution
        
        return electric_field
    
    def compute_electric_flux(
        self, 
        electric_field: np.ndarray,
        surface_points: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        
        if surface_points is None:
            surface_points = np.linspace(
                0, 
                len(electric_field) - 1, 
                DiracConfig.FLUX_INTEGRATION_SAMPLES
            )
        
        field_interpolated = np.interp(
            surface_points, 
            np.arange(len(electric_field)), 
            electric_field
        )
        
        flux_outward = np.sum(field_interpolated[field_interpolated > 0])
        flux_inward = np.sum(np.abs(field_interpolated[field_interpolated < 0]))
        flux_net = flux_outward - flux_inward
        
        enclosed_charge = flux_net * DiracConfig.EPSILON_0
        
        return {
            'flux_outward': float(flux_outward),
            'flux_inward': float(flux_inward),
            'flux_net': float(flux_net),
            'enclosed_charge': float(enclosed_charge),
            'gauss_law_verification': float(
                np.abs(flux_net * DiracConfig.EPSILON_0)
            )
        }
    
    def compute_divergence(
        self, 
        electric_field: np.ndarray
    ) -> np.ndarray:
        
        divergence = np.gradient(electric_field)
        return divergence
    
    def verify_gauss_law(
        self, 
        dirac_data: Dict[str, Any],
        flux_data: Dict[str, float]
    ) -> Dict[str, Any]:
        
        total_charge = dirac_data['total_mass']
        enclosed_from_flux = flux_data['enclosed_charge']
        
        relative_error = np.abs(
            (total_charge - enclosed_from_flux) / (total_charge + 1e-10)
        )
        
        is_consistent = relative_error < 0.1
        
        return {
            'total_charge_direct': float(total_charge),
            'enclosed_charge_flux': float(enclosed_from_flux),
            'relative_error': float(relative_error),
            'is_gauss_consistent': bool(is_consistent),
            'divergence_theorem_holds': is_consistent
        }
    
    def analyze_all(self) -> Dict[str, Any]:
        
        print(f"\nAnalyzing checkpoint: {self.checkpoint_path}")
        print(f"Epoch: {self.epoch}\n")
        
        charge_density = self.extract_charge_distribution()
        
        dirac_data = self.compute_dirac_delta_approximation(charge_density)
        
        electric_field = self.compute_electric_field(dirac_data)
        
        flux_data = self.compute_electric_flux(electric_field)
        
        divergence = self.compute_divergence(electric_field)
        
        gauss_verification = self.verify_gauss_law(dirac_data, flux_data)
        
        results = {
            'metadata': {
                'checkpoint_path': self.checkpoint_path,
                'epoch': self.epoch,
                'timestamp': datetime.now().isoformat(),
                'analysis_version': '1.0.0'
            },
            'dirac_distribution': {
                'num_point_charges': dirac_data['num_point_charges'],
                'delta_strength': dirac_data['delta_strength'],
                'discrete_mass': dirac_data['discrete_mass'],
                'continuous_mass': dirac_data['continuous_mass'],
                'total_mass': dirac_data['total_mass']
            },
            'electric_field': {
                'max_magnitude': float(np.max(np.abs(electric_field))),
                'mean_magnitude': float(np.mean(np.abs(electric_field))),
                'std_magnitude': float(np.std(electric_field))
            },
            'electric_flux': flux_data,
            'gauss_law': gauss_verification,
            'divergence': {
                'max': float(np.max(divergence)),
                'min': float(np.min(divergence)),
                'mean': float(np.mean(divergence)),
                'std': float(np.std(divergence))
            }
        }
        
        self._print_report(results)
        
        return results, {
            'charge_density': dirac_data['charge_density_full'],
            'point_charges': dirac_data['point_charges'],
            'point_positions': dirac_data['point_positions'],
            'electric_field': electric_field,
            'divergence': divergence
        }
    
    def _print_report(self, results: Dict):
        
        print("=" * 70)
        print("DIRAC DELTA AND ELECTRIC FLUX ANALYSIS")
        print("=" * 70)
        
        print(f"\n[METADATA]")
        print(f"  Checkpoint: {results['metadata']['checkpoint_path']}")
        print(f"  Epoch: {results['metadata']['epoch']}")
        
        print(f"\n[DIRAC DISTRIBUTION]")
        dd = results['dirac_distribution']
        print(f"  Point charges: {dd['num_point_charges']}")
        print(f"  Delta strength: {dd['delta_strength']:.6f}")
        print(f"  Discrete mass: {dd['discrete_mass']:.6e}")
        print(f"  Continuous mass: {dd['continuous_mass']:.6e}")
        print(f"  Total mass: {dd['total_mass']:.6e}")
        
        print(f"\n[ELECTRIC FIELD]")
        ef = results['electric_field']
        print(f"  Max magnitude: {ef['max_magnitude']:.6e} V/m")
        print(f"  Mean magnitude: {ef['mean_magnitude']:.6e} V/m")
        print(f"  Std deviation: {ef['std_magnitude']:.6e} V/m")
        
        print(f"\n[ELECTRIC FLUX]")
        flux = results['electric_flux']
        print(f"  Outward flux: {flux['flux_outward']:.6e} V·m")
        print(f"  Inward flux: {flux['flux_inward']:.6e} V·m")
        print(f"  Net flux: {flux['flux_net']:.6e} V·m")
        print(f"  Enclosed charge: {flux['enclosed_charge']:.6e} C")
        
        print(f"\n[GAUSS LAW VERIFICATION]")
        gauss = results['gauss_law']
        print(f"  Total charge (direct): {gauss['total_charge_direct']:.6e} C")
        print(f"  Enclosed charge (flux): {gauss['enclosed_charge_flux']:.6e} C")
        print(f"  Relative error: {gauss['relative_error']:.6f}")
        print(f"  Gauss consistent: {gauss['is_gauss_consistent']}")
        print(f"  Divergence theorem: {gauss['divergence_theorem_holds']}")
        
        print(f"\n[DIVERGENCE]")
        div = results['divergence']
        print(f"  Max: {div['max']:.6e}")
        print(f"  Min: {div['min']:.6e}")
        print(f"  Mean: {div['mean']:.6e}")
        print(f"  Std: {div['std']:.6e}")
        
        print("=" * 70)


class DiracVisualizer:
    
    @staticmethod
    def plot_charge_distribution(
        charge_density: np.ndarray,
        point_positions: np.ndarray,
        point_charges: np.ndarray,
        output_path: str
    ):
        
        fig, ax = plt.subplots(figsize=(12, 6), dpi=DiracConfig.FIGURE_DPI)
        
        positions = np.arange(len(charge_density))
        ax.plot(
            positions, 
            charge_density, 
            color='#2E86AB', 
            linewidth=1.5, 
            label='Continuous charge density'
        )
        
        ax.scatter(
            point_positions,
            point_charges,
            color='#A23B72',
            s=100,
            marker='o',
            edgecolors='black',
            linewidths=1.5,
            label='Point charges (Dirac delta)',
            zorder=5
        )
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Position index', fontsize=12)
        ax.set_ylabel('Charge density', fontsize=12)
        ax.set_title('Spatial Charge Distribution', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        plt.savefig(
            output_path, 
            dpi=DiracConfig.FIGURE_DPI, 
            format=DiracConfig.SAVE_FORMAT
        )
        plt.close()
        
        print(f"Saved charge distribution plot: {output_path}")
    
    @staticmethod
    def plot_electric_field(
        electric_field: np.ndarray,
        output_path: str
    ):
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1, 
            figsize=(12, 10), 
            dpi=DiracConfig.FIGURE_DPI
        )
        
        positions = np.linspace(0, len(electric_field) - 1, len(electric_field))
        
        ax1.plot(
            positions, 
            electric_field, 
            color='#F18F01', 
            linewidth=2,
            label='Electric field E(x)'
        )
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_xlabel('Position', fontsize=12)
        ax1.set_ylabel('Electric field (V/m)', fontsize=12)
        ax1.set_title('Electric Field Distribution', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle=':')
        
        field_magnitude = np.abs(electric_field)
        norm = Normalize(vmin=field_magnitude.min(), vmax=field_magnitude.max())
        cmap = plt.get_cmap(DiracConfig.COLORMAP)
        colors = cmap(norm(field_magnitude))
        
        for i in range(len(positions) - 1):
            ax2.plot(
                positions[i:i+2], 
                electric_field[i:i+2], 
                color=colors[i], 
                linewidth=2
            )
        
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2)
        cbar.set_label('Field magnitude', fontsize=10)
        
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Position', fontsize=12)
        ax2.set_ylabel('Electric field (V/m)', fontsize=12)
        ax2.set_title('Electric Field (Color-coded by magnitude)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        plt.savefig(
            output_path, 
            dpi=DiracConfig.FIGURE_DPI, 
            format=DiracConfig.SAVE_FORMAT
        )
        plt.close()
        
        print(f"Saved electric field plot: {output_path}")
    
    @staticmethod
    def plot_divergence(
        divergence: np.ndarray,
        output_path: str
    ):
        
        fig, ax = plt.subplots(figsize=(12, 6), dpi=DiracConfig.FIGURE_DPI)
        
        positions = np.linspace(0, len(divergence) - 1, len(divergence))
        
        ax.fill_between(
            positions, 
            divergence, 
            0,
            where=(divergence >= 0),
            color='#06A77D',
            alpha=0.6,
            label='Positive divergence (sources)'
        )
        
        ax.fill_between(
            positions, 
            divergence, 
            0,
            where=(divergence < 0),
            color='#D62828',
            alpha=0.6,
            label='Negative divergence (sinks)'
        )
        
        ax.plot(positions, divergence, color='black', linewidth=1.5, alpha=0.8)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Divergence of E', fontsize=12)
        ax.set_title('Divergence of Electric Field', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        plt.savefig(
            output_path, 
            dpi=DiracConfig.FIGURE_DPI, 
            format=DiracConfig.SAVE_FORMAT
        )
        plt.close()
        
        print(f"Saved divergence plot: {output_path}")
    
    @staticmethod
    def plot_combined_analysis(
        charge_density: np.ndarray,
        point_positions: np.ndarray,
        point_charges: np.ndarray,
        electric_field: np.ndarray,
        divergence: np.ndarray,
        output_path: str
    ):
        
        fig, axes = plt.subplots(
            3, 1, 
            figsize=(14, 12), 
            dpi=DiracConfig.FIGURE_DPI
        )
        
        positions_charge = np.arange(len(charge_density))
        axes[0].plot(
            positions_charge, 
            charge_density, 
            color='#2E86AB', 
            linewidth=1.5
        )
        axes[0].scatter(
            point_positions,
            point_charges,
            color='#A23B72',
            s=80,
            marker='o',
            edgecolors='black',
            linewidths=1.2,
            zorder=5
        )
        axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        axes[0].set_ylabel('Charge density', fontsize=11)
        axes[0].set_title('Charge Distribution', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, linestyle=':')
        
        positions_field = np.linspace(0, len(electric_field) - 1, len(electric_field))
        axes[1].plot(
            positions_field, 
            electric_field, 
            color='#F18F01', 
            linewidth=2
        )
        axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].set_ylabel('Electric field (V/m)', fontsize=11)
        axes[1].set_title('Electric Field', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, linestyle=':')
        
        positions_div = np.linspace(0, len(divergence) - 1, len(divergence))
        axes[2].fill_between(
            positions_div, 
            divergence, 
            0,
            where=(divergence >= 0),
            color='#06A77D',
            alpha=0.6
        )
        axes[2].fill_between(
            positions_div, 
            divergence, 
            0,
            where=(divergence < 0),
            color='#D62828',
            alpha=0.6
        )
        axes[2].plot(positions_div, divergence, color='black', linewidth=1.5, alpha=0.8)
        axes[2].axhline(y=0, color='gray', linestyle='--', linewidth=1)
        axes[2].set_xlabel('Position', fontsize=11)
        axes[2].set_ylabel('Divergence of E', fontsize=11)
        axes[2].set_title('Divergence', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        plt.savefig(
            output_path, 
            dpi=DiracConfig.FIGURE_DPI, 
            format=DiracConfig.SAVE_FORMAT
        )
        plt.close()
        
        print(f"Saved combined analysis plot: {output_path}")


def analyze_checkpoint(
    checkpoint_path: str,
    output_dir: str = 'dirac_analysis'
):
    
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = DiracDeltaAnalyzer(checkpoint_path)
    results, plot_data = analyzer.analyze_all()
    
    base_name = Path(checkpoint_path).stem
    
    DiracVisualizer.plot_charge_distribution(
        plot_data['charge_density'],
        plot_data['point_positions'],
        plot_data['point_charges'],
        os.path.join(output_dir, f'{base_name}_charge_distribution.{DiracConfig.SAVE_FORMAT}')
    )
    
    DiracVisualizer.plot_electric_field(
        plot_data['electric_field'],
        os.path.join(output_dir, f'{base_name}_electric_field.{DiracConfig.SAVE_FORMAT}')
    )
    
    DiracVisualizer.plot_divergence(
        plot_data['divergence'],
        os.path.join(output_dir, f'{base_name}_divergence.{DiracConfig.SAVE_FORMAT}')
    )
    
    DiracVisualizer.plot_combined_analysis(
        plot_data['charge_density'],
        plot_data['point_positions'],
        plot_data['point_charges'],
        plot_data['electric_field'],
        plot_data['divergence'],
        os.path.join(output_dir, f'{base_name}_combined.{DiracConfig.SAVE_FORMAT}')
    )
    
    results_path = os.path.join(output_dir, f'{base_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results: {results_path}")
    
    return results


def analyze_multiple_checkpoints(
    checkpoint_dir: str = 'crystal_checkpoints',
    n_latest: int = 5,
    output_dir: str = 'dirac_analysis'
):
    
    pattern = os.path.join(checkpoint_dir, '*.pth')
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    checkpoints = checkpoints[:n_latest]
    
    print(f"\nAnalyzing {len(checkpoints)} checkpoints...\n")
    
    all_results = []
    for cp in checkpoints:
        try:
            results = analyze_checkpoint(cp, output_dir)
            all_results.append(results)
        except Exception as e:
            print(f"Error analyzing {cp}: {e}")
    
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Dirac delta and electric flux analysis for HPU checkpoints'
    )
    parser.add_argument(
        'checkpoint',
        nargs='?',
        default=None,
        help='Path to checkpoint'
    )
    parser.add_argument(
        '--latest',
        type=int,
        default=None,
        help='Analyze N latest checkpoints'
    )
    parser.add_argument(
        '--dir',
        default='crystal_checkpoints',
        help='Checkpoint directory'
    )
    parser.add_argument(
        '--output',
        default='dirac_analysis',
        help='Output directory for plots and results'
    )
    
    args = parser.parse_args()
    
    if args.latest:
        analyze_multiple_checkpoints(args.dir, args.latest, args.output)
    elif args.checkpoint:
        analyze_checkpoint(args.checkpoint, args.output)
    else:
        latest = os.path.join(args.dir, 'latest.pth')
        if os.path.exists(latest):
            analyze_checkpoint(latest, args.output)
        else:
            analyze_multiple_checkpoints(args.dir, 1, args.output)


if __name__ == '__main__':
    main()