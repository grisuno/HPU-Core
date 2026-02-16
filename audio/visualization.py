"""
Hamiltonian Audio Visualization Module.

Generates scientific visualizations of the Hamiltonian field analysis:
- Energy density (amplitude) heatmaps
- Topological phase portraits
- Action density maps
- Waveform comparison plots
- Spectral analysis plots
- Energy landscape evolution

Uses matplotlib for publication-quality figures.

Follows Single Responsibility Principle: solely responsible for
rendering and exporting visual representations.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Tuple, Optional
from config import VisualizationConfig, AudioProcessingConfig


class HamiltonianAudioVisualizer:
    """
    Generates all scientific visualizations for Hamiltonian audio analysis.
    """

    def __init__(
        self,
        vis_config: VisualizationConfig,
        audio_config: AudioProcessingConfig,
    ) -> None:
        self._vis_config = vis_config
        self._audio_config = audio_config
        os.makedirs(vis_config.output_directory, exist_ok=True)

    def render_complete_analysis(
        self,
        amplitude_map: torch.Tensor,
        phase_map: torch.Tensor,
        action_map: torch.Tensor,
        original_spectrogram: torch.Tensor,
        reconstructed_spectrogram: torch.Tensor,
        original_waveform: Optional[torch.Tensor] = None,
        reconstructed_waveform: Optional[torch.Tensor] = None,
        output_prefix: str = "hamiltonian_audio",
    ) -> None:
        """
        Generate the complete suite of Hamiltonian analysis visualizations.

        Args:
            amplitude_map: Energy density field [H, W].
            phase_map: Phase topology field [H, W].
            action_map: Action density field [H, W].
            original_spectrogram: Original mel spectrogram [1, 1, H, W].
            reconstructed_spectrogram: Reconstructed mel spectrogram [1, 1, H, W].
            original_waveform: Original audio waveform [1, T].
            reconstructed_waveform: Reconstructed audio waveform [1, T].
            output_prefix: Filename prefix for all outputs.
        """
        if self._vis_config.export_spectrograms:
            self._render_hamiltonian_fields(
                amplitude_map, phase_map, action_map, output_prefix
            )
            self._render_spectrogram_comparison(
                original_spectrogram, reconstructed_spectrogram, output_prefix
            )
        if self._vis_config.export_phase_portraits:
            self._render_phase_portrait(amplitude_map, phase_map, output_prefix)
        if self._vis_config.export_energy_landscapes:
            self._render_energy_landscape(amplitude_map, action_map, output_prefix)
        if (
            self._vis_config.export_waveform_comparison
            and original_waveform is not None
            and reconstructed_waveform is not None
        ):
            self._render_waveform_comparison(
                original_waveform, reconstructed_waveform, output_prefix
            )
        print(
            f"[VISUALIZATION] All outputs saved to: "
            f"{self._vis_config.output_directory}/"
        )

    def _render_hamiltonian_fields(
        self,
        amplitude_map: torch.Tensor,
        phase_map: torch.Tensor,
        action_map: torch.Tensor,
        output_prefix: str,
    ) -> None:
        """Render the three Hamiltonian field visualizations."""
        amp_np = amplitude_map.detach().cpu().numpy()
        phase_np = phase_map.detach().cpu().numpy()
        act_np = action_map.detach().cpu().numpy()
        fig, axes = plt.subplots(
            1, 3,
            figsize=(self._vis_config.spectrogram_figure_width, self._vis_config.spectrogram_figure_height),
            dpi=self._vis_config.figure_dpi,
        )
        fig.suptitle(
            "Hamiltonian Field Analysis of Audio Signal",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        im0 = axes[0].imshow(amp_np, aspect="auto", origin="lower", cmap="inferno")
        axes[0].set_title("Energy Density (Resonance)", fontsize=10)
        axes[0].set_xlabel("Time Frame")
        axes[0].set_ylabel("Mel Frequency Bin")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Amplitude")
        im1 = axes[1].imshow(
            phase_np, aspect="auto", origin="lower", cmap="twilight",
            vmin=-np.pi, vmax=np.pi,
        )
        axes[1].set_title("Topological Phase (Vortices)", fontsize=10)
        axes[1].set_xlabel("Time Frame")
        axes[1].set_ylabel("Mel Frequency Bin")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Phase (rad)")
        im2 = axes[2].imshow(act_np, aspect="auto", origin="lower", cmap="magma")
        axes[2].set_title("Action Density (Least Action)", fontsize=10)
        axes[2].set_xlabel("Time Frame")
        axes[2].set_ylabel("Mel Frequency Bin")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Action")
        plt.tight_layout()
        path = os.path.join(
            self._vis_config.output_directory,
            f"{output_prefix}_hamiltonian_fields.png",
        )
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [SAVED] {path}")

    def _render_spectrogram_comparison(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        output_prefix: str,
    ) -> None:
        """Render original vs reconstructed spectrogram comparison."""
        orig_np = original.squeeze().detach().cpu().numpy()
        recon_np = reconstructed.squeeze().detach().cpu().numpy()
        diff_np = np.abs(orig_np - recon_np)
        fig, axes = plt.subplots(
            1, 3,
            figsize=(self._vis_config.spectrogram_figure_width, self._vis_config.spectrogram_figure_height),
            dpi=self._vis_config.figure_dpi,
        )
        fig.suptitle(
            "Spectrogram Reconstruction Analysis",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        im0 = axes[0].imshow(orig_np, aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_title("Original Spectrogram", fontsize=10)
        axes[0].set_xlabel("Time Frame")
        axes[0].set_ylabel("Mel Frequency Bin")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(recon_np, aspect="auto", origin="lower", cmap="viridis")
        axes[1].set_title("Hamiltonian Reconstruction", fontsize=10)
        axes[1].set_xlabel("Time Frame")
        axes[1].set_ylabel("Mel Frequency Bin")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        im2 = axes[2].imshow(diff_np, aspect="auto", origin="lower", cmap="hot")
        axes[2].set_title("Absolute Reconstruction Error", fontsize=10)
        axes[2].set_xlabel("Time Frame")
        axes[2].set_ylabel("Mel Frequency Bin")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Error")
        plt.tight_layout()
        path = os.path.join(
            self._vis_config.output_directory,
            f"{output_prefix}_spectrogram_comparison.png",
        )
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [SAVED] {path}")

    def _render_phase_portrait(
        self,
        amplitude_map: torch.Tensor,
        phase_map: torch.Tensor,
        output_prefix: str,
    ) -> None:
        """Render 2D phase portrait (amplitude vs phase histogram)."""
        amp_flat = amplitude_map.detach().cpu().numpy().flatten()
        phase_flat = phase_map.detach().cpu().numpy().flatten()
        fig, ax = plt.subplots(
            figsize=(8, 8),
            dpi=self._vis_config.figure_dpi,
        )
        h = ax.hist2d(
            phase_flat,
            amp_flat,
            bins=self._vis_config.phase_portrait_bins,
            cmap="inferno",
            density=True,
        )
        ax.set_xlabel("Phase (radians)", fontsize=11)
        ax.set_ylabel("Amplitude (energy density)", fontsize=11)
        ax.set_title("Hamiltonian Phase Portrait", fontsize=13, fontweight="bold")
        plt.colorbar(h[3], ax=ax, label="Density")
        plt.tight_layout()
        path = os.path.join(
            self._vis_config.output_directory,
            f"{output_prefix}_phase_portrait.png",
        )
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [SAVED] {path}")

    def _render_energy_landscape(
        self,
        amplitude_map: torch.Tensor,
        action_map: torch.Tensor,
        output_prefix: str,
    ) -> None:
        """Render energy landscape as a 3D surface plot."""
        amp_np = amplitude_map.detach().cpu().numpy()
        act_np = action_map.detach().cpu().numpy()
        subsample = max(1, amp_np.shape[0] // 64)
        amp_sub = amp_np[::subsample, ::subsample]
        act_sub = act_np[::subsample, ::subsample]
        x = np.arange(amp_sub.shape[1])
        y = np.arange(amp_sub.shape[0])
        x_grid, y_grid = np.meshgrid(x, y)
        fig = plt.figure(
            figsize=(self._vis_config.spectrogram_figure_width, self._vis_config.spectrogram_figure_height),
            dpi=self._vis_config.figure_dpi,
        )
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.plot_surface(
            x_grid, y_grid, amp_sub,
            cmap="inferno", alpha=0.8, rstride=1, cstride=1,
        )
        ax1.set_title("Energy Density Landscape", fontsize=10)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Frequency")
        ax1.set_zlabel("Energy")
        ax2 = fig.add_subplot(122, projection="3d")
        ax2.plot_surface(
            x_grid, y_grid, act_sub,
            cmap="magma", alpha=0.8, rstride=1, cstride=1,
        )
        ax2.set_title("Action Density Landscape", fontsize=10)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Frequency")
        ax2.set_zlabel("Action")
        fig.suptitle(
            "Hamiltonian Energy-Action Landscape",
            fontsize=13,
            fontweight="bold",
            y=0.95,
        )
        plt.tight_layout()
        path = os.path.join(
            self._vis_config.output_directory,
            f"{output_prefix}_energy_landscape.png",
        )
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [SAVED] {path}")

    def _render_waveform_comparison(
        self,
        original_waveform: torch.Tensor,
        reconstructed_waveform: torch.Tensor,
        output_prefix: str,
    ) -> None:
        """Render original vs reconstructed waveform comparison."""
        orig_np = original_waveform.squeeze().detach().cpu().numpy()
        recon_np = reconstructed_waveform.squeeze().detach().cpu().numpy()
        min_len = min(len(orig_np), len(recon_np))
        orig_np = orig_np[:min_len]
        recon_np = recon_np[:min_len]
        time_axis = np.arange(min_len) / self._audio_config.sample_rate
        fig, axes = plt.subplots(
            3, 1,
            figsize=(self._vis_config.waveform_figure_width, self._vis_config.waveform_figure_height * 3),
            dpi=self._vis_config.figure_dpi,
        )
        fig.suptitle(
            "Waveform Reconstruction Analysis",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        axes[0].plot(time_axis, orig_np, color="#2196F3", linewidth=0.3, alpha=0.8)
        axes[0].set_title("Original Waveform", fontsize=10)
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(time_axis, recon_np, color="#FF5722", linewidth=0.3, alpha=0.8)
        axes[1].set_title("Hamiltonian Reconstruction", fontsize=10)
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True, alpha=0.3)
        residual = orig_np - recon_np
        axes[2].plot(time_axis, residual, color="#4CAF50", linewidth=0.3, alpha=0.8)
        axes[2].set_title("Residual (Original - Reconstructed)", fontsize=10)
        axes[2].set_xlabel("Time (seconds)")
        axes[2].set_ylabel("Amplitude")
        axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(
            self._vis_config.output_directory,
            f"{output_prefix}_waveform_comparison.png",
        )
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [SAVED] {path}")
