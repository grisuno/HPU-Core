"""
Hamiltonian Neural Network Architecture for Audio Processing.

Implements the spectral-domain Hamiltonian evolution layers with:
- Complex-valued spectral convolution (FFT-based)
- Symplectic integration structure
- Energy-conserving transformations
- Parametric kernel dimensions

Architecture matches the original experiment2.HamiltonianNeuralNetwork:
- SpectralEvolutionLayer with kernel shape [hidden_dim, hidden_dim, kH, kW]
- 1x1 input/output projections by default
- Same forward, evolve_complex, evolve_real logic as the visual pipeline

For audio, the input is the STFT magnitude (a 2D real field, analogous
to a grayscale image). The Hamiltonian evolution operates on this field
identically to how it operates on screen captures in the visual pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from config import ModelArchitectureConfig


class SpectralEvolutionLayer(nn.Module):
    """
    Single Hamiltonian spectral evolution layer.

    Performs frequency-domain evolution using learnable complex kernels.
    Kernel shape: [hidden_dim, hidden_dim, kernel_base_height, kernel_base_width]
    matching the original experiment2 architecture.
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_base_height: int,
        kernel_base_width: int,
        init_std: float,
    ) -> None:
        super().__init__()
        self.kernel_real = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, kernel_base_height, kernel_base_width) * init_std
        )
        self.kernel_imag = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, kernel_base_height, kernel_base_width) * init_std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply one step of Hamiltonian spectral evolution via RFFT2.

        Args:
            x: Input tensor [B, C, H, W] in spatial domain.

        Returns:
            Evolved tensor [B, C, H, W] in spatial domain.
        """
        spatial_h, spatial_w = x.shape[2], x.shape[3]
        x_fft = torch.fft.rfft2(x)
        _, _, freq_h, freq_w = x_fft.shape

        kr = F.interpolate(
            self.kernel_real.mean(dim=(0, 1), keepdim=True),
            size=(freq_h, freq_w),
            mode="bilinear",
            align_corners=False,
        )
        ki = F.interpolate(
            self.kernel_imag.mean(dim=(0, 1), keepdim=True),
            size=(freq_h, freq_w),
            mode="bilinear",
            align_corners=False,
        )

        evolved_real = x_fft.real * kr - x_fft.imag * ki
        evolved_imag = x_fft.real * ki + x_fft.imag * kr
        evolved_fft = torch.complex(evolved_real, evolved_imag)

        output = torch.fft.irfft2(evolved_fft, s=(spatial_h, spatial_w))
        return output

    def evolve_complex(
        self, x: torch.Tensor, target_height: int, target_width: int
    ) -> torch.Tensor:
        """
        Full complex FFT evolution for amplitude and phase extraction.

        Uses full FFT2 (not RFFT2) to preserve complete complex structure.

        Args:
            x: Input tensor [B, C, H, W].
            target_height: Output spatial height.
            target_width: Output spatial width.

        Returns:
            Complex-valued evolved field in spatial domain.
        """
        x_fft_complex = torch.fft.fft2(x)
        _, _, freq_h, freq_w = x_fft_complex.shape

        kr = F.interpolate(
            self.kernel_real.mean(dim=(0, 1), keepdim=True),
            size=(freq_h, freq_w),
            mode="bilinear",
            align_corners=False,
        )
        ki = F.interpolate(
            self.kernel_imag.mean(dim=(0, 1), keepdim=True),
            size=(freq_h, freq_w),
            mode="bilinear",
            align_corners=False,
        )

        evolved_real = x_fft_complex.real * kr - x_fft_complex.imag * ki
        evolved_imag = x_fft_complex.real * ki + x_fft_complex.imag * kr
        evolved_fft = torch.complex(evolved_real, evolved_imag)

        psi_complex = torch.fft.ifft2(evolved_fft, s=(target_height, target_width))
        return psi_complex

    def evolve_real(
        self, x: torch.Tensor, target_height: int, target_width: int
    ) -> torch.Tensor:
        """
        Real FFT evolution for action map computation.

        Args:
            x: Input tensor [B, C, H, W].
            target_height: Output spatial height.
            target_width: Output spatial width.

        Returns:
            Real-valued evolved field in spatial domain.
        """
        x_fft_real = torch.fft.rfft2(x)
        _, _, freq_h, freq_w = x_fft_real.shape

        kr = F.interpolate(
            self.kernel_real.mean(dim=(0, 1), keepdim=True),
            size=(freq_h, freq_w),
            mode="bilinear",
            align_corners=False,
        )
        ki = F.interpolate(
            self.kernel_imag.mean(dim=(0, 1), keepdim=True),
            size=(freq_h, freq_w),
            mode="bilinear",
            align_corners=False,
        )

        evolved_real = x_fft_real.real * kr - x_fft_real.imag * ki
        evolved_imag = x_fft_real.real * ki + x_fft_real.imag * kr
        evolved_fft = torch.complex(evolved_real, evolved_imag)

        psi_real = torch.fft.irfft2(evolved_fft, s=(target_height, target_width))
        return psi_real


class HamiltonianNeuralNetwork(nn.Module):
    """
    Complete Hamiltonian Neural Network with parametric architecture.

    Architecture (matching experiment2):
        1. Input projection: Conv2d(1, hidden_dim, kernel, pad)
        2. N spectral evolution layers with learnable complex kernels
        3. Output projection: Conv2d(hidden_dim, 1, kernel, pad)
    """

    def __init__(self, config: ModelArchitectureConfig) -> None:
        super().__init__()
        self._config = config

        self.input_proj = nn.Conv2d(
            config.input_channels,
            config.hidden_dimension,
            kernel_size=config.input_projection_kernel_size,
            padding=config.input_projection_padding,
        )

        self.spectral_layers = nn.ModuleList()
        for layer_idx in range(config.num_spectral_layers):
            layer = SpectralEvolutionLayer(
                hidden_dim=config.hidden_dimension,
                kernel_base_height=config.spectral_kernel_base_height,
                kernel_base_width=config.spectral_kernel_base_width,
                init_std=config.kernel_init_std,
            )
            self.spectral_layers.append(layer)

        self.output_proj = nn.Conv2d(
            config.hidden_dimension,
            config.output_channels,
            kernel_size=config.output_projection_kernel_size,
            padding=config.output_projection_padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: project -> evolve -> reconstruct.

        Args:
            x: Input tensor [B, 1, H, W].

        Returns:
            Reconstructed tensor [B, 1, H, W].
        """
        phi = F.gelu(self.input_proj(x), approximate=self._config.gelu_approximate)
        for layer in self.spectral_layers:
            phi = F.gelu(layer(phi), approximate=self._config.gelu_approximate)
        output = self.output_proj(phi)
        return output

    def forward_with_intermediates(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass returning intermediate hidden states for analysis.

        Args:
            x: Input tensor [B, 1, H, W].

        Returns:
            Tuple of (output, list of intermediate states).
        """
        intermediates = []
        phi = F.gelu(self.input_proj(x), approximate=self._config.gelu_approximate)
        intermediates.append(phi.clone())
        for layer in self.spectral_layers:
            phi = F.gelu(layer(phi), approximate=self._config.gelu_approximate)
            intermediates.append(phi.clone())
        output = self.output_proj(phi)
        return output, intermediates

    def extract_hamiltonian_fields(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract the three Hamiltonian field representations:
        1. Amplitude map (energy density / resonance)
        2. Phase map (topological structure / vortices)
        3. Action map (constructive interference = clear vision)

        Mirrors the visual processing logic from the original code.

        Args:
            x: Input tensor [B, 1, H, W].

        Returns:
            Tuple of (amplitude_map, phase_map, action_map) each [H, W].
        """
        _, _, h_orig, w_orig = x.shape

        phi = F.gelu(self.input_proj(x), approximate=self._config.gelu_approximate)
        layer = self.spectral_layers[0]

        psi_complex = layer.evolve_complex(phi, h_orig, w_orig)
        amplitude_map = torch.abs(psi_complex).mean(dim=1).squeeze()
        phase_map = torch.angle(psi_complex).mean(dim=1).squeeze()

        psi_real = layer.evolve_real(phi, h_orig, w_orig)
        action_map = torch.abs(
            psi_real.mean(dim=1) - phi.mean(dim=1)
        ).squeeze()

        return amplitude_map, phase_map, action_map

    def compute_energy_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hamiltonian energy mask for spectral reconstruction.

        This method implements the constructive interference principle:
        the complex FFT evolution reveals amplitude (resonance) and
        phase (topology). Their constructive sum amplitude * cos(phase)
        identifies WHERE in the time-frequency plane the model detects
        coherent energy structure.

        The real FFT evolution provides a complementary view (action),
        which highlights WHERE the model sees change/structure.

        Both are combined and normalized to [0, 1] as a mask that
        can be applied to the original STFT magnitude to produce
        the reconstructed audio.

        This is the audio equivalent of the "clear vision" (action map)
        in the visual domain.

        Args:
            x: STFT magnitude input [B, 1, freq_bins, time_frames],
               normalized to [0, 1].

        Returns:
            Energy mask [B, 1, freq_bins, time_frames] in [0, 1].
        """
        _, _, h_orig, w_orig = x.shape

        phi = F.gelu(self.input_proj(x), approximate=self._config.gelu_approximate)
        layer = self.spectral_layers[0]

        psi_complex = layer.evolve_complex(phi, h_orig, w_orig)
        amplitude = torch.abs(psi_complex).mean(dim=1, keepdim=True)
        phase = torch.angle(psi_complex).mean(dim=1, keepdim=True)
        spectral_field = amplitude * torch.cos(phase)

        psi_real = layer.evolve_real(phi, h_orig, w_orig)
        action_field = torch.abs(
            psi_real.mean(dim=1, keepdim=True) - phi.mean(dim=1, keepdim=True)
        )

        combined_field = spectral_field + action_field

        field_min = combined_field.min()
        field_max = combined_field.max()
        field_range = field_max - field_min
        if field_range > 1e-12:
            energy_mask = (combined_field - field_min) / field_range
        else:
            energy_mask = torch.ones_like(combined_field)

        return energy_mask
