"""
Hamiltonian Audio Inference Module.

Pipeline (STFT complex domain):
1. Load trained model from checkpoint
2. Load audio -> waveform
3. Waveform -> complex STFT (2D complex field: freq x time)
4. Split STFT into magnitude (energy) and phase (topology)
5. Normalize magnitude -> feed to Hamiltonian network
6. Network produces energy mask via constructive interference
7. Apply energy mask to ORIGINAL magnitude
8. Recombine with ORIGINAL phase -> modified complex STFT
9. ISTFT -> reconstructed waveform
10. Generate visualizations and export audio

The key insight: the Hamiltonian network operates on the STFT magnitude
(a 2D real field, identical in structure to a grayscale image).
The original phase is preserved and recombined, ensuring temporal
coherence in the reconstruction. This is why ISTFT works cleanly
while Griffin-Lim (phase estimation) produced noise.

Follows Single Responsibility Principle: orchestrates inference only.
"""

import os
import torch
from typing import Tuple, Optional

from config import HamiltonianAudioConfig
from model import HamiltonianNeuralNetwork
from audio_io import AudioProcessor
from visualization import HamiltonianAudioVisualizer
from metrics import HamiltonianMetricsTracker
from checkpoint_manager import CheckpointManager


class HamiltonianAudioInference:
    """
    Performs complete Hamiltonian audio analysis on a given audio file.
    """

    def __init__(
        self,
        config: HamiltonianAudioConfig,
        load_best: bool = False,
    ) -> None:
        self._config = config
        self._device = torch.device(config.device)
        config.ensure_directories()
        self._model = HamiltonianNeuralNetwork(config.model).to(self._device)
        self._audio_processor = AudioProcessor(config.audio, config.device)
        self._visualizer = HamiltonianAudioVisualizer(
            config.visualization, config.audio
        )
        self._metrics_tracker = HamiltonianMetricsTracker(config.metrics)
        self._checkpoint_manager = CheckpointManager(config.checkpoint)
        metadata = self._checkpoint_manager.load_checkpoint(
            self._model, load_best=load_best
        )
        if metadata is not None:
            print(
                f"[INFERENCE] Model loaded from checkpoint. "
                f"Epoch: {metadata.get('epoch', 'unknown')}, "
                f"Loss: {metadata.get('current_loss', 'unknown')}"
            )
        else:
            print(
                "[INFERENCE] WARNING: No checkpoint found. "
                "Using randomly initialized model."
            )
        self._model.eval()

    def analyze_audio(
        self,
        audio_file_path: str,
        output_prefix: Optional[str] = None,
    ) -> None:
        """
        Perform complete Hamiltonian analysis on an audio file.

        Args:
            audio_file_path: Path to the audio file to analyze.
            output_prefix: Optional prefix for output filenames.
        """
        if output_prefix is None:
            output_prefix = os.path.splitext(os.path.basename(audio_file_path))[0]

        print(f"[INFERENCE] Loading audio: {audio_file_path}")
        waveform, sample_rate = self._audio_processor.load_audio(audio_file_path)
        print(
            f"[INFERENCE] Audio loaded: {waveform.shape[1]} samples, "
            f"{waveform.shape[1] / sample_rate:.2f} seconds"
        )

        print("[INFERENCE] Computing complex STFT...")
        stft_complex = self._audio_processor.waveform_to_stft_complex(waveform)
        magnitude, phase = self._audio_processor.stft_to_magnitude_phase(stft_complex)
        print(
            f"[INFERENCE] STFT shape: {stft_complex.shape} "
            f"(freq_bins={stft_complex.shape[0]}, time_frames={stft_complex.shape[1]})"
        )

        print("[INFERENCE] Computing Hamiltonian energy mask on STFT magnitude...")
        model_input = self._audio_processor.stft_magnitude_to_model_input(magnitude)
        print(f"[INFERENCE] Model input shape: {model_input.shape}")

        energy_mask = self._compute_energy_mask_patched(model_input)
        print(
            f"[INFERENCE] Energy mask range: "
            f"[{energy_mask.min().item():.6f}, {energy_mask.max().item():.6f}]"
        )

        print("[INFERENCE] Extracting Hamiltonian fields for visualization...")
        amplitude_map, phase_map, action_map = self._extract_hamiltonian_fields_patched(
            model_input
        )

        print("[INFERENCE] Applying energy mask to STFT magnitude...")
        mask_2d = energy_mask.squeeze(0).squeeze(0)
        reconstructed_magnitude = magnitude * mask_2d

        print("[INFERENCE] Recombining with original phase via ISTFT...")
        reconstructed_stft = self._audio_processor.magnitude_phase_to_stft(
            reconstructed_magnitude, phase
        )
        reconstructed_waveform = self._audio_processor.stft_complex_to_waveform(
            reconstructed_stft
        )

        self._compute_inference_metrics(magnitude, reconstructed_magnitude, stft_complex)
        self._print_inference_metrics()

        if self._config.visualization.export_reconstructed_audio:
            output_audio_path = os.path.join(
                self._config.visualization.output_directory,
                f"{output_prefix}_reconstructed.wav",
            )
            self._audio_processor.save_audio(reconstructed_waveform, output_audio_path)
            print(f"[INFERENCE] Reconstructed audio saved: {output_audio_path}")

        mel_spec_original = self._audio_processor.waveform_to_mel_spectrogram(waveform)
        mel_spec_reconstructed = self._audio_processor.waveform_to_mel_spectrogram(
            reconstructed_waveform
        )

        self._visualizer.render_complete_analysis(
            amplitude_map=amplitude_map,
            phase_map=phase_map,
            action_map=action_map,
            original_spectrogram=mel_spec_original,
            reconstructed_spectrogram=mel_spec_reconstructed,
            original_waveform=waveform,
            reconstructed_waveform=reconstructed_waveform,
            output_prefix=output_prefix,
        )
        print("[INFERENCE] Analysis complete.")

    def _compute_energy_mask_patched(
        self, model_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute energy mask over the full STFT magnitude, processing
        in patches along the time axis if the input exceeds patch width.

        Args:
            model_input: Normalized STFT magnitude [1, 1, freq_bins, time_frames].

        Returns:
            Energy mask [1, 1, freq_bins, time_frames] in [0, 1].
        """
        _, _, freq_bins, total_frames = model_input.shape
        patch_width = self._config.model.matrix_size_width

        if total_frames <= patch_width:
            padded = torch.zeros(
                1, 1, freq_bins, patch_width, device=self._device
            )
            padded[:, :, :, :total_frames] = model_input
            with torch.no_grad():
                mask = self._model.compute_energy_mask(padded)
            return mask[:, :, :, :total_frames]

        mask_parts = []
        stride = patch_width
        start = 0
        while start < total_frames:
            end = min(start + patch_width, total_frames)
            patch = model_input[:, :, :, start:end]
            actual_width = patch.shape[3]

            if actual_width < patch_width:
                padded = torch.zeros(
                    1, 1, freq_bins, patch_width, device=self._device
                )
                padded[:, :, :, :actual_width] = patch
                patch = padded

            with torch.no_grad():
                mask_patch = self._model.compute_energy_mask(patch)

            if actual_width < patch_width:
                mask_patch = mask_patch[:, :, :, :actual_width]

            mask_parts.append(mask_patch)
            start += stride

        return torch.cat(mask_parts, dim=3)

    def _extract_hamiltonian_fields_patched(
        self, model_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract Hamiltonian fields over full STFT magnitude with patching.

        Args:
            model_input: Normalized STFT magnitude [1, 1, freq_bins, time_frames].

        Returns:
            Tuple of (amplitude_map, phase_map, action_map).
        """
        _, _, freq_bins, total_frames = model_input.shape
        patch_width = self._config.model.matrix_size_width

        if total_frames <= patch_width:
            padded = torch.zeros(
                1, 1, freq_bins, patch_width, device=self._device
            )
            padded[:, :, :, :total_frames] = model_input
            with torch.no_grad():
                amp, ph, act = self._model.extract_hamiltonian_fields(padded)
            return amp[:, :total_frames], ph[:, :total_frames], act[:, :total_frames]

        amp_parts = []
        phase_parts = []
        action_parts = []
        stride = patch_width
        start = 0

        while start < total_frames:
            end = min(start + patch_width, total_frames)
            patch = model_input[:, :, :, start:end]
            actual_width = patch.shape[3]

            if actual_width < patch_width:
                padded = torch.zeros(
                    1, 1, freq_bins, patch_width, device=self._device
                )
                padded[:, :, :, :actual_width] = patch
                patch = padded

            with torch.no_grad():
                amp, ph, act = self._model.extract_hamiltonian_fields(patch)

            if actual_width < patch_width:
                amp = amp[:, :actual_width]
                ph = ph[:, :actual_width]
                act = act[:, :actual_width]

            amp_parts.append(amp)
            phase_parts.append(ph)
            action_parts.append(act)
            start += stride

        return (
            torch.cat(amp_parts, dim=-1),
            torch.cat(phase_parts, dim=-1),
            torch.cat(action_parts, dim=-1),
        )

    def _compute_inference_metrics(
        self,
        original_magnitude: torch.Tensor,
        reconstructed_magnitude: torch.Tensor,
        original_stft: torch.Tensor,
    ) -> None:
        """Compute all inference-time metrics on the STFT domain."""
        with torch.no_grad():
            orig_4d = original_magnitude.unsqueeze(0).unsqueeze(0)
            recon_4d = reconstructed_magnitude.unsqueeze(0).unsqueeze(0)
            self._metrics_tracker.compute_reconstruction_snr(orig_4d, recon_4d)
            self._metrics_tracker.compute_spectral_convergence(
                original_magnitude, reconstructed_magnitude
            )
            self._metrics_tracker.compute_spectral_entropy(reconstructed_magnitude)
            phase_orig = torch.angle(original_stft)
            self._metrics_tracker.compute_phase_coherence(phase_orig, phase_orig)

    def _print_inference_metrics(self) -> None:
        """Print all computed inference metrics."""
        metrics = self._metrics_tracker.get_current_metrics()
        print("[INFERENCE METRICS]")
        for name, value in sorted(metrics.items()):
            if isinstance(value, float):
                if abs(value) < 1e-3 or abs(value) > 1e4:
                    print(f"  {name}: {value:.6e}")
                else:
                    print(f"  {name}: {value:.6f}")
            else:
                print(f"  {name}: {value}")
