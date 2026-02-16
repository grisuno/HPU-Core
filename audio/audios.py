"""
Hamiltonian Perception Unit - Audio Modality
A SOLID-compliant implementation demonstrating that sensory perception
is an epiphenomenon of underlying Hamiltonian field dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import time
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from pathlib import Path
from safetensors.torch import load_model, save_model

# Importar la arquitectura exacta del experimento original
from experiment2 import HamiltonianNeuralNetwork

# Audio processing imports - using only scipy, no librosa/numba dependency
try:
    from scipy.io import wavfile
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Install with: pip install scipy")

# Visualization imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Processing without visualization.")


# =============================================================================
# CONFIGURATION (Single Source of Truth)
# =============================================================================

@dataclass(frozen=True)
class HamiltonianConfig:
    """
    Immutable configuration container for all hyperparameters.
    Eliminates magic numbers and provides single point of control.
    """
    
    # Audio Processing Parameters
    target_sample_rate: int = 22050
    fft_size: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    segment_duration: float = 2.0
    
    # Neural Architecture Parameters - DEBEN COINCIDIR CON experiment2
    input_channels: int = 1
    hidden_dimensions: int = 32  # Coincide con Config.HIDDEN_DIM en experiment2
    spectral_matrix_size: int = 16  # Coincide con Config.GRID_SIZE en experiment2
    num_spectral_layers: int = 2  # Coincide con Config.NUM_SPECTRAL_LAYERS en experiment2
    expansion_factor: float = 2.0
    
    # Hamiltonian Dynamics Parameters
    time_step: float = 0.01
    dissipation_rate: float = 0.001
    resonance_frequency: float = 1.0
    
    # Training Parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    batch_size: int = 8
    checkpoint_interval_minutes: float = 5.0
    
    # Visualization Parameters (OpenCV colormap constants as integers)
    colormap_energy: int = 2  # cv2.COLORMAP_JET
    colormap_phase: int = 16  # cv2.COLORMAP_TWILIGHT
    colormap_action: int = 11  # cv2.COLORMAP_HOT
    
    # System Parameters
    device: str = "cpu"
    random_seed: int = 42
    
    @property
    def segment_samples(self) -> int:
        """Calculate segment length in samples."""
        return int(self.segment_duration * self.target_sample_rate)
    
    @property
    def freq_bins(self) -> int:
        """Calculate frequency bins for real FFT."""
        return self.fft_size // 2 + 1


# =============================================================================
# ABSTRACTIONS (Interfaces)
# =============================================================================

class IAudioSource(ABC):
    """Interface for audio input sources."""
    
    @abstractmethod
    def read_segment(self) -> Optional[np.ndarray]:
        """Read audio segment. Returns None when exhausted."""
        pass
    
    @abstractmethod
    def get_properties(self) -> Dict[str, any]:
        """Return audio properties."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        pass


class IFieldOperator(ABC):
    """Interface for Hamiltonian field evolution operators."""
    
    @abstractmethod
    def evolve(self, field_state: torch.Tensor) -> torch.Tensor:
        """Evolve field state through Hamiltonian dynamics."""
        pass


class IMetricCollector(ABC):
    """Interface for training metrics collection."""
    
    @abstractmethod
    def record(self, metrics: Dict[str, float]) -> None:
        """Record metric values."""
        pass
    
    @abstractmethod
    def get_summary(self) -> Dict[str, float]:
        """Return aggregated metrics."""
        pass


# =============================================================================
# AUDIO RESAMPLING UTILITIES (Pure NumPy/SciPy, no Librosa)
# =============================================================================

class AudioResampler:
    """
    Handles audio resampling using scipy.signal, avoiding librosa/numba dependencies.
    """
    
    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio from orig_sr to target_sr using polyphase filtering.
        """
        if orig_sr == target_sr:
            return audio
        
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        
        if up == 1 and down == 1:
            return audio
        
        resampled = signal.resample_poly(audio, up, down)
        return resampled.astype(np.float32)
    
    @staticmethod
    def load_wav_with_resample(file_path: str, target_sr: int) -> Tuple[np.ndarray, int]:
        """
        Load WAV file and resample to target sample rate.
        Returns (audio_data, original_sample_rate).
        """
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy is required for audio loading")
        
        orig_sr, data = wavfile.read(file_path)
        
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128) / 128.0
        else:
            data = data.astype(np.float32)
        
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        
        resampled = AudioResampler.resample(data, orig_sr, target_sr)
        
        return resampled.astype(np.float32), orig_sr


# =============================================================================
# CONCRETE IMPLEMENTATIONS
# =============================================================================

class WaveFileSource(IAudioSource):
    """
    Concrete implementation of audio source from file.
    Supports automatic resampling to target sample rate using scipy.
    """
    
    def __init__(self, file_path: str, config: HamiltonianConfig):
        self._config = config
        self._file_path = file_path
        self._audio_data: Optional[np.ndarray] = None
        self._original_sample_rate: int = 0
        self._total_samples: int = 0
        self._current_position: int = 0
        
        self._validate_and_load()
    
    def _validate_and_load(self) -> None:
        """Validate file format and load with automatic resampling."""
        if not os.path.exists(self._file_path):
            raise FileNotFoundError(f"Audio file not found: {self._file_path}")
        
        if not SCIPY_AVAILABLE:
            raise RuntimeError(
                "scipy is required for audio processing. "
                "Install with: pip install scipy"
            )
        
        try:
            self._audio_data, self._original_sample_rate = AudioResampler.load_wav_with_resample(
                self._file_path,
                self._config.target_sample_rate
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {e}")
        
        self._total_samples = len(self._audio_data)
        
        if self._total_samples == 0:
            raise ValueError("Audio file contains no samples")
    
    def read_segment(self) -> Optional[np.ndarray]:
        """Read next audio segment."""
        if self._current_position >= self._total_samples:
            return None
        
        segment_samples = self._config.segment_samples
        end_position = min(self._current_position + segment_samples, self._total_samples)
        
        segment = self._audio_data[self._current_position:end_position]
        self._current_position = end_position
        
        if len(segment) < segment_samples:
            padding = np.zeros(segment_samples - len(segment), dtype=np.float32)
            segment = np.concatenate([segment, padding])
        
        return segment.astype(np.float32)
    
    def get_properties(self) -> Dict[str, any]:
        """Return audio file properties."""
        return {
            'file_path': self._file_path,
            'original_sample_rate': self._original_sample_rate,
            'target_sample_rate': self._config.target_sample_rate,
            'total_samples': self._total_samples,
            'duration_seconds': self._total_samples / self._config.target_sample_rate,
            'num_channels': 1,
            'segments_total': int(np.ceil(self._total_samples / self._config.segment_samples))
        }
    
    def close(self) -> None:
        """Release resources."""
        self._audio_data = None


class ComprehensiveMetricCollector(IMetricCollector):
    """
    Collects all metrics from Hamiltonian paper, activation functions,
    and architectural diagnostics for informed decision-making.
    """
    
    def __init__(self, config: HamiltonianConfig):
        self._config = config
        self._metrics_history: List[Dict[str, float]] = []
        self._start_time = time.time()
    
    def record(self, metrics: Dict[str, float]) -> None:
        """Record comprehensive metrics."""
        enriched_metrics = {
            **metrics,
            'elapsed_time_seconds': time.time() - self._start_time,
            'timestamp': time.time()
        }
        self._metrics_history.append(enriched_metrics)
    
    def get_summary(self) -> Dict[str, float]:
        """Return statistical summary of all metrics."""
        if not self._metrics_history:
            return {}
        
        summary = {}
        keys = self._metrics_history[0].keys()
        
        for key in keys:
            if key == 'timestamp':
                continue
            
            values = [m[key] for m in self._metrics_history if key in m]
            if values:
                summary[f"{key}_mean"] = np.mean(values)
                summary[f"{key}_std"] = np.std(values)
                summary[f"{key}_min"] = np.min(values)
                summary[f"{key}_max"] = np.max(values)
                summary[f"{key}_last"] = values[-1]
        
        return summary
    
    def export_to_json(self, path: str) -> None:
        """Export full history to JSON."""
        with open(path, 'w') as f:
            json.dump(self._metrics_history, f, indent=2)


class CheckpointManager:
    """
    Manages periodic checkpointing with atomic writes.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: HamiltonianConfig,
        checkpoint_dir: str = "checkpoints"
    ):
        self._model = model
        self._config = config
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._last_checkpoint_time = time.time()
        self._checkpoint_count = 0
    
    def check_and_save(self, force: bool = False) -> Optional[str]:
        """
        Check if checkpoint interval elapsed and save if necessary.
        Returns path if saved, None otherwise.
        """
        current_time = time.time()
        elapsed_minutes = (current_time - self._last_checkpoint_time) / 60.0
        
        if force or elapsed_minutes >= self._config.checkpoint_interval_minutes:
            return self._save_checkpoint()
        
        return None
    
    def _save_checkpoint(self) -> str:
        """Atomic checkpoint save."""
        self._checkpoint_count += 1
        timestamp = int(time.time())
        temp_path = self._checkpoint_dir / f"temp_{timestamp}.safetensors"
        final_path = self._checkpoint_dir / "latest.safetensors"
        
        save_model(self._model, str(temp_path))
        temp_path.replace(final_path)
        
        metadata_path = self._checkpoint_dir / "checkpoint_metadata.json"
        metadata = {
            'checkpoint_number': self._checkpoint_count,
            'timestamp': timestamp,
            'elapsed_minutes': (time.time() - self._last_checkpoint_time) / 60.0,
            'config': {
                'hidden_dimensions': self._config.hidden_dimensions,
                'spectral_matrix_size': self._config.spectral_matrix_size,
                'num_spectral_layers': self._config.num_spectral_layers
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self._last_checkpoint_time = time.time()
        
        return str(final_path)


class AudioSpectrogramConverter:
    """
    Converts between audio waveforms and 2D field representations.
    Adaptado para la arquitectura de experiment2 (grid_size=16).
    """
    
    def __init__(self, config: HamiltonianConfig):
        self._config = config
    
    def waveform_to_field(self, waveform: np.ndarray) -> torch.Tensor:
        """
        Convert 1D audio to 2D field representation via STFT.
        Returns (1, 1, grid_size, grid_size) tensor compatible con experiment2.
        """
        waveform_tensor = torch.from_numpy(waveform).float()
        
        # Calcular STFT
        stft = torch.stft(
            waveform_tensor,
            n_fft=self._config.fft_size,
            hop_length=self._config.hop_length,
            win_length=self._config.fft_size,
            window=torch.hann_window(self._config.fft_size),
            return_complex=True
        )
        
        magnitude = torch.abs(stft)
        
        # Adaptar a la arquitectura de experiment2 (grid_size x grid_size)
        target_size = self._config.spectral_matrix_size
        
        # Redimensionar espectrograma a grid_size x grid_size
        field_2d = magnitude.unsqueeze(0).unsqueeze(0)
        
        # Interpolar al tamaño esperado por el modelo
        if field_2d.shape[-2] != target_size or field_2d.shape[-1] != target_size:
            field_2d = F.interpolate(
                field_2d,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            )
        
        return field_2d
    
    def field_to_waveform(self, field: torch.Tensor, original_length: int) -> np.ndarray:
        """
        Reconstruct waveform from 2D field representation.
        """
        field = field.squeeze().numpy()
        
        griffin_lim_iterations = 60
        
        spectrogram = field
        
        angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))
        x = spectrogram * angles
        
        for _ in range(griffin_lim_iterations):
            y = self._inverse_spectrogram(x)
            y = y[:original_length]
            
            if len(y) < original_length:
                y = np.pad(y, (0, original_length - len(y)))
            
            new_stft = self._forward_spectrogram(y)
            angles = np.exp(1j * np.angle(new_stft))
            x = spectrogram * angles
        
        return self._inverse_spectrogram(x)[:original_length]
    
    def _forward_spectrogram(self, x: np.ndarray) -> np.ndarray:
        """Compute magnitude spectrogram."""
        stft = np.fft.rfft(x)
        return np.abs(stft)
    
    def _inverse_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """Griffin-Lim inverse."""
        return np.fft.irfft(spectrogram)


class HamiltonianAudioProcessor:
    """
    Main orchestrator for Hamiltonian audio processing.
    Demonstrates that auditory perception is epiphenomenon of Hamiltonian dynamics.
    Usa la arquitectura exacta de experiment2.
    """
    
    def __init__(
        self,
        config: HamiltonianConfig,
        model: Optional[HamiltonianNeuralNetwork] = None,
        source: Optional[IAudioSource] = None
    ):
        self._config = config
        self._device = torch.device(config.device)
        
        # Usar la arquitectura importada de experiment2 con parámetros correctos
        if model is None:
            self._model = HamiltonianNeuralNetwork(
                grid_size=config.spectral_matrix_size,
                hidden_dim=config.hidden_dimensions,
                num_spectral_layers=config.num_spectral_layers
            )
        else:
            self._model = model
        
        self._model.to(self._device)
        self._model.eval()
        
        self._source = source
        self._converter = AudioSpectrogramConverter(config)
        self._metrics = ComprehensiveMetricCollector(config)
        self._checkpoint_manager = CheckpointManager(self._model, config)
        
        self._visualization_active = False
    
    def load_model_weights(self, path: str) -> None:
        """Load pretrained Hamiltonian operator desde safetensors."""
        try:
            load_model(self._model, path)
            print(f"Hamiltonian operator loaded from {path}")
        except Exception as e:
            print(f"Critical error loading Hamiltonian operator: {e}")
            sys.exit(1)
    
    def attach_source(self, source: IAudioSource) -> None:
        """Attach audio source via dependency injection."""
        self._source = source
    
    def process_stream(self) -> None:
        """
        Process audio stream through Hamiltonian perception.
        Generates three epiphenomenal representations:
        1. Energy Density (Resonance)
        2. Topological Phase (Vortices)
        3. Action Map (Perceptual Clarity)
        """
        if self._source is None:
            raise RuntimeError("No audio source attached")
        
        # Detectar si hay display disponible para OpenCV
        if CV2_AVAILABLE:
            try:
                cv2.namedWindow("test", cv2.WINDOW_NORMAL)
                cv2.destroyWindow("test")
                self._visualization_active = True
            except cv2.error:
                print("Warning: No display available, processing without visualization")
                self._visualization_active = False
        else:
            self._visualization_active = False
        
        print("Initializing Hamiltonian Perception of Acoustic Fields...")
        print("Epiphenomenon 1: Energy Density (Resonance)")
        print("Epiphenomenon 2: Topological Phase (Vortices)")
        print("Epiphenomenon 3: Action Map (Perceptual Clarity)")
        
        if self._visualization_active:
            print("Press 'q' to terminate observation, 'c' to force checkpoint.")
        else:
            print("Running in headless mode. Press Ctrl+C to terminate.")
        
        segment_count = 0
        
        try:
            while True:
                waveform = self._source.read_segment()
                if waveform is None:
                    print("Audio stream exhausted.")
                    break
                
                segment_count += 1
                
                metrics = self._process_single_segment(waveform, segment_count)
                self._metrics.record(metrics)
                
                checkpoint_path = self._checkpoint_manager.check_and_save()
                if checkpoint_path:
                    print(f"Checkpoint saved: {checkpoint_path}")
                
                if self._visualization_active:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('c'):
                        forced_path = self._checkpoint_manager.check_and_save(force=True)
                        print(f"Forced checkpoint: {forced_path}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        
        if self._visualization_active:
            cv2.destroyAllWindows()
        
        self._source.close()
        
        summary = self._metrics.get_summary()
        print("\nHamiltonian Perception Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
    
    def _process_single_segment(self, waveform: np.ndarray, index: int) -> Dict[str, float]:
        """Process single audio segment and return metrics."""
        field = self._converter.waveform_to_field(waveform).to(self._device)
        
        with torch.no_grad():
            # El modelo de experiment2 maneja internamente las dimensiones
            # input: (B, 1, H, W) -> output: (B, H, W) debido al squeeze(1)
            reconstruction = self._model(field)
            
            # Añadir dimensión de canal para comparación con field
            if reconstruction.dim() == 3:
                reconstruction = reconstruction.unsqueeze(1)
            
            # Calcular pérdida de reconstrucción con dimensiones correctas
            loss = F.mse_loss(reconstruction, field).item()
            
            # Extraer representaciones intermedias manualmente para visualización
            phi = F.gelu(self._model.input_proj(field))
            
            # Acceder a la primera capa espectral
            layer = self._model.spectral_layers[0]
            
            # ---------------------------------------------------------
            # VISIÓN 1 & 2: DOMINIO COMPLEJO (Topología y Resonancia)
            # ---------------------------------------------------------
            x_fft_complex = torch.fft.fft2(phi)
            _, _, freq_h_c, freq_w_c = x_fft_complex.shape
            
            # Adaptar kernels de la capa espectral al tamaño FFT
            kr_c = F.interpolate(
                layer.kernel_real.mean(dim=(0, 1), keepdim=True),
                size=(freq_h_c, freq_w_c),
                mode='bilinear',
                align_corners=False
            )
            ki_c = F.interpolate(
                layer.kernel_imag.mean(dim=(0, 1), keepdim=True),
                size=(freq_h_c, freq_w_c),
                mode='bilinear',
                align_corners=False
            )
            
            res_real_c = x_fft_complex.real * kr_c - x_fft_complex.imag * ki_c
            res_imag_c = x_fft_complex.real * ki_c + x_fft_complex.imag * kr_c
            evolved_fft_complex = torch.complex(res_real_c, res_imag_c)
            
            psi_t_complex = torch.fft.ifft2(evolved_fft_complex, s=phi.shape[-2:])
            
            amplitude_map = torch.abs(psi_t_complex).mean(dim=1).squeeze()
            phase_map = torch.angle(psi_t_complex).mean(dim=1).squeeze()
            
            # ---------------------------------------------------------
            # VISIÓN 3: ACTION MAP (Lo que "ve" el modelo claramente)
            # ---------------------------------------------------------
            x_fft_real = torch.fft.rfft2(phi)
            _, _, freq_h_r, freq_w_r = x_fft_real.shape
            
            kr_r = F.interpolate(
                layer.kernel_real.mean(dim=(0, 1), keepdim=True),
                size=(freq_h_r, freq_w_r),
                mode='bilinear',
                align_corners=False
            )
            ki_r = F.interpolate(
                layer.kernel_imag.mean(dim=(0, 1), keepdim=True),
                size=(freq_h_r, freq_w_r),
                mode='bilinear',
                align_corners=False
            )
            
            res_real_r = x_fft_real.real * kr_r - x_fft_real.imag * ki_r
            res_imag_r = x_fft_real.real * ki_r + x_fft_real.imag * kr_r
            evolved_fft_real = torch.complex(res_real_r, res_imag_r)
            psi_t_real = torch.fft.irfft2(evolved_fft_real, s=phi.shape[-2:])
            
            action = torch.abs(psi_t_real.mean(dim=1) - phi.mean(dim=1)).squeeze()
        
        if self._visualization_active:
            self._render_epiphenomena(amplitude_map, phase_map, action)
        
        return {
            'segment_index': index,
            'hamiltonian_loss': loss,
            'amplitude_mean': amplitude_map.mean().item(),
            'amplitude_std': amplitude_map.std().item(),
            'phase_entropy': self._calculate_phase_entropy(phase_map),
            'action_mean': action.mean().item(),
            'action_variance': action.var().item(),
            'gelu_activation_mean': phi.mean().item(),
            'gelu_activation_sparsity': (phi == 0).float().mean().item()
        }
    
    def _calculate_phase_entropy(self, phase_map: torch.Tensor) -> float:
        """Calculate topological entropy from phase distribution."""
        phase_np = phase_map.cpu().numpy().flatten()
        hist, _ = np.histogram(phase_np, bins=50, range=(-np.pi, np.pi), density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        return float(entropy)
    
    def _render_epiphenomena(
        self,
        amplitude: torch.Tensor,
        phase: torch.Tensor,
        action: torch.Tensor
    ) -> None:
        """Render three epiphenomenal visualizations."""
        if not CV2_AVAILABLE:
            return
        
        amp_np = amplitude.cpu().numpy()
        v_min, v_max = amp_np.min(), amp_np.max()
        if v_max > v_min:
            amp_norm = ((amp_np - v_min) / (v_max - v_min) * 255).astype(np.uint8)
        else:
            amp_norm = np.zeros_like(amp_np, dtype=np.uint8)
        amp_color = cv2.applyColorMap(amp_norm, self._config.colormap_energy)
        amp_color = cv2.resize(amp_color, (512, 512))
        
        phase_np = phase.cpu().numpy()
        phase_norm = ((phase_np + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        phase_color = cv2.applyColorMap(phase_norm, self._config.colormap_phase)
        phase_color = cv2.resize(phase_color, (512, 512))
        
        act_np = action.cpu().numpy()
        v_min_a, v_max_a = act_np.min(), act_np.max()
        if v_max_a > v_min_a:
            act_norm = ((act_np - v_min_a) / (v_max_a - v_min_a + 1e-8) * 255).astype(np.uint8)
        else:
            act_norm = np.zeros_like(act_np, dtype=np.uint8)
        act_color = cv2.applyColorMap(act_norm, self._config.colormap_action)
        act_color = cv2.resize(act_color, (512, 512))
        
        cv2.imshow("Epiphenomenon 1: Energy Density (Resonance)", amp_color)
        cv2.imshow("Epiphenomenon 2: Topological Phase (Vortices)", phase_color)
        cv2.imshow("Epiphenomenon 3: Action Map (Perceptual Clarity)", act_color)
    
    def export_metrics(self, path: str) -> None:
        """Export comprehensive metrics to file."""
        self._metrics.export_to_json(path)
    
    def force_checkpoint(self) -> str:
        """Force immediate checkpoint save."""
        return self._checkpoint_manager.check_and_save(force=True)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Hamiltonian Perception Unit - Audio Modality. "
                   "Demonstrates that sensory perception is epiphenomenon "
                   "of underlying Hamiltonian field dynamics."
    )
    parser.add_argument(
        'audio_file',
        type=str,
        help='Path to input audio file (WAV format)'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=22050,
        help='Target sample rate for processing (will resample if different)'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=32,
        help='Hidden dimensionality of Hamiltonian operator (must match checkpoint)'
    )
    parser.add_argument(
        '--matrix-size',
        type=int,
        default=16,
        help='Spectral matrix size for field representation (must match checkpoint)'
    )
    parser.add_argument(
        '--spectral-layers',
        type=int,
        default=2,
        help='Number of spectral layers (must match checkpoint)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/latest.safetensors',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Computation device'
    )
    parser.add_argument(
        '--segment-duration',
        type=float,
        default=2.0,
        help='Duration of each audio segment in seconds'
    )
    
    args = parser.parse_args()
    
    # Crear configuración con los parámetros exactos del modelo entrenado
    config = HamiltonianConfig(
        target_sample_rate=args.sample_rate,
        hidden_dimensions=args.hidden_dim,
        spectral_matrix_size=args.matrix_size,
        num_spectral_layers=args.spectral_layers,
        device=args.device,
        segment_duration=args.segment_duration
    )
    
    source = WaveFileSource(args.audio_file, config)
    properties = source.get_properties()
    print(f"Audio source properties: {properties}")
    
    processor = HamiltonianAudioProcessor(config)
    
    if os.path.exists(args.checkpoint):
        processor.load_model_weights(args.checkpoint)
    else:
        print(f"No checkpoint found at {args.checkpoint}, using initialized operator")
    
    processor.attach_source(source)
    processor.process_stream()
    
    metrics_path = f"metrics_{int(time.time())}.json"
    processor.export_metrics(metrics_path)
    print(f"Metrics exported to {metrics_path}")


if __name__ == "__main__":
    main()