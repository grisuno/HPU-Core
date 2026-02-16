"""
Audio I/O and Spectral Transform Module.

Core design principle: audio is processed in the COMPLEX STFT domain,
which is the natural 2D (time x frequency) complex representation
analogous to how images are 2D spatial fields.

The complex STFT preserves both magnitude AND phase, which is essential
for the Hamiltonian evolution (the learned kernel operates on complex
fields via real+imaginary components). This is why the same architecture
that works on images can work on audio: both are 2D complex fields
when viewed in the correct domain.

Pipeline:
    WAV -> STFT complex [freq_bins, time_frames] -> Hamiltonian evolution
    -> Modified STFT complex -> ISTFT -> Reconstructed WAV

Mel spectrograms are computed only for visualization purposes.

Follows Single Responsibility Principle: solely responsible for
the boundary between raw audio waveforms and spectral tensors.
"""

import torch
import torchaudio
import torchaudio.transforms as T
from typing import Tuple, Optional
from config import AudioProcessingConfig


class AudioProcessor:
    """
    Audio processing pipeline centered on the complex STFT domain.

    The complex STFT is the audio equivalent of a grayscale image:
    a 2D field where one axis is time, the other is frequency, and
    each point carries a complex value (magnitude + phase). This is
    the correct domain for applying the Hamiltonian spectral evolution.
    """

    def __init__(self, config: AudioProcessingConfig, device: str = "cpu") -> None:
        self._config = config
        self._device = torch.device(device)
        self._window = torch.hann_window(config.n_fft).to(self._device)
        self._mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            power=config.power_spectrogram,
        ).to(self._device)
        self._amplitude_to_db = T.AmplitudeToDB(
            stype="power",
            top_db=config.amplitude_to_db_top_db,
        ).to(self._device)

    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load an audio file and convert to mono at the target sample rate.

        Args:
            file_path: Path to the audio file.

        Returns:
            Tuple of (waveform tensor [1, T], sample_rate).
        """
        waveform, original_sr = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if original_sr != self._config.sample_rate:
            resampler = T.Resample(
                orig_freq=original_sr,
                new_freq=self._config.sample_rate,
            ).to(self._device)
            waveform = resampler(waveform.to(self._device))
        else:
            waveform = waveform.to(self._device)
        return waveform, self._config.sample_rate

    def waveform_to_stft_complex(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute the complex STFT of a waveform.

        This is the primary transform: converts 1D audio into a 2D complex
        field (freq_bins x time_frames) that the Hamiltonian network
        can process identically to how it processes images.

        Args:
            waveform: Audio waveform [1, T] or [T].

        Returns:
            Complex STFT tensor [freq_bins, time_frames].
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)
        stft_complex = torch.stft(
            waveform,
            n_fft=self._config.n_fft,
            hop_length=self._config.hop_length,
            window=self._window,
            return_complex=True,
        )
        return stft_complex

    def stft_complex_to_waveform(self, stft_complex: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct waveform from complex STFT via inverse STFT.

        Unlike Griffin-Lim (which estimates phase), ISTFT uses the
        EXACT phase from the complex STFT, producing a faithful
        reconstruction when the magnitude/phase have been coherently
        modified by the Hamiltonian evolution.

        Args:
            stft_complex: Complex STFT tensor [freq_bins, time_frames].

        Returns:
            Reconstructed waveform [1, T].
        """
        waveform = torch.istft(
            stft_complex,
            n_fft=self._config.n_fft,
            hop_length=self._config.hop_length,
            window=self._window,
        )
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return waveform

    def stft_to_magnitude_phase(
        self, stft_complex: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose complex STFT into magnitude and phase.

        Args:
            stft_complex: Complex STFT [freq_bins, time_frames].

        Returns:
            Tuple of (magnitude, phase) each [freq_bins, time_frames].
        """
        magnitude = torch.abs(stft_complex)
        phase = torch.angle(stft_complex)
        return magnitude, phase

    def magnitude_phase_to_stft(
        self, magnitude: torch.Tensor, phase: torch.Tensor
    ) -> torch.Tensor:
        """
        Recombine magnitude and phase into complex STFT.

        Args:
            magnitude: Magnitude spectrum [freq_bins, time_frames].
            phase: Phase spectrum [freq_bins, time_frames].

        Returns:
            Complex STFT [freq_bins, time_frames].
        """
        return magnitude * torch.exp(1j * phase)

    def stft_magnitude_to_model_input(
        self, magnitude: torch.Tensor
    ) -> torch.Tensor:
        """
        Prepare STFT magnitude for input to the Hamiltonian network.

        Normalizes magnitude to [0, 1] range and shapes as [1, 1, H, W],
        matching the expected input format (analogous to a grayscale image).

        Args:
            magnitude: STFT magnitude [freq_bins, time_frames].

        Returns:
            Normalized tensor [1, 1, freq_bins, time_frames].
        """
        mag_db = 20.0 * torch.log10(magnitude + self._config.normalization_floor)
        mag_min = mag_db.min()
        mag_max = mag_db.max()
        mag_range = mag_max - mag_min
        if mag_range > self._config.normalization_floor:
            normalized = (mag_db - mag_min) / mag_range
        else:
            normalized = torch.zeros_like(mag_db)
        return normalized.unsqueeze(0).unsqueeze(0)

    def model_output_to_stft_magnitude(
        self, model_output: torch.Tensor, original_magnitude: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert model output (energy mask in [0, 1]) back to STFT magnitude scale.

        The model output represents the Hamiltonian energy structure --
        which regions of the time-frequency plane carry coherent energy.
        This is used to modulate the original magnitude.

        Args:
            model_output: Energy mask [1, 1, freq_bins, time_frames] in [0, 1].
            original_magnitude: Original STFT magnitude [freq_bins, time_frames].

        Returns:
            Reconstructed magnitude [freq_bins, time_frames].
        """
        mask = model_output.squeeze(0).squeeze(0)
        return original_magnitude * mask

    def waveform_to_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to normalized mel spectrogram (for visualization only).

        Args:
            waveform: Audio waveform tensor [1, T] or [B, 1, T].

        Returns:
            Normalized mel spectrogram [B, 1, n_mels, time_frames].
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        mel_spec = self._mel_spectrogram_transform(waveform)
        mel_spec_db = self._amplitude_to_db(mel_spec)
        spec_min = mel_spec_db.min()
        spec_max = mel_spec_db.max()
        spec_range = spec_max - spec_min
        if spec_range > self._config.normalization_floor:
            mel_spec_normalized = (mel_spec_db - spec_min) / spec_range
        else:
            mel_spec_normalized = torch.zeros_like(mel_spec_db)
        return mel_spec_normalized

    def save_audio(
        self, waveform: torch.Tensor, file_path: str, sample_rate: Optional[int] = None
    ) -> None:
        """
        Save a waveform tensor to an audio file.

        Args:
            waveform: Audio tensor [1, T] or [T].
            file_path: Output file path.
            sample_rate: Sample rate (defaults to config sample rate).
        """
        sr = sample_rate if sample_rate is not None else self._config.sample_rate
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform_cpu = waveform.detach().cpu()
        max_val = waveform_cpu.abs().max()
        if max_val > self._config.normalization_floor:
            waveform_cpu = waveform_cpu / max_val
        torchaudio.save(file_path, waveform_cpu, sr)

    def get_spectrogram_db_range(self, waveform: torch.Tensor) -> Tuple[float, float]:
        """
        Compute the dB range of a waveform's mel spectrogram.

        Args:
            waveform: Audio waveform [1, T].

        Returns:
            Tuple of (db_min, db_max).
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        mel_spec = self._mel_spectrogram_transform(waveform)
        mel_spec_db = self._amplitude_to_db(mel_spec)
        return mel_spec_db.min().item(), mel_spec_db.max().item()
