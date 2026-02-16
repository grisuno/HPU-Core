"""
Hamiltonian Audio Processing Configuration Module.

Centralizes all hyperparameters, architectural constants, training parameters,
checkpoint settings, and audio processing parameters. No magic numbers exist
outside this module.

Follows Single Responsibility Principle: this module is solely responsible
for holding and validating configuration state.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class AudioProcessingConfig:
    """Parameters governing raw audio ingestion and spectrogram computation."""

    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    power_spectrogram: float = 2.0
    amplitude_to_db_ref: float = 1.0
    amplitude_to_db_top_db: float = 80.0
    normalization_floor: float = 1e-8
    mono_channel_index: int = 0
    supported_extensions: tuple = (".wav", ".mp3", ".flac", ".ogg", ".aiff")


@dataclass
class ModelArchitectureConfig:
    """
    Parametric architecture dimensions for the Hamiltonian Neural Network.

    All hidden dimensions, matrix sizes, expansion factors, and layer counts
    are configurable from this single source of truth.
    """

    input_channels: int = 1
    hidden_dimension: int = 32
    matrix_size_height: int = 128
    matrix_size_width: int = 128
    num_spectral_layers: int = 2
    kernel_expansion_factors: List[int] = field(default_factory=lambda: [1, 1])
    output_channels: int = 1
    activation_negative_slope: float = 0.01
    gelu_approximate: str = "none"
    input_projection_kernel_size: int = 1
    input_projection_padding: int = 0
    output_projection_kernel_size: int = 1
    output_projection_padding: int = 0
    kernel_init_std: float = 0.02
    spectral_kernel_base_height: int = 9
    spectral_kernel_base_width: int = 16
    symplectic_integration_steps: int = 4
    energy_conservation_weight: float = 0.1
    symplectic_regularization_weight: float = 0.01

    def validate(self) -> None:
        """Ensure architectural coherence."""
        if len(self.kernel_expansion_factors) != self.num_spectral_layers:
            raise ValueError(
                f"kernel_expansion_factors length ({len(self.kernel_expansion_factors)}) "
                f"must match num_spectral_layers ({self.num_spectral_layers})"
            )
        if self.hidden_dimension <= 0:
            raise ValueError(
                f"hidden_dimension must be positive, got {self.hidden_dimension}"
            )
        if self.matrix_size_height <= 0 or self.matrix_size_width <= 0:
            raise ValueError(
                "matrix_size_height and matrix_size_width must be positive"
            )


@dataclass
class TrainingConfig:
    """All training loop hyperparameters and scheduling constants."""

    num_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_scheduler_step_size: int = 20
    lr_scheduler_gamma: float = 0.5
    gradient_clip_max_norm: float = 1.0
    validation_split_ratio: float = 0.1
    random_seed: int = 42
    num_data_loader_workers: int = 4
    pin_memory: bool = True
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-6
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1
    loss_reconstruction_weight: float = 1.0
    loss_energy_conservation_weight: float = 0.1
    loss_symplectic_weight: float = 0.01
    loss_spectral_consistency_weight: float = 0.05
    loss_phase_coherence_weight: float = 0.02
    loss_action_minimization_weight: float = 0.03
    loss_liouville_weight: float = 0.01
    loss_hamiltonian_constraint_weight: float = 0.05


@dataclass
class CheckpointConfig:
    """Checkpoint persistence parameters."""

    checkpoint_directory: str = "checkpoints"
    checkpoint_filename: str = "latest.safetensors"
    checkpoint_file_path: Optional[str] = None
    checkpoint_interval_minutes: int = 5
    keep_best_model: bool = True
    best_model_filename: str = "best.safetensors"
    metadata_filename: str = "training_metadata.json"

    @property
    def checkpoint_path(self) -> str:
        if self.checkpoint_file_path is not None:
            return self.checkpoint_file_path
        return os.path.join(self.checkpoint_directory, self.checkpoint_filename)

    @property
    def best_model_path(self) -> str:
        return os.path.join(self.checkpoint_directory, self.best_model_filename)

    @property
    def metadata_path(self) -> str:
        return os.path.join(self.checkpoint_directory, self.metadata_filename)


@dataclass
class VisualizationConfig:
    """Parameters for audio reconstruction visualization and output."""

    output_directory: str = "output"
    colormap_amplitude: str = "JET"
    colormap_phase: str = "TWILIGHT"
    colormap_action: str = "JET"
    figure_dpi: int = 150
    spectrogram_figure_width: int = 12
    spectrogram_figure_height: int = 8
    waveform_figure_width: int = 14
    waveform_figure_height: int = 4
    phase_portrait_bins: int = 256
    energy_plot_window_size: int = 50
    export_reconstructed_audio: bool = True
    export_spectrograms: bool = True
    export_phase_portraits: bool = True
    export_energy_landscapes: bool = True
    export_waveform_comparison: bool = True


@dataclass
class MetricsConfig:
    """Configuration for all tracked metrics during training and inference."""

    log_interval_steps: int = 10
    track_hamiltonian_energy: bool = True
    track_symplectic_form: bool = True
    track_liouville_measure: bool = True
    track_phase_space_volume: bool = True
    track_action_integral: bool = True
    track_poisson_bracket: bool = True
    track_spectral_entropy: bool = True
    track_reconstruction_snr: bool = True
    track_spectral_convergence: bool = True
    track_phase_coherence: bool = True
    track_energy_drift: bool = True
    track_gradient_norm: bool = True
    track_parameter_norm: bool = True
    track_learning_rate: bool = True
    track_loss_components: bool = True
    moving_average_window: int = 100
    energy_drift_tolerance: float = 1e-4


@dataclass
class HamiltonianAudioConfig:
    """
    Top-level configuration aggregator.

    Composes all sub-configurations into a single injectable dependency,
    following the Dependency Inversion Principle.
    """

    audio: AudioProcessingConfig = field(default_factory=AudioProcessingConfig)
    model: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    device: str = "cpu"

    def validate_all(self) -> None:
        """Run validation on all sub-configurations."""
        self.model.validate()
        if self.training.validation_split_ratio < 0.0 or self.training.validation_split_ratio > 1.0:
            raise ValueError("validation_split_ratio must be in [0.0, 1.0]")
        if self.checkpoint.checkpoint_interval_minutes <= 0:
            raise ValueError("checkpoint_interval_minutes must be positive")

    def ensure_directories(self) -> None:
        """Create required output directories if they do not exist."""
        os.makedirs(self.checkpoint.checkpoint_directory, exist_ok=True)
        os.makedirs(self.visualization.output_directory, exist_ok=True)
