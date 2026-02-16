"""
Hamiltonian Audio Processing System - Main Entry Point.

Demonstrates that sensory perception (hearing) is reducible to
physical field dynamics through Hamiltonian mechanics. The same
network architecture that reconstructs visual data from thermal
energy density can reconstruct audio from spectral energy fields.

Usage:
    Training:
        python main.py train --audio path/to/audio.wav [options]

    Inference:
        python main.py infer --audio path/to/audio.wav [options]

All architectural parameters are configurable via command-line arguments.
"""

import argparse
import sys
import os

from config import (
    HamiltonianAudioConfig,
    AudioProcessingConfig,
    ModelArchitectureConfig,
    TrainingConfig,
    CheckpointConfig,
    VisualizationConfig,
    MetricsConfig,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the complete argument parser with all configurable parameters."""
    parser = argparse.ArgumentParser(
        description=(
            "Hamiltonian Audio Processing System: "
            "Reconstructs audio signals through Hamiltonian spectral evolution, "
            "demonstrating the physical substrate of auditory perception."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")
    subparsers.required = True

    train_parser = subparsers.add_parser(
        "train", help="Train the Hamiltonian network on an audio file"
    )
    infer_parser = subparsers.add_parser(
        "infer", help="Run Hamiltonian analysis on an audio file using a trained model"
    )

    for sub in [train_parser, infer_parser]:
        sub.add_argument(
            "--audio", type=str, required=True,
            help="Path to input audio file (WAV, MP3, FLAC, OGG, AIFF)",
        )
        sub.add_argument("--device", type=str, default="cpu", help="Computation device")
        sub.add_argument("--sample-rate", type=int, default=22050, help="Audio sample rate in Hz")
        sub.add_argument("--n-fft", type=int, default=2048, help="FFT window size")
        sub.add_argument("--hop-length", type=int, default=512, help="STFT hop length")
        sub.add_argument("--n-mels", type=int, default=128, help="Number of mel frequency bins")
        sub.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension of the network")
        sub.add_argument("--matrix-height", type=int, default=128, help="Internal matrix height")
        sub.add_argument("--matrix-width", type=int, default=128, help="Internal matrix width / patch width")
        sub.add_argument("--num-spectral-layers", type=int, default=2, help="Number of spectral evolution layers")
        sub.add_argument("--expansion-factors", type=int, nargs="+", default=[1, 1], help="Kernel expansion factors per layer")
        sub.add_argument("--spectral-kernel-base-height", type=int, default=9, help="Base height for spectral kernels")
        sub.add_argument("--spectral-kernel-base-width", type=int, default=16, help="Base width for spectral kernels")
        sub.add_argument("--symplectic-steps", type=int, default=4, help="Symplectic integration steps")
        sub.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
        sub.add_argument("--checkpoint-path", type=str, default=None, help="Direct path to a .safetensors checkpoint file (overrides --checkpoint-dir)")
        sub.add_argument("--output-dir", type=str, default="output", help="Output directory for visualizations")

    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    train_parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay")
    train_parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping max norm")
    train_parser.add_argument("--checkpoint-interval", type=int, default=5, help="Checkpoint interval in minutes")
    train_parser.add_argument("--early-stopping-patience", type=int, default=15, help="Early stopping patience in epochs")
    train_parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split ratio")
    train_parser.add_argument("--warmup-epochs", type=int, default=5, help="Number of warmup epochs")
    train_parser.add_argument("--lr-step-size", type=int, default=20, help="LR scheduler step size")
    train_parser.add_argument("--lr-gamma", type=float, default=0.5, help="LR scheduler gamma")
    train_parser.add_argument("--loss-reconstruction-weight", type=float, default=1.0, help="Reconstruction loss weight")
    train_parser.add_argument("--loss-energy-weight", type=float, default=0.1, help="Energy conservation loss weight")
    train_parser.add_argument("--loss-symplectic-weight", type=float, default=0.01, help="Symplectic structure loss weight")
    train_parser.add_argument("--loss-spectral-weight", type=float, default=0.05, help="Spectral consistency loss weight")
    train_parser.add_argument("--loss-phase-weight", type=float, default=0.02, help="Phase coherence loss weight")
    train_parser.add_argument("--loss-action-weight", type=float, default=0.03, help="Action minimization loss weight")
    train_parser.add_argument("--loss-liouville-weight", type=float, default=0.01, help="Liouville theorem loss weight")
    train_parser.add_argument("--loss-hamiltonian-weight", type=float, default=0.05, help="Hamiltonian constraint loss weight")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    infer_parser.add_argument("--load-best", action="store_true", default=False, help="Load best model instead of latest checkpoint")
    infer_parser.add_argument("--output-prefix", type=str, default=None, help="Prefix for output filenames")

    return parser


def build_config_from_args(args: argparse.Namespace) -> HamiltonianAudioConfig:
    """Construct the full configuration from parsed CLI arguments."""
    audio_config = AudioProcessingConfig(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    )
    model_config = ModelArchitectureConfig(
        hidden_dimension=args.hidden_dim,
        matrix_size_height=args.matrix_height,
        matrix_size_width=args.matrix_width,
        num_spectral_layers=args.num_spectral_layers,
        kernel_expansion_factors=args.expansion_factors,
        spectral_kernel_base_height=args.spectral_kernel_base_height,
        spectral_kernel_base_width=args.spectral_kernel_base_width,
        symplectic_integration_steps=args.symplectic_steps,
    )
    checkpoint_config = CheckpointConfig(
        checkpoint_directory=args.checkpoint_dir,
        checkpoint_file_path=args.checkpoint_path,
    )
    visualization_config = VisualizationConfig(output_directory=args.output_dir)
    training_config = TrainingConfig()

    if args.mode == "train":
        training_config = TrainingConfig(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip_max_norm=args.gradient_clip,
            validation_split_ratio=args.validation_split,
            random_seed=args.seed,
            early_stopping_patience=args.early_stopping_patience,
            warmup_epochs=args.warmup_epochs,
            lr_scheduler_step_size=args.lr_step_size,
            lr_scheduler_gamma=args.lr_gamma,
            loss_reconstruction_weight=args.loss_reconstruction_weight,
            loss_energy_conservation_weight=args.loss_energy_weight,
            loss_symplectic_weight=args.loss_symplectic_weight,
            loss_spectral_consistency_weight=args.loss_spectral_weight,
            loss_phase_coherence_weight=args.loss_phase_weight,
            loss_action_minimization_weight=args.loss_action_weight,
            loss_liouville_weight=args.loss_liouville_weight,
            loss_hamiltonian_constraint_weight=args.loss_hamiltonian_weight,
        )
        checkpoint_config.checkpoint_interval_minutes = args.checkpoint_interval

    config = HamiltonianAudioConfig(
        audio=audio_config,
        model=model_config,
        training=training_config,
        checkpoint=checkpoint_config,
        visualization=visualization_config,
        device=args.device,
    )
    return config


def validate_audio_file(file_path: str) -> None:
    """Validate that the audio file exists and has a supported extension."""
    if not os.path.exists(file_path):
        print(f"[ERROR] Audio file not found: {file_path}")
        sys.exit(1)
    supported = AudioProcessingConfig().supported_extensions
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in supported:
        print(f"[ERROR] Unsupported audio format '{ext}'. Supported: {', '.join(supported)}")
        sys.exit(1)


def print_configuration_banner(config: HamiltonianAudioConfig, mode: str, audio_path: str) -> None:
    """Print a formatted configuration summary."""
    print("=" * 80)
    print(f"HAMILTONIAN AUDIO PROCESSING SYSTEM - {mode.upper()} MODE")
    print("=" * 80)
    print(f"Audio file:                {audio_path}")
    print(f"Device:                    {config.device}")
    print(f"Sample rate:               {config.audio.sample_rate} Hz")
    print(f"N-FFT:                     {config.audio.n_fft}")
    print(f"Hop length:                {config.audio.hop_length}")
    print(f"Mel bins:                  {config.audio.n_mels}")
    print(f"Hidden dimension:          {config.model.hidden_dimension}")
    print(f"Spectral layers:           {config.model.num_spectral_layers}")
    print(f"Expansion factors:         {config.model.kernel_expansion_factors}")
    print(f"Matrix size:               {config.model.matrix_size_height}x{config.model.matrix_size_width}")
    print(f"Kernel base size:          {config.model.spectral_kernel_base_height}x{config.model.spectral_kernel_base_width}")
    print(f"Symplectic steps:          {config.model.symplectic_integration_steps}")
    print(f"Checkpoint path:           {os.path.abspath(config.checkpoint.checkpoint_path)}")
    if mode == "train":
        print(f"Epochs:                    {config.training.num_epochs}")
        print(f"Batch size:                {config.training.batch_size}")
        print(f"Learning rate:             {config.training.learning_rate}")
        print(f"Weight decay:              {config.training.weight_decay}")
        print(f"Gradient clip:             {config.training.gradient_clip_max_norm}")
        print(f"Warmup epochs:             {config.training.warmup_epochs}")
        print(f"LR step / gamma:           {config.training.lr_scheduler_step_size} / {config.training.lr_scheduler_gamma}")
        print(f"Checkpoint interval:       {config.checkpoint.checkpoint_interval_minutes} min")
        print(f"Early stopping patience:   {config.training.early_stopping_patience}")
        print(f"Loss weights:")
        print(f"  Reconstruction:          {config.training.loss_reconstruction_weight}")
        print(f"  Energy conservation:     {config.training.loss_energy_conservation_weight}")
        print(f"  Symplectic:              {config.training.loss_symplectic_weight}")
        print(f"  Spectral consistency:    {config.training.loss_spectral_consistency_weight}")
        print(f"  Phase coherence:         {config.training.loss_phase_coherence_weight}")
        print(f"  Action minimization:     {config.training.loss_action_minimization_weight}")
        print(f"  Liouville:               {config.training.loss_liouville_weight}")
        print(f"  Hamiltonian constraint:   {config.training.loss_hamiltonian_constraint_weight}")
    print("=" * 80)


def run_training(args: argparse.Namespace) -> None:
    """Execute the training pipeline."""
    validate_audio_file(args.audio)
    config = build_config_from_args(args)
    config.validate_all()
    print_configuration_banner(config, "training", args.audio)
    from trainer import HamiltonianAudioTrainer
    trainer = HamiltonianAudioTrainer(config)
    trainer.train(args.audio)


def run_inference(args: argparse.Namespace) -> None:
    """Execute the inference pipeline."""
    validate_audio_file(args.audio)
    config = build_config_from_args(args)
    print_configuration_banner(config, "inference", args.audio)
    from inference import HamiltonianAudioInference
    engine = HamiltonianAudioInference(config, load_best=args.load_best)
    engine.analyze_audio(args.audio, output_prefix=args.output_prefix)


def main() -> None:
    """Main entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.mode == "train":
        run_training(args)
    elif args.mode == "infer":
        run_inference(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
