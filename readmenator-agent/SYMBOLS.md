# Symbols (page 1 of 2)
Pages: [SYMBOLS.md](SYMBOLS.md), [SYMBOLS_p2.md](SYMBOLS_p2.md)

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
| `FastDataset` | class | `app.py:113` | `class FastDataset(Dataset)` |
| `HamiltonianOperator` | class | `app.py:88` | `class HamiltonianOperator` |
| `SimpleConfig` | class | `app.py:29` | `class SimpleConfig` |
| `SimpleHamiltonianNet` | class | `app.py:222` | `class SimpleHamiltonianNet(Module)` |
| `SpectralLayer` | class | `app.py:170` | `class SpectralLayer(Module)` |
| `__getitem__` | method | `app.py:163` | `def __getitem__(self, idx)` |
| `__init__` | method | `app.py:30` | `def __init__(self, grid_size, hidden_dim, num_spectral_layers, target_accuracy, learning_rate)` |
| `__init__` | method | `app.py:91` | `def __init__(self, grid_size)` |
| `__init__` | method | `app.py:116` | `def __init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)` |
| `__init__` | method | `app.py:173` | `def __init__(self, channels, grid_size)` |
| `__init__` | method | `app.py:225` | `def __init__(self, grid_size, hidden_dim, num_spectral_layers)` |
| `__len__` | method | `app.py:160` | `def __len__(self)` |
| `_precompute_spectral_operators` | method | `app.py:95` | `def _precompute_spectral_operators(self)` |
| `apply` | method | `app.py:102` | `def apply(self, field)` |
| `compute_local_complexity` | method | `app.py:45` | `def compute_local_complexity(weights, epsilon)` |
| `compute_superposition` | method | `app.py:60` | `def compute_superposition(weights)` |
| `forward` | method | `app.py:186` | `def forward(self, x)` |
| `forward` | method | `app.py:246` | `def forward(self, x)` |
| `get_val_batch` | method | `app.py:166` | `def get_val_batch(self)` |
| `main` | method | `app.py:369` | `def main()` |
| `time_evolution` | method | `app.py:107` | `def time_evolution(self, field, dt)` |
| `train_model` | method | `app.py:260` | `def train_model(grid_size, epochs, hidden_dim, num_spectral_layers, lr)` |
| `AudioProcessor` | class | `audio/audio_io.py:31` | `class AudioProcessor` |
| `__init__` | method | `audio/audio_io.py:41` | `def __init__(self, config, device)` |
| `get_spectrogram_db_range` | method | `audio/audio_io.py:249` | `def get_spectrogram_db_range(self, waveform)` |
| `load_audio` | method | `audio/audio_io.py:57` | `def load_audio(self, file_path)` |
| `magnitude_phase_to_stft` | method | `audio/audio_io.py:146` | `def magnitude_phase_to_stft(self, magnitude, phase)` |
| `model_output_to_stft_magnitude` | method | `audio/audio_io.py:186` | `def model_output_to_stft_magnitude(self, model_output, original_magnitude)` |
| `save_audio` | method | `audio/audio_io.py:229` | `def save_audio(self, waveform, file_path, sample_rate)` |
| `stft_complex_to_waveform` | method | `audio/audio_io.py:105` | `def stft_complex_to_waveform(self, stft_complex)` |
| `stft_magnitude_to_model_input` | method | `audio/audio_io.py:161` | `def stft_magnitude_to_model_input(self, magnitude)` |
| `stft_to_magnitude_phase` | method | `audio/audio_io.py:130` | `def stft_to_magnitude_phase(self, stft_complex)` |
| `waveform_to_mel_spectrogram` | method | `audio/audio_io.py:206` | `def waveform_to_mel_spectrogram(self, waveform)` |
| `waveform_to_stft_complex` | method | `audio/audio_io.py:80` | `def waveform_to_stft_complex(self, waveform)` |
| `AudioResampler` | class | `audio/audios.py:149` | `class AudioResampler` |
| `AudioSpectrogramConverter` | class | `audio/audios.py:387` | `class AudioSpectrogramConverter` |
| `CheckpointManager` | class | `audio/audios.py:326` | `class CheckpointManager` |
| `ComprehensiveMetricCollector` | class | `audio/audios.py:278` | `class ComprehensiveMetricCollector(IMetricCollector)` |
| `HamiltonianAudioProcessor` | class | `audio/audios.py:468` | `class HamiltonianAudioProcessor` |
| `HamiltonianConfig` | class | `audio/audios.py:48` | `class HamiltonianConfig` |
| `IAudioSource` | class | `audio/audios.py:103` | `class IAudioSource(ABC)` |
| `IFieldOperator` | class | `audio/audios.py:122` | `class IFieldOperator(ABC)` |
| `IMetricCollector` | class | `audio/audios.py:131` | `class IMetricCollector(ABC)` |
| `WaveFileSource` | class | `audio/audios.py:204` | `class WaveFileSource(IAudioSource)` |
| `__init__` | method | `audio/audios.py:210` | `def __init__(self, file_path, config)` |
| `__init__` | method | `audio/audios.py:284` | `def __init__(self, config)` |
| `__init__` | method | `audio/audios.py:331` | `def __init__(self, model, config, checkpoint_dir)` |
| `__init__` | method | `audio/audios.py:393` | `def __init__(self, config)` |
| `__init__` | method | `audio/audios.py:475` | `def __init__(self, config, model, source)` |
| `_calculate_phase_entropy` | method | `audio/audios.py:684` | `def _calculate_phase_entropy(self, phase_map)` |
| `_forward_spectrogram` | method | `audio/audios.py:458` | `def _forward_spectrogram(self, x)` |
| `_inverse_spectrogram` | method | `audio/audios.py:463` | `def _inverse_spectrogram(self, spectrogram)` |
| `_process_single_segment` | method | `audio/audios.py:592` | `def _process_single_segment(self, waveform, index)` |
| `_render_epiphenomena` | method | `audio/audios.py:692` | `def _render_epiphenomena(self, amplitude, phase, action)` |
| `_save_checkpoint` | method | `audio/audios.py:357` | `def _save_checkpoint(self)` |
| `_validate_and_load` | method | `audio/audios.py:220` | `def _validate_and_load(self)` |
| `attach_source` | method | `audio/audios.py:513` | `def attach_source(self, source)` |
| `check_and_save` | method | `audio/audios.py:344` | `def check_and_save(self, force)` |
| `close` | method | `audio/audios.py:117` | `def close(self)` |
| `close` | method | `audio/audios.py:273` | `def close(self)` |
| `evolve` | method | `audio/audios.py:126` | `def evolve(self, field_state)` |
| `export_metrics` | method | `audio/audios.py:729` | `def export_metrics(self, path)` |
| `export_to_json` | method | `audio/audios.py:320` | `def export_to_json(self, path)` |
| `field_to_waveform` | method | `audio/audios.py:432` | `def field_to_waveform(self, field, original_length)` |
| `force_checkpoint` | method | `audio/audios.py:733` | `def force_checkpoint(self)` |
| `freq_bins` | method | `audio/audios.py:94` | `def freq_bins(self)` |
| `get_properties` | method | `audio/audios.py:112` | `def get_properties(self)` |
| `get_properties` | method | `audio/audios.py:261` | `def get_properties(self)` |
| `get_summary` | method | `audio/audios.py:140` | `def get_summary(self)` |
| `get_summary` | method | `audio/audios.py:298` | `def get_summary(self)` |
| `load_model_weights` | method | `audio/audios.py:504` | `def load_model_weights(self, path)` |
| `load_wav_with_resample` | method | `audio/audios.py:173` | `def load_wav_with_resample(file_path, target_sr)` |
| `main` | method | `audio/audios.py:742` | `def main()` |
| `process_stream` | method | `audio/audios.py:517` | `def process_stream(self)` |
| `read_segment` | method | `audio/audios.py:107` | `def read_segment(self)` |
| `read_segment` | method | `audio/audios.py:244` | `def read_segment(self)` |
| `record` | method | `audio/audios.py:135` | `def record(self, metrics)` |
| `record` | method | `audio/audios.py:289` | `def record(self, metrics)` |
| `resample` | method | `audio/audios.py:155` | `def resample(audio, orig_sr, target_sr)` |
| `segment_samples` | method | `audio/audios.py:89` | `def segment_samples(self)` |
| `waveform_to_field` | method | `audio/audios.py:396` | `def waveform_to_field(self, waveform)` |
| `CheckpointManager` | class | `audio/checkpoint_manager.py:28` | `class CheckpointManager` |
| `__init__` | method | `audio/checkpoint_manager.py:34` | `def __init__(self, config)` |
| `best_loss` | method | `audio/checkpoint_manager.py:156` | `def best_loss(self)` |
| `load_checkpoint` | method | `audio/checkpoint_manager.py:98` | `def load_checkpoint(self, model, load_best)` |
| `save_checkpoint` | method | `audio/checkpoint_manager.py:46` | `def save_checkpoint(self, model, optimizer, scheduler, epoch, step, metrics, current_loss)` |
| `should_save_checkpoint` | method | `audio/checkpoint_manager.py:41` | `def should_save_checkpoint(self)` |
| `AudioProcessingConfig` | class | `audio/config.py:18` | `class AudioProcessingConfig` |
| `CheckpointConfig` | class | `audio/config.py:109` | `class CheckpointConfig` |
| `HamiltonianAudioConfig` | class | `audio/config.py:182` | `class HamiltonianAudioConfig` |
| `MetricsConfig` | class | `audio/config.py:158` | `class MetricsConfig` |
| `ModelArchitectureConfig` | class | `audio/config.py:34` | `class ModelArchitectureConfig` |
| `TrainingConfig` | class | `audio/config.py:80` | `class TrainingConfig` |
| `VisualizationConfig` | class | `audio/config.py:136` | `class VisualizationConfig` |
| `best_model_path` | method | `audio/config.py:127` | `def best_model_path(self)` |
| `checkpoint_path` | method | `audio/config.py:121` | `def checkpoint_path(self)` |
| `ensure_directories` | method | `audio/config.py:206` | `def ensure_directories(self)` |
| `metadata_path` | method | `audio/config.py:131` | `def metadata_path(self)` |
| `validate` | method | `audio/config.py:62` | `def validate(self)` |
| `validate_all` | method | `audio/config.py:198` | `def validate_all(self)` |
| `Application` | class | `audio/experiment2.py:1275` | `class Application` |
| `CheckpointAnalyzer` | class | `audio/experiment2.py:1223` | `class CheckpointAnalyzer` |
| `CheckpointManager` | class | `audio/experiment2.py:804` | `class CheckpointManager` |
| `Config` | class | `audio/experiment2.py:22` | `class Config` |
| `CrystallographyMetricsCalculator` | class | `audio/experiment2.py:302` | `class CrystallographyMetricsCalculator(IMetricsCalculator)` |
| `GlassStateDetector` | class | `audio/experiment2.py:912` | `class GlassStateDetector` |
| `HamiltonianDataset` | class | `audio/experiment2.py:133` | `class HamiltonianDataset(Dataset)` |
| `HamiltonianNeuralNetwork` | class | `audio/experiment2.py:227` | `class HamiltonianNeuralNetwork(Module)` |
| `HamiltonianOperator` | class | `audio/experiment2.py:111` | `class HamiltonianOperator` |
| `IAnalysisStrategy` | class | `audio/experiment2.py:99` | `class IAnalysisStrategy(ABC)` |
| `IMetricsCalculator` | class | `audio/experiment2.py:105` | `class IMetricsCalculator(ABC)` |
| `LocalComplexityAnalyzer` | class | `audio/experiment2.py:257` | `class LocalComplexityAnalyzer` |
| `LoggerFactory` | class | `audio/experiment2.py:84` | `class LoggerFactory` |
| `SeedManager` | class | `audio/experiment2.py:74` | `class SeedManager` |
| `SeedMiningSystem` | class | `audio/experiment2.py:1113` | `class SeedMiningSystem` |
| `SingleExperimentRunner` | class | `audio/experiment2.py:1164` | `class SingleExperimentRunner` |
| `SpectralLayer` | class | `audio/experiment2.py:183` | `class SpectralLayer(Module)` |
| `SpectroscopyMetricsCalculator` | class | `audio/experiment2.py:770` | `class SpectroscopyMetricsCalculator(IMetricsCalculator)` |
| `SuperpositionAnalyzer` | class | `audio/experiment2.py:273` | `class SuperpositionAnalyzer` |
| `ThermodynamicMetricsCalculator` | class | `audio/experiment2.py:737` | `class ThermodynamicMetricsCalculator(IMetricsCalculator)` |
| `TrainingEngine` | class | `audio/experiment2.py:973` | `class TrainingEngine` |
| `TrainingMetricsMonitor` | class | `audio/experiment2.py:874` | `class TrainingMetricsMonitor` |
| `__getitem__` | method | `audio/experiment2.py:176` | `def __getitem__(self, idx)` |
| `__init__` | method | `audio/experiment2.py:112` | `def __init__(self, grid_size)` |
| `__init__` | method | `audio/experiment2.py:134` | `def __init__(self, num_samples, grid_size, time_steps, dt, train_ratio)` |
| `__init__` | method | `audio/experiment2.py:184` | `def __init__(self, channels, grid_size)` |
| `__init__` | method | `audio/experiment2.py:228` | `def __init__(self, grid_size, hidden_dim, num_spectral_layers)` |
| `__init__` | method | `audio/experiment2.py:805` | `def __init__(self, interval_minutes, max_checkpoints)` |
| `__init__` | method | `audio/experiment2.py:875` | `def __init__(self)` |
| `__init__` | method | `audio/experiment2.py:913` | `def __init__(self, patience_epochs)` |
| `__init__` | method | `audio/experiment2.py:974` | `def __init__(self, model, optimizer, device, logger)` |
| `__init__` | method | `audio/experiment2.py:1114` | `def __init__(self, max_attempts)` |
| `__init__` | method | `audio/experiment2.py:1165` | `def __init__(self, seed, epochs, grid_size, hidden_dim, num_spectral_layers, learning_rate)` |
| `__init__` | method | `audio/experiment2.py:1224` | `def __init__(self, checkpoint_path, results_dir)` |
| `__init__` | method | `audio/experiment2.py:1276` | `def __init__(self)` |
| `__len__` | method | `audio/experiment2.py:173` | `def __len__(self)` |
| `_check_weight_integrity` | method | `audio/experiment2.py:539` | `def _check_weight_integrity(self, model)` |
| `_compute_crystallography_metrics` | method | `audio/experiment2.py:511` | `def _compute_crystallography_metrics(self, model, val_x, val_y)` |
| `_compute_spectral_entropy` | method | `audio/experiment2.py:795` | `def _compute_spectral_entropy(power_spectrum)` |
| `_create_argument_parser` | method | `audio/experiment2.py:1280` | `def _create_argument_parser(self)` |
| `_precompute_spectral_operators` | method | `audio/experiment2.py:116` | `def _precompute_spectral_operators(self)` |
| `analyze` | method | `audio/experiment2.py:101` | `def analyze(self, model)` |
| `analyze` | method | `audio/experiment2.py:1230` | `def analyze(self)` |
| `apply` | method | `audio/experiment2.py:122` | `def apply(self, field)` |
| `compute` | method | `audio/experiment2.py:107` | `def compute(self, model)` |
| `compute` | method | `audio/experiment2.py:303` | `def compute(self, model, val_x, val_y)` |
| `compute` | method | `audio/experiment2.py:738` | `def compute(self, model, gradient_buffer, learning_rate, loss_history, temp_history)` |
| `compute` | method | `audio/experiment2.py:771` | `def compute(self, model)` |
| `compute_all_metrics` | method | `audio/experiment2.py:679` | `def compute_all_metrics(model, val_x, val_y)` |
| `compute_alpha_purity` | method | `audio/experiment2.py:383` | `def compute_alpha_purity(coeffs)` |
| `compute_alpha_purity_from_model` | method | `audio/experiment2.py:373` | `def compute_alpha_purity_from_model(model)` |
| `compute_discretization_margin` | method | `audio/experiment2.py:361` | `def compute_discretization_margin(coeffs)` |
| `compute_discretization_margin_from_state_dict` | method | `audio/experiment2.py:348` | `def compute_discretization_margin_from_state_dict(model)` |
| `compute_effective_temperature` | method | `audio/experiment2.py:747` | `def compute_effective_temperature(gradient_buffer, learning_rate)` |
| `compute_gradient_covariance_kappa` | method | `audio/experiment2.py:311` | `def compute_gradient_covariance_kappa(model, dataloader, num_batches)` |
| `compute_kappa` | method | `audio/experiment2.py:393` | `def compute_kappa(model, val_x, val_y, num_batches)` |
| `compute_kappa_quantum` | method | `audio/experiment2.py:464` | `def compute_kappa_quantum(model, hbar)` |
| `compute_kappa_quantum_from_coeffs` | method | `audio/experiment2.py:492` | `def compute_kappa_quantum_from_coeffs(coeffs, hbar)` |
| `compute_local_complexity` | method | `audio/experiment2.py:259` | `def compute_local_complexity(weights, epsilon)` |
| `compute_poynting_vector` | method | `audio/experiment2.py:603` | `def compute_poynting_vector(model)` |
| `compute_specific_heat` | method | `audio/experiment2.py:760` | `def compute_specific_heat(loss_history, temp_history, cv_threshold)` |
| `compute_superposition` | method | `audio/experiment2.py:275` | `def compute_superposition(weights)` |
| `compute_weight_diffraction` | method | `audio/experiment2.py:776` | `def compute_weight_diffraction(coeffs)` |
| `compute_weight_metrics` | method | `audio/experiment2.py:1040` | `def compute_weight_metrics(self)` |
| `create_logger` | method | `audio/experiment2.py:86` | `def create_logger(name, level)` |
| `execute_training` | method | `audio/experiment2.py:1056` | `def execute_training(self, dataloader, val_x, val_y, epochs, seed, early_stopping)` |
| `forward` | method | `audio/experiment2.py:195` | `def forward(self, x)` |
| `forward` | method | `audio/experiment2.py:243` | `def forward(self, x)` |
| `get_validation_batch` | method | `audio/experiment2.py:179` | `def get_validation_batch(self)` |
| `is_crystal_formed` | method | `audio/experiment2.py:963` | `def is_crystal_formed(self, lc, sp, kappa, delta, temp, cv)` |
| `main` | method | `audio/experiment2.py:1334` | `def main()` |
| `mine` | method | `audio/experiment2.py:1118` | `def mine(self)` |
| `run` | method | `audio/experiment2.py:1175` | `def run(self)` |
| `run` | method | `audio/experiment2.py:1294` | `def run(self)` |
| `safe_compute` | method | `audio/experiment2.py:694` | `def safe_compute(func)` |
| `save_checkpoint` | method | `audio/experiment2.py:818` | `def save_checkpoint(self, model, optimizer, epoch, metrics)` |
| `set_seed` | method | `audio/experiment2.py:76` | `def set_seed(seed)` |
| `should_save_checkpoint` | method | `audio/experiment2.py:813` | `def should_save_checkpoint(self)` |
| `should_stop` | method | `audio/experiment2.py:918` | `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)` |
| `time_evolution` | method | `audio/experiment2.py:127` | `def time_evolution(self, field, dt)` |
| `train_epoch` | method | `audio/experiment2.py:997` | `def train_epoch(self, dataloader, epoch)` |
| `update_metrics` | method | `audio/experiment2.py:895` | `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat...` |
| `validate` | method | `audio/experiment2.py:1027` | `def validate(self, val_x, val_y)` |
| `HamiltonianAudioInference` | class | `audio/inference.py:37` | `class HamiltonianAudioInference` |
| `__init__` | method | `audio/inference.py:42` | `def __init__(self, config, load_best)` |
| `_compute_energy_mask_patched` | method | `audio/inference.py:158` | `def _compute_energy_mask_patched(self, model_input)` |
| `_compute_inference_metrics` | method | `audio/inference.py:270` | `def _compute_inference_metrics(self, original_magnitude, reconstructed_magnitude, original_stft)` |
| `_extract_hamiltonian_fields_patched` | method | `audio/inference.py:209` | `def _extract_hamiltonian_fields_patched(self, model_input)` |
| `_print_inference_metrics` | method | `audio/inference.py:288` | `def _print_inference_metrics(self)` |
| `analyze_audio` | method | `audio/inference.py:73` | `def analyze_audio(self, audio_file_path, output_prefix)` |
| `HamiltonianLossComputer` | class | `audio/losses.py:28` | `class HamiltonianLossComputer` |
| `__init__` | method | `audio/losses.py:37` | `def __init__(self, config)` |
| `_compute_action_minimization_loss` | method | `audio/losses.py:177` | `def _compute_action_minimization_loss(self, intermediates)` |
| `_compute_energy_conservation_loss` | method | `audio/losses.py:100` | `def _compute_energy_conservation_loss(self, intermediates)` |
| `_compute_hamiltonian_constraint_loss` | method | `audio/losses.py:214` | `def _compute_hamiltonian_constraint_loss(self, intermediates)` |
| `_compute_liouville_loss` | method | `audio/losses.py:192` | `def _compute_liouville_loss(self, intermediates)` |
| `_compute_phase_coherence_loss` | method | `audio/losses.py:161` | `def _compute_phase_coherence_loss(self, prediction, target)` |
| `_compute_reconstruction_loss` | method | `audio/losses.py:94` | `def _compute_reconstruction_loss(self, prediction, target)` |
| `_compute_spectral_consistency_loss` | method | `audio/losses.py:145` | `def _compute_spectral_consistency_loss(self, prediction, target)` |
| `_compute_symplectic_loss` | method | `audio/losses.py:122` | `def _compute_symplectic_loss(self, intermediates)` |
| `compute_total_loss` | method | `audio/losses.py:41` | `def compute_total_loss(self, prediction, target, intermediates, model)` |
| `build_argument_parser` | function | `audio/main.py:34` | `def build_argument_parser()` |
| `build_config_from_args` | function | `audio/main.py:103` | `def build_config_from_args(args)` |
| `main` | function | `audio/main.py:236` | `def main()` |
| `print_configuration_banner` | function | `audio/main.py:175` | `def print_configuration_banner(config, mode, audio_path)` |
| `run_inference` | function | `audio/main.py:226` | `def run_inference(args)` |
| `run_training` | function | `audio/main.py:215` | `def run_training(args)` |
| `validate_audio_file` | function | `audio/main.py:163` | `def validate_audio_file(file_path)` |
| `HamiltonianMetricsTracker` | class | `audio/metrics.py:26` | `class HamiltonianMetricsTracker` |
| `__init__` | method | `audio/metrics.py:36` | `def __init__(self, config)` |
| `_initialize_history_buffers` | method | `audio/metrics.py:43` | `def _initialize_history_buffers(self)` |
| `_record` | method | `audio/metrics.py:379` | `def _record(self, metric_name, value)` |
| `compute_action_integral` | method | `audio/metrics.py:181` | `def compute_action_integral(self, q_trajectory, p_trajectory, dt)` |
| `compute_energy_drift` | method | `audio/metrics.py:328` | `def compute_energy_drift(self, energy_initial, energy_current)` |
| `compute_hamiltonian_energy` | method | `audio/metrics.py:74` | `def compute_hamiltonian_energy(self, q, p)` |
| `compute_liouville_measure` | method | `audio/metrics.py:125` | `def compute_liouville_measure(self, jacobian)` |
| `compute_phase_coherence` | method | `audio/metrics.py:305` | `def compute_phase_coherence(self, phase_original, phase_reconstructed)` |
| `compute_phase_space_volume` | method | `audio/metrics.py:153` | `def compute_phase_space_volume(self, q, p)` |
| `compute_poisson_bracket` | method | `audio/metrics.py:208` | `def compute_poisson_bracket(self, f_values, g_values, q, p)` |
| `compute_reconstruction_snr` | method | `audio/metrics.py:259` | `def compute_reconstruction_snr(self, original, reconstructed)` |
| `compute_spectral_convergence` | method | `audio/metrics.py:281` | `def compute_spectral_convergence(self, original_spectrum, reconstructed_spectrum)` |
| `compute_spectral_entropy` | method | `audio/metrics.py:238` | `def compute_spectral_entropy(self, spectrum)` |
| `compute_symplectic_form` | method | `audio/metrics.py:97` | `def compute_symplectic_form(self, q, p, dq, dp)` |
| `get_current_metrics` | method | `audio/metrics.py:388` | `def get_current_metrics(self)` |
| `get_formatted_metrics_string` | method | `audio/metrics.py:400` | `def get_formatted_metrics_string(self)` |
| `get_moving_averages` | method | `audio/metrics.py:392` | `def get_moving_averages(self)` |
| `increment_step` | method | `audio/metrics.py:413` | `def increment_step(self)` |
| `record_gradient_norm` | method | `audio/metrics.py:348` | `def record_gradient_norm(self, model_parameters)` |
| `record_learning_rate` | method | `audio/metrics.py:369` | `def record_learning_rate(self, lr)` |
| `record_loss_component` | method | `audio/metrics.py:374` | `def record_loss_component(self, name, value)` |
| `record_parameter_norm` | method | `audio/metrics.py:359` | `def record_parameter_norm(self, model_parameters)` |
| `should_log` | method | `audio/metrics.py:421` | `def should_log(self)` |
| `step_count` | method | `audio/metrics.py:418` | `def step_count(self)` |
| `HamiltonianNeuralNetwork` | class | `audio/model.py:162` | `class HamiltonianNeuralNetwork(Module)` |
| `SpectralEvolutionLayer` | class | `audio/model.py:27` | `class SpectralEvolutionLayer(Module)` |
| `__init__` | method | `audio/model.py:36` | `def __init__(self, hidden_dim, kernel_base_height, kernel_base_width, init_std)` |
| `__init__` | method | `audio/model.py:172` | `def __init__(self, config)` |
| `compute_energy_mask` | method | `audio/model.py:270` | `def compute_energy_mask(self, x)` |
| `evolve_complex` | method | `audio/model.py:85` | `def evolve_complex(self, x, target_height, target_width)` |
| `evolve_real` | method | `audio/model.py:124` | `def evolve_real(self, x, target_height, target_width)` |
| `extract_hamiltonian_fields` | method | `audio/model.py:237` | `def extract_hamiltonian_fields(self, x)` |
| `forward` | method | `audio/model.py:51` | `def forward(self, x)` |
| `forward` | method | `audio/model.py:200` | `def forward(self, x)` |
| `forward_with_intermediates` | method | `audio/model.py:216` | `def forward_with_intermediates(self, x)` |
| `AudioSpectrogramDatasetBuilder` | class | `audio/trainer.py:38` | `class AudioSpectrogramDatasetBuilder` |
| `HamiltonianAudioTrainer` | class | `audio/trainer.py:79` | `class HamiltonianAudioTrainer` |
| `__init__` | method | `audio/trainer.py:46` | `def __init__(self, config)` |
| `__init__` | method | `audio/trainer.py:92` | `def __init__(self, config)` |
| `_attempt_checkpoint_recovery` | method | `audio/trainer.py:129` | `def _attempt_checkpoint_recovery(self)` |
| `_train_one_epoch` | method | `audio/trainer.py:255` | `def _train_one_epoch(self, train_loader, epoch)` |
| `_validate` | method | `audio/trainer.py:344` | `def _validate(self, val_loader, epoch)` |
| `audio_processor` | method | `audio/trainer.py:381` | `def audio_processor(self)` |
| `build_dataset` | method | `audio/trainer.py:49` | `def build_dataset(self, mel_spectrogram)` |
| `model` | method | `audio/trainer.py:377` | `def model(self)` |
| `train` | method | `audio/trainer.py:151` | `def train(self, audio_file_path)` |
| `HamiltonianAudioVisualizer` | class | `audio/visualization.py:29` | `class HamiltonianAudioVisualizer` |
| `__init__` | method | `audio/visualization.py:34` | `def __init__(self, vis_config, audio_config)` |
| `_render_energy_landscape` | method | `audio/visualization.py:218` | `def _render_energy_landscape(self, amplitude_map, action_map, output_prefix)` |
| `_render_hamiltonian_fields` | method | `audio/visualization.py:91` | `def _render_hamiltonian_fields(self, amplitude_map, phase_map, action_map, output_prefix)` |
| `_render_phase_portrait` | method | `audio/visualization.py:185` | `def _render_phase_portrait(self, amplitude_map, phase_map, output_prefix)` |
| `_render_spectrogram_comparison` | method | `audio/visualization.py:140` | `def _render_spectrogram_comparison(self, original, reconstructed, output_prefix)` |
| `_render_waveform_comparison` | method | `audio/visualization.py:270` | `def _render_waveform_comparison(self, original_waveform, reconstructed_waveform, output_prefix)` |
| `render_complete_analysis` | method | `audio/visualization.py:43` | `def render_complete_analysis(self, amplitude_map, phase_map, action_map, original_spectrogram...` |
| `analize_checkpoint` | function | `diff_weights.py:5` | `def analize_checkpoint(path)` |
| `DiracConfig` | class | `dirac.py:25` | `class DiracConfig` |
| `DiracDeltaAnalyzer` | class | `dirac.py:38` | `class DiracDeltaAnalyzer` |
| `DiracVisualizer` | class | `dirac.py:328` | `class DiracVisualizer` |
| `__init__` | method | `dirac.py:40` | `def __init__(self, checkpoint_path, device)` |
| `_print_report` | method | `dirac.py:279` | `def _print_report(self, results)` |
| `analyze_all` | method | `dirac.py:223` | `def analyze_all(self)` |
| `analyze_checkpoint` | method | `dirac.py:574` | `def analyze_checkpoint(checkpoint_path, output_dir)` |
| `analyze_multiple_checkpoints` | method | `dirac.py:620` | `def analyze_multiple_checkpoints(checkpoint_dir, n_latest, output_dir)` |
| `compute_dirac_delta_approximation` | method | `dirac.py:77` | `def compute_dirac_delta_approximation(self, charge_density)` |
| `compute_divergence` | method | `dirac.py:192` | `def compute_divergence(self, electric_field)` |
| `compute_electric_field` | method | `dirac.py:112` | `def compute_electric_field(self, dirac_data, eval_points)` |
| `compute_electric_flux` | method | `dirac.py:157` | `def compute_electric_flux(self, electric_field, surface_points)` |
| `extract_charge_distribution` | method | `dirac.py:64` | `def extract_charge_distribution(self)` |
| `main` | method | `dirac.py:652` | `def main()` |
| `plot_charge_distribution` | method | `dirac.py:331` | `def plot_charge_distribution(charge_density, point_positions, point_charges, output_path)` |
| `plot_combined_analysis` | method | `dirac.py:490` | `def plot_combined_analysis(charge_density, point_positions, point_charges, electric_field, divergence, output_path)` |
| `plot_divergence` | method | `dirac.py:441` | `def plot_divergence(divergence, output_path)` |
| `plot_electric_field` | method | `dirac.py:379` | `def plot_electric_field(electric_field, output_path)` |
| `verify_gauss_law` | method | `dirac.py:200` | `def verify_gauss_law(self, dirac_data, flux_data)` |
| `evaluate_model` | function | `expand.py:74` | `def evaluate_model(model, resolution, device)` |
| `expand_model` | function | `expand.py:43` | `def expand_model(model, target_resolution, source_resolution)` |
| `expand_spectral_weights` | function | `expand.py:23` | `def expand_spectral_weights(kernel_real, kernel_imag, target_size, source_size)` |
| `load_config` | function | `expand.py:18` | `def load_config(toml_path)` |
| `main` | function | `expand.py:105` | `def main()` |
| `BoltzmannAnalysisProgram` | class | `experiment.py:879` | `class BoltzmannAnalysisProgram` |
| `CheckpointManager` | class | `experiment.py:500` | `class CheckpointManager` |
| `Config` | class | `experiment.py:41` | `class Config` |
| `CrystallographyMetrics` | class | `experiment.py:340` | `class CrystallographyMetrics` |
| `FastDataset` | class | `experiment.py:146` | `class FastDataset(Dataset)` |
| `GlassStopper` | class | `experiment.py:611` | `class GlassStopper` |
| `HamiltonianOperator` | class | `experiment.py:121` | `class HamiltonianOperator` |
| `IAnalysisStrategy` | class | `experiment.py:109` | `class IAnalysisStrategy(ABC)` |
| `IMetricsCalculator` | class | `experiment.py:115` | `class IMetricsCalculator(ABC)` |
| `LocalComplexityAnalyzer` | class | `experiment.py:293` | `class LocalComplexityAnalyzer` |
| `SimpleHamiltonianNet` | class | `experiment.py:255` | `class SimpleHamiltonianNet(Module)` |
| `SpectralLayer` | class | `experiment.py:203` | `class SpectralLayer(Module)` |
| `SpectroscopyMetrics` | class | `experiment.py:470` | `class SpectroscopyMetrics` |
| `SuperpositionAnalyzer` | class | `experiment.py:310` | `class SuperpositionAnalyzer` |
| `ThermodynamicMetrics` | class | `experiment.py:444` | `class ThermodynamicMetrics` |
| `TrainingMonitor` | class | `experiment.py:573` | `class TrainingMonitor` |
| `__getitem__` | method | `experiment.py:196` | `def __getitem__(self, idx)` |
| `__init__` | method | `experiment.py:124` | `def __init__(self, grid_size)` |
| `__init__` | method | `experiment.py:149` | `def __init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)` |
| `__init__` | method | `experiment.py:206` | `def __init__(self, channels, grid_size)` |
| `__init__` | method | `experiment.py:258` | `def __init__(self, grid_size, hidden_dim, num_spectral_layers)` |
| `__init__` | method | `experiment.py:501` | `def __init__(self, interval_minutes, max_checkpoints)` |
| `__init__` | method | `experiment.py:574` | `def __init__(self)` |
| `__init__` | method | `experiment.py:612` | `def __init__(self, patience_epochs)` |
| `__init__` | method | `experiment.py:880` | `def __init__(self, checkpoint_path, results_dir)` |
| `__len__` | method | `experiment.py:193` | `def __len__(self)` |
| `_compute_spectral_entropy` | method | `experiment.py:491` | `def _compute_spectral_entropy(power_spectrum)` |
| `_precompute_spectral_operators` | method | `experiment.py:128` | `def _precompute_spectral_operators(self)` |
| `analyze` | method | `experiment.py:111` | `def analyze(self, model)` |
| `apply` | method | `experiment.py:135` | `def apply(self, field)` |
| `compute` | method | `experiment.py:117` | `def compute(self, model)` |
| `compute_all_metrics` | method | `experiment.py:426` | `def compute_all_metrics(model, dataloader)` |
| `compute_alpha_purity` | method | `experiment.py:387` | `def compute_alpha_purity(coeffs)` |
| `compute_discretization_margin` | method | `experiment.py:378` | `def compute_discretization_margin(coeffs)` |
| `compute_effective_temperature` | method | `experiment.py:446` | `def compute_effective_temperature(gradient_buffer, learning_rate)` |
| `compute_kappa` | method | `experiment.py:342` | `def compute_kappa(model, dataloader, num_batches)` |
| `compute_kappa_quantum` | method | `experiment.py:394` | `def compute_kappa_quantum(coeffs, hbar)` |
| `compute_local_complexity` | method | `experiment.py:295` | `def compute_local_complexity(weights, epsilon)` |
| `compute_poynting_vector` | method | `experiment.py:411` | `def compute_poynting_vector(coeffs)` |
| `compute_specific_heat` | method | `experiment.py:460` | `def compute_specific_heat(loss_history, temp_history, cv_threshold)` |
| `compute_superposition` | method | `experiment.py:312` | `def compute_superposition(weights)` |
| `compute_weight_diffraction` | method | `experiment.py:472` | `def compute_weight_diffraction(coeffs)` |
| `dataloader` | method | `experiment.py:903` | `def dataloader()` |
| `forward` | method | `experiment.py:219` | `def forward(self, x)` |
| `forward` | method | `experiment.py:279` | `def forward(self, x)` |
| `get_val_batch` | method | `experiment.py:199` | `def get_val_batch(self)` |
| `load_and_analyze_checkpoint` | method | `experiment.py:886` | `def load_and_analyze_checkpoint(self)` |
| `main` | method | `experiment.py:856` | `def main()` |
| `save_checkpoint` | method | `experiment.py:514` | `def save_checkpoint(self, model, optimizer, epoch, metrics)` |
| `seed_miner` | method | `experiment.py:803` | `def seed_miner(total_attempts)` |
| `set_seed` | method | `experiment.py:88` | `def set_seed(seed)` |
| `setup_logger` | method | `experiment.py:96` | `def setup_logger(name, level)` |
| `should_save_checkpoint` | method | `experiment.py:509` | `def should_save_checkpoint(self)` |
| `should_stop` | method | `experiment.py:616` | `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)` |
| `time_evolution` | method | `experiment.py:140` | `def time_evolution(self, field, dt)` |
| `train_with_early_glass_stop` | method | `experiment.py:670` | `def train_with_early_glass_stop(model, optimizer, seed, epochs)` |
| `update_metrics` | method | `experiment.py:594` | `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat...` |
| `Application` | class | `experiment2.py:1275` | `class Application` |
| `CheckpointAnalyzer` | class | `experiment2.py:1223` | `class CheckpointAnalyzer` |
| `CheckpointManager` | class | `experiment2.py:804` | `class CheckpointManager` |
| `Config` | class | `experiment2.py:22` | `class Config` |
| `CrystallographyMetricsCalculator` | class | `experiment2.py:302` | `class CrystallographyMetricsCalculator(IMetricsCalculator)` |
| `GlassStateDetector` | class | `experiment2.py:912` | `class GlassStateDetector` |
| `HamiltonianDataset` | class | `experiment2.py:133` | `class HamiltonianDataset(Dataset)` |
| `HamiltonianNeuralNetwork` | class | `experiment2.py:227` | `class HamiltonianNeuralNetwork(Module)` |
| `HamiltonianOperator` | class | `experiment2.py:111` | `class HamiltonianOperator` |
| `IAnalysisStrategy` | class | `experiment2.py:99` | `class IAnalysisStrategy(ABC)` |
| `IMetricsCalculator` | class | `experiment2.py:105` | `class IMetricsCalculator(ABC)` |
| `LocalComplexityAnalyzer` | class | `experiment2.py:257` | `class LocalComplexityAnalyzer` |
| `LoggerFactory` | class | `experiment2.py:84` | `class LoggerFactory` |
| `SeedManager` | class | `experiment2.py:74` | `class SeedManager` |
| `SeedMiningSystem` | class | `experiment2.py:1113` | `class SeedMiningSystem` |
| `SingleExperimentRunner` | class | `experiment2.py:1164` | `class SingleExperimentRunner` |
| `SpectralLayer` | class | `experiment2.py:183` | `class SpectralLayer(Module)` |
| `SpectroscopyMetricsCalculator` | class | `experiment2.py:770` | `class SpectroscopyMetricsCalculator(IMetricsCalculator)` |
| `SuperpositionAnalyzer` | class | `experiment2.py:273` | `class SuperpositionAnalyzer` |
| `ThermodynamicMetricsCalculator` | class | `experiment2.py:737` | `class ThermodynamicMetricsCalculator(IMetricsCalculator)` |
| `TrainingEngine` | class | `experiment2.py:973` | `class TrainingEngine` |
| `TrainingMetricsMonitor` | class | `experiment2.py:874` | `class TrainingMetricsMonitor` |
| `__getitem__` | method | `experiment2.py:176` | `def __getitem__(self, idx)` |
| `__init__` | method | `experiment2.py:112` | `def __init__(self, grid_size)` |
| `__init__` | method | `experiment2.py:134` | `def __init__(self, num_samples, grid_size, time_steps, dt, train_ratio)` |
| `__init__` | method | `experiment2.py:184` | `def __init__(self, channels, grid_size)` |
| `__init__` | method | `experiment2.py:228` | `def __init__(self, grid_size, hidden_dim, num_spectral_layers)` |
| `__init__` | method | `experiment2.py:805` | `def __init__(self, interval_minutes, max_checkpoints)` |
| `__init__` | method | `experiment2.py:875` | `def __init__(self)` |
| `__init__` | method | `experiment2.py:913` | `def __init__(self, patience_epochs)` |
| `__init__` | method | `experiment2.py:974` | `def __init__(self, model, optimizer, device, logger)` |
| `__init__` | method | `experiment2.py:1114` | `def __init__(self, max_attempts)` |
| `__init__` | method | `experiment2.py:1165` | `def __init__(self, seed, epochs, grid_size, hidden_dim, num_spectral_layers, learning_rate)` |
| `__init__` | method | `experiment2.py:1224` | `def __init__(self, checkpoint_path, results_dir)` |
| `__init__` | method | `experiment2.py:1276` | `def __init__(self)` |
| `__len__` | method | `experiment2.py:173` | `def __len__(self)` |
| `_check_weight_integrity` | method | `experiment2.py:539` | `def _check_weight_integrity(self, model)` |
| `_compute_crystallography_metrics` | method | `experiment2.py:511` | `def _compute_crystallography_metrics(self, model, val_x, val_y)` |
| `_compute_spectral_entropy` | method | `experiment2.py:795` | `def _compute_spectral_entropy(power_spectrum)` |
| `_create_argument_parser` | method | `experiment2.py:1280` | `def _create_argument_parser(self)` |
| `_precompute_spectral_operators` | method | `experiment2.py:116` | `def _precompute_spectral_operators(self)` |
| `analyze` | method | `experiment2.py:101` | `def analyze(self, model)` |
| `analyze` | method | `experiment2.py:1230` | `def analyze(self)` |
| `apply` | method | `experiment2.py:122` | `def apply(self, field)` |
| `compute` | method | `experiment2.py:107` | `def compute(self, model)` |
| `compute` | method | `experiment2.py:303` | `def compute(self, model, val_x, val_y)` |
| `compute` | method | `experiment2.py:738` | `def compute(self, model, gradient_buffer, learning_rate, loss_history, temp_history)` |
| `compute` | method | `experiment2.py:771` | `def compute(self, model)` |
| `compute_all_metrics` | method | `experiment2.py:679` | `def compute_all_metrics(model, val_x, val_y)` |
| `compute_alpha_purity` | method | `experiment2.py:383` | `def compute_alpha_purity(coeffs)` |
| `compute_alpha_purity_from_model` | method | `experiment2.py:373` | `def compute_alpha_purity_from_model(model)` |
| `compute_discretization_margin` | method | `experiment2.py:361` | `def compute_discretization_margin(coeffs)` |
| `compute_discretization_margin_from_state_dict` | method | `experiment2.py:348` | `def compute_discretization_margin_from_state_dict(model)` |
| `compute_effective_temperature` | method | `experiment2.py:747` | `def compute_effective_temperature(gradient_buffer, learning_rate)` |
| `compute_gradient_covariance_kappa` | method | `experiment2.py:311` | `def compute_gradient_covariance_kappa(model, dataloader, num_batches)` |
| `compute_kappa` | method | `experiment2.py:393` | `def compute_kappa(model, val_x, val_y, num_batches)` |
| `compute_kappa_quantum` | method | `experiment2.py:464` | `def compute_kappa_quantum(model, hbar)` |
| `compute_kappa_quantum_from_coeffs` | method | `experiment2.py:492` | `def compute_kappa_quantum_from_coeffs(coeffs, hbar)` |
| `compute_local_complexity` | method | `experiment2.py:259` | `def compute_local_complexity(weights, epsilon)` |
| `compute_poynting_vector` | method | `experiment2.py:603` | `def compute_poynting_vector(model)` |
| `compute_specific_heat` | method | `experiment2.py:760` | `def compute_specific_heat(loss_history, temp_history, cv_threshold)` |
| `compute_superposition` | method | `experiment2.py:275` | `def compute_superposition(weights)` |
| `compute_weight_diffraction` | method | `experiment2.py:776` | `def compute_weight_diffraction(coeffs)` |
| `compute_weight_metrics` | method | `experiment2.py:1040` | `def compute_weight_metrics(self)` |
| `create_logger` | method | `experiment2.py:86` | `def create_logger(name, level)` |
| `execute_training` | method | `experiment2.py:1056` | `def execute_training(self, dataloader, val_x, val_y, epochs, seed, early_stopping)` |
| `forward` | method | `experiment2.py:195` | `def forward(self, x)` |
| `forward` | method | `experiment2.py:243` | `def forward(self, x)` |
| `get_validation_batch` | method | `experiment2.py:179` | `def get_validation_batch(self)` |
| `is_crystal_formed` | method | `experiment2.py:963` | `def is_crystal_formed(self, lc, sp, kappa, delta, temp, cv)` |
| `main` | method | `experiment2.py:1334` | `def main()` |
| `mine` | method | `experiment2.py:1118` | `def mine(self)` |
| `run` | method | `experiment2.py:1175` | `def run(self)` |
| `run` | method | `experiment2.py:1294` | `def run(self)` |
| `safe_compute` | method | `experiment2.py:694` | `def safe_compute(func)` |
| `save_checkpoint` | method | `experiment2.py:818` | `def save_checkpoint(self, model, optimizer, epoch, metrics)` |
| `set_seed` | method | `experiment2.py:76` | `def set_seed(seed)` |
| `should_save_checkpoint` | method | `experiment2.py:813` | `def should_save_checkpoint(self)` |
| `should_stop` | method | `experiment2.py:918` | `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)` |
| `time_evolution` | method | `experiment2.py:127` | `def time_evolution(self, field, dt)` |
| `train_epoch` | method | `experiment2.py:997` | `def train_epoch(self, dataloader, epoch)` |
| `update_metrics` | method | `experiment2.py:895` | `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat...` |
| `validate` | method | `experiment2.py:1027` | `def validate(self, val_x, val_y)` |
| `CheckpointVerifier` | class | `get_meditions.py:740` | `class CheckpointVerifier` |
| `CrystallographyMetrics` | class | `get_meditions.py:98` | `class CrystallographyMetrics` |
| `SpectralCoefficients` | class | `get_meditions.py:105` | `class SpectralCoefficients` |
| `SpectroscopyMetrics` | class | `get_meditions.py:634` | `class SpectroscopyMetrics` |
| `ThermodynamicConfig` | class | `get_meditions.py:28` | `class ThermodynamicConfig` |
| `ThermodynamicMetrics` | class | `get_meditions.py:394` | `class ThermodynamicMetrics` |
| `ThermodynamicPotential` | class | `get_meditions.py:72` | `class ThermodynamicPotential` |
| `__init__` | method | `get_meditions.py:741` | `def __init__(self, checkpoint_path, device)` |
| `_approximate_ricci_curvature` | method | `get_meditions.py:1113` | `def _approximate_ricci_curvature(self)` |
| `_assign_crystallographic_grade` | method | `get_meditions.py:1336` | `def _assign_crystallographic_grade(self, delta, alpha)` |
| `_check_internal_consistency` | method | `get_meditions.py:1226` | `def _check_internal_consistency(self, results)` |
| `_check_weight_integrity` | method | `get_meditions.py:849` | `def _check_weight_integrity(self)` |
| `_compare_with_stored` | method | `get_meditions.py:1188` | `def _compare_with_stored(self, computed)` |
| `_compute_crystallography_metrics` | method | `get_meditions.py:1031` | `def _compute_crystallography_metrics(self)` |
| `_compute_discretization_metrics` | method | `get_meditions.py:942` | `def _compute_discretization_metrics(self)` |
| `_compute_health_score` | method | `get_meditions.py:1267` | `def _compute_health_score(self, results)` |
| `_compute_kappa_iterative` | method | `get_meditions.py:255` | `def _compute_kappa_iterative(params, hbar, n, max_iters, tol)` |
| `_compute_loss_metrics` | method | `get_meditions.py:1005` | `def _compute_loss_metrics(self)` |
| `_compute_quantization_metrics` | method | `get_meditions.py:986` | `def _compute_quantization_metrics(self)` |
| `_compute_spectral_entropy` | method | `get_meditions.py:672` | `def _compute_spectral_entropy(power_spectrum)` |
| `_compute_spectroscopy` | method | `get_meditions.py:1134` | `def _compute_spectroscopy(self)` |
| `_compute_thermodynamic_metrics` | method | `get_meditions.py:1038` | `def _compute_thermodynamic_metrics(self)` |
| `_compute_thermodynamic_potential` | method | `get_meditions.py:1164` | `def _compute_thermodynamic_potential(self, results)` |
| `_compute_validation_metrics` | method | `get_meditions.py:915` | `def _compute_validation_metrics(self)` |
| `_print_report` | method | `get_meditions.py:1349` | `def _print_report(self, results)` |
| `calculate_carnot_efficiency` | method | `get_meditions.py:608` | `def calculate_carnot_efficiency(delta_alpha, total_flops, initial_alpha)` |
| `compute_all_metrics` | method | `get_meditions.py:366` | `def compute_all_metrics(model, val_x, val_y)` |
| `compute_alpha_purity` | method | `get_meditions.py:203` | `def compute_alpha_purity(model)` |
| `compute_critical_exponents` | method | `get_meditions.py:436` | `def compute_critical_exponents(temp_history, cv_history, alpha_history)` |
| `compute_discretization_margin` | method | `get_meditions.py:187` | `def compute_discretization_margin(model)` |
| `compute_effective_temperature` | method | `get_meditions.py:401` | `def compute_effective_temperature(gradient_buffer, learning_rate)` |
| `compute_equation_of_state` | method | `get_meditions.py:504` | `def compute_equation_of_state(temp_eff, alpha, kappa)` |
| `compute_fisher_information_matrix` | method | `get_meditions.py:572` | `def compute_fisher_information_matrix(model, samples)` |
| `compute_gibbs_free_energy` | method | `get_meditions.py:732` | `def compute_gibbs_free_energy(loss, temp, entropy)` |
| `compute_kappa` | method | `get_meditions.py:127` | `def compute_kappa(model, val_x, val_y, num_batches)` |
| `compute_kappa_quantum` | method | `get_meditions.py:226` | `def compute_kappa_quantum(model, hbar)` |
| `compute_local_complexity` | method | `get_meditions.py:214` | `def compute_local_complexity(model)` |
| `compute_mutual_information` | method | `get_meditions.py:539` | `def compute_mutual_information(weights, gradients)` |
| `compute_poynting_vector` | method | `get_meditions.py:312` | `def compute_poynting_vector(model)` |
| `compute_ricci_curvature` | method | `get_meditions.py:593` | `def compute_ricci_curvature(fisher_matrix)` |
| `compute_specific_heat` | method | `get_meditions.py:418` | `def compute_specific_heat(loss_history, temp_history, cv_threshold)` |
| `compute_weight_diffraction` | method | `get_meditions.py:640` | `def compute_weight_diffraction(model)` |
| `estimate_hbar_algorithmic` | method | `get_meditions.py:561` | `def estimate_hbar_algorithmic(model_complexity, weight_dim, mutual_information)` |
| `extract_lattice_parameters` | method | `get_meditions.py:680` | `def extract_lattice_parameters(weight_tensor, rank)` |
| `from_model` | method | `get_meditions.py:112` | `def from_model(cls, model)` |
| `gibbs_free_energy` | method | `get_meditions.py:85` | `def gibbs_free_energy(self)` |
| `helmholtz_free_energy` | method | `get_meditions.py:81` | `def helmholtz_free_energy(self)` |
| `is_stable` | method | `get_meditions.py:90` | `def is_stable(self)` |
| `main` | method | `get_meditions.py:1500` | `def main()` |
| `setup_logger` | method | `get_meditions.py:51` | `def setup_logger(name, level)` |
| `verify_all_metrics` | method | `get_meditions.py:782` | `def verify_all_metrics(self)` |
| `verify_latest_checkpoints` | method | `get_meditions.py:1439` | `def verify_latest_checkpoints(checkpoint_dir, n)` |
| `ArchitectureMigrator` | class | `hamiltonian_mbl.py:209` | `class ArchitectureMigrator` |
| `CheckpointMigrator` | class | `hamiltonian_mbl.py:1270` | `class CheckpointMigrator` |
| `CrystallinityIndexCalculator` | class | `hamiltonian_mbl.py:1089` | `class CrystallinityIndexCalculator` |
| `DiscretizationDialAnalyzer` | class | `hamiltonian_mbl.py:844` | `class DiscretizationDialAnalyzer` |
| `EffectiveTemperatureCalculator` | class | `hamiltonian_mbl.py:1004` | `class EffectiveTemperatureCalculator` |
| `HamiltonianArchitectureConfig` | class | `hamiltonian_mbl.py:37` | `class HamiltonianArchitectureConfig` |
| `HamiltonianCheckpointAnalyzer` | class | `hamiltonian_mbl.py:1710` | `class HamiltonianCheckpointAnalyzer` |
| `HamiltonianDataset` | class | `hamiltonian_mbl.py:555` | `class HamiltonianDataset` |
| `HamiltonianMBLMetricsCollector` | class | `hamiltonian_mbl.py:1386` | `class HamiltonianMBLMetricsCollector` |
| `HamiltonianMBLPipeline` | class | `hamiltonian_mbl.py:1875` | `class HamiltonianMBLPipeline` |
| `HamiltonianNeuralNetwork` | class | `hamiltonian_mbl.py:412` | `class HamiltonianNeuralNetwork(Module)` |
| `HamiltonianTrainer` | class | `hamiltonian_mbl.py:1540` | `class HamiltonianTrainer` |
| `ICheckpointManager` | class | `hamiltonian_mbl.py:191` | `class ICheckpointManager(Protocol)` |

Next: [SYMBOLS_p2.md](SYMBOLS_p2.md)
