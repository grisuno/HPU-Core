# Symbols

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
| `update_metrics` | method | `audio/experiment2.py:895` | `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat, poynti` |
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
| `render_complete_analysis` | method | `audio/visualization.py:43` | `def render_complete_analysis(self, amplitude_map, phase_map, action_map, original_spectrogram, reconstructed_spectrogram` |
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
| `update_metrics` | method | `experiment.py:594` | `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat, poynti` |
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
| `update_metrics` | method | `experiment2.py:895` | `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat, poynti` |
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
| `IDiscretizationDialAnalyzer` | class | `hamiltonian_mbl.py:185` | `class IDiscretizationDialAnalyzer(Protocol)` |
| `ILevelSpacingCalculator` | class | `hamiltonian_mbl.py:167` | `class ILevelSpacingCalculator(Protocol)` |
| `IModel` | class | `hamiltonian_mbl.py:160` | `class IModel(Protocol)` |
| `IParticipationRatioCalculator` | class | `hamiltonian_mbl.py:173` | `class IParticipationRatioCalculator(Protocol)` |
| `ISyntheticPlanckCalculator` | class | `hamiltonian_mbl.py:179` | `class ISyntheticPlanckCalculator(Protocol)` |
| `ITrainingMetricsCollector` | class | `hamiltonian_mbl.py:199` | `class ITrainingMetricsCollector(Protocol)` |
| `KrylovComplexityCalculator` | class | `hamiltonian_mbl.py:1048` | `class KrylovComplexityCalculator` |
| `LevelSpacingRatioCalculator` | class | `hamiltonian_mbl.py:608` | `class LevelSpacingRatioCalculator` |
| `MBLAnalysisConfig` | class | `hamiltonian_mbl.py:67` | `class MBLAnalysisConfig` |
| `MBLCheckpointManager` | class | `hamiltonian_mbl.py:1310` | `class MBLCheckpointManager` |
| `ParticipationRatioCalculator` | class | `hamiltonian_mbl.py:719` | `class ParticipationRatioCalculator` |
| `PhaseClassifier` | class | `hamiltonian_mbl.py:1243` | `class PhaseClassifier` |
| `PurityIndexCalculator` | class | `hamiltonian_mbl.py:947` | `class PurityIndexCalculator` |
| `ResilienceSpectrometer` | class | `hamiltonian_mbl.py:1144` | `class ResilienceSpectrometer` |
| `SpectralHamiltonianLayer` | class | `hamiltonian_mbl.py:346` | `class SpectralHamiltonianLayer(Module)` |
| `SyntheticPlanckConstantCalculator` | class | `hamiltonian_mbl.py:801` | `class SyntheticPlanckConstantCalculator` |
| `TrainingConfig` | class | `hamiltonian_mbl.py:141` | `class TrainingConfig` |
| `__init__` | method | `hamiltonian_mbl.py:214` | `def __init__(self, source_config, target_config)` |
| `__init__` | method | `hamiltonian_mbl.py:352` | `def __init__(self, config)` |
| `__init__` | method | `hamiltonian_mbl.py:418` | `def __init__(self, config)` |
| `__init__` | method | `hamiltonian_mbl.py:561` | `def __init__(self, grid_size, num_samples, device)` |
| `__init__` | method | `hamiltonian_mbl.py:616` | `def __init__(self, config)` |
| `__init__` | method | `hamiltonian_mbl.py:725` | `def __init__(self, config)` |
| `__init__` | method | `hamiltonian_mbl.py:807` | `def __init__(self, config)` |
| `__init__` | method | `hamiltonian_mbl.py:849` | `def __init__(self, config)` |
| `__init__` | method | `hamiltonian_mbl.py:950` | `def __init__(self, config)` |
| `__init__` | method | `hamiltonian_mbl.py:1007` | `def __init__(self, config)` |
| `__init__` | method | `hamiltonian_mbl.py:1054` | `def __init__(self, config)` |
| `__init__` | method | `hamiltonian_mbl.py:1095` | `def __init__(self, config)` |
| `__init__` | method | `hamiltonian_mbl.py:1150` | `def __init__(self, config)` |
| `__init__` | method | `hamiltonian_mbl.py:1246` | `def __init__(self, config)` |
| `__init__` | method | `hamiltonian_mbl.py:1273` | `def __init__(self, arch_config)` |
| `__init__` | method | `hamiltonian_mbl.py:1315` | `def __init__(self, config, arch_config)` |
| `__init__` | method | `hamiltonian_mbl.py:1392` | `def __init__(self, config)` |
| `__init__` | method | `hamiltonian_mbl.py:1545` | `def __init__(self, model, arch_config, mbl_config, train_config)` |
| `__init__` | method | `hamiltonian_mbl.py:1713` | `def __init__(self, checkpoint_path, arch_config, mbl_config)` |
| `__init__` | method | `hamiltonian_mbl.py:1878` | `def __init__(self, arch_config, mbl_config)` |
| `_aggregate_by_dimension` | method | `hamiltonian_mbl.py:1220` | `def _aggregate_by_dimension(self, results)` |
| `_aggregate_by_noise` | method | `hamiltonian_mbl.py:1231` | `def _aggregate_by_noise(self, results)` |
| `_assess_purity_quality` | method | `hamiltonian_mbl.py:993` | `def _assess_purity_quality(self, alpha, variance)` |
| `_calculate_fractal_dimension` | method | `hamiltonian_mbl.py:793` | `def _calculate_fractal_dimension(self, ipr, n)` |
| `_calculate_ipr` | method | `hamiltonian_mbl.py:771` | `def _calculate_ipr(self, coefficients)` |
| `_calculate_renyi_ipr` | method | `hamiltonian_mbl.py:782` | `def _calculate_renyi_ipr(self, coefficients, q)` |
| `_calculate_spacing_ratios` | method | `hamiltonian_mbl.py:671` | `def _calculate_spacing_ratios(self, spacings)` |
| `_classify_phase` | method | `hamiltonian_mbl.py:684` | `def _classify_phase(self, mean_ratio)` |
| `_classify_quantum_phase` | method | `hamiltonian_mbl.py:1520` | `def _classify_quantum_phase(self, level_spacing, hbar_results)` |
| `_compute_eigenvalues` | method | `hamiltonian_mbl.py:666` | `def _compute_eigenvalues(self, hessian)` |
| `_compute_layer_purity` | method | `hamiltonian_mbl.py:982` | `def _compute_layer_purity(self, weights)` |
| `_construct_hessian_from_weights` | method | `hamiltonian_mbl.py:651` | `def _construct_hessian_from_weights(self, model)` |
| `_create_default_parameter` | method | `hamiltonian_mbl.py:320` | `def _create_default_parameter(self, key)` |
| `_delta_to_alpha` | method | `hamiltonian_mbl.py:940` | `def _delta_to_alpha(self, delta)` |
| `_delta_to_alpha` | method | `hamiltonian_mbl.py:988` | `def _delta_to_alpha(self, delta)` |
| `_estimate_brody_parameter` | method | `hamiltonian_mbl.py:699` | `def _estimate_brody_parameter(self, ratios)` |
| `_generate_summary` | method | `hamiltonian_mbl.py:1787` | `def _generate_summary(self, metrics)` |
| `_generate_text_report` | method | `hamiltonian_mbl.py:1982` | `def _generate_text_report(self, summary, output_dir)` |
| `_initialize_spectral_parameters` | method | `hamiltonian_mbl.py:366` | `def _initialize_spectral_parameters(self)` |
| `_initialize_weights` | method | `hamiltonian_mbl.py:438` | `def _initialize_weights(self)` |
| `_load_checkpoint` | method | `hamiltonian_mbl.py:1723` | `def _load_checkpoint(self)` |
| `_log_metrics` | method | `hamiltonian_mbl.py:1658` | `def _log_metrics(self, metrics)` |
| `_measure_base_performance` | method | `hamiltonian_mbl.py:1176` | `def _measure_base_performance(self, model)` |
| `_migrate_if_needed` | method | `hamiltonian_mbl.py:1290` | `def _migrate_if_needed(self, state_dict, device)` |
| `_perturb_and_measure` | method | `hamiltonian_mbl.py:921` | `def _perturb_and_measure(self, model, noise_level)` |
| `_print_report` | method | `hamiltonian_mbl.py:1807` | `def _print_report(self, results)` |
| `_test_perturbation` | method | `hamiltonian_mbl.py:1195` | `def _test_perturbation(self, model, dimension, noise_level)` |
| `analyze` | method | `hamiltonian_mbl.py:1763` | `def analyze(self)` |
| `analyze_robustness` | method | `hamiltonian_mbl.py:187` | `def analyze_robustness(self, model, noise_levels)` |
| `analyze_robustness` | method | `hamiltonian_mbl.py:877` | `def analyze_robustness(self, model, noise_levels)` |
| `calculate` | method | `hamiltonian_mbl.py:169` | `def calculate(self, model)` |
| `calculate` | method | `hamiltonian_mbl.py:175` | `def calculate(self, model)` |
| `calculate` | method | `hamiltonian_mbl.py:181` | `def calculate(self, participation_ratio, energy_gap)` |
| `calculate` | method | `hamiltonian_mbl.py:619` | `def calculate(self, model)` |
| `calculate` | method | `hamiltonian_mbl.py:728` | `def calculate(self, model)` |
| `calculate` | method | `hamiltonian_mbl.py:810` | `def calculate(self, participation_ratio, energy_gap)` |
| `calculate` | method | `hamiltonian_mbl.py:953` | `def calculate(self, model)` |
| `calculate` | method | `hamiltonian_mbl.py:1010` | `def calculate(self, loss_history)` |
| `calculate` | method | `hamiltonian_mbl.py:1057` | `def calculate(self, model)` |
| `calculate` | method | `hamiltonian_mbl.py:1098` | `def calculate(self, model)` |
| `calculate_base_discretization` | method | `hamiltonian_mbl.py:853` | `def calculate_base_discretization(self, model)` |
| `calculate_from_model` | method | `hamiltonian_mbl.py:820` | `def calculate_from_model(self, model, level_spacing_results, pr_results)` |
| `classify` | method | `hamiltonian_mbl.py:1249` | `def classify(self, alpha, temperature)` |
| `collect` | method | `hamiltonian_mbl.py:201` | `def collect(self, model, loss, epoch, loss_history)` |
| `collect` | method | `hamiltonian_mbl.py:1405` | `def collect(self, model, loss, epoch, loss_history, step)` |
| `collect_comprehensive` | method | `hamiltonian_mbl.py:1493` | `def collect_comprehensive(self, model, loss, epoch, loss_history, step)` |
| `construct_hessian_approximation` | method | `hamiltonian_mbl.py:503` | `def construct_hessian_approximation(self, max_dim, method)` |
| `forward` | method | `hamiltonian_mbl.py:163` | `def forward(self)` |
| `forward` | method | `hamiltonian_mbl.py:373` | `def forward(self, q, p, dt)` |
| `forward` | method | `hamiltonian_mbl.py:443` | `def forward(self, q, p, dt)` |
| `generate_double_well` | method | `hamiltonian_mbl.py:591` | `def generate_double_well(self, barrier_height)` |
| `generate_harmonic_oscillator` | method | `hamiltonian_mbl.py:567` | `def generate_harmonic_oscillator(self, omega)` |
| `generate_summary` | method | `hamiltonian_mbl.py:1941` | `def generate_summary(self, all_results, output_dir)` |
| `get_coefficients` | method | `hamiltonian_mbl.py:162` | `def get_coefficients(self)` |
| `get_coefficients` | method | `hamiltonian_mbl.py:486` | `def get_coefficients(self)` |
| `get_flat_parameters` | method | `hamiltonian_mbl.py:496` | `def get_flat_parameters(self)` |
| `get_hamiltonian` | method | `hamiltonian_mbl.py:401` | `def get_hamiltonian(self, q, p)` |
| `get_hamiltonian` | method | `hamiltonian_mbl.py:474` | `def get_hamiltonian(self, q, p)` |
| `get_input_dim` | method | `hamiltonian_mbl.py:55` | `def get_input_dim(self)` |
| `get_reduced_dimension` | method | `hamiltonian_mbl.py:135` | `def get_reduced_dimension(self)` |
| `get_total_parameters` | method | `hamiltonian_mbl.py:59` | `def get_total_parameters(self)` |
| `load_checkpoint` | method | `hamiltonian_mbl.py:195` | `def load_checkpoint(self, path)` |
| `load_checkpoint` | method | `hamiltonian_mbl.py:1361` | `def load_checkpoint(self, path)` |
| `main` | method | `hamiltonian_mbl.py:2031` | `def main()` |
| `measure` | method | `hamiltonian_mbl.py:1153` | `def measure(self, model)` |
| `migrate` | method | `hamiltonian_mbl.py:1277` | `def migrate(self, raw_data, device)` |
| `migrate_state_dict` | method | `hamiltonian_mbl.py:218` | `def migrate_state_dict(self, source_state)` |
| `process_checkpoint` | method | `hamiltonian_mbl.py:1882` | `def process_checkpoint(self, checkpoint_path, output_dir)` |
| `process_directory` | method | `hamiltonian_mbl.py:1901` | `def process_directory(self, checkpoint_dir, n_latest, output_dir)` |
| `save_checkpoint` | method | `hamiltonian_mbl.py:193` | `def save_checkpoint(self, model, epoch, metrics, loss_history, path)` |
| `save_checkpoint` | method | `hamiltonian_mbl.py:1328` | `def save_checkpoint(self, model, epoch, metrics, loss_history, checkpoint_dir)` |
| `should_save_checkpoint` | method | `hamiltonian_mbl.py:1322` | `def should_save_checkpoint(self)` |
| `time_evolution` | method | `hamiltonian_mbl.py:459` | `def time_evolution(self, q_initial, p_initial, num_steps, dt)` |
| `train` | method | `hamiltonian_mbl.py:1672` | `def train(self, dataset, num_epochs)` |
| `train_epoch` | method | `hamiltonian_mbl.py:1603` | `def train_epoch(self, dataset, epoch)` |
| `train_step` | method | `hamiltonian_mbl.py:1571` | `def train_step(self, q_batch, p_batch, q_target, p_target)` |
| `BoltzmannAnalysisProgram` | class | `mining_seeds.py:879` | `class BoltzmannAnalysisProgram` |
| `CheckpointManager` | class | `mining_seeds.py:500` | `class CheckpointManager` |
| `Config` | class | `mining_seeds.py:41` | `class Config` |
| `CrystallographyMetrics` | class | `mining_seeds.py:340` | `class CrystallographyMetrics` |
| `FastDataset` | class | `mining_seeds.py:146` | `class FastDataset(Dataset)` |
| `GlassStopper` | class | `mining_seeds.py:611` | `class GlassStopper` |
| `HamiltonianOperator` | class | `mining_seeds.py:121` | `class HamiltonianOperator` |
| `IAnalysisStrategy` | class | `mining_seeds.py:109` | `class IAnalysisStrategy(ABC)` |
| `IMetricsCalculator` | class | `mining_seeds.py:115` | `class IMetricsCalculator(ABC)` |
| `LocalComplexityAnalyzer` | class | `mining_seeds.py:293` | `class LocalComplexityAnalyzer` |
| `SimpleHamiltonianNet` | class | `mining_seeds.py:255` | `class SimpleHamiltonianNet(Module)` |
| `SpectralLayer` | class | `mining_seeds.py:203` | `class SpectralLayer(Module)` |
| `SpectroscopyMetrics` | class | `mining_seeds.py:470` | `class SpectroscopyMetrics` |
| `SuperpositionAnalyzer` | class | `mining_seeds.py:310` | `class SuperpositionAnalyzer` |
| `ThermodynamicMetrics` | class | `mining_seeds.py:444` | `class ThermodynamicMetrics` |
| `TrainingMonitor` | class | `mining_seeds.py:573` | `class TrainingMonitor` |
| `__getitem__` | method | `mining_seeds.py:196` | `def __getitem__(self, idx)` |
| `__init__` | method | `mining_seeds.py:124` | `def __init__(self, grid_size)` |
| `__init__` | method | `mining_seeds.py:149` | `def __init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)` |
| `__init__` | method | `mining_seeds.py:206` | `def __init__(self, channels, grid_size)` |
| `__init__` | method | `mining_seeds.py:258` | `def __init__(self, grid_size, hidden_dim, num_spectral_layers)` |
| `__init__` | method | `mining_seeds.py:501` | `def __init__(self, interval_minutes, max_checkpoints)` |
| `__init__` | method | `mining_seeds.py:574` | `def __init__(self)` |
| `__init__` | method | `mining_seeds.py:612` | `def __init__(self, patience_epochs)` |
| `__init__` | method | `mining_seeds.py:880` | `def __init__(self, checkpoint_path, results_dir)` |
| `__len__` | method | `mining_seeds.py:193` | `def __len__(self)` |
| `_compute_spectral_entropy` | method | `mining_seeds.py:491` | `def _compute_spectral_entropy(power_spectrum)` |
| `_precompute_spectral_operators` | method | `mining_seeds.py:128` | `def _precompute_spectral_operators(self)` |
| `analyze` | method | `mining_seeds.py:111` | `def analyze(self, model)` |
| `apply` | method | `mining_seeds.py:135` | `def apply(self, field)` |
| `compute` | method | `mining_seeds.py:117` | `def compute(self, model)` |
| `compute_all_metrics` | method | `mining_seeds.py:426` | `def compute_all_metrics(model, dataloader)` |
| `compute_alpha_purity` | method | `mining_seeds.py:387` | `def compute_alpha_purity(coeffs)` |
| `compute_discretization_margin` | method | `mining_seeds.py:378` | `def compute_discretization_margin(coeffs)` |
| `compute_effective_temperature` | method | `mining_seeds.py:446` | `def compute_effective_temperature(gradient_buffer, learning_rate)` |
| `compute_kappa` | method | `mining_seeds.py:342` | `def compute_kappa(model, dataloader, num_batches)` |
| `compute_kappa_quantum` | method | `mining_seeds.py:394` | `def compute_kappa_quantum(coeffs, hbar)` |
| `compute_local_complexity` | method | `mining_seeds.py:295` | `def compute_local_complexity(weights, epsilon)` |
| `compute_poynting_vector` | method | `mining_seeds.py:411` | `def compute_poynting_vector(coeffs)` |
| `compute_specific_heat` | method | `mining_seeds.py:460` | `def compute_specific_heat(loss_history, temp_history, cv_threshold)` |
| `compute_superposition` | method | `mining_seeds.py:312` | `def compute_superposition(weights)` |
| `compute_weight_diffraction` | method | `mining_seeds.py:472` | `def compute_weight_diffraction(coeffs)` |
| `dataloader` | method | `mining_seeds.py:903` | `def dataloader()` |
| `forward` | method | `mining_seeds.py:219` | `def forward(self, x)` |
| `forward` | method | `mining_seeds.py:279` | `def forward(self, x)` |
| `get_val_batch` | method | `mining_seeds.py:199` | `def get_val_batch(self)` |
| `load_and_analyze_checkpoint` | method | `mining_seeds.py:886` | `def load_and_analyze_checkpoint(self)` |
| `main` | method | `mining_seeds.py:856` | `def main()` |
| `save_checkpoint` | method | `mining_seeds.py:514` | `def save_checkpoint(self, model, optimizer, epoch, metrics)` |
| `seed_miner` | method | `mining_seeds.py:803` | `def seed_miner(total_attempts)` |
| `set_seed` | method | `mining_seeds.py:88` | `def set_seed(seed)` |
| `setup_logger` | method | `mining_seeds.py:96` | `def setup_logger(name, level)` |
| `should_save_checkpoint` | method | `mining_seeds.py:509` | `def should_save_checkpoint(self)` |
| `should_stop` | method | `mining_seeds.py:616` | `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)` |
| `time_evolution` | method | `mining_seeds.py:140` | `def time_evolution(self, field, dt)` |
| `train_with_early_glass_stop` | method | `mining_seeds.py:670` | `def train_with_early_glass_stop(model, optimizer, seed, epochs)` |
| `update_metrics` | method | `mining_seeds.py:594` | `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat, poynti` |
| `HBarCalculator` | class | `plank.py:26` | `class HBarCalculator` |
| `__init__` | method | `plank.py:29` | `def __init__(self, checkpoint_path, device)` |
| `calculate_all` | method | `plank.py:54` | `def calculate_all(self)` |
| `main` | method | `plank.py:213` | `def main()` |
| `print_report` | method | `plank.py:170` | `def print_report(self, results)` |
| `ControlConfig` | class | `polos.py:27` | `class ControlConfig` |
| `ControlSystemAnalyzer` | class | `polos.py:599` | `class ControlSystemAnalyzer` |
| `ControlVisualizer` | class | `polos.py:798` | `class ControlVisualizer` |
| `ControllerDesigner` | class | `polos.py:516` | `class ControllerDesigner` |
| `FrequencyResponseAnalyzer` | class | `polos.py:299` | `class FrequencyResponseAnalyzer` |
| `PoleZeroAnalyzer` | class | `polos.py:142` | `class PoleZeroAnalyzer` |
| `TimeResponseAnalyzer` | class | `polos.py:415` | `class TimeResponseAnalyzer` |
| `TransferFunctionExtractor` | class | `polos.py:48` | `class TransferFunctionExtractor` |
| `__init__` | method | `polos.py:50` | `def __init__(self, model, device)` |
| `__init__` | method | `polos.py:144` | `def __init__(self, numerator, denominator)` |
| `__init__` | method | `polos.py:301` | `def __init__(self, numerator, denominator)` |
| `__init__` | method | `polos.py:417` | `def __init__(self, numerator, denominator)` |
| `__init__` | method | `polos.py:518` | `def __init__(self, poles, zeros)` |
| `__init__` | method | `polos.py:601` | `def __init__(self, checkpoint_path, device)` |
| `_compute_poles_zeros` | method | `polos.py:153` | `def _compute_poles_zeros(self)` |
| `_print_report` | method | `polos.py:715` | `def _print_report(self, results)` |
| `analyze_checkpoint` | method | `polos.py:1152` | `def analyze_checkpoint(checkpoint_path, output_dir)` |
| `analyze_complete_system` | method | `polos.py:625` | `def analyze_complete_system(self)` |
| `analyze_multiple_checkpoints` | method | `polos.py:1208` | `def analyze_multiple_checkpoints(checkpoint_dir, n_latest, output_dir)` |
| `analyze_stability` | method | `polos.py:166` | `def analyze_stability(self)` |
| `analyze_step_response_characteristics` | method | `polos.py:457` | `def analyze_step_response_characteristics(self, step_data)` |
| `classify_poles` | method | `polos.py:207` | `def classify_poles(self)` |
| `compute_bode_plot_data` | method | `polos.py:309` | `def compute_bode_plot_data(self)` |
| `compute_damping_frequency` | method | `polos.py:238` | `def compute_damping_frequency(self)` |
| `compute_gain_phase_margins` | method | `polos.py:328` | `def compute_gain_phase_margins(self)` |
| `compute_impulse_response` | method | `polos.py:441` | `def compute_impulse_response(self)` |
| `compute_nyquist_data` | method | `polos.py:357` | `def compute_nyquist_data(self)` |
| `compute_root_locus` | method | `polos.py:572` | `def compute_root_locus(self, num, den)` |
| `compute_step_response` | method | `polos.py:425` | `def compute_step_response(self)` |
| `compute_time_constants` | method | `polos.py:278` | `def compute_time_constants(self)` |
| `compute_transfer_function` | method | `polos.py:105` | `def compute_transfer_function(self, A, B, C, D)` |
| `design_lead_compensator` | method | `polos.py:542` | `def design_lead_compensator(self, desired_phase_margin)` |
| `design_pid_controller` | method | `polos.py:522` | `def design_pid_controller(self, desired_damping, desired_settling_time)` |
| `evaluate_nyquist_stability` | method | `polos.py:379` | `def evaluate_nyquist_stability(self, nyquist_data)` |
| `extract_state_space_representation` | method | `polos.py:55` | `def extract_state_space_representation(self)` |
| `main` | method | `polos.py:1240` | `def main()` |
| `plot_bode_diagram` | method | `polos.py:865` | `def plot_bode_diagram(bode_data, margins, output_path)` |
| `plot_combined_analysis` | method | `polos.py:1096` | `def plot_combined_analysis(poles, zeros, bode_data, step_data, output_path)` |
| `plot_nyquist_diagram` | method | `polos.py:940` | `def plot_nyquist_diagram(nyquist_data, output_path)` |
| `plot_pole_zero_map` | method | `polos.py:801` | `def plot_pole_zero_map(poles, zeros, output_path)` |
| `plot_root_locus` | method | `polos.py:1036` | `def plot_root_locus(root_locus_data, output_path)` |
| `plot_time_responses` | method | `polos.py:990` | `def plot_time_responses(step_data, impulse_data, output_path)` |
| `ContinuationEngine` | class | `precision.py:72` | `class ContinuationEngine` |
| `CrystallizationLossMassive` | class | `precision.py:36` | `class CrystallizationLossMassive(Module)` |
| `MassiveLambdaConfig` | class | `precision.py:26` | `class MassiveLambdaConfig` |
| `__init__` | method | `precision.py:37` | `def __init__(self, lambda_quant)` |
| `__init__` | method | `precision.py:73` | `def __init__(self, checkpoint_path, device)` |
| `_compile_results` | method | `precision.py:490` | `def _compile_results(self, success, final_epoch)` |
| `_compute_initial_metrics` | method | `precision.py:191` | `def _compute_initial_metrics(self, model)` |
| `_find_latest_checkpoint` | method | `precision.py:162` | `def _find_latest_checkpoint(self)` |
| `_save_crystal_checkpoint` | method | `precision.py:456` | `def _save_crystal_checkpoint(self, epoch, metrics, val_acc, final, force_save, emergency)` |
| `_save_latest_checkpoint` | method | `precision.py:430` | `def _save_latest_checkpoint(self, epoch, metrics, val_acc)` |
| `_setup_logger` | method | `precision.py:150` | `def _setup_logger(self)` |
| `compute_discretization_metrics` | method | `precision.py:208` | `def compute_discretization_metrics(self)` |
| `forward` | method | `precision.py:54` | `def forward(self, predictions, targets, model)` |
| `main` | method | `precision.py:506` | `def main()` |
| `quantization_penalty` | method | `precision.py:42` | `def quantization_penalty(self, model)` |
| `refine` | method | `precision.py:288` | `def refine(self)` |
| `train_epoch` | method | `precision.py:250` | `def train_epoch(self, epoch)` |
| `validate` | method | `precision.py:241` | `def validate(self)` |
| `CrystallizationConfig` | class | `refinamiento.py:33` | `class CrystallizationConfig` |
| `CrystallizationEngine` | class | `refinamiento.py:144` | `class CrystallizationEngine` |
| `CrystallizationLoss` | class | `refinamiento.py:57` | `class CrystallizationLoss(Module)` |
| `StructuralPruner` | class | `refinamiento.py:96` | `class StructuralPruner` |
| `__init__` | method | `refinamiento.py:62` | `def __init__(self, lambda_quant)` |
| `__init__` | method | `refinamiento.py:98` | `def __init__(self, thresholds)` |
| `__init__` | method | `refinamiento.py:148` | `def __init__(self, checkpoint_path, device)` |
| `_compile_results` | method | `refinamiento.py:482` | `def _compile_results(self, success, final_epoch)` |
| `_compute_initial_metrics` | method | `refinamiento.py:234` | `def _compute_initial_metrics(self, model)` |
| `_load_checkpoint` | method | `refinamiento.py:198` | `def _load_checkpoint(self)` |
| `_save_crystal_checkpoint` | method | `refinamiento.py:459` | `def _save_crystal_checkpoint(self, epoch, metrics, val_acc, final)` |
| `_setup_logger` | method | `refinamiento.py:186` | `def _setup_logger(self)` |
| `analyze_discretization` | method | `refinamiento.py:498` | `def analyze_discretization(checkpoint_path)` |
| `compute_discretization_metrics` | method | `refinamiento.py:253` | `def compute_discretization_metrics(self)` |
| `forward` | method | `refinamiento.py:81` | `def forward(self, predictions, targets, model)` |
| `get_sparsity` | method | `refinamiento.py:131` | `def get_sparsity(self, model)` |
| `main` | method | `refinamiento.py:575` | `def main()` |
| `prune` | method | `refinamiento.py:107` | `def prune(self, model, force_threshold)` |
| `quantization_penalty` | method | `refinamiento.py:67` | `def quantization_penalty(self, model)` |
| `refine` | method | `refinamiento.py:344` | `def refine(self)` |
| `should_prune` | method | `refinamiento.py:103` | `def should_prune(self, epoch)` |
| `train_epoch` | method | `refinamiento.py:302` | `def train_epoch(self, epoch)` |
| `validate` | method | `refinamiento.py:291` | `def validate(self)` |
| `GrokkingValidator` | class | `test_grokkit.py:41` | `class GrokkingValidator` |
| `__init__` | method | `test_grokkit.py:56` | `def __init__(self, weights_dir)` |
| `compute_local_complexity` | method | `test_grokkit.py:158` | `def compute_local_complexity(self, model)` |
| `compute_operator_error` | method | `test_grokkit.py:203` | `def compute_operator_error(self, model, inputs, targets)` |
| `compute_spectral_gap` | method | `test_grokkit.py:228` | `def compute_spectral_gap(self, model)` |
| `compute_superposition` | method | `test_grokkit.py:181` | `def compute_superposition(self, model)` |
| `generate_report` | method | `test_grokkit.py:382` | `def generate_report(self)` |
| `generate_test_dataset` | method | `test_grokkit.py:119` | `def generate_test_dataset(self, num_samples)` |
| `load_model` | method | `test_grokkit.py:64` | `def load_model(self)` |
| `run_quick_test` | method | `test_grokkit.py:507` | `def run_quick_test()` |
| `run_validation` | method | `test_grokkit.py:259` | `def run_validation(self)` |
| `CheckpointVerifier` | class | `verify.py:14` | `class CheckpointVerifier` |
| `__init__` | method | `verify.py:15` | `def __init__(self, checkpoint_path, device)` |
| `_check_internal_consistency` | method | `verify.py:302` | `def _check_internal_consistency(self, results)` |
| `_check_weight_integrity` | method | `verify.py:101` | `def _check_weight_integrity(self)` |
| `_compare_with_stored` | method | `verify.py:269` | `def _compare_with_stored(self, computed)` |
| `_compute_discretization_metrics` | method | `verify.py:173` | `def _compute_discretization_metrics(self)` |
| `_compute_health_score` | method | `verify.py:331` | `def _compute_health_score(self, results)` |
| `_compute_loss_metrics` | method | `verify.py:244` | `def _compute_loss_metrics(self)` |
| `_compute_quantization_metrics` | method | `verify.py:225` | `def _compute_quantization_metrics(self)` |
| `_compute_validation_metrics` | method | `verify.py:146` | `def _compute_validation_metrics(self)` |
| `_print_report` | method | `verify.py:374` | `def _print_report(self, results)` |
| `main` | method | `verify.py:486` | `def main()` |
| `verify_all_metrics` | method | `verify.py:50` | `def verify_all_metrics(self)` |
| `verify_latest_checkpoints` | method | `verify.py:444` | `def verify_latest_checkpoints(checkpoint_dir, n)` |
