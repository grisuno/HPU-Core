# API (page 1 of 2)
Pages: [API.md](API.md), [API_p2.md](API_p2.md)

## app.py
- `SimpleConfig.__init__` (method) `app.py:30` `def __init__(self, grid_size, hidden_dim, num_spectral_layers, target_accuracy, learning_rate)`
- `SimpleConfig.compute_local_complexity` (method) `app.py:45` `def compute_local_complexity(weights, epsilon)` -- Compute Local Complexity (LC) metric for weight matrix.
- `SimpleConfig.compute_superposition` (method) `app.py:60` `def compute_superposition(weights)` -- Compute Superposition (SP) metric for weight matrix.
- `HamiltonianOperator.__init__` (method) `app.py:91` `def __init__(self, grid_size)`
- `HamiltonianOperator.apply` (method) `app.py:102` `def apply(self, field)`
- `HamiltonianOperator.time_evolution` (method) `app.py:107` `def time_evolution(self, field, dt)`
- `FastDataset.__init__` (method) `app.py:116` `def __init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)`
- `FastDataset.get_val_batch` (method) `app.py:166` `def get_val_batch(self)`
- `SpectralLayer.__init__` (method) `app.py:173` `def __init__(self, channels, grid_size)`
- `SpectralLayer.forward` (method) `app.py:186` `def forward(self, x)`
- `SimpleHamiltonianNet.__init__` (method) `app.py:225` `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `SimpleHamiltonianNet.forward` (method) `app.py:246` `def forward(self, x)`
- `SimpleHamiltonianNet.train_model` (method) `app.py:260` `def train_model(grid_size, epochs, hidden_dim, num_spectral_layers, lr)` -- Train the Hamiltonian operator model.
- `SimpleHamiltonianNet.main` (method) `app.py:369` `def main()`

## audio/audio_io.py
Depends on: `audio/config.py`
Imported by: `audio/inference.py`, `audio/trainer.py`
- `AudioProcessor.__init__` (method) `audio/audio_io.py:41` `def __init__(self, config, device)`
- `AudioProcessor.load_audio` (method) `audio/audio_io.py:57` `def load_audio(self, file_path)` -- Load an audio file and convert to mono at the target sample rate.
- `AudioProcessor.waveform_to_stft_complex` (method) `audio/audio_io.py:80` `def waveform_to_stft_complex(self, waveform)` -- Compute the complex STFT of a waveform.
- `AudioProcessor.stft_complex_to_waveform` (method) `audio/audio_io.py:105` `def stft_complex_to_waveform(self, stft_complex)` -- Reconstruct waveform from complex STFT via inverse STFT.
- `AudioProcessor.stft_to_magnitude_phase` (method) `audio/audio_io.py:130` `def stft_to_magnitude_phase(self, stft_complex)` -- Decompose complex STFT into magnitude and phase.
- `AudioProcessor.magnitude_phase_to_stft` (method) `audio/audio_io.py:146` `def magnitude_phase_to_stft(self, magnitude, phase)` -- Recombine magnitude and phase into complex STFT.
- `AudioProcessor.stft_magnitude_to_model_input` (method) `audio/audio_io.py:161` `def stft_magnitude_to_model_input(self, magnitude)` -- Prepare STFT magnitude for input to the Hamiltonian network.
- `AudioProcessor.model_output_to_stft_magnitude` (method) `audio/audio_io.py:186` `def model_output_to_stft_magnitude(self, model_output, original_magnitude)` -- Convert model output (energy mask in [0, 1]) back to STFT magnitude scale.
- `AudioProcessor.waveform_to_mel_spectrogram` (method) `audio/audio_io.py:206` `def waveform_to_mel_spectrogram(self, waveform)` -- Convert waveform to normalized mel spectrogram (for visualization only).
- `AudioProcessor.save_audio` (method) `audio/audio_io.py:229` `def save_audio(self, waveform, file_path, sample_rate)` -- Save a waveform tensor to an audio file.
- `AudioProcessor.get_spectrogram_db_range` (method) `audio/audio_io.py:249` `def get_spectrogram_db_range(self, waveform)` -- Compute the dB range of a waveform's mel spectrogram.

## audio/audios.py
Depends on: `audio/experiment2.py`
- `HamiltonianConfig.segment_samples` (method) `audio/audios.py:89` `def segment_samples(self)` -- Calculate segment length in samples.
- `HamiltonianConfig.freq_bins` (method) `audio/audios.py:94` `def freq_bins(self)` -- Calculate frequency bins for real FFT.
- `IAudioSource.read_segment` (method) `audio/audios.py:107` `def read_segment(self)` -- Read audio segment.
- `IAudioSource.get_properties` (method) `audio/audios.py:112` `def get_properties(self)` -- Return audio properties.
- `IAudioSource.close` (method) `audio/audios.py:117` `def close(self)` -- Release resources.
- `IFieldOperator.evolve` (method) `audio/audios.py:126` `def evolve(self, field_state)` -- Evolve field state through Hamiltonian dynamics.
- `IMetricCollector.record` (method) `audio/audios.py:135` `def record(self, metrics)` -- Record metric values.
- `IMetricCollector.get_summary` (method) `audio/audios.py:140` `def get_summary(self)` -- Return aggregated metrics.
- `AudioResampler.resample` (method) `audio/audios.py:155` `def resample(audio, orig_sr, target_sr)` -- Resample audio from orig_sr to target_sr using polyphase filtering.
- `AudioResampler.load_wav_with_resample` (method) `audio/audios.py:173` `def load_wav_with_resample(file_path, target_sr)` -- Load WAV file and resample to target sample rate.
- `WaveFileSource.__init__` (method) `audio/audios.py:210` `def __init__(self, file_path, config)`
- `WaveFileSource.read_segment` (method) `audio/audios.py:244` `def read_segment(self)` -- Read next audio segment.
- `WaveFileSource.get_properties` (method) `audio/audios.py:261` `def get_properties(self)` -- Return audio file properties.
- `WaveFileSource.close` (method) `audio/audios.py:273` `def close(self)` -- Release resources.
- `ComprehensiveMetricCollector.__init__` (method) `audio/audios.py:284` `def __init__(self, config)`
- `ComprehensiveMetricCollector.record` (method) `audio/audios.py:289` `def record(self, metrics)` -- Record comprehensive metrics.
- `ComprehensiveMetricCollector.get_summary` (method) `audio/audios.py:298` `def get_summary(self)` -- Return statistical summary of all metrics.
- `ComprehensiveMetricCollector.export_to_json` (method) `audio/audios.py:320` `def export_to_json(self, path)` -- Export full history to JSON.
- `CheckpointManager.__init__` (method) `audio/audios.py:331` `def __init__(self, model, config, checkpoint_dir)`
- `CheckpointManager.check_and_save` (method) `audio/audios.py:344` `def check_and_save(self, force)` -- Check if checkpoint interval elapsed and save if necessary.
- `AudioSpectrogramConverter.__init__` (method) `audio/audios.py:393` `def __init__(self, config)`
- `AudioSpectrogramConverter.waveform_to_field` (method) `audio/audios.py:396` `def waveform_to_field(self, waveform)` -- Convert 1D audio to 2D field representation via STFT.
- `AudioSpectrogramConverter.field_to_waveform` (method) `audio/audios.py:432` `def field_to_waveform(self, field, original_length)` -- Reconstruct waveform from 2D field representation.
- `HamiltonianAudioProcessor.__init__` (method) `audio/audios.py:475` `def __init__(self, config, model, source)`
- `HamiltonianAudioProcessor.load_model_weights` (method) `audio/audios.py:504` `def load_model_weights(self, path)` -- Load pretrained Hamiltonian operator desde safetensors.
- `HamiltonianAudioProcessor.attach_source` (method) `audio/audios.py:513` `def attach_source(self, source)` -- Attach audio source via dependency injection.
- `HamiltonianAudioProcessor.process_stream` (method) `audio/audios.py:517` `def process_stream(self)` -- Process audio stream through Hamiltonian perception.
- `HamiltonianAudioProcessor.export_metrics` (method) `audio/audios.py:729` `def export_metrics(self, path)` -- Export comprehensive metrics to file.
- `HamiltonianAudioProcessor.force_checkpoint` (method) `audio/audios.py:733` `def force_checkpoint(self)` -- Force immediate checkpoint save.
- `HamiltonianAudioProcessor.main` (method) `audio/audios.py:742` `def main()` -- Entry point with argument parsing.

## audio/checkpoint_manager.py
Depends on: `audio/config.py`
Imported by: `audio/inference.py`, `audio/trainer.py`
- `CheckpointManager.__init__` (method) `audio/checkpoint_manager.py:34` `def __init__(self, config)`
- `CheckpointManager.should_save_checkpoint` (method) `audio/checkpoint_manager.py:41` `def should_save_checkpoint(self)` -- Check if enough time has elapsed since the last checkpoint.
- `CheckpointManager.save_checkpoint` (method) `audio/checkpoint_manager.py:46` `def save_checkpoint(self, model, optimizer, scheduler, epoch, step, metrics, current_loss)` -- Save the current model state and training metadata.
- `CheckpointManager.load_checkpoint` (method) `audio/checkpoint_manager.py:98` `def load_checkpoint(self, model, load_best)` -- Load a model checkpoint and return training metadata.
- `CheckpointManager.best_loss` (method) `audio/checkpoint_manager.py:156` `def best_loss(self)`

## audio/config.py
Imported by: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/inference.py`, `audio/losses.py`, `audio/main.py`, `audio/metrics.py`, `audio/model.py`, `audio/trainer.py`, `audio/visualization.py`
- `ModelArchitectureConfig.validate` (method) `audio/config.py:62` `def validate(self)` -- Ensure architectural coherence.
- `CheckpointConfig.checkpoint_path` (method) `audio/config.py:121` `def checkpoint_path(self)`
- `CheckpointConfig.best_model_path` (method) `audio/config.py:127` `def best_model_path(self)`
- `CheckpointConfig.metadata_path` (method) `audio/config.py:131` `def metadata_path(self)`
- `HamiltonianAudioConfig.validate_all` (method) `audio/config.py:198` `def validate_all(self)` -- Run validation on all sub-configurations.
- `HamiltonianAudioConfig.ensure_directories` (method) `audio/config.py:206` `def ensure_directories(self)` -- Create required output directories if they do not exist.

## audio/experiment2.py
Imported by: `audio/audios.py`
- `SeedManager.set_seed` (method) `audio/experiment2.py:76` `def set_seed(seed)`
- `LoggerFactory.create_logger` (method) `audio/experiment2.py:86` `def create_logger(name, level)`
- `IAnalysisStrategy.analyze` (method) `audio/experiment2.py:101` `def analyze(self, model)`
- `IMetricsCalculator.compute` (method) `audio/experiment2.py:107` `def compute(self, model)`
- `HamiltonianOperator.__init__` (method) `audio/experiment2.py:112` `def __init__(self, grid_size)`
- `HamiltonianOperator.apply` (method) `audio/experiment2.py:122` `def apply(self, field)`
- `HamiltonianOperator.time_evolution` (method) `audio/experiment2.py:127` `def time_evolution(self, field, dt)`
- `HamiltonianDataset.__init__` (method) `audio/experiment2.py:134` `def __init__(self, num_samples, grid_size, time_steps, dt, train_ratio)`
- `HamiltonianDataset.get_validation_batch` (method) `audio/experiment2.py:179` `def get_validation_batch(self)`
- `SpectralLayer.__init__` (method) `audio/experiment2.py:184` `def __init__(self, channels, grid_size)`
- `SpectralLayer.forward` (method) `audio/experiment2.py:195` `def forward(self, x)`
- `HamiltonianNeuralNetwork.__init__` (method) `audio/experiment2.py:228` `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `HamiltonianNeuralNetwork.forward` (method) `audio/experiment2.py:243` `def forward(self, x)`
- `LocalComplexityAnalyzer.compute_local_complexity` (method) `audio/experiment2.py:259` `def compute_local_complexity(weights, epsilon)`
- `SuperpositionAnalyzer.compute_superposition` (method) `audio/experiment2.py:275` `def compute_superposition(weights)`
- `CrystallographyMetricsCalculator.compute` (method) `audio/experiment2.py:303` `def compute(self, model, val_x, val_y)` -- Implementación de interfaz IMetricsCalculator.
- `CrystallographyMetricsCalculator.compute_gradient_covariance_kappa` (method) `audio/experiment2.py:311` `def compute_gradient_covariance_kappa(model, dataloader, num_batches)`
- `CrystallographyMetricsCalculator.compute_discretization_margin_from_state_dict` (method) `audio/experiment2.py:348` `def compute_discretization_margin_from_state_dict(model)` -- Calcula el margen de discretización desde los parámetros del modelo.
- `CrystallographyMetricsCalculator.compute_discretization_margin` (method) `audio/experiment2.py:361` `def compute_discretization_margin(coeffs)` -- Calcula el margen de discretización desde un diccionario de coeficientes.
- `CrystallographyMetricsCalculator.compute_alpha_purity_from_model` (method) `audio/experiment2.py:373` `def compute_alpha_purity_from_model(model)` -- Calcula el índice de pureza alpha directamente desde el modelo.
- `CrystallographyMetricsCalculator.compute_alpha_purity` (method) `audio/experiment2.py:383` `def compute_alpha_purity(coeffs)` -- Calcula el índice de pureza alpha desde un diccionario de coeficientes.
- `CrystallographyMetricsCalculator.compute_kappa` (method) `audio/experiment2.py:393` `def compute_kappa(model, val_x, val_y, num_batches)` -- Número de condición de la matriz de covarianza de gradientes.
- `CrystallographyMetricsCalculator.compute_kappa_quantum` (method) `audio/experiment2.py:464` `def compute_kappa_quantum(model, hbar)` -- Versión del cálculo cuántico de kappa que opera directamente sobre el modelo.
- `CrystallographyMetricsCalculator.compute_kappa_quantum_from_coeffs` (method) `audio/experiment2.py:492` `def compute_kappa_quantum_from_coeffs(coeffs, hbar)` -- Versión del cálculo cuántico de kappa desde diccionario de coeficientes.
- `CrystallographyMetricsCalculator.compute_poynting_vector` (method) `audio/experiment2.py:603` `def compute_poynting_vector(model)` -- Vector de Poynting: flujo de energía en el espacio de parámetros.
- `CrystallographyMetricsCalculator.compute_all_metrics` (method) `audio/experiment2.py:679` `def compute_all_metrics(model, val_x, val_y)` -- Calcula todas las métricas cristalográficas con manejo de errores.
- `CrystallographyMetricsCalculator.safe_compute` (method) `audio/experiment2.py:694` `def safe_compute(func)`
- `ThermodynamicMetricsCalculator.compute` (method) `audio/experiment2.py:738` `def compute(self, model, gradient_buffer, learning_rate, loss_history, temp_history)`
- `ThermodynamicMetricsCalculator.compute_effective_temperature` (method) `audio/experiment2.py:747` `def compute_effective_temperature(gradient_buffer, learning_rate)`
- `ThermodynamicMetricsCalculator.compute_specific_heat` (method) `audio/experiment2.py:760` `def compute_specific_heat(loss_history, temp_history, cv_threshold)`
- `SpectroscopyMetricsCalculator.compute` (method) `audio/experiment2.py:771` `def compute(self, model)`
- `SpectroscopyMetricsCalculator.compute_weight_diffraction` (method) `audio/experiment2.py:776` `def compute_weight_diffraction(coeffs)`
- `CheckpointManager.__init__` (method) `audio/experiment2.py:805` `def __init__(self, interval_minutes, max_checkpoints)`
- `CheckpointManager.should_save_checkpoint` (method) `audio/experiment2.py:813` `def should_save_checkpoint(self)`
- `CheckpointManager.save_checkpoint` (method) `audio/experiment2.py:818` `def save_checkpoint(self, model, optimizer, epoch, metrics)`
- `TrainingMetricsMonitor.__init__` (method) `audio/experiment2.py:875` `def __init__(self)`
- `TrainingMetricsMonitor.update_metrics` (method) `audio/experiment2.py:895` `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat...`
- `GlassStateDetector.__init__` (method) `audio/experiment2.py:913` `def __init__(self, patience_epochs)`
- `GlassStateDetector.should_stop` (method) `audio/experiment2.py:918` `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)`
- `GlassStateDetector.is_crystal_formed` (method) `audio/experiment2.py:963` `def is_crystal_formed(self, lc, sp, kappa, delta, temp, cv)`
- `TrainingEngine.__init__` (method) `audio/experiment2.py:974` `def __init__(self, model, optimizer, device, logger)`
- `TrainingEngine.train_epoch` (method) `audio/experiment2.py:997` `def train_epoch(self, dataloader, epoch)`
- `TrainingEngine.validate` (method) `audio/experiment2.py:1027` `def validate(self, val_x, val_y)`
- `TrainingEngine.compute_weight_metrics` (method) `audio/experiment2.py:1040` `def compute_weight_metrics(self)`
- `TrainingEngine.execute_training` (method) `audio/experiment2.py:1056` `def execute_training(self, dataloader, val_x, val_y, epochs, seed, early_stopping)`
- `SeedMiningSystem.__init__` (method) `audio/experiment2.py:1114` `def __init__(self, max_attempts)`
- `SeedMiningSystem.mine` (method) `audio/experiment2.py:1118` `def mine(self)`
- `SingleExperimentRunner.__init__` (method) `audio/experiment2.py:1165` `def __init__(self, seed, epochs, grid_size, hidden_dim, num_spectral_layers, learning_rate)`
- `SingleExperimentRunner.run` (method) `audio/experiment2.py:1175` `def run(self)`
- `CheckpointAnalyzer.__init__` (method) `audio/experiment2.py:1224` `def __init__(self, checkpoint_path, results_dir)`
- `CheckpointAnalyzer.analyze` (method) `audio/experiment2.py:1230` `def analyze(self)`
- `Application.__init__` (method) `audio/experiment2.py:1276` `def __init__(self)`
- `Application.run` (method) `audio/experiment2.py:1294` `def run(self)`
- `Application.main` (method) `audio/experiment2.py:1334` `def main()`

## audio/inference.py
Depends on: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/config.py`, `audio/metrics.py`, `audio/model.py`, `audio/visualization.py`
Imported by: `audio/main.py`
- `HamiltonianAudioInference.__init__` (method) `audio/inference.py:42` `def __init__(self, config, load_best)`
- `HamiltonianAudioInference.analyze_audio` (method) `audio/inference.py:73` `def analyze_audio(self, audio_file_path, output_prefix)` -- Perform complete Hamiltonian analysis on an audio file.

## audio/losses.py
Depends on: `audio/config.py`
Imported by: `audio/trainer.py`
- `HamiltonianLossComputer.__init__` (method) `audio/losses.py:37` `def __init__(self, config)`
- `HamiltonianLossComputer.compute_total_loss` (method) `audio/losses.py:41` `def compute_total_loss(self, prediction, target, intermediates, model)` -- Compute the complete weighted loss with all Hamiltonian terms.

## audio/main.py
Depends on: `audio/config.py`, `audio/inference.py`, `audio/trainer.py`
Imported by: `test_grokkit.py`
- `build_argument_parser` (function) `audio/main.py:34` `def build_argument_parser()` -- Construct the complete argument parser with all configurable parameters.
- `build_config_from_args` (function) `audio/main.py:103` `def build_config_from_args(args)` -- Construct the full configuration from parsed CLI arguments.
- `validate_audio_file` (function) `audio/main.py:163` `def validate_audio_file(file_path)` -- Validate that the audio file exists and has a supported extension.
- `print_configuration_banner` (function) `audio/main.py:175` `def print_configuration_banner(config, mode, audio_path)` -- Print a formatted configuration summary.
- `run_training` (function) `audio/main.py:215` `def run_training(args)` -- Execute the training pipeline.
- `run_inference` (function) `audio/main.py:226` `def run_inference(args)` -- Execute the inference pipeline.
- `main` (function) `audio/main.py:236` `def main()` -- Main entry point.

## audio/metrics.py
Depends on: `audio/config.py`
Imported by: `audio/inference.py`, `audio/trainer.py`
- `HamiltonianMetricsTracker.__init__` (method) `audio/metrics.py:36` `def __init__(self, config)`
- `HamiltonianMetricsTracker.compute_hamiltonian_energy` (method) `audio/metrics.py:74` `def compute_hamiltonian_energy(self, q, p)` -- Compute the Hamiltonian H(q, p) = T(p) + V(q).
- `HamiltonianMetricsTracker.compute_symplectic_form` (method) `audio/metrics.py:97` `def compute_symplectic_form(self, q, p, dq, dp)` -- Compute the symplectic 2-form omega(dq, dp) = sum(dq_i ^ dp_i).
- `HamiltonianMetricsTracker.compute_liouville_measure` (method) `audio/metrics.py:125` `def compute_liouville_measure(self, jacobian)` -- Compute Liouville measure |det(J)| for the flow map Jacobian.
- `HamiltonianMetricsTracker.compute_phase_space_volume` (method) `audio/metrics.py:153` `def compute_phase_space_volume(self, q, p)` -- Estimate phase space volume occupied by the state (q, p).
- `HamiltonianMetricsTracker.compute_action_integral` (method) `audio/metrics.py:181` `def compute_action_integral(self, q_trajectory, p_trajectory, dt)` -- Compute the action integral S = integral(L dt) along a trajectory.
- `HamiltonianMetricsTracker.compute_poisson_bracket` (method) `audio/metrics.py:208` `def compute_poisson_bracket(self, f_values, g_values, q, p)` -- Estimate the Poisson bracket {f, g} = sum(df/dq * dg/dp - df/dp * dg/dq).
- `HamiltonianMetricsTracker.compute_spectral_entropy` (method) `audio/metrics.py:238` `def compute_spectral_entropy(self, spectrum)` -- Compute spectral entropy H = -sum(p_i * log(p_i)).
- `HamiltonianMetricsTracker.compute_reconstruction_snr` (method) `audio/metrics.py:259` `def compute_reconstruction_snr(self, original, reconstructed)` -- Compute Signal-to-Noise Ratio in dB.
- `HamiltonianMetricsTracker.compute_spectral_convergence` (method) `audio/metrics.py:281` `def compute_spectral_convergence(self, original_spectrum, reconstructed_spectrum)` -- Compute spectral convergence metric.
- `HamiltonianMetricsTracker.compute_phase_coherence` (method) `audio/metrics.py:305` `def compute_phase_coherence(self, phase_original, phase_reconstructed)` -- Compute phase coherence between original and reconstructed signals.
- `HamiltonianMetricsTracker.compute_energy_drift` (method) `audio/metrics.py:328` `def compute_energy_drift(self, energy_initial, energy_current)` -- Compute relative energy drift from initial state.
- `HamiltonianMetricsTracker.record_gradient_norm` (method) `audio/metrics.py:348` `def record_gradient_norm(self, model_parameters)` -- Compute and record the total gradient norm across all parameters.
- `HamiltonianMetricsTracker.record_parameter_norm` (method) `audio/metrics.py:359` `def record_parameter_norm(self, model_parameters)` -- Compute and record the total parameter norm.
- `HamiltonianMetricsTracker.record_learning_rate` (method) `audio/metrics.py:369` `def record_learning_rate(self, lr)` -- Record current learning rate.
- `HamiltonianMetricsTracker.record_loss_component` (method) `audio/metrics.py:374` `def record_loss_component(self, name, value)` -- Record an individual loss component value.
- `HamiltonianMetricsTracker.get_current_metrics` (method) `audio/metrics.py:388` `def get_current_metrics(self)` -- Return a snapshot of all current metric values.
- `HamiltonianMetricsTracker.get_moving_averages` (method) `audio/metrics.py:392` `def get_moving_averages(self)` -- Compute moving averages for all tracked metrics.
- `HamiltonianMetricsTracker.get_formatted_metrics_string` (method) `audio/metrics.py:400` `def get_formatted_metrics_string(self)` -- Format all current metrics into a human-readable string for progress bars.
- `HamiltonianMetricsTracker.increment_step` (method) `audio/metrics.py:413` `def increment_step(self)` -- Advance the global step counter.
- `HamiltonianMetricsTracker.step_count` (method) `audio/metrics.py:418` `def step_count(self)`
- `HamiltonianMetricsTracker.should_log` (method) `audio/metrics.py:421` `def should_log(self)` -- Determine if metrics should be logged at this step.

## audio/model.py
Depends on: `audio/config.py`
Imported by: `audio/inference.py`, `audio/trainer.py`
- `SpectralEvolutionLayer.__init__` (method) `audio/model.py:36` `def __init__(self, hidden_dim, kernel_base_height, kernel_base_width, init_std)`
- `SpectralEvolutionLayer.forward` (method) `audio/model.py:51` `def forward(self, x)` -- Apply one step of Hamiltonian spectral evolution via RFFT2.
- `SpectralEvolutionLayer.evolve_complex` (method) `audio/model.py:85` `def evolve_complex(self, x, target_height, target_width)` -- Full complex FFT evolution for amplitude and phase extraction.
- `SpectralEvolutionLayer.evolve_real` (method) `audio/model.py:124` `def evolve_real(self, x, target_height, target_width)` -- Real FFT evolution for action map computation.
- `HamiltonianNeuralNetwork.__init__` (method) `audio/model.py:172` `def __init__(self, config)`
- `HamiltonianNeuralNetwork.forward` (method) `audio/model.py:200` `def forward(self, x)` -- Full forward pass: project -> evolve -> reconstruct.
- `HamiltonianNeuralNetwork.forward_with_intermediates` (method) `audio/model.py:216` `def forward_with_intermediates(self, x)` -- Forward pass returning intermediate hidden states for analysis.
- `HamiltonianNeuralNetwork.extract_hamiltonian_fields` (method) `audio/model.py:237` `def extract_hamiltonian_fields(self, x)` -- Extract the three Hamiltonian field representations: 1.
- `HamiltonianNeuralNetwork.compute_energy_mask` (method) `audio/model.py:270` `def compute_energy_mask(self, x)` -- Compute the Hamiltonian energy mask for spectral reconstruction.

## audio/trainer.py
Depends on: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/config.py`, `audio/losses.py`, `audio/metrics.py`, `audio/model.py`
Imported by: `audio/main.py`
- `AudioSpectrogramDatasetBuilder.__init__` (method) `audio/trainer.py:46` `def __init__(self, config)`
- `AudioSpectrogramDatasetBuilder.build_dataset` (method) `audio/trainer.py:49` `def build_dataset(self, mel_spectrogram)` -- Segment a mel spectrogram into training patches.
- `HamiltonianAudioTrainer.__init__` (method) `audio/trainer.py:92` `def __init__(self, config)`
- `HamiltonianAudioTrainer.train` (method) `audio/trainer.py:151` `def train(self, audio_file_path)` -- Execute the full training pipeline on an audio file.
- `HamiltonianAudioTrainer.model` (method) `audio/trainer.py:377` `def model(self)`
- `HamiltonianAudioTrainer.audio_processor` (method) `audio/trainer.py:381` `def audio_processor(self)`

## audio/visualization.py
Depends on: `audio/config.py`
Imported by: `audio/inference.py`
- `HamiltonianAudioVisualizer.__init__` (method) `audio/visualization.py:34` `def __init__(self, vis_config, audio_config)`
- `HamiltonianAudioVisualizer.render_complete_analysis` (method) `audio/visualization.py:43` `def render_complete_analysis(self, amplitude_map, phase_map, action_map, original_spectrogram...` -- Generate the complete suite of Hamiltonian analysis visualizations.

## diff_weights.py
- `analize_checkpoint` (function) `diff_weights.py:5` `def analize_checkpoint(path)`

## dirac.py
Depends on: `experiment2.py`
- `DiracDeltaAnalyzer.__init__` (method) `dirac.py:40` `def __init__(self, checkpoint_path, device)`
- `DiracDeltaAnalyzer.extract_charge_distribution` (method) `dirac.py:64` `def extract_charge_distribution(self)`
- `DiracDeltaAnalyzer.compute_dirac_delta_approximation` (method) `dirac.py:77` `def compute_dirac_delta_approximation(self, charge_density)`
- `DiracDeltaAnalyzer.compute_electric_field` (method) `dirac.py:112` `def compute_electric_field(self, dirac_data, eval_points)`
- `DiracDeltaAnalyzer.compute_electric_flux` (method) `dirac.py:157` `def compute_electric_flux(self, electric_field, surface_points)`
- `DiracDeltaAnalyzer.compute_divergence` (method) `dirac.py:192` `def compute_divergence(self, electric_field)`
- `DiracDeltaAnalyzer.verify_gauss_law` (method) `dirac.py:200` `def verify_gauss_law(self, dirac_data, flux_data)`
- `DiracDeltaAnalyzer.analyze_all` (method) `dirac.py:223` `def analyze_all(self)`
- `DiracVisualizer.plot_charge_distribution` (method) `dirac.py:331` `def plot_charge_distribution(charge_density, point_positions, point_charges, output_path)`
- `DiracVisualizer.plot_electric_field` (method) `dirac.py:379` `def plot_electric_field(electric_field, output_path)`
- `DiracVisualizer.plot_divergence` (method) `dirac.py:441` `def plot_divergence(divergence, output_path)`
- `DiracVisualizer.plot_combined_analysis` (method) `dirac.py:490` `def plot_combined_analysis(charge_density, point_positions, point_charges, electric_field, divergence, output_path)`
- `DiracVisualizer.analyze_checkpoint` (method) `dirac.py:574` `def analyze_checkpoint(checkpoint_path, output_dir)`
- `DiracVisualizer.analyze_multiple_checkpoints` (method) `dirac.py:620` `def analyze_multiple_checkpoints(checkpoint_dir, n_latest, output_dir)`
- `DiracVisualizer.main` (method) `dirac.py:652` `def main()`

## expand.py
- `load_config` (function) `expand.py:18` `def load_config(toml_path)`
- `expand_spectral_weights` (function) `expand.py:23` `def expand_spectral_weights(kernel_real, kernel_imag, target_size, source_size)` -- Expand spectral kernels via zero-padding in frequency domain.
- `expand_model` (function) `expand.py:43` `def expand_model(model, target_resolution, source_resolution)` -- Create a new model with expanded spectral weights.
- `evaluate_model` (function) `expand.py:74` `def evaluate_model(model, resolution, device)` -- Evaluate expanded model on synthetic data.
- `main` (function) `expand.py:105` `def main()`

## experiment.py
- `Config.set_seed` (method) `experiment.py:88` `def set_seed(seed)`
- `Config.setup_logger` (method) `experiment.py:96` `def setup_logger(name, level)`
- `IAnalysisStrategy.analyze` (method) `experiment.py:111` `def analyze(self, model)`
- `IMetricsCalculator.compute` (method) `experiment.py:117` `def compute(self, model)`
- `HamiltonianOperator.__init__` (method) `experiment.py:124` `def __init__(self, grid_size)`
- `HamiltonianOperator.apply` (method) `experiment.py:135` `def apply(self, field)`
- `HamiltonianOperator.time_evolution` (method) `experiment.py:140` `def time_evolution(self, field, dt)`
- `FastDataset.__init__` (method) `experiment.py:149` `def __init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)`
- `FastDataset.get_val_batch` (method) `experiment.py:199` `def get_val_batch(self)`
- `SpectralLayer.__init__` (method) `experiment.py:206` `def __init__(self, channels, grid_size)`
- `SpectralLayer.forward` (method) `experiment.py:219` `def forward(self, x)`
- `SimpleHamiltonianNet.__init__` (method) `experiment.py:258` `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `SimpleHamiltonianNet.forward` (method) `experiment.py:279` `def forward(self, x)`
- `LocalComplexityAnalyzer.compute_local_complexity` (method) `experiment.py:295` `def compute_local_complexity(weights, epsilon)` -- Compute Local Complexity (LC) metric for weight matrix.
- `SuperpositionAnalyzer.compute_superposition` (method) `experiment.py:312` `def compute_superposition(weights)` -- Compute Superposition (SP) metric for weight matrix.
- `CrystallographyMetrics.compute_kappa` (method) `experiment.py:342` `def compute_kappa(model, dataloader, num_batches)`
- `CrystallographyMetrics.compute_discretization_margin` (method) `experiment.py:378` `def compute_discretization_margin(coeffs)`
- `CrystallographyMetrics.compute_alpha_purity` (method) `experiment.py:387` `def compute_alpha_purity(coeffs)`
- `CrystallographyMetrics.compute_kappa_quantum` (method) `experiment.py:394` `def compute_kappa_quantum(coeffs, hbar)`
- `CrystallographyMetrics.compute_poynting_vector` (method) `experiment.py:411` `def compute_poynting_vector(coeffs)`
- `CrystallographyMetrics.compute_all_metrics` (method) `experiment.py:426` `def compute_all_metrics(model, dataloader)`
- `ThermodynamicMetrics.compute_effective_temperature` (method) `experiment.py:446` `def compute_effective_temperature(gradient_buffer, learning_rate)`
- `ThermodynamicMetrics.compute_specific_heat` (method) `experiment.py:460` `def compute_specific_heat(loss_history, temp_history, cv_threshold)`
- `SpectroscopyMetrics.compute_weight_diffraction` (method) `experiment.py:472` `def compute_weight_diffraction(coeffs)`
- `CheckpointManager.__init__` (method) `experiment.py:501` `def __init__(self, interval_minutes, max_checkpoints)`
- `CheckpointManager.should_save_checkpoint` (method) `experiment.py:509` `def should_save_checkpoint(self)`
- `CheckpointManager.save_checkpoint` (method) `experiment.py:514` `def save_checkpoint(self, model, optimizer, epoch, metrics)`
- `TrainingMonitor.__init__` (method) `experiment.py:574` `def __init__(self)`
- `TrainingMonitor.update_metrics` (method) `experiment.py:594` `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat...`
- `GlassStopper.__init__` (method) `experiment.py:612` `def __init__(self, patience_epochs)`
- `GlassStopper.should_stop` (method) `experiment.py:616` `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)` -- Check if the system is in glass state and should stop mining.
- `GlassStopper.train_with_early_glass_stop` (method) `experiment.py:670` `def train_with_early_glass_stop(model, optimizer, seed, epochs)` -- Train model with early stopping for glass detection.
- `GlassStopper.seed_miner` (method) `experiment.py:803` `def seed_miner(total_attempts)` -- Mine for crystal seeds by trying sequential seeds.
- `GlassStopper.main` (method) `experiment.py:856` `def main()`
- `BoltzmannAnalysisProgram.__init__` (method) `experiment.py:880` `def __init__(self, checkpoint_path, results_dir)`
- `BoltzmannAnalysisProgram.load_and_analyze_checkpoint` (method) `experiment.py:886` `def load_and_analyze_checkpoint(self)`
- `BoltzmannAnalysisProgram.dataloader` (method) `experiment.py:903` `def dataloader()`

## experiment2.py
Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`
- `SeedManager.set_seed` (method) `experiment2.py:76` `def set_seed(seed)`
- `LoggerFactory.create_logger` (method) `experiment2.py:86` `def create_logger(name, level)`
- `IAnalysisStrategy.analyze` (method) `experiment2.py:101` `def analyze(self, model)`
- `IMetricsCalculator.compute` (method) `experiment2.py:107` `def compute(self, model)`
- `HamiltonianOperator.__init__` (method) `experiment2.py:112` `def __init__(self, grid_size)`
- `HamiltonianOperator.apply` (method) `experiment2.py:122` `def apply(self, field)`
- `HamiltonianOperator.time_evolution` (method) `experiment2.py:127` `def time_evolution(self, field, dt)`
- `HamiltonianDataset.__init__` (method) `experiment2.py:134` `def __init__(self, num_samples, grid_size, time_steps, dt, train_ratio)`
- `HamiltonianDataset.get_validation_batch` (method) `experiment2.py:179` `def get_validation_batch(self)`
- `SpectralLayer.__init__` (method) `experiment2.py:184` `def __init__(self, channels, grid_size)`
- `SpectralLayer.forward` (method) `experiment2.py:195` `def forward(self, x)`
- `HamiltonianNeuralNetwork.__init__` (method) `experiment2.py:228` `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `HamiltonianNeuralNetwork.forward` (method) `experiment2.py:243` `def forward(self, x)`
- `LocalComplexityAnalyzer.compute_local_complexity` (method) `experiment2.py:259` `def compute_local_complexity(weights, epsilon)`
- `SuperpositionAnalyzer.compute_superposition` (method) `experiment2.py:275` `def compute_superposition(weights)`
- `CrystallographyMetricsCalculator.compute` (method) `experiment2.py:303` `def compute(self, model, val_x, val_y)` -- Implementación de interfaz IMetricsCalculator.
- `CrystallographyMetricsCalculator.compute_gradient_covariance_kappa` (method) `experiment2.py:311` `def compute_gradient_covariance_kappa(model, dataloader, num_batches)`
- `CrystallographyMetricsCalculator.compute_discretization_margin_from_state_dict` (method) `experiment2.py:348` `def compute_discretization_margin_from_state_dict(model)` -- Calcula el margen de discretización desde los parámetros del modelo.
- `CrystallographyMetricsCalculator.compute_discretization_margin` (method) `experiment2.py:361` `def compute_discretization_margin(coeffs)` -- Calcula el margen de discretización desde un diccionario de coeficientes.
- `CrystallographyMetricsCalculator.compute_alpha_purity_from_model` (method) `experiment2.py:373` `def compute_alpha_purity_from_model(model)` -- Calcula el índice de pureza alpha directamente desde el modelo.
- `CrystallographyMetricsCalculator.compute_alpha_purity` (method) `experiment2.py:383` `def compute_alpha_purity(coeffs)` -- Calcula el índice de pureza alpha desde un diccionario de coeficientes.
- `CrystallographyMetricsCalculator.compute_kappa` (method) `experiment2.py:393` `def compute_kappa(model, val_x, val_y, num_batches)` -- Número de condición de la matriz de covarianza de gradientes.
- `CrystallographyMetricsCalculator.compute_kappa_quantum` (method) `experiment2.py:464` `def compute_kappa_quantum(model, hbar)` -- Versión del cálculo cuántico de kappa que opera directamente sobre el modelo.
- `CrystallographyMetricsCalculator.compute_kappa_quantum_from_coeffs` (method) `experiment2.py:492` `def compute_kappa_quantum_from_coeffs(coeffs, hbar)` -- Versión del cálculo cuántico de kappa desde diccionario de coeficientes.
- `CrystallographyMetricsCalculator.compute_poynting_vector` (method) `experiment2.py:603` `def compute_poynting_vector(model)` -- Vector de Poynting: flujo de energía en el espacio de parámetros.
- `CrystallographyMetricsCalculator.compute_all_metrics` (method) `experiment2.py:679` `def compute_all_metrics(model, val_x, val_y)` -- Calcula todas las métricas cristalográficas con manejo de errores.
- `CrystallographyMetricsCalculator.safe_compute` (method) `experiment2.py:694` `def safe_compute(func)`
- `ThermodynamicMetricsCalculator.compute` (method) `experiment2.py:738` `def compute(self, model, gradient_buffer, learning_rate, loss_history, temp_history)`
- `ThermodynamicMetricsCalculator.compute_effective_temperature` (method) `experiment2.py:747` `def compute_effective_temperature(gradient_buffer, learning_rate)`
- `ThermodynamicMetricsCalculator.compute_specific_heat` (method) `experiment2.py:760` `def compute_specific_heat(loss_history, temp_history, cv_threshold)`
- `SpectroscopyMetricsCalculator.compute` (method) `experiment2.py:771` `def compute(self, model)`
- `SpectroscopyMetricsCalculator.compute_weight_diffraction` (method) `experiment2.py:776` `def compute_weight_diffraction(coeffs)`
- `CheckpointManager.__init__` (method) `experiment2.py:805` `def __init__(self, interval_minutes, max_checkpoints)`
- `CheckpointManager.should_save_checkpoint` (method) `experiment2.py:813` `def should_save_checkpoint(self)`
- `CheckpointManager.save_checkpoint` (method) `experiment2.py:818` `def save_checkpoint(self, model, optimizer, epoch, metrics)`
- `TrainingMetricsMonitor.__init__` (method) `experiment2.py:875` `def __init__(self)`
- `TrainingMetricsMonitor.update_metrics` (method) `experiment2.py:895` `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat...`
- `GlassStateDetector.__init__` (method) `experiment2.py:913` `def __init__(self, patience_epochs)`
- `GlassStateDetector.should_stop` (method) `experiment2.py:918` `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)`
- `GlassStateDetector.is_crystal_formed` (method) `experiment2.py:963` `def is_crystal_formed(self, lc, sp, kappa, delta, temp, cv)`
- `TrainingEngine.__init__` (method) `experiment2.py:974` `def __init__(self, model, optimizer, device, logger)`
- `TrainingEngine.train_epoch` (method) `experiment2.py:997` `def train_epoch(self, dataloader, epoch)`
- `TrainingEngine.validate` (method) `experiment2.py:1027` `def validate(self, val_x, val_y)`
- `TrainingEngine.compute_weight_metrics` (method) `experiment2.py:1040` `def compute_weight_metrics(self)`
- `TrainingEngine.execute_training` (method) `experiment2.py:1056` `def execute_training(self, dataloader, val_x, val_y, epochs, seed, early_stopping)`
- `SeedMiningSystem.__init__` (method) `experiment2.py:1114` `def __init__(self, max_attempts)`
- `SeedMiningSystem.mine` (method) `experiment2.py:1118` `def mine(self)`
- `SingleExperimentRunner.__init__` (method) `experiment2.py:1165` `def __init__(self, seed, epochs, grid_size, hidden_dim, num_spectral_layers, learning_rate)`
- `SingleExperimentRunner.run` (method) `experiment2.py:1175` `def run(self)`
- `CheckpointAnalyzer.__init__` (method) `experiment2.py:1224` `def __init__(self, checkpoint_path, results_dir)`
- `CheckpointAnalyzer.analyze` (method) `experiment2.py:1230` `def analyze(self)`
- `Application.__init__` (method) `experiment2.py:1276` `def __init__(self)`
- `Application.run` (method) `experiment2.py:1294` `def run(self)`
- `Application.main` (method) `experiment2.py:1334` `def main()`

## get_meditions.py
Depends on: `experiment2.py`
- `ThermodynamicConfig.setup_logger` (method) `get_meditions.py:51` `def setup_logger(name, level)`
- `ThermodynamicPotential.helmholtz_free_energy` (method) `get_meditions.py:81` `def helmholtz_free_energy(self)` -- F = U - T*S (a μ y N constantes)
- `ThermodynamicPotential.gibbs_free_energy` (method) `get_meditions.py:85` `def gibbs_free_energy(self)` -- G = F + μ*N + P*V (presión algorítmica)
- `ThermodynamicPotential.is_stable` (method) `get_meditions.py:90` `def is_stable(self)` -- Criterio de estabilidad: dG < 0
- `SpectralCoefficients.from_model` (method) `get_meditions.py:112` `def from_model(cls, model)` -- Extrae coeficientes del modelo HPU Core
- `SpectralCoefficients.compute_kappa` (method) `get_meditions.py:127` `def compute_kappa(model, val_x, val_y, num_batches)` -- Número de condición de la matriz de covarianza de gradientes.
- `SpectralCoefficients.compute_discretization_margin` (method) `get_meditions.py:187` `def compute_discretization_margin(model)` -- δ = max |w - round(w)| sobre todos los parámetros.
- `SpectralCoefficients.compute_alpha_purity` (method) `get_meditions.py:203` `def compute_alpha_purity(model)` -- α = -log(δ).
- `SpectralCoefficients.compute_local_complexity` (method) `get_meditions.py:214` `def compute_local_complexity(model)` -- Fracción de parámetros "activos" (no cerca de cero).
- `SpectralCoefficients.compute_kappa_quantum` (method) `get_meditions.py:226` `def compute_kappa_quantum(model, hbar)` -- κ cuántico: número de condición con regularización cuántica.
- `SpectralCoefficients.compute_poynting_vector` (method) `get_meditions.py:312` `def compute_poynting_vector(model)` -- Vector de Poynting: flujo de energía en el espacio de parámetros.
- `SpectralCoefficients.compute_all_metrics` (method) `get_meditions.py:366` `def compute_all_metrics(model, val_x, val_y)` -- Calcula todas las métricas cristalográficas.
- `ThermodynamicMetrics.compute_effective_temperature` (method) `get_meditions.py:401` `def compute_effective_temperature(gradient_buffer, learning_rate)` -- T_eff = (lr/2) * Var(∇L).
- `ThermodynamicMetrics.compute_specific_heat` (method) `get_meditions.py:418` `def compute_specific_heat(loss_history, temp_history, cv_threshold)` -- C_v = Var(U) / T^2.
- `ThermodynamicMetrics.compute_critical_exponents` (method) `get_meditions.py:436` `def compute_critical_exponents(temp_history, cv_history, alpha_history)` -- Exponentes críticos cerca de transiciones de fase.
- `ThermodynamicMetrics.compute_equation_of_state` (method) `get_meditions.py:504` `def compute_equation_of_state(temp_eff, alpha, kappa)` -- Ecuación de estado: T_c(α) = T_0 * exp(-c*α) Relación constitutiva cristal-vidrio.
- `ThermodynamicMetrics.compute_mutual_information` (method) `get_meditions.py:539` `def compute_mutual_information(weights, gradients)` -- Información mutua pesos-gradientes.
- `ThermodynamicMetrics.estimate_hbar_algorithmic` (method) `get_meditions.py:561` `def estimate_hbar_algorithmic(model_complexity, weight_dim, mutual_information)` -- ħ algorítmico efectivo.
- `ThermodynamicMetrics.compute_fisher_information_matrix` (method) `get_meditions.py:572` `def compute_fisher_information_matrix(model, samples)` -- Matriz de información de Fisher.
- `ThermodynamicMetrics.compute_ricci_curvature` (method) `get_meditions.py:593` `def compute_ricci_curvature(fisher_matrix)` -- Curvatura de Ricci escalar.
- `ThermodynamicMetrics.calculate_carnot_efficiency` (method) `get_meditions.py:608` `def calculate_carnot_efficiency(delta_alpha, total_flops, initial_alpha)` -- Eficiencia de Carnot del proceso de aprendizaje.
- `SpectroscopyMetrics.compute_weight_diffraction` (method) `get_meditions.py:640` `def compute_weight_diffraction(model)` -- Patrón de difracción de pesos (FFT).
- `SpectroscopyMetrics.extract_lattice_parameters` (method) `get_meditions.py:680` `def extract_lattice_parameters(weight_tensor, rank)` -- Extrae parámetros de red vía SVD.
- `SpectroscopyMetrics.compute_gibbs_free_energy` (method) `get_meditions.py:732` `def compute_gibbs_free_energy(loss, temp, entropy)` -- Energía libre de Gibbs.
- `CheckpointVerifier.__init__` (method) `get_meditions.py:741` `def __init__(self, checkpoint_path, device)`
- `CheckpointVerifier.verify_all_metrics` (method) `get_meditions.py:782` `def verify_all_metrics(self)` -- Calcula TODAS las métricas desde cero y compara con las guardadas
- `CheckpointVerifier.verify_latest_checkpoints` (method) `get_meditions.py:1439` `def verify_latest_checkpoints(checkpoint_dir, n)` -- Verifica los N checkpoints más recientes con análisis completo
- `CheckpointVerifier.main` (method) `get_meditions.py:1500` `def main()`

## hamiltonian_mbl.py
- `HamiltonianArchitectureConfig.get_input_dim` (method) `hamiltonian_mbl.py:55` `def get_input_dim(self)` -- Calculate input dimension from grid size.
- `HamiltonianArchitectureConfig.get_total_parameters` (method) `hamiltonian_mbl.py:59` `def get_total_parameters(self)` -- Estimate total parameter count.
- `MBLAnalysisConfig.get_reduced_dimension` (method) `hamiltonian_mbl.py:135` `def get_reduced_dimension(self)` -- Calculate reduced dimension for analysis.
- `IModel.get_coefficients` (method) `hamiltonian_mbl.py:162` `def get_coefficients(self)`
- `IModel.forward` (method) `hamiltonian_mbl.py:163` `def forward(self)`
- `ILevelSpacingCalculator.calculate` (method) `hamiltonian_mbl.py:169` `def calculate(self, model)`
- `IParticipationRatioCalculator.calculate` (method) `hamiltonian_mbl.py:175` `def calculate(self, model)`
- `ISyntheticPlanckCalculator.calculate` (method) `hamiltonian_mbl.py:181` `def calculate(self, participation_ratio, energy_gap)`
- `IDiscretizationDialAnalyzer.analyze_robustness` (method) `hamiltonian_mbl.py:187` `def analyze_robustness(self, model, noise_levels)`
- `ICheckpointManager.save_checkpoint` (method) `hamiltonian_mbl.py:193` `def save_checkpoint(self, model, epoch, metrics, loss_history, path)`
- `ICheckpointManager.load_checkpoint` (method) `hamiltonian_mbl.py:195` `def load_checkpoint(self, path)`
- `ITrainingMetricsCollector.collect` (method) `hamiltonian_mbl.py:201` `def collect(self, model, loss, epoch, loss_history)`
- `ArchitectureMigrator.__init__` (method) `hamiltonian_mbl.py:214` `def __init__(self, source_config, target_config)`
- `ArchitectureMigrator.migrate_state_dict` (method) `hamiltonian_mbl.py:218` `def migrate_state_dict(self, source_state)` -- Migra estado de SimpleHamiltonianNet a HamiltonianNeuralNetwork.
- `SpectralHamiltonianLayer.__init__` (method) `hamiltonian_mbl.py:352` `def __init__(self, config)`
- `SpectralHamiltonianLayer.forward` (method) `hamiltonian_mbl.py:373` `def forward(self, q, p, dt)` -- Symplectic Euler integration of Hamilton's equations.
- `SpectralHamiltonianLayer.get_hamiltonian` (method) `hamiltonian_mbl.py:401` `def get_hamiltonian(self, q, p)` -- Compute Hamiltonian H = T + V in spectral space.
- `HamiltonianNeuralNetwork.__init__` (method) `hamiltonian_mbl.py:418` `def __init__(self, config)`
- `HamiltonianNeuralNetwork.forward` (method) `hamiltonian_mbl.py:443` `def forward(self, q, p, dt)` -- Forward pass through Hamiltonian dynamics.
- `HamiltonianNeuralNetwork.time_evolution` (method) `hamiltonian_mbl.py:459` `def time_evolution(self, q_initial, p_initial, num_steps, dt)` -- Generate trajectory through time evolution.
- `HamiltonianNeuralNetwork.get_hamiltonian` (method) `hamiltonian_mbl.py:474` `def get_hamiltonian(self, q, p)` -- Compute total Hamiltonian.
- `HamiltonianNeuralNetwork.get_coefficients` (method) `hamiltonian_mbl.py:486` `def get_coefficients(self)`
- `HamiltonianNeuralNetwork.get_flat_parameters` (method) `hamiltonian_mbl.py:496` `def get_flat_parameters(self)` -- Returns all parameters flattened for Hamiltonian construction.
- `HamiltonianNeuralNetwork.construct_hessian_approximation` (method) `hamiltonian_mbl.py:503` `def construct_hessian_approximation(self, max_dim, method)` -- MÉTODO CORREGIDO - No usa 65GB de RAM.
- `HamiltonianDataset.__init__` (method) `hamiltonian_mbl.py:561` `def __init__(self, grid_size, num_samples, device)`
- `HamiltonianDataset.generate_harmonic_oscillator` (method) `hamiltonian_mbl.py:567` `def generate_harmonic_oscillator(self, omega)` -- Generate harmonic oscillator initial conditions.
- `HamiltonianDataset.generate_double_well` (method) `hamiltonian_mbl.py:591` `def generate_double_well(self, barrier_height)` -- Generate double-well potential trajectories.
- `LevelSpacingRatioCalculator.__init__` (method) `hamiltonian_mbl.py:616` `def __init__(self, config)`
- `LevelSpacingRatioCalculator.calculate` (method) `hamiltonian_mbl.py:619` `def calculate(self, model)` -- Calculate level spacing statistics from model weights.
- `ParticipationRatioCalculator.__init__` (method) `hamiltonian_mbl.py:725` `def __init__(self, config)`
- `ParticipationRatioCalculator.calculate` (method) `hamiltonian_mbl.py:728` `def calculate(self, model)` -- Calculate participation ratios for all weight layers.
- `SyntheticPlanckConstantCalculator.__init__` (method) `hamiltonian_mbl.py:807` `def __init__(self, config)`
- `SyntheticPlanckConstantCalculator.calculate` (method) `hamiltonian_mbl.py:810` `def calculate(self, participation_ratio, energy_gap)` -- Calculate synthetic Planck's constant.
- `SyntheticPlanckConstantCalculator.calculate_from_model` (method) `hamiltonian_mbl.py:820` `def calculate_from_model(self, model, level_spacing_results, pr_results)` -- Comprehensive calculation from model and previous analyses.
- `DiscretizationDialAnalyzer.__init__` (method) `hamiltonian_mbl.py:849` `def __init__(self, config)`
- `DiscretizationDialAnalyzer.calculate_base_discretization` (method) `hamiltonian_mbl.py:853` `def calculate_base_discretization(self, model)` -- Calculate the base discretization level from weight rounding error.
- `DiscretizationDialAnalyzer.analyze_robustness` (method) `hamiltonian_mbl.py:877` `def analyze_robustness(self, model, noise_levels)` -- Test robustness by applying noise and measuring gap collapse.
- `PurityIndexCalculator.__init__` (method) `hamiltonian_mbl.py:950` `def __init__(self, config)`
- `PurityIndexCalculator.calculate` (method) `hamiltonian_mbl.py:953` `def calculate(self, model)`
- `EffectiveTemperatureCalculator.__init__` (method) `hamiltonian_mbl.py:1007` `def __init__(self, config)`
- `EffectiveTemperatureCalculator.calculate` (method) `hamiltonian_mbl.py:1010` `def calculate(self, loss_history)`
- `KrylovComplexityCalculator.__init__` (method) `hamiltonian_mbl.py:1054` `def __init__(self, config)`
- `KrylovComplexityCalculator.calculate` (method) `hamiltonian_mbl.py:1057` `def calculate(self, model)` -- Calculate Krylov complexity from model dynamics.
- `CrystallinityIndexCalculator.__init__` (method) `hamiltonian_mbl.py:1095` `def __init__(self, config)`
- `CrystallinityIndexCalculator.calculate` (method) `hamiltonian_mbl.py:1098` `def calculate(self, model)` -- Calculate crystallinity index from weight spectra.
- `ResilienceSpectrometer.__init__` (method) `hamiltonian_mbl.py:1150` `def __init__(self, config)`
- `ResilienceSpectrometer.measure` (method) `hamiltonian_mbl.py:1153` `def measure(self, model)` -- Comprehensive resilience measurement.
- `PhaseClassifier.__init__` (method) `hamiltonian_mbl.py:1246` `def __init__(self, config)`
- `PhaseClassifier.classify` (method) `hamiltonian_mbl.py:1249` `def classify(self, alpha, temperature)`
- `CheckpointMigrator.__init__` (method) `hamiltonian_mbl.py:1273` `def __init__(self, arch_config)`
- `CheckpointMigrator.migrate` (method) `hamiltonian_mbl.py:1277` `def migrate(self, raw_data, device)`
- `MBLCheckpointManager.__init__` (method) `hamiltonian_mbl.py:1315` `def __init__(self, config, arch_config)`
- `MBLCheckpointManager.should_save_checkpoint` (method) `hamiltonian_mbl.py:1322` `def should_save_checkpoint(self)` -- Check if 5 minutes have elapsed since last checkpoint.
- `MBLCheckpointManager.save_checkpoint` (method) `hamiltonian_mbl.py:1328` `def save_checkpoint(self, model, epoch, metrics, loss_history, checkpoint_dir)` -- Save checkpoint with all MBL metrics.
- `MBLCheckpointManager.load_checkpoint` (method) `hamiltonian_mbl.py:1361` `def load_checkpoint(self, path)` -- Load checkpoint with automatic device placement and migration.
- `HamiltonianMBLMetricsCollector.__init__` (method) `hamiltonian_mbl.py:1392` `def __init__(self, config)`
- `HamiltonianMBLMetricsCollector.collect` (method) `hamiltonian_mbl.py:1405` `def collect(self, model, loss, epoch, loss_history, step)` -- Collect core metrics for the current training state.
- `HamiltonianMBLMetricsCollector.collect_comprehensive` (method) `hamiltonian_mbl.py:1493` `def collect_comprehensive(self, model, loss, epoch, loss_history, step)` -- Collect comprehensive metrics including expensive calculations.
- `HamiltonianTrainer.__init__` (method) `hamiltonian_mbl.py:1545` `def __init__(self, model, arch_config, mbl_config, train_config)`
- `HamiltonianTrainer.train_step` (method) `hamiltonian_mbl.py:1571` `def train_step(self, q_batch, p_batch, q_target, p_target)` -- Single training step with Hamiltonian loss.
- `HamiltonianTrainer.train_epoch` (method) `hamiltonian_mbl.py:1603` `def train_epoch(self, dataset, epoch)` -- Train for one epoch with MBL monitoring.
- `HamiltonianTrainer.train` (method) `hamiltonian_mbl.py:1672` `def train(self, dataset, num_epochs)` -- Full training loop.
- `HamiltonianCheckpointAnalyzer.__init__` (method) `hamiltonian_mbl.py:1713` `def __init__(self, checkpoint_path, arch_config, mbl_config)`
- `HamiltonianCheckpointAnalyzer.analyze` (method) `hamiltonian_mbl.py:1763` `def analyze(self)` -- Perform complete MBL analysis.
- `HamiltonianMBLPipeline.__init__` (method) `hamiltonian_mbl.py:1878` `def __init__(self, arch_config, mbl_config)`
- `HamiltonianMBLPipeline.process_checkpoint` (method) `hamiltonian_mbl.py:1882` `def process_checkpoint(self, checkpoint_path, output_dir)` -- Process single checkpoint and save results.
- `HamiltonianMBLPipeline.process_directory` (method) `hamiltonian_mbl.py:1901` `def process_directory(self, checkpoint_dir, n_latest, output_dir)` -- Process multiple checkpoints from directory.
- `HamiltonianMBLPipeline.generate_summary` (method) `hamiltonian_mbl.py:1941` `def generate_summary(self, all_results, output_dir)` -- Generate aggregate summary report.
- `HamiltonianMBLPipeline.main` (method) `hamiltonian_mbl.py:2031` `def main()`

## mining_seeds.py
- `Config.set_seed` (method) `mining_seeds.py:88` `def set_seed(seed)`
- `Config.setup_logger` (method) `mining_seeds.py:96` `def setup_logger(name, level)`
- `IAnalysisStrategy.analyze` (method) `mining_seeds.py:111` `def analyze(self, model)`
- `IMetricsCalculator.compute` (method) `mining_seeds.py:117` `def compute(self, model)`
- `HamiltonianOperator.__init__` (method) `mining_seeds.py:124` `def __init__(self, grid_size)`
- `HamiltonianOperator.apply` (method) `mining_seeds.py:135` `def apply(self, field)`
- `HamiltonianOperator.time_evolution` (method) `mining_seeds.py:140` `def time_evolution(self, field, dt)`
- `FastDataset.__init__` (method) `mining_seeds.py:149` `def __init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)`
- `FastDataset.get_val_batch` (method) `mining_seeds.py:199` `def get_val_batch(self)`
- `SpectralLayer.__init__` (method) `mining_seeds.py:206` `def __init__(self, channels, grid_size)`
- `SpectralLayer.forward` (method) `mining_seeds.py:219` `def forward(self, x)`
- `SimpleHamiltonianNet.__init__` (method) `mining_seeds.py:258` `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `SimpleHamiltonianNet.forward` (method) `mining_seeds.py:279` `def forward(self, x)`
- `LocalComplexityAnalyzer.compute_local_complexity` (method) `mining_seeds.py:295` `def compute_local_complexity(weights, epsilon)` -- Compute Local Complexity (LC) metric for weight matrix.
- `SuperpositionAnalyzer.compute_superposition` (method) `mining_seeds.py:312` `def compute_superposition(weights)` -- Compute Superposition (SP) metric for weight matrix.
- `CrystallographyMetrics.compute_kappa` (method) `mining_seeds.py:342` `def compute_kappa(model, dataloader, num_batches)`
- `CrystallographyMetrics.compute_discretization_margin` (method) `mining_seeds.py:378` `def compute_discretization_margin(coeffs)`
- `CrystallographyMetrics.compute_alpha_purity` (method) `mining_seeds.py:387` `def compute_alpha_purity(coeffs)`
- `CrystallographyMetrics.compute_kappa_quantum` (method) `mining_seeds.py:394` `def compute_kappa_quantum(coeffs, hbar)`
- `CrystallographyMetrics.compute_poynting_vector` (method) `mining_seeds.py:411` `def compute_poynting_vector(coeffs)`
- `CrystallographyMetrics.compute_all_metrics` (method) `mining_seeds.py:426` `def compute_all_metrics(model, dataloader)`
- `ThermodynamicMetrics.compute_effective_temperature` (method) `mining_seeds.py:446` `def compute_effective_temperature(gradient_buffer, learning_rate)`
- `ThermodynamicMetrics.compute_specific_heat` (method) `mining_seeds.py:460` `def compute_specific_heat(loss_history, temp_history, cv_threshold)`
- `SpectroscopyMetrics.compute_weight_diffraction` (method) `mining_seeds.py:472` `def compute_weight_diffraction(coeffs)`
- `CheckpointManager.__init__` (method) `mining_seeds.py:501` `def __init__(self, interval_minutes, max_checkpoints)`
- `CheckpointManager.should_save_checkpoint` (method) `mining_seeds.py:509` `def should_save_checkpoint(self)`
- `CheckpointManager.save_checkpoint` (method) `mining_seeds.py:514` `def save_checkpoint(self, model, optimizer, epoch, metrics)`
- `TrainingMonitor.__init__` (method) `mining_seeds.py:574` `def __init__(self)`
- `TrainingMonitor.update_metrics` (method) `mining_seeds.py:594` `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat...`
- `GlassStopper.__init__` (method) `mining_seeds.py:612` `def __init__(self, patience_epochs)`
- `GlassStopper.should_stop` (method) `mining_seeds.py:616` `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)` -- Check if the system is in glass state and should stop mining.
- `GlassStopper.train_with_early_glass_stop` (method) `mining_seeds.py:670` `def train_with_early_glass_stop(model, optimizer, seed, epochs)` -- Train model with early stopping for glass detection.
- `GlassStopper.seed_miner` (method) `mining_seeds.py:803` `def seed_miner(total_attempts)` -- Mine for crystal seeds by trying sequential seeds.
- `GlassStopper.main` (method) `mining_seeds.py:856` `def main()`
- `BoltzmannAnalysisProgram.__init__` (method) `mining_seeds.py:880` `def __init__(self, checkpoint_path, results_dir)`
- `BoltzmannAnalysisProgram.load_and_analyze_checkpoint` (method) `mining_seeds.py:886` `def load_and_analyze_checkpoint(self)`
- `BoltzmannAnalysisProgram.dataloader` (method) `mining_seeds.py:903` `def dataloader()`

## plank.py
- `HBarCalculator.__init__` (method) `plank.py:29` `def __init__(self, checkpoint_path, device)`
- `HBarCalculator.calculate_all` (method) `plank.py:54` `def calculate_all(self)` -- Ejecuta todos los cálculos de ħ.
- `HBarCalculator.print_report` (method) `plank.py:170` `def print_report(self, results)` -- Imprime reporte formateado.
- `HBarCalculator.main` (method) `plank.py:213` `def main()`


Next: [API_p2.md](API_p2.md)
