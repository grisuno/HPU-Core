# API

## app.py

### compute_local_complexity `def compute_local_complexity(weights, epsilon)`
- Defined: `app.py:45`
- Doc: Compute Local Complexity (LC) metric for weight matrix.

### compute_superposition `def compute_superposition(weights)`
- Defined: `app.py:60`
- Doc: Compute Superposition (SP) metric for weight matrix.

### train_model `def train_model(grid_size, epochs, hidden_dim, num_spectral_layers, lr)`
- Defined: `app.py:260`
- Doc: Train the Hamiltonian operator model.

### main `def main()`
- Defined: `app.py:369`

### __init__ `def __init__(self, grid_size, hidden_dim, num_spectral_layers, target_accuracy, learning_rate)`
- Defined: `app.py:30`

### __init__ `def __init__(self, grid_size)`
- Defined: `app.py:91`

### _precompute_spectral_operators `def _precompute_spectral_operators(self)`
- Defined: `app.py:95`

### apply `def apply(self, field)`
- Defined: `app.py:102`

### time_evolution `def time_evolution(self, field, dt)`
- Defined: `app.py:107`

### __init__ `def __init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)`
- Defined: `app.py:116`

### __len__ `def __len__(self)`
- Defined: `app.py:160`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `app.py:163`

### get_val_batch `def get_val_batch(self)`
- Defined: `app.py:166`

### __init__ `def __init__(self, channels, grid_size)`
- Defined: `app.py:173`

### forward `def forward(self, x)`
- Defined: `app.py:186`

### __init__ `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- Defined: `app.py:225`

### forward `def forward(self, x)`
- Defined: `app.py:246`

## audio/audio_io.py

### __init__ `def __init__(self, config, device)`
- Defined: `audio/audio_io.py:41`
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### load_audio `def load_audio(self, file_path)`
- Defined: `audio/audio_io.py:57`
- Doc: Load an audio file and convert to mono at the target sample rate.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### waveform_to_stft_complex `def waveform_to_stft_complex(self, waveform)`
- Defined: `audio/audio_io.py:80`
- Doc: Compute the complex STFT of a waveform.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### stft_complex_to_waveform `def stft_complex_to_waveform(self, stft_complex)`
- Defined: `audio/audio_io.py:105`
- Doc: Reconstruct waveform from complex STFT via inverse STFT.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### stft_to_magnitude_phase `def stft_to_magnitude_phase(self, stft_complex)`
- Defined: `audio/audio_io.py:130`
- Doc: Decompose complex STFT into magnitude and phase.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### magnitude_phase_to_stft `def magnitude_phase_to_stft(self, magnitude, phase)`
- Defined: `audio/audio_io.py:146`
- Doc: Recombine magnitude and phase into complex STFT.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### stft_magnitude_to_model_input `def stft_magnitude_to_model_input(self, magnitude)`
- Defined: `audio/audio_io.py:161`
- Doc: Prepare STFT magnitude for input to the Hamiltonian network.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### model_output_to_stft_magnitude `def model_output_to_stft_magnitude(self, model_output, original_magnitude)`
- Defined: `audio/audio_io.py:186`
- Doc: Convert model output (energy mask in [0, 1]) back to STFT magnitude scale.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### waveform_to_mel_spectrogram `def waveform_to_mel_spectrogram(self, waveform)`
- Defined: `audio/audio_io.py:206`
- Doc: Convert waveform to normalized mel spectrogram (for visualization only).
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### save_audio `def save_audio(self, waveform, file_path, sample_rate)`
- Defined: `audio/audio_io.py:229`
- Doc: Save a waveform tensor to an audio file.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### get_spectrogram_db_range `def get_spectrogram_db_range(self, waveform)`
- Defined: `audio/audio_io.py:249`
- Doc: Compute the dB range of a waveform's mel spectrogram.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

## audio/audios.py

### main `def main()`
- Defined: `audio/audios.py:742`
- Doc: Entry point with argument parsing.
- Depends on: `audio/experiment2.py`

### segment_samples `def segment_samples(self)`
- Defined: `audio/audios.py:89`
- Doc: Calculate segment length in samples.
- Depends on: `audio/experiment2.py`

### freq_bins `def freq_bins(self)`
- Defined: `audio/audios.py:94`
- Doc: Calculate frequency bins for real FFT.
- Depends on: `audio/experiment2.py`

### read_segment `def read_segment(self)`
- Defined: `audio/audios.py:107`
- Doc: Read audio segment. Returns None when exhausted.
- Depends on: `audio/experiment2.py`

### get_properties `def get_properties(self)`
- Defined: `audio/audios.py:112`
- Doc: Return audio properties.
- Depends on: `audio/experiment2.py`

### close `def close(self)`
- Defined: `audio/audios.py:117`
- Doc: Release resources.
- Depends on: `audio/experiment2.py`

### evolve `def evolve(self, field_state)`
- Defined: `audio/audios.py:126`
- Doc: Evolve field state through Hamiltonian dynamics.
- Depends on: `audio/experiment2.py`

### record `def record(self, metrics)`
- Defined: `audio/audios.py:135`
- Doc: Record metric values.
- Depends on: `audio/experiment2.py`

### get_summary `def get_summary(self)`
- Defined: `audio/audios.py:140`
- Doc: Return aggregated metrics.
- Depends on: `audio/experiment2.py`

### resample `def resample(audio, orig_sr, target_sr)`
- Defined: `audio/audios.py:155`
- Doc: Resample audio from orig_sr to target_sr using polyphase filtering.
- Depends on: `audio/experiment2.py`

### load_wav_with_resample `def load_wav_with_resample(file_path, target_sr)`
- Defined: `audio/audios.py:173`
- Doc: Load WAV file and resample to target sample rate.
- Depends on: `audio/experiment2.py`

### __init__ `def __init__(self, file_path, config)`
- Defined: `audio/audios.py:210`
- Depends on: `audio/experiment2.py`

### _validate_and_load `def _validate_and_load(self)`
- Defined: `audio/audios.py:220`
- Doc: Validate file format and load with automatic resampling.
- Depends on: `audio/experiment2.py`

### read_segment `def read_segment(self)`
- Defined: `audio/audios.py:244`
- Doc: Read next audio segment.
- Depends on: `audio/experiment2.py`

### get_properties `def get_properties(self)`
- Defined: `audio/audios.py:261`
- Doc: Return audio file properties.
- Depends on: `audio/experiment2.py`

### close `def close(self)`
- Defined: `audio/audios.py:273`
- Doc: Release resources.
- Depends on: `audio/experiment2.py`

### __init__ `def __init__(self, config)`
- Defined: `audio/audios.py:284`
- Depends on: `audio/experiment2.py`

### record `def record(self, metrics)`
- Defined: `audio/audios.py:289`
- Doc: Record comprehensive metrics.
- Depends on: `audio/experiment2.py`

### get_summary `def get_summary(self)`
- Defined: `audio/audios.py:298`
- Doc: Return statistical summary of all metrics.
- Depends on: `audio/experiment2.py`

### export_to_json `def export_to_json(self, path)`
- Defined: `audio/audios.py:320`
- Doc: Export full history to JSON.
- Depends on: `audio/experiment2.py`

### __init__ `def __init__(self, model, config, checkpoint_dir)`
- Defined: `audio/audios.py:331`
- Depends on: `audio/experiment2.py`

### check_and_save `def check_and_save(self, force)`
- Defined: `audio/audios.py:344`
- Doc: Check if checkpoint interval elapsed and save if necessary.
- Depends on: `audio/experiment2.py`

### _save_checkpoint `def _save_checkpoint(self)`
- Defined: `audio/audios.py:357`
- Doc: Atomic checkpoint save.
- Depends on: `audio/experiment2.py`

### __init__ `def __init__(self, config)`
- Defined: `audio/audios.py:393`
- Depends on: `audio/experiment2.py`

### waveform_to_field `def waveform_to_field(self, waveform)`
- Defined: `audio/audios.py:396`
- Doc: Convert 1D audio to 2D field representation via STFT.
- Depends on: `audio/experiment2.py`

### field_to_waveform `def field_to_waveform(self, field, original_length)`
- Defined: `audio/audios.py:432`
- Doc: Reconstruct waveform from 2D field representation.
- Depends on: `audio/experiment2.py`

### _forward_spectrogram `def _forward_spectrogram(self, x)`
- Defined: `audio/audios.py:458`
- Doc: Compute magnitude spectrogram.
- Depends on: `audio/experiment2.py`

### _inverse_spectrogram `def _inverse_spectrogram(self, spectrogram)`
- Defined: `audio/audios.py:463`
- Doc: Griffin-Lim inverse.
- Depends on: `audio/experiment2.py`

### __init__ `def __init__(self, config, model, source)`
- Defined: `audio/audios.py:475`
- Depends on: `audio/experiment2.py`

### load_model_weights `def load_model_weights(self, path)`
- Defined: `audio/audios.py:504`
- Doc: Load pretrained Hamiltonian operator desde safetensors.
- Depends on: `audio/experiment2.py`

### attach_source `def attach_source(self, source)`
- Defined: `audio/audios.py:513`
- Doc: Attach audio source via dependency injection.
- Depends on: `audio/experiment2.py`

### process_stream `def process_stream(self)`
- Defined: `audio/audios.py:517`
- Doc: Process audio stream through Hamiltonian perception.
- Depends on: `audio/experiment2.py`

### _process_single_segment `def _process_single_segment(self, waveform, index)`
- Defined: `audio/audios.py:592`
- Doc: Process single audio segment and return metrics.
- Depends on: `audio/experiment2.py`

### _calculate_phase_entropy `def _calculate_phase_entropy(self, phase_map)`
- Defined: `audio/audios.py:684`
- Doc: Calculate topological entropy from phase distribution.
- Depends on: `audio/experiment2.py`

### _render_epiphenomena `def _render_epiphenomena(self, amplitude, phase, action)`
- Defined: `audio/audios.py:692`
- Doc: Render three epiphenomenal visualizations.
- Depends on: `audio/experiment2.py`

### export_metrics `def export_metrics(self, path)`
- Defined: `audio/audios.py:729`
- Doc: Export comprehensive metrics to file.
- Depends on: `audio/experiment2.py`

### force_checkpoint `def force_checkpoint(self)`
- Defined: `audio/audios.py:733`
- Doc: Force immediate checkpoint save.
- Depends on: `audio/experiment2.py`

## audio/checkpoint_manager.py

### __init__ `def __init__(self, config)`
- Defined: `audio/checkpoint_manager.py:34`
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### should_save_checkpoint `def should_save_checkpoint(self)`
- Defined: `audio/checkpoint_manager.py:41`
- Doc: Check if enough time has elapsed since the last checkpoint.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### save_checkpoint `def save_checkpoint(self, model, optimizer, scheduler, epoch, step, metrics, current_loss)`
- Defined: `audio/checkpoint_manager.py:46`
- Doc: Save the current model state and training metadata.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### load_checkpoint `def load_checkpoint(self, model, load_best)`
- Defined: `audio/checkpoint_manager.py:98`
- Doc: Load a model checkpoint and return training metadata.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### best_loss `def best_loss(self)`
- Defined: `audio/checkpoint_manager.py:156`
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

## audio/config.py

### validate `def validate(self)`
- Defined: `audio/config.py:62`
- Doc: Ensure architectural coherence.
- Imported by: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/inference.py`, `audio/losses.py`, `audio/main.py`, `audio/metrics.py`, `audio/model.py`, `audio/trainer.py`, `audio/visualization.py`

### checkpoint_path `def checkpoint_path(self)`
- Defined: `audio/config.py:121`
- Imported by: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/inference.py`, `audio/losses.py`, `audio/main.py`, `audio/metrics.py`, `audio/model.py`, `audio/trainer.py`, `audio/visualization.py`

### best_model_path `def best_model_path(self)`
- Defined: `audio/config.py:127`
- Imported by: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/inference.py`, `audio/losses.py`, `audio/main.py`, `audio/metrics.py`, `audio/model.py`, `audio/trainer.py`, `audio/visualization.py`

### metadata_path `def metadata_path(self)`
- Defined: `audio/config.py:131`
- Imported by: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/inference.py`, `audio/losses.py`, `audio/main.py`, `audio/metrics.py`, `audio/model.py`, `audio/trainer.py`, `audio/visualization.py`

### validate_all `def validate_all(self)`
- Defined: `audio/config.py:198`
- Doc: Run validation on all sub-configurations.
- Imported by: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/inference.py`, `audio/losses.py`, `audio/main.py`, `audio/metrics.py`, `audio/model.py`, `audio/trainer.py`, `audio/visualization.py`

### ensure_directories `def ensure_directories(self)`
- Defined: `audio/config.py:206`
- Doc: Create required output directories if they do not exist.
- Imported by: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/inference.py`, `audio/losses.py`, `audio/main.py`, `audio/metrics.py`, `audio/model.py`, `audio/trainer.py`, `audio/visualization.py`

## audio/experiment2.py

### main `def main()`
- Defined: `audio/experiment2.py:1334`
- Imported by: `audio/audios.py`

### set_seed `def set_seed(seed)`
- Defined: `audio/experiment2.py:76`
- Imported by: `audio/audios.py`

### create_logger `def create_logger(name, level)`
- Defined: `audio/experiment2.py:86`
- Imported by: `audio/audios.py`

### analyze `def analyze(self, model)`
- Defined: `audio/experiment2.py:101`
- Imported by: `audio/audios.py`

### compute `def compute(self, model)`
- Defined: `audio/experiment2.py:107`
- Imported by: `audio/audios.py`

### __init__ `def __init__(self, grid_size)`
- Defined: `audio/experiment2.py:112`
- Imported by: `audio/audios.py`

### _precompute_spectral_operators `def _precompute_spectral_operators(self)`
- Defined: `audio/experiment2.py:116`
- Imported by: `audio/audios.py`

### apply `def apply(self, field)`
- Defined: `audio/experiment2.py:122`
- Imported by: `audio/audios.py`

### time_evolution `def time_evolution(self, field, dt)`
- Defined: `audio/experiment2.py:127`
- Imported by: `audio/audios.py`

### __init__ `def __init__(self, num_samples, grid_size, time_steps, dt, train_ratio)`
- Defined: `audio/experiment2.py:134`
- Imported by: `audio/audios.py`

### __len__ `def __len__(self)`
- Defined: `audio/experiment2.py:173`
- Imported by: `audio/audios.py`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `audio/experiment2.py:176`
- Imported by: `audio/audios.py`

### get_validation_batch `def get_validation_batch(self)`
- Defined: `audio/experiment2.py:179`
- Imported by: `audio/audios.py`

### __init__ `def __init__(self, channels, grid_size)`
- Defined: `audio/experiment2.py:184`
- Imported by: `audio/audios.py`

### forward `def forward(self, x)`
- Defined: `audio/experiment2.py:195`
- Imported by: `audio/audios.py`

### __init__ `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- Defined: `audio/experiment2.py:228`
- Imported by: `audio/audios.py`

### forward `def forward(self, x)`
- Defined: `audio/experiment2.py:243`
- Imported by: `audio/audios.py`

### compute_local_complexity `def compute_local_complexity(weights, epsilon)`
- Defined: `audio/experiment2.py:259`
- Imported by: `audio/audios.py`

### compute_superposition `def compute_superposition(weights)`
- Defined: `audio/experiment2.py:275`
- Imported by: `audio/audios.py`

### compute `def compute(self, model, val_x, val_y)`
- Defined: `audio/experiment2.py:303`
- Doc: Implementación de interfaz IMetricsCalculator.
- Imported by: `audio/audios.py`

### compute_gradient_covariance_kappa `def compute_gradient_covariance_kappa(model, dataloader, num_batches)`
- Defined: `audio/experiment2.py:311`
- Imported by: `audio/audios.py`

### compute_discretization_margin_from_state_dict `def compute_discretization_margin_from_state_dict(model)`
- Defined: `audio/experiment2.py:348`
- Doc: Calcula el margen de discretización desde los parámetros del modelo.
- Imported by: `audio/audios.py`

### compute_discretization_margin `def compute_discretization_margin(coeffs)`
- Defined: `audio/experiment2.py:361`
- Doc: Calcula el margen de discretización desde un diccionario de coeficientes.
- Imported by: `audio/audios.py`

### compute_alpha_purity_from_model `def compute_alpha_purity_from_model(model)`
- Defined: `audio/experiment2.py:373`
- Doc: Calcula el índice de pureza alpha directamente desde el modelo.
- Imported by: `audio/audios.py`

### compute_alpha_purity `def compute_alpha_purity(coeffs)`
- Defined: `audio/experiment2.py:383`
- Doc: Calcula el índice de pureza alpha desde un diccionario de coeficientes.
- Imported by: `audio/audios.py`

### compute_kappa `def compute_kappa(model, val_x, val_y, num_batches)`
- Defined: `audio/experiment2.py:393`
- Doc: Número de condición de la matriz de covarianza de gradientes.
- Imported by: `audio/audios.py`

### compute_kappa_quantum `def compute_kappa_quantum(model, hbar)`
- Defined: `audio/experiment2.py:464`
- Doc: Versión del cálculo cuántico de kappa que opera directamente sobre el modelo.
- Imported by: `audio/audios.py`

### compute_kappa_quantum_from_coeffs `def compute_kappa_quantum_from_coeffs(coeffs, hbar)`
- Defined: `audio/experiment2.py:492`
- Doc: Versión del cálculo cuántico de kappa desde diccionario de coeficientes.
- Imported by: `audio/audios.py`

### _compute_crystallography_metrics `def _compute_crystallography_metrics(self, model, val_x, val_y)`
- Defined: `audio/experiment2.py:511`
- Doc: Métricas cristalográficas con aislamiento completo de errores.
- Imported by: `audio/audios.py`

### _check_weight_integrity `def _check_weight_integrity(self, model)`
- Defined: `audio/experiment2.py:539`
- Doc: Verifica integridad de pesos: NaN, Inf, y estadísticas básicas.
- Imported by: `audio/audios.py`

### compute_poynting_vector `def compute_poynting_vector(model)`
- Defined: `audio/experiment2.py:603`
- Doc: Vector de Poynting: flujo de energía en el espacio de parámetros.
- Imported by: `audio/audios.py`

### compute_all_metrics `def compute_all_metrics(model, val_x, val_y)`
- Defined: `audio/experiment2.py:679`
- Doc: Calcula todas las métricas cristalográficas con manejo de errores.
- Imported by: `audio/audios.py`

### compute `def compute(self, model, gradient_buffer, learning_rate, loss_history, temp_history)`
- Defined: `audio/experiment2.py:738`
- Imported by: `audio/audios.py`

### compute_effective_temperature `def compute_effective_temperature(gradient_buffer, learning_rate)`
- Defined: `audio/experiment2.py:747`
- Imported by: `audio/audios.py`

### compute_specific_heat `def compute_specific_heat(loss_history, temp_history, cv_threshold)`
- Defined: `audio/experiment2.py:760`
- Imported by: `audio/audios.py`

### compute `def compute(self, model)`
- Defined: `audio/experiment2.py:771`
- Imported by: `audio/audios.py`

### compute_weight_diffraction `def compute_weight_diffraction(coeffs)`
- Defined: `audio/experiment2.py:776`
- Imported by: `audio/audios.py`

### _compute_spectral_entropy `def _compute_spectral_entropy(power_spectrum)`
- Defined: `audio/experiment2.py:795`
- Imported by: `audio/audios.py`

### __init__ `def __init__(self, interval_minutes, max_checkpoints)`
- Defined: `audio/experiment2.py:805`
- Imported by: `audio/audios.py`

### should_save_checkpoint `def should_save_checkpoint(self)`
- Defined: `audio/experiment2.py:813`
- Imported by: `audio/audios.py`

### save_checkpoint `def save_checkpoint(self, model, optimizer, epoch, metrics)`
- Defined: `audio/experiment2.py:818`
- Imported by: `audio/audios.py`

### __init__ `def __init__(self)`
- Defined: `audio/experiment2.py:875`
- Imported by: `audio/audios.py`

### update_metrics `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat, poynting_magnitude)`
- Defined: `audio/experiment2.py:895`
- Imported by: `audio/audios.py`

### __init__ `def __init__(self, patience_epochs)`
- Defined: `audio/experiment2.py:913`
- Imported by: `audio/audios.py`

### should_stop `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)`
- Defined: `audio/experiment2.py:918`
- Imported by: `audio/audios.py`

### is_crystal_formed `def is_crystal_formed(self, lc, sp, kappa, delta, temp, cv)`
- Defined: `audio/experiment2.py:963`
- Imported by: `audio/audios.py`

### __init__ `def __init__(self, model, optimizer, device, logger)`
- Defined: `audio/experiment2.py:974`
- Imported by: `audio/audios.py`

### train_epoch `def train_epoch(self, dataloader, epoch)`
- Defined: `audio/experiment2.py:997`
- Imported by: `audio/audios.py`

### validate `def validate(self, val_x, val_y)`
- Defined: `audio/experiment2.py:1027`
- Imported by: `audio/audios.py`

### compute_weight_metrics `def compute_weight_metrics(self)`
- Defined: `audio/experiment2.py:1040`
- Imported by: `audio/audios.py`

### execute_training `def execute_training(self, dataloader, val_x, val_y, epochs, seed, early_stopping)`
- Defined: `audio/experiment2.py:1056`
- Imported by: `audio/audios.py`

### __init__ `def __init__(self, max_attempts)`
- Defined: `audio/experiment2.py:1114`
- Imported by: `audio/audios.py`

### mine `def mine(self)`
- Defined: `audio/experiment2.py:1118`
- Imported by: `audio/audios.py`

### __init__ `def __init__(self, seed, epochs, grid_size, hidden_dim, num_spectral_layers, learning_rate)`
- Defined: `audio/experiment2.py:1165`
- Imported by: `audio/audios.py`

### run `def run(self)`
- Defined: `audio/experiment2.py:1175`
- Imported by: `audio/audios.py`

### __init__ `def __init__(self, checkpoint_path, results_dir)`
- Defined: `audio/experiment2.py:1224`
- Imported by: `audio/audios.py`

### analyze `def analyze(self)`
- Defined: `audio/experiment2.py:1230`
- Imported by: `audio/audios.py`

### __init__ `def __init__(self)`
- Defined: `audio/experiment2.py:1276`
- Imported by: `audio/audios.py`

### _create_argument_parser `def _create_argument_parser(self)`
- Defined: `audio/experiment2.py:1280`
- Imported by: `audio/audios.py`

### run `def run(self)`
- Defined: `audio/experiment2.py:1294`
- Imported by: `audio/audios.py`

### safe_compute `def safe_compute(func)`
- Defined: `audio/experiment2.py:694`
- Imported by: `audio/audios.py`

## audio/inference.py

### __init__ `def __init__(self, config, load_best)`
- Defined: `audio/inference.py:42`
- Depends on: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/config.py`, `audio/metrics.py`, `audio/model.py`, `audio/visualization.py`
- Imported by: `audio/main.py`

### analyze_audio `def analyze_audio(self, audio_file_path, output_prefix)`
- Defined: `audio/inference.py:73`
- Doc: Perform complete Hamiltonian analysis on an audio file.
- Depends on: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/config.py`, `audio/metrics.py`, `audio/model.py`, `audio/visualization.py`
- Imported by: `audio/main.py`

### _compute_energy_mask_patched `def _compute_energy_mask_patched(self, model_input)`
- Defined: `audio/inference.py:158`
- Doc: Compute energy mask over the full STFT magnitude, processing
- Depends on: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/config.py`, `audio/metrics.py`, `audio/model.py`, `audio/visualization.py`
- Imported by: `audio/main.py`

### _extract_hamiltonian_fields_patched `def _extract_hamiltonian_fields_patched(self, model_input)`
- Defined: `audio/inference.py:209`
- Doc: Extract Hamiltonian fields over full STFT magnitude with patching.
- Depends on: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/config.py`, `audio/metrics.py`, `audio/model.py`, `audio/visualization.py`
- Imported by: `audio/main.py`

### _compute_inference_metrics `def _compute_inference_metrics(self, original_magnitude, reconstructed_magnitude, original_stft)`
- Defined: `audio/inference.py:270`
- Doc: Compute all inference-time metrics on the STFT domain.
- Depends on: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/config.py`, `audio/metrics.py`, `audio/model.py`, `audio/visualization.py`
- Imported by: `audio/main.py`

### _print_inference_metrics `def _print_inference_metrics(self)`
- Defined: `audio/inference.py:288`
- Doc: Print all computed inference metrics.
- Depends on: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/config.py`, `audio/metrics.py`, `audio/model.py`, `audio/visualization.py`
- Imported by: `audio/main.py`

## audio/losses.py

### __init__ `def __init__(self, config)`
- Defined: `audio/losses.py:37`
- Depends on: `audio/config.py`
- Imported by: `audio/trainer.py`

### compute_total_loss `def compute_total_loss(self, prediction, target, intermediates, model)`
- Defined: `audio/losses.py:41`
- Doc: Compute the complete weighted loss with all Hamiltonian terms.
- Depends on: `audio/config.py`
- Imported by: `audio/trainer.py`

### _compute_reconstruction_loss `def _compute_reconstruction_loss(self, prediction, target)`
- Defined: `audio/losses.py:94`
- Doc: MSE reconstruction loss between predicted and target spectrograms.
- Depends on: `audio/config.py`
- Imported by: `audio/trainer.py`

### _compute_energy_conservation_loss `def _compute_energy_conservation_loss(self, intermediates)`
- Defined: `audio/losses.py:100`
- Doc: Penalize energy drift across layers.
- Depends on: `audio/config.py`
- Imported by: `audio/trainer.py`

### _compute_symplectic_loss `def _compute_symplectic_loss(self, intermediates)`
- Defined: `audio/losses.py:122`
- Doc: Penalize violation of symplectic structure.
- Depends on: `audio/config.py`
- Imported by: `audio/trainer.py`

### _compute_spectral_consistency_loss `def _compute_spectral_consistency_loss(self, prediction, target)`
- Defined: `audio/losses.py:145`
- Doc: Penalize spectral divergence in frequency domain.
- Depends on: `audio/config.py`
- Imported by: `audio/trainer.py`

### _compute_phase_coherence_loss `def _compute_phase_coherence_loss(self, prediction, target)`
- Defined: `audio/losses.py:161`
- Doc: Penalize phase misalignment between prediction and target.
- Depends on: `audio/config.py`
- Imported by: `audio/trainer.py`

### _compute_action_minimization_loss `def _compute_action_minimization_loss(self, intermediates)`
- Defined: `audio/losses.py:177`
- Doc: Principle of least action: minimize the total action
- Depends on: `audio/config.py`
- Imported by: `audio/trainer.py`

### _compute_liouville_loss `def _compute_liouville_loss(self, intermediates)`
- Defined: `audio/losses.py:192`
- Doc: Liouville theorem: phase space volume should be preserved.
- Depends on: `audio/config.py`
- Imported by: `audio/trainer.py`

### _compute_hamiltonian_constraint_loss `def _compute_hamiltonian_constraint_loss(self, intermediates)`
- Defined: `audio/losses.py:214`
- Doc: Hamilton's equations: dq/dt = dH/dp, dp/dt = -dH/dq.
- Depends on: `audio/config.py`
- Imported by: `audio/trainer.py`

## audio/main.py

### build_argument_parser `def build_argument_parser()`
- Defined: `audio/main.py:34`
- Doc: Construct the complete argument parser with all configurable parameters.
- Depends on: `audio/config.py`, `audio/inference.py`, `audio/trainer.py`
- Imported by: `test_grokkit.py`

### build_config_from_args `def build_config_from_args(args)`
- Defined: `audio/main.py:103`
- Doc: Construct the full configuration from parsed CLI arguments.
- Depends on: `audio/config.py`, `audio/inference.py`, `audio/trainer.py`
- Imported by: `test_grokkit.py`

### validate_audio_file `def validate_audio_file(file_path)`
- Defined: `audio/main.py:163`
- Doc: Validate that the audio file exists and has a supported extension.
- Depends on: `audio/config.py`, `audio/inference.py`, `audio/trainer.py`
- Imported by: `test_grokkit.py`

### print_configuration_banner `def print_configuration_banner(config, mode, audio_path)`
- Defined: `audio/main.py:175`
- Doc: Print a formatted configuration summary.
- Depends on: `audio/config.py`, `audio/inference.py`, `audio/trainer.py`
- Imported by: `test_grokkit.py`

### run_training `def run_training(args)`
- Defined: `audio/main.py:215`
- Doc: Execute the training pipeline.
- Depends on: `audio/config.py`, `audio/inference.py`, `audio/trainer.py`
- Imported by: `test_grokkit.py`

### run_inference `def run_inference(args)`
- Defined: `audio/main.py:226`
- Doc: Execute the inference pipeline.
- Depends on: `audio/config.py`, `audio/inference.py`, `audio/trainer.py`
- Imported by: `test_grokkit.py`

### main `def main()`
- Defined: `audio/main.py:236`
- Doc: Main entry point.
- Depends on: `audio/config.py`, `audio/inference.py`, `audio/trainer.py`
- Imported by: `test_grokkit.py`

## audio/metrics.py

### __init__ `def __init__(self, config)`
- Defined: `audio/metrics.py:36`
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### _initialize_history_buffers `def _initialize_history_buffers(self)`
- Defined: `audio/metrics.py:43`
- Doc: Pre-allocate deque buffers for each tracked metric.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### compute_hamiltonian_energy `def compute_hamiltonian_energy(self, q, p)`
- Defined: `audio/metrics.py:74`
- Doc: Compute the Hamiltonian H(q, p) = T(p) + V(q).
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### compute_symplectic_form `def compute_symplectic_form(self, q, p, dq, dp)`
- Defined: `audio/metrics.py:97`
- Doc: Compute the symplectic 2-form omega(dq, dp) = sum(dq_i ^ dp_i).
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### compute_liouville_measure `def compute_liouville_measure(self, jacobian)`
- Defined: `audio/metrics.py:125`
- Doc: Compute Liouville measure |det(J)| for the flow map Jacobian.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### compute_phase_space_volume `def compute_phase_space_volume(self, q, p)`
- Defined: `audio/metrics.py:153`
- Doc: Estimate phase space volume occupied by the state (q, p).
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### compute_action_integral `def compute_action_integral(self, q_trajectory, p_trajectory, dt)`
- Defined: `audio/metrics.py:181`
- Doc: Compute the action integral S = integral(L dt) along a trajectory.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### compute_poisson_bracket `def compute_poisson_bracket(self, f_values, g_values, q, p)`
- Defined: `audio/metrics.py:208`
- Doc: Estimate the Poisson bracket {f, g} = sum(df/dq * dg/dp - df/dp * dg/dq).
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### compute_spectral_entropy `def compute_spectral_entropy(self, spectrum)`
- Defined: `audio/metrics.py:238`
- Doc: Compute spectral entropy H = -sum(p_i * log(p_i)).
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### compute_reconstruction_snr `def compute_reconstruction_snr(self, original, reconstructed)`
- Defined: `audio/metrics.py:259`
- Doc: Compute Signal-to-Noise Ratio in dB.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### compute_spectral_convergence `def compute_spectral_convergence(self, original_spectrum, reconstructed_spectrum)`
- Defined: `audio/metrics.py:281`
- Doc: Compute spectral convergence metric.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### compute_phase_coherence `def compute_phase_coherence(self, phase_original, phase_reconstructed)`
- Defined: `audio/metrics.py:305`
- Doc: Compute phase coherence between original and reconstructed signals.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### compute_energy_drift `def compute_energy_drift(self, energy_initial, energy_current)`
- Defined: `audio/metrics.py:328`
- Doc: Compute relative energy drift from initial state.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### record_gradient_norm `def record_gradient_norm(self, model_parameters)`
- Defined: `audio/metrics.py:348`
- Doc: Compute and record the total gradient norm across all parameters.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### record_parameter_norm `def record_parameter_norm(self, model_parameters)`
- Defined: `audio/metrics.py:359`
- Doc: Compute and record the total parameter norm.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### record_learning_rate `def record_learning_rate(self, lr)`
- Defined: `audio/metrics.py:369`
- Doc: Record current learning rate.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### record_loss_component `def record_loss_component(self, name, value)`
- Defined: `audio/metrics.py:374`
- Doc: Record an individual loss component value.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### _record `def _record(self, metric_name, value)`
- Defined: `audio/metrics.py:379`
- Doc: Store a metric value in history and current snapshot.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### get_current_metrics `def get_current_metrics(self)`
- Defined: `audio/metrics.py:388`
- Doc: Return a snapshot of all current metric values.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### get_moving_averages `def get_moving_averages(self)`
- Defined: `audio/metrics.py:392`
- Doc: Compute moving averages for all tracked metrics.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### get_formatted_metrics_string `def get_formatted_metrics_string(self)`
- Defined: `audio/metrics.py:400`
- Doc: Format all current metrics into a human-readable string for progress bars.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### increment_step `def increment_step(self)`
- Defined: `audio/metrics.py:413`
- Doc: Advance the global step counter.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### step_count `def step_count(self)`
- Defined: `audio/metrics.py:418`
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### should_log `def should_log(self)`
- Defined: `audio/metrics.py:421`
- Doc: Determine if metrics should be logged at this step.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

## audio/model.py

### __init__ `def __init__(self, hidden_dim, kernel_base_height, kernel_base_width, init_std)`
- Defined: `audio/model.py:36`
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### forward `def forward(self, x)`
- Defined: `audio/model.py:51`
- Doc: Apply one step of Hamiltonian spectral evolution via RFFT2.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### evolve_complex `def evolve_complex(self, x, target_height, target_width)`
- Defined: `audio/model.py:85`
- Doc: Full complex FFT evolution for amplitude and phase extraction.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### evolve_real `def evolve_real(self, x, target_height, target_width)`
- Defined: `audio/model.py:124`
- Doc: Real FFT evolution for action map computation.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### __init__ `def __init__(self, config)`
- Defined: `audio/model.py:172`
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### forward `def forward(self, x)`
- Defined: `audio/model.py:200`
- Doc: Full forward pass: project -> evolve -> reconstruct.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### forward_with_intermediates `def forward_with_intermediates(self, x)`
- Defined: `audio/model.py:216`
- Doc: Forward pass returning intermediate hidden states for analysis.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### extract_hamiltonian_fields `def extract_hamiltonian_fields(self, x)`
- Defined: `audio/model.py:237`
- Doc: Extract the three Hamiltonian field representations:
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

### compute_energy_mask `def compute_energy_mask(self, x)`
- Defined: `audio/model.py:270`
- Doc: Compute the Hamiltonian energy mask for spectral reconstruction.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`, `audio/trainer.py`

## audio/trainer.py

### __init__ `def __init__(self, config)`
- Defined: `audio/trainer.py:46`
- Depends on: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/config.py`, `audio/losses.py`, `audio/metrics.py`, `audio/model.py`
- Imported by: `audio/main.py`

### build_dataset `def build_dataset(self, mel_spectrogram)`
- Defined: `audio/trainer.py:49`
- Doc: Segment a mel spectrogram into training patches.
- Depends on: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/config.py`, `audio/losses.py`, `audio/metrics.py`, `audio/model.py`
- Imported by: `audio/main.py`

### __init__ `def __init__(self, config)`
- Defined: `audio/trainer.py:92`
- Depends on: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/config.py`, `audio/losses.py`, `audio/metrics.py`, `audio/model.py`
- Imported by: `audio/main.py`

### _attempt_checkpoint_recovery `def _attempt_checkpoint_recovery(self)`
- Defined: `audio/trainer.py:129`
- Doc: Load existing checkpoint if available.
- Depends on: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/config.py`, `audio/losses.py`, `audio/metrics.py`, `audio/model.py`
- Imported by: `audio/main.py`

### train `def train(self, audio_file_path)`
- Defined: `audio/trainer.py:151`
- Doc: Execute the full training pipeline on an audio file.
- Depends on: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/config.py`, `audio/losses.py`, `audio/metrics.py`, `audio/model.py`
- Imported by: `audio/main.py`

### _train_one_epoch `def _train_one_epoch(self, train_loader, epoch)`
- Defined: `audio/trainer.py:255`
- Doc: Execute one training epoch with full metric tracking.
- Depends on: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/config.py`, `audio/losses.py`, `audio/metrics.py`, `audio/model.py`
- Imported by: `audio/main.py`

### _validate `def _validate(self, val_loader, epoch)`
- Defined: `audio/trainer.py:344`
- Doc: Run validation pass and return metrics.
- Depends on: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/config.py`, `audio/losses.py`, `audio/metrics.py`, `audio/model.py`
- Imported by: `audio/main.py`

### model `def model(self)`
- Defined: `audio/trainer.py:377`
- Depends on: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/config.py`, `audio/losses.py`, `audio/metrics.py`, `audio/model.py`
- Imported by: `audio/main.py`

### audio_processor `def audio_processor(self)`
- Defined: `audio/trainer.py:381`
- Depends on: `audio/audio_io.py`, `audio/checkpoint_manager.py`, `audio/config.py`, `audio/losses.py`, `audio/metrics.py`, `audio/model.py`
- Imported by: `audio/main.py`

## audio/visualization.py

### __init__ `def __init__(self, vis_config, audio_config)`
- Defined: `audio/visualization.py:34`
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`

### render_complete_analysis `def render_complete_analysis(self, amplitude_map, phase_map, action_map, original_spectrogram, reconstructed_spectrogram, original_waveform, reconstructed_waveform, output_prefix)`
- Defined: `audio/visualization.py:43`
- Doc: Generate the complete suite of Hamiltonian analysis visualizations.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`

### _render_hamiltonian_fields `def _render_hamiltonian_fields(self, amplitude_map, phase_map, action_map, output_prefix)`
- Defined: `audio/visualization.py:91`
- Doc: Render the three Hamiltonian field visualizations.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`

### _render_spectrogram_comparison `def _render_spectrogram_comparison(self, original, reconstructed, output_prefix)`
- Defined: `audio/visualization.py:140`
- Doc: Render original vs reconstructed spectrogram comparison.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`

### _render_phase_portrait `def _render_phase_portrait(self, amplitude_map, phase_map, output_prefix)`
- Defined: `audio/visualization.py:185`
- Doc: Render 2D phase portrait (amplitude vs phase histogram).
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`

### _render_energy_landscape `def _render_energy_landscape(self, amplitude_map, action_map, output_prefix)`
- Defined: `audio/visualization.py:218`
- Doc: Render energy landscape as a 3D surface plot.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`

### _render_waveform_comparison `def _render_waveform_comparison(self, original_waveform, reconstructed_waveform, output_prefix)`
- Defined: `audio/visualization.py:270`
- Doc: Render original vs reconstructed waveform comparison.
- Depends on: `audio/config.py`
- Imported by: `audio/inference.py`

## diff_weights.py

### analize_checkpoint `def analize_checkpoint(path)`
- Defined: `diff_weights.py:5`

## dirac.py

### analyze_checkpoint `def analyze_checkpoint(checkpoint_path, output_dir)`
- Defined: `dirac.py:574`
- Depends on: `experiment2.py`

### analyze_multiple_checkpoints `def analyze_multiple_checkpoints(checkpoint_dir, n_latest, output_dir)`
- Defined: `dirac.py:620`
- Depends on: `experiment2.py`

### main `def main()`
- Defined: `dirac.py:652`
- Depends on: `experiment2.py`

### __init__ `def __init__(self, checkpoint_path, device)`
- Defined: `dirac.py:40`
- Depends on: `experiment2.py`

### extract_charge_distribution `def extract_charge_distribution(self)`
- Defined: `dirac.py:64`
- Depends on: `experiment2.py`

### compute_dirac_delta_approximation `def compute_dirac_delta_approximation(self, charge_density)`
- Defined: `dirac.py:77`
- Depends on: `experiment2.py`

### compute_electric_field `def compute_electric_field(self, dirac_data, eval_points)`
- Defined: `dirac.py:112`
- Depends on: `experiment2.py`

### compute_electric_flux `def compute_electric_flux(self, electric_field, surface_points)`
- Defined: `dirac.py:157`
- Depends on: `experiment2.py`

### compute_divergence `def compute_divergence(self, electric_field)`
- Defined: `dirac.py:192`
- Depends on: `experiment2.py`

### verify_gauss_law `def verify_gauss_law(self, dirac_data, flux_data)`
- Defined: `dirac.py:200`
- Depends on: `experiment2.py`

### analyze_all `def analyze_all(self)`
- Defined: `dirac.py:223`
- Depends on: `experiment2.py`

### _print_report `def _print_report(self, results)`
- Defined: `dirac.py:279`
- Depends on: `experiment2.py`

### plot_charge_distribution `def plot_charge_distribution(charge_density, point_positions, point_charges, output_path)`
- Defined: `dirac.py:331`
- Depends on: `experiment2.py`

### plot_electric_field `def plot_electric_field(electric_field, output_path)`
- Defined: `dirac.py:379`
- Depends on: `experiment2.py`

### plot_divergence `def plot_divergence(divergence, output_path)`
- Defined: `dirac.py:441`
- Depends on: `experiment2.py`

### plot_combined_analysis `def plot_combined_analysis(charge_density, point_positions, point_charges, electric_field, divergence, output_path)`
- Defined: `dirac.py:490`
- Depends on: `experiment2.py`

## expand.py

### load_config `def load_config(toml_path)`
- Defined: `expand.py:18`

### expand_spectral_weights `def expand_spectral_weights(kernel_real, kernel_imag, target_size, source_size)`
- Defined: `expand.py:23`
- Doc: Expand spectral kernels via zero-padding in frequency domain.

### expand_model `def expand_model(model, target_resolution, source_resolution)`
- Defined: `expand.py:43`
- Doc: Create a new model with expanded spectral weights.

### evaluate_model `def evaluate_model(model, resolution, device)`
- Defined: `expand.py:74`
- Doc: Evaluate expanded model on synthetic data.

### main `def main()`
- Defined: `expand.py:105`

## experiment.py

### set_seed `def set_seed(seed)`
- Defined: `experiment.py:88`

### setup_logger `def setup_logger(name, level)`
- Defined: `experiment.py:96`

### train_with_early_glass_stop `def train_with_early_glass_stop(model, optimizer, seed, epochs)`
- Defined: `experiment.py:670`
- Doc: Train model with early stopping for glass detection.

### seed_miner `def seed_miner(total_attempts)`
- Defined: `experiment.py:803`
- Doc: Mine for crystal seeds by trying sequential seeds.

### main `def main()`
- Defined: `experiment.py:856`

### analyze `def analyze(self, model)`
- Defined: `experiment.py:111`

### compute `def compute(self, model)`
- Defined: `experiment.py:117`

### __init__ `def __init__(self, grid_size)`
- Defined: `experiment.py:124`

### _precompute_spectral_operators `def _precompute_spectral_operators(self)`
- Defined: `experiment.py:128`

### apply `def apply(self, field)`
- Defined: `experiment.py:135`

### time_evolution `def time_evolution(self, field, dt)`
- Defined: `experiment.py:140`

### __init__ `def __init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)`
- Defined: `experiment.py:149`

### __len__ `def __len__(self)`
- Defined: `experiment.py:193`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `experiment.py:196`

### get_val_batch `def get_val_batch(self)`
- Defined: `experiment.py:199`

### __init__ `def __init__(self, channels, grid_size)`
- Defined: `experiment.py:206`

### forward `def forward(self, x)`
- Defined: `experiment.py:219`

### __init__ `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- Defined: `experiment.py:258`

### forward `def forward(self, x)`
- Defined: `experiment.py:279`

### compute_local_complexity `def compute_local_complexity(weights, epsilon)`
- Defined: `experiment.py:295`
- Doc: Compute Local Complexity (LC) metric for weight matrix.

### compute_superposition `def compute_superposition(weights)`
- Defined: `experiment.py:312`
- Doc: Compute Superposition (SP) metric for weight matrix.

### compute_kappa `def compute_kappa(model, dataloader, num_batches)`
- Defined: `experiment.py:342`

### compute_discretization_margin `def compute_discretization_margin(coeffs)`
- Defined: `experiment.py:378`

### compute_alpha_purity `def compute_alpha_purity(coeffs)`
- Defined: `experiment.py:387`

### compute_kappa_quantum `def compute_kappa_quantum(coeffs, hbar)`
- Defined: `experiment.py:394`

### compute_poynting_vector `def compute_poynting_vector(coeffs)`
- Defined: `experiment.py:411`

### compute_all_metrics `def compute_all_metrics(model, dataloader)`
- Defined: `experiment.py:426`

### compute_effective_temperature `def compute_effective_temperature(gradient_buffer, learning_rate)`
- Defined: `experiment.py:446`

### compute_specific_heat `def compute_specific_heat(loss_history, temp_history, cv_threshold)`
- Defined: `experiment.py:460`

### compute_weight_diffraction `def compute_weight_diffraction(coeffs)`
- Defined: `experiment.py:472`

### _compute_spectral_entropy `def _compute_spectral_entropy(power_spectrum)`
- Defined: `experiment.py:491`

### __init__ `def __init__(self, interval_minutes, max_checkpoints)`
- Defined: `experiment.py:501`

### should_save_checkpoint `def should_save_checkpoint(self)`
- Defined: `experiment.py:509`

### save_checkpoint `def save_checkpoint(self, model, optimizer, epoch, metrics)`
- Defined: `experiment.py:514`

### __init__ `def __init__(self)`
- Defined: `experiment.py:574`

### update_metrics `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat, poynting_magnitude)`
- Defined: `experiment.py:594`

### __init__ `def __init__(self, patience_epochs)`
- Defined: `experiment.py:612`

### should_stop `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)`
- Defined: `experiment.py:616`
- Doc: Check if the system is in glass state and should stop mining.

### __init__ `def __init__(self, checkpoint_path, results_dir)`
- Defined: `experiment.py:880`

### load_and_analyze_checkpoint `def load_and_analyze_checkpoint(self)`
- Defined: `experiment.py:886`

### dataloader `def dataloader()`
- Defined: `experiment.py:903`

## experiment2.py

### main `def main()`
- Defined: `experiment2.py:1334`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### set_seed `def set_seed(seed)`
- Defined: `experiment2.py:76`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### create_logger `def create_logger(name, level)`
- Defined: `experiment2.py:86`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### analyze `def analyze(self, model)`
- Defined: `experiment2.py:101`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### compute `def compute(self, model)`
- Defined: `experiment2.py:107`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### __init__ `def __init__(self, grid_size)`
- Defined: `experiment2.py:112`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### _precompute_spectral_operators `def _precompute_spectral_operators(self)`
- Defined: `experiment2.py:116`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### apply `def apply(self, field)`
- Defined: `experiment2.py:122`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### time_evolution `def time_evolution(self, field, dt)`
- Defined: `experiment2.py:127`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### __init__ `def __init__(self, num_samples, grid_size, time_steps, dt, train_ratio)`
- Defined: `experiment2.py:134`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### __len__ `def __len__(self)`
- Defined: `experiment2.py:173`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `experiment2.py:176`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### get_validation_batch `def get_validation_batch(self)`
- Defined: `experiment2.py:179`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### __init__ `def __init__(self, channels, grid_size)`
- Defined: `experiment2.py:184`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### forward `def forward(self, x)`
- Defined: `experiment2.py:195`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### __init__ `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- Defined: `experiment2.py:228`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### forward `def forward(self, x)`
- Defined: `experiment2.py:243`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### compute_local_complexity `def compute_local_complexity(weights, epsilon)`
- Defined: `experiment2.py:259`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### compute_superposition `def compute_superposition(weights)`
- Defined: `experiment2.py:275`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### compute `def compute(self, model, val_x, val_y)`
- Defined: `experiment2.py:303`
- Doc: Implementación de interfaz IMetricsCalculator.
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### compute_gradient_covariance_kappa `def compute_gradient_covariance_kappa(model, dataloader, num_batches)`
- Defined: `experiment2.py:311`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### compute_discretization_margin_from_state_dict `def compute_discretization_margin_from_state_dict(model)`
- Defined: `experiment2.py:348`
- Doc: Calcula el margen de discretización desde los parámetros del modelo.
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### compute_discretization_margin `def compute_discretization_margin(coeffs)`
- Defined: `experiment2.py:361`
- Doc: Calcula el margen de discretización desde un diccionario de coeficientes.
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### compute_alpha_purity_from_model `def compute_alpha_purity_from_model(model)`
- Defined: `experiment2.py:373`
- Doc: Calcula el índice de pureza alpha directamente desde el modelo.
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### compute_alpha_purity `def compute_alpha_purity(coeffs)`
- Defined: `experiment2.py:383`
- Doc: Calcula el índice de pureza alpha desde un diccionario de coeficientes.
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### compute_kappa `def compute_kappa(model, val_x, val_y, num_batches)`
- Defined: `experiment2.py:393`
- Doc: Número de condición de la matriz de covarianza de gradientes.
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### compute_kappa_quantum `def compute_kappa_quantum(model, hbar)`
- Defined: `experiment2.py:464`
- Doc: Versión del cálculo cuántico de kappa que opera directamente sobre el modelo.
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### compute_kappa_quantum_from_coeffs `def compute_kappa_quantum_from_coeffs(coeffs, hbar)`
- Defined: `experiment2.py:492`
- Doc: Versión del cálculo cuántico de kappa desde diccionario de coeficientes.
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### _compute_crystallography_metrics `def _compute_crystallography_metrics(self, model, val_x, val_y)`
- Defined: `experiment2.py:511`
- Doc: Métricas cristalográficas con aislamiento completo de errores.
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### _check_weight_integrity `def _check_weight_integrity(self, model)`
- Defined: `experiment2.py:539`
- Doc: Verifica integridad de pesos: NaN, Inf, y estadísticas básicas.
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### compute_poynting_vector `def compute_poynting_vector(model)`
- Defined: `experiment2.py:603`
- Doc: Vector de Poynting: flujo de energía en el espacio de parámetros.
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### compute_all_metrics `def compute_all_metrics(model, val_x, val_y)`
- Defined: `experiment2.py:679`
- Doc: Calcula todas las métricas cristalográficas con manejo de errores.
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### compute `def compute(self, model, gradient_buffer, learning_rate, loss_history, temp_history)`
- Defined: `experiment2.py:738`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### compute_effective_temperature `def compute_effective_temperature(gradient_buffer, learning_rate)`
- Defined: `experiment2.py:747`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### compute_specific_heat `def compute_specific_heat(loss_history, temp_history, cv_threshold)`
- Defined: `experiment2.py:760`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### compute `def compute(self, model)`
- Defined: `experiment2.py:771`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### compute_weight_diffraction `def compute_weight_diffraction(coeffs)`
- Defined: `experiment2.py:776`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### _compute_spectral_entropy `def _compute_spectral_entropy(power_spectrum)`
- Defined: `experiment2.py:795`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### __init__ `def __init__(self, interval_minutes, max_checkpoints)`
- Defined: `experiment2.py:805`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### should_save_checkpoint `def should_save_checkpoint(self)`
- Defined: `experiment2.py:813`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### save_checkpoint `def save_checkpoint(self, model, optimizer, epoch, metrics)`
- Defined: `experiment2.py:818`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### __init__ `def __init__(self)`
- Defined: `experiment2.py:875`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### update_metrics `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat, poynting_magnitude)`
- Defined: `experiment2.py:895`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### __init__ `def __init__(self, patience_epochs)`
- Defined: `experiment2.py:913`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### should_stop `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)`
- Defined: `experiment2.py:918`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### is_crystal_formed `def is_crystal_formed(self, lc, sp, kappa, delta, temp, cv)`
- Defined: `experiment2.py:963`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### __init__ `def __init__(self, model, optimizer, device, logger)`
- Defined: `experiment2.py:974`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### train_epoch `def train_epoch(self, dataloader, epoch)`
- Defined: `experiment2.py:997`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### validate `def validate(self, val_x, val_y)`
- Defined: `experiment2.py:1027`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### compute_weight_metrics `def compute_weight_metrics(self)`
- Defined: `experiment2.py:1040`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### execute_training `def execute_training(self, dataloader, val_x, val_y, epochs, seed, early_stopping)`
- Defined: `experiment2.py:1056`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### __init__ `def __init__(self, max_attempts)`
- Defined: `experiment2.py:1114`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### mine `def mine(self)`
- Defined: `experiment2.py:1118`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### __init__ `def __init__(self, seed, epochs, grid_size, hidden_dim, num_spectral_layers, learning_rate)`
- Defined: `experiment2.py:1165`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### run `def run(self)`
- Defined: `experiment2.py:1175`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### __init__ `def __init__(self, checkpoint_path, results_dir)`
- Defined: `experiment2.py:1224`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### analyze `def analyze(self)`
- Defined: `experiment2.py:1230`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### __init__ `def __init__(self)`
- Defined: `experiment2.py:1276`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### _create_argument_parser `def _create_argument_parser(self)`
- Defined: `experiment2.py:1280`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### run `def run(self)`
- Defined: `experiment2.py:1294`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

### safe_compute `def safe_compute(func)`
- Defined: `experiment2.py:694`
- Imported by: `check_fase_berry.py`, `dirac.py`, `get_meditions.py`, `hpu_view.py`, `polos.py`, `precision.py`, `refinamiento.py`, `simple_hpu_view.py`, `verify.py`

## get_meditions.py

### setup_logger `def setup_logger(name, level)`
- Defined: `get_meditions.py:51`
- Depends on: `experiment2.py`

### verify_latest_checkpoints `def verify_latest_checkpoints(checkpoint_dir, n)`
- Defined: `get_meditions.py:1439`
- Doc: Verifica los N checkpoints más recientes con análisis completo
- Depends on: `experiment2.py`

### main `def main()`
- Defined: `get_meditions.py:1500`
- Depends on: `experiment2.py`

### helmholtz_free_energy `def helmholtz_free_energy(self)`
- Defined: `get_meditions.py:81`
- Doc: F = U - T*S (a μ y N constantes)
- Depends on: `experiment2.py`

### gibbs_free_energy `def gibbs_free_energy(self)`
- Defined: `get_meditions.py:85`
- Doc: G = F + μ*N + P*V (presión algorítmica)
- Depends on: `experiment2.py`

### is_stable `def is_stable(self)`
- Defined: `get_meditions.py:90`
- Doc: Criterio de estabilidad: dG < 0
- Depends on: `experiment2.py`

### compute_kappa `def compute_kappa(model, val_x, val_y, num_batches)`
- Defined: `get_meditions.py:127`
- Doc: Número de condición de la matriz de covarianza de gradientes.
- Depends on: `experiment2.py`

### compute_discretization_margin `def compute_discretization_margin(model)`
- Defined: `get_meditions.py:187`
- Doc: δ = max |w - round(w)| sobre todos los parámetros.
- Depends on: `experiment2.py`

### compute_alpha_purity `def compute_alpha_purity(model)`
- Defined: `get_meditions.py:203`
- Doc: α = -log(δ). Pureza cristalina.
- Depends on: `experiment2.py`

### compute_local_complexity `def compute_local_complexity(model)`
- Defined: `get_meditions.py:214`
- Doc: Fracción de parámetros "activos" (no cerca de cero).
- Depends on: `experiment2.py`

### compute_kappa_quantum `def compute_kappa_quantum(model, hbar)`
- Defined: `get_meditions.py:226`
- Doc: κ cuántico: número de condición con regularización cuántica.
- Depends on: `experiment2.py`

### _compute_kappa_iterative `def _compute_kappa_iterative(params, hbar, n, max_iters, tol)`
- Defined: `get_meditions.py:255`
- Doc: Método de potencia para estimar κ sin construir matriz.
- Depends on: `experiment2.py`

### compute_poynting_vector `def compute_poynting_vector(model)`
- Defined: `get_meditions.py:312`
- Doc: Vector de Poynting: flujo de energía en el espacio de parámetros.
- Depends on: `experiment2.py`

### compute_all_metrics `def compute_all_metrics(model, val_x, val_y)`
- Defined: `get_meditions.py:366`
- Doc: Calcula todas las métricas cristalográficas.
- Depends on: `experiment2.py`

### compute_effective_temperature `def compute_effective_temperature(gradient_buffer, learning_rate)`
- Defined: `get_meditions.py:401`
- Doc: T_eff = (lr/2) * Var(∇L). Temperatura de fluctuaciones.
- Depends on: `experiment2.py`

### compute_specific_heat `def compute_specific_heat(loss_history, temp_history, cv_threshold)`
- Defined: `get_meditions.py:418`
- Doc: C_v = Var(U) / T^2. Detecta transiciones de fase (picos en C_v).
- Depends on: `experiment2.py`

### compute_critical_exponents `def compute_critical_exponents(temp_history, cv_history, alpha_history)`
- Defined: `get_meditions.py:436`
- Doc: Exponentes críticos cerca de transiciones de fase.
- Depends on: `experiment2.py`

### compute_equation_of_state `def compute_equation_of_state(temp_eff, alpha, kappa)`
- Defined: `get_meditions.py:504`
- Doc: Ecuación de estado: T_c(α) = T_0 * exp(-c*α)
- Depends on: `experiment2.py`

### compute_mutual_information `def compute_mutual_information(weights, gradients)`
- Defined: `get_meditions.py:539`
- Doc: Información mutua pesos-gradientes.
- Depends on: `experiment2.py`

### estimate_hbar_algorithmic `def estimate_hbar_algorithmic(model_complexity, weight_dim, mutual_information)`
- Defined: `get_meditions.py:561`
- Doc: ħ algorítmico efectivo.
- Depends on: `experiment2.py`

### compute_fisher_information_matrix `def compute_fisher_information_matrix(model, samples)`
- Defined: `get_meditions.py:572`
- Doc: Matriz de información de Fisher.
- Depends on: `experiment2.py`

### compute_ricci_curvature `def compute_ricci_curvature(fisher_matrix)`
- Defined: `get_meditions.py:593`
- Doc: Curvatura de Ricci escalar.
- Depends on: `experiment2.py`

### calculate_carnot_efficiency `def calculate_carnot_efficiency(delta_alpha, total_flops, initial_alpha)`
- Defined: `get_meditions.py:608`
- Doc: Eficiencia de Carnot del proceso de aprendizaje.
- Depends on: `experiment2.py`

### compute_weight_diffraction `def compute_weight_diffraction(model)`
- Defined: `get_meditions.py:640`
- Doc: Patrón de difracción de pesos (FFT).
- Depends on: `experiment2.py`

### _compute_spectral_entropy `def _compute_spectral_entropy(power_spectrum)`
- Defined: `get_meditions.py:672`
- Doc: Entropía espectral de Shannon.
- Depends on: `experiment2.py`

### extract_lattice_parameters `def extract_lattice_parameters(weight_tensor, rank)`
- Defined: `get_meditions.py:680`
- Doc: Extrae parámetros de red vía SVD.
- Depends on: `experiment2.py`

### compute_gibbs_free_energy `def compute_gibbs_free_energy(loss, temp, entropy)`
- Defined: `get_meditions.py:732`
- Doc: Energía libre de Gibbs.
- Depends on: `experiment2.py`

### __init__ `def __init__(self, checkpoint_path, device)`
- Defined: `get_meditions.py:741`
- Depends on: `experiment2.py`

### verify_all_metrics `def verify_all_metrics(self)`
- Defined: `get_meditions.py:782`
- Doc: Calcula TODAS las métricas desde cero y compara con las guardadas
- Depends on: `experiment2.py`

### _check_weight_integrity `def _check_weight_integrity(self)`
- Defined: `get_meditions.py:849`
- Doc: Verifica que los pesos no tengan NaN/Inf.
- Depends on: `experiment2.py`

### _compute_validation_metrics `def _compute_validation_metrics(self)`
- Defined: `get_meditions.py:915`
- Doc: Calcula MSE y accuracy de validación desde cero
- Depends on: `experiment2.py`

### _compute_discretization_metrics `def _compute_discretization_metrics(self)`
- Defined: `get_meditions.py:942`
- Doc: Calcula delta, alpha, purity, etc.
- Depends on: `experiment2.py`

### _compute_quantization_metrics `def _compute_quantization_metrics(self)`
- Defined: `get_meditions.py:986`
- Doc: Calcula la penalización de cuantización
- Depends on: `experiment2.py`

### _compute_loss_metrics `def _compute_loss_metrics(self)`
- Defined: `get_meditions.py:1005`
- Doc: Reconstruye el loss total
- Depends on: `experiment2.py`

### _compute_crystallography_metrics `def _compute_crystallography_metrics(self)`
- Defined: `get_meditions.py:1031`
- Doc: Métricas cristalográficas completas
- Depends on: `experiment2.py`

### _compute_thermodynamic_metrics `def _compute_thermodynamic_metrics(self)`
- Defined: `get_meditions.py:1038`
- Doc: Métricas termodinámicas
- Depends on: `experiment2.py`

### _approximate_ricci_curvature `def _approximate_ricci_curvature(self)`
- Defined: `get_meditions.py:1113`
- Doc: Aproximación de curvatura de Ricci para HPU Core
- Depends on: `experiment2.py`

### _compute_spectroscopy `def _compute_spectroscopy(self)`
- Defined: `get_meditions.py:1134`
- Doc: Análisis espectroscópico
- Depends on: `experiment2.py`

### _compute_thermodynamic_potential `def _compute_thermodynamic_potential(self, results)`
- Defined: `get_meditions.py:1164`
- Doc: Calcula potencial termodinámico completo
- Depends on: `experiment2.py`

### _compare_with_stored `def _compare_with_stored(self, computed)`
- Defined: `get_meditions.py:1188`
- Doc: Compara métricas calculadas vs almacenadas en el checkpoint
- Depends on: `experiment2.py`

### _check_internal_consistency `def _check_internal_consistency(self, results)`
- Defined: `get_meditions.py:1226`
- Doc: Verifica consistencia entre métricas relacionadas
- Depends on: `experiment2.py`

### _compute_health_score `def _compute_health_score(self, results)`
- Defined: `get_meditions.py:1267`
- Doc: Calcula un score de salud del checkpoint (0-100)
- Depends on: `experiment2.py`

### _assign_crystallographic_grade `def _assign_crystallographic_grade(self, delta, alpha)`
- Defined: `get_meditions.py:1336`
- Doc: Asigna grado cristalográfico
- Depends on: `experiment2.py`

### _print_report `def _print_report(self, results)`
- Defined: `get_meditions.py:1349`
- Doc: Imprime reporte formateado con todas las métricas nuevas
- Depends on: `experiment2.py`

### from_model `def from_model(cls, model)`
- Defined: `get_meditions.py:112`
- Doc: Extrae coeficientes del modelo HPU Core
- Depends on: `experiment2.py`

## hamiltonian_mbl.py

### main `def main()`
- Defined: `hamiltonian_mbl.py:2031`

### get_input_dim `def get_input_dim(self)`
- Defined: `hamiltonian_mbl.py:55`
- Doc: Calculate input dimension from grid size.

### get_total_parameters `def get_total_parameters(self)`
- Defined: `hamiltonian_mbl.py:59`
- Doc: Estimate total parameter count.

### get_reduced_dimension `def get_reduced_dimension(self)`
- Defined: `hamiltonian_mbl.py:135`
- Doc: Calculate reduced dimension for analysis.

### get_coefficients `def get_coefficients(self)`
- Defined: `hamiltonian_mbl.py:162`

### forward `def forward(self)`
- Defined: `hamiltonian_mbl.py:163`

### calculate `def calculate(self, model)`
- Defined: `hamiltonian_mbl.py:169`

### calculate `def calculate(self, model)`
- Defined: `hamiltonian_mbl.py:175`

### calculate `def calculate(self, participation_ratio, energy_gap)`
- Defined: `hamiltonian_mbl.py:181`

### analyze_robustness `def analyze_robustness(self, model, noise_levels)`
- Defined: `hamiltonian_mbl.py:187`

### save_checkpoint `def save_checkpoint(self, model, epoch, metrics, loss_history, path)`
- Defined: `hamiltonian_mbl.py:193`

### load_checkpoint `def load_checkpoint(self, path)`
- Defined: `hamiltonian_mbl.py:195`

### collect `def collect(self, model, loss, epoch, loss_history)`
- Defined: `hamiltonian_mbl.py:201`

### __init__ `def __init__(self, source_config, target_config)`
- Defined: `hamiltonian_mbl.py:214`

### migrate_state_dict `def migrate_state_dict(self, source_state)`
- Defined: `hamiltonian_mbl.py:218`
- Doc: Migra estado de SimpleHamiltonianNet a HamiltonianNeuralNetwork.

### _create_default_parameter `def _create_default_parameter(self, key)`
- Defined: `hamiltonian_mbl.py:320`
- Doc: Crea parámetro por defecto.

### __init__ `def __init__(self, config)`
- Defined: `hamiltonian_mbl.py:352`

### _initialize_spectral_parameters `def _initialize_spectral_parameters(self)`
- Defined: `hamiltonian_mbl.py:366`
- Doc: Initialize with physics-informed priors.

### forward `def forward(self, q, p, dt)`
- Defined: `hamiltonian_mbl.py:373`
- Doc: Symplectic Euler integration of Hamilton's equations.

### get_hamiltonian `def get_hamiltonian(self, q, p)`
- Defined: `hamiltonian_mbl.py:401`
- Doc: Compute Hamiltonian H = T + V in spectral space.

### __init__ `def __init__(self, config)`
- Defined: `hamiltonian_mbl.py:418`

### _initialize_weights `def _initialize_weights(self)`
- Defined: `hamiltonian_mbl.py:438`
- Doc: Orthogonal initialization for Hamiltonian structure preservation.

### forward `def forward(self, q, p, dt)`
- Defined: `hamiltonian_mbl.py:443`
- Doc: Forward pass through Hamiltonian dynamics.

### time_evolution `def time_evolution(self, q_initial, p_initial, num_steps, dt)`
- Defined: `hamiltonian_mbl.py:459`
- Doc: Generate trajectory through time evolution.

### get_hamiltonian `def get_hamiltonian(self, q, p)`
- Defined: `hamiltonian_mbl.py:474`
- Doc: Compute total Hamiltonian.

### get_coefficients `def get_coefficients(self)`
- Defined: `hamiltonian_mbl.py:486`

### get_flat_parameters `def get_flat_parameters(self)`
- Defined: `hamiltonian_mbl.py:496`
- Doc: Returns all parameters flattened for Hamiltonian construction.

### construct_hessian_approximation `def construct_hessian_approximation(self, max_dim, method)`
- Defined: `hamiltonian_mbl.py:503`
- Doc: MÉTODO CORREGIDO - No usa 65GB de RAM.

### __init__ `def __init__(self, grid_size, num_samples, device)`
- Defined: `hamiltonian_mbl.py:561`

### generate_harmonic_oscillator `def generate_harmonic_oscillator(self, omega)`
- Defined: `hamiltonian_mbl.py:567`
- Doc: Generate harmonic oscillator initial conditions.

### generate_double_well `def generate_double_well(self, barrier_height)`
- Defined: `hamiltonian_mbl.py:591`
- Doc: Generate double-well potential trajectories.

### __init__ `def __init__(self, config)`
- Defined: `hamiltonian_mbl.py:616`

### calculate `def calculate(self, model)`
- Defined: `hamiltonian_mbl.py:619`
- Doc: Calculate level spacing statistics from model weights.

### _construct_hessian_from_weights `def _construct_hessian_from_weights(self, model)`
- Defined: `hamiltonian_mbl.py:651`
- Doc: Alternative Hessian construction for generic models.

### _compute_eigenvalues `def _compute_eigenvalues(self, hessian)`
- Defined: `hamiltonian_mbl.py:666`
- Doc: Compute sorted eigenvalues of the Hamiltonian.

### _calculate_spacing_ratios `def _calculate_spacing_ratios(self, spacings)`
- Defined: `hamiltonian_mbl.py:671`
- Doc: Calculate adjacent gap ratios r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}).

### _classify_phase `def _classify_phase(self, mean_ratio)`
- Defined: `hamiltonian_mbl.py:684`
- Doc: Classify the quantum phase based on level spacing ratio.

### _estimate_brody_parameter `def _estimate_brody_parameter(self, ratios)`
- Defined: `hamiltonian_mbl.py:699`
- Doc: Estimate Brody parameter for intermediate statistics.

### __init__ `def __init__(self, config)`
- Defined: `hamiltonian_mbl.py:725`

### calculate `def calculate(self, model)`
- Defined: `hamiltonian_mbl.py:728`
- Doc: Calculate participation ratios for all weight layers.

### _calculate_ipr `def _calculate_ipr(self, coefficients)`
- Defined: `hamiltonian_mbl.py:771`
- Doc: Calculate standard Inverse Participation Ratio.

### _calculate_renyi_ipr `def _calculate_renyi_ipr(self, coefficients, q)`
- Defined: `hamiltonian_mbl.py:782`
- Doc: Calculate q-th order Rényi IPR.

### _calculate_fractal_dimension `def _calculate_fractal_dimension(self, ipr, n)`
- Defined: `hamiltonian_mbl.py:793`
- Doc: Calculate fractal dimension D_q from IPR.

### __init__ `def __init__(self, config)`
- Defined: `hamiltonian_mbl.py:807`

### calculate `def calculate(self, participation_ratio, energy_gap)`
- Defined: `hamiltonian_mbl.py:810`
- Doc: Calculate synthetic Planck's constant.

### calculate_from_model `def calculate_from_model(self, model, level_spacing_results, pr_results)`
- Defined: `hamiltonian_mbl.py:820`
- Doc: Comprehensive calculation from model and previous analyses.

### __init__ `def __init__(self, config)`
- Defined: `hamiltonian_mbl.py:849`

### calculate_base_discretization `def calculate_base_discretization(self, model)`
- Defined: `hamiltonian_mbl.py:853`
- Doc: Calculate the base discretization level from weight rounding error.

### analyze_robustness `def analyze_robustness(self, model, noise_levels)`
- Defined: `hamiltonian_mbl.py:877`
- Doc: Test robustness by applying noise and measuring gap collapse.

### _perturb_and_measure `def _perturb_and_measure(self, model, noise_level)`
- Defined: `hamiltonian_mbl.py:921`
- Doc: Apply noise to model and measure resulting metrics.

### _delta_to_alpha `def _delta_to_alpha(self, delta)`
- Defined: `hamiltonian_mbl.py:940`
- Doc: Convert discretization error to purity alpha.

### __init__ `def __init__(self, config)`
- Defined: `hamiltonian_mbl.py:950`

### calculate `def calculate(self, model)`
- Defined: `hamiltonian_mbl.py:953`

### _compute_layer_purity `def _compute_layer_purity(self, weights)`
- Defined: `hamiltonian_mbl.py:982`

### _delta_to_alpha `def _delta_to_alpha(self, delta)`
- Defined: `hamiltonian_mbl.py:988`

### _assess_purity_quality `def _assess_purity_quality(self, alpha, variance)`
- Defined: `hamiltonian_mbl.py:993`

### __init__ `def __init__(self, config)`
- Defined: `hamiltonian_mbl.py:1007`

### calculate `def calculate(self, loss_history)`
- Defined: `hamiltonian_mbl.py:1010`

### __init__ `def __init__(self, config)`
- Defined: `hamiltonian_mbl.py:1054`

### calculate `def calculate(self, model)`
- Defined: `hamiltonian_mbl.py:1057`
- Doc: Calculate Krylov complexity from model dynamics.

### __init__ `def __init__(self, config)`
- Defined: `hamiltonian_mbl.py:1095`

### calculate `def calculate(self, model)`
- Defined: `hamiltonian_mbl.py:1098`
- Doc: Calculate crystallinity index from weight spectra.

### __init__ `def __init__(self, config)`
- Defined: `hamiltonian_mbl.py:1150`

### measure `def measure(self, model)`
- Defined: `hamiltonian_mbl.py:1153`
- Doc: Comprehensive resilience measurement.

### _measure_base_performance `def _measure_base_performance(self, model)`
- Defined: `hamiltonian_mbl.py:1176`
- Doc: Measure baseline performance metrics.

### _test_perturbation `def _test_perturbation(self, model, dimension, noise_level)`
- Defined: `hamiltonian_mbl.py:1195`
- Doc: Test resilience to specific perturbation.

### _aggregate_by_dimension `def _aggregate_by_dimension(self, results)`
- Defined: `hamiltonian_mbl.py:1220`
- Doc: Aggregate resilience scores by perturbation dimension.

### _aggregate_by_noise `def _aggregate_by_noise(self, results)`
- Defined: `hamiltonian_mbl.py:1231`
- Doc: Aggregate resilience scores by noise level.

### __init__ `def __init__(self, config)`
- Defined: `hamiltonian_mbl.py:1246`

### classify `def classify(self, alpha, temperature)`
- Defined: `hamiltonian_mbl.py:1249`

### __init__ `def __init__(self, arch_config)`
- Defined: `hamiltonian_mbl.py:1273`

### migrate `def migrate(self, raw_data, device)`
- Defined: `hamiltonian_mbl.py:1277`

### _migrate_if_needed `def _migrate_if_needed(self, state_dict, device)`
- Defined: `hamiltonian_mbl.py:1290`
- Doc: Detecta el formato y aplica migración si es necesario.

### __init__ `def __init__(self, config, arch_config)`
- Defined: `hamiltonian_mbl.py:1315`

### should_save_checkpoint `def should_save_checkpoint(self)`
- Defined: `hamiltonian_mbl.py:1322`
- Doc: Check if 5 minutes have elapsed since last checkpoint.

### save_checkpoint `def save_checkpoint(self, model, epoch, metrics, loss_history, checkpoint_dir)`
- Defined: `hamiltonian_mbl.py:1328`
- Doc: Save checkpoint with all MBL metrics.

### load_checkpoint `def load_checkpoint(self, path)`
- Defined: `hamiltonian_mbl.py:1361`
- Doc: Load checkpoint with automatic device placement and migration.

### __init__ `def __init__(self, config)`
- Defined: `hamiltonian_mbl.py:1392`

### collect `def collect(self, model, loss, epoch, loss_history, step)`
- Defined: `hamiltonian_mbl.py:1405`
- Doc: Collect core metrics for the current training state.

### collect_comprehensive `def collect_comprehensive(self, model, loss, epoch, loss_history, step)`
- Defined: `hamiltonian_mbl.py:1493`
- Doc: Collect comprehensive metrics including expensive calculations.

### _classify_quantum_phase `def _classify_quantum_phase(self, level_spacing, hbar_results)`
- Defined: `hamiltonian_mbl.py:1520`
- Doc: Classify combined quantum phase.

### __init__ `def __init__(self, model, arch_config, mbl_config, train_config)`
- Defined: `hamiltonian_mbl.py:1545`

### train_step `def train_step(self, q_batch, p_batch, q_target, p_target)`
- Defined: `hamiltonian_mbl.py:1571`
- Doc: Single training step with Hamiltonian loss.

### train_epoch `def train_epoch(self, dataset, epoch)`
- Defined: `hamiltonian_mbl.py:1603`
- Doc: Train for one epoch with MBL monitoring.

### _log_metrics `def _log_metrics(self, metrics)`
- Defined: `hamiltonian_mbl.py:1658`
- Doc: Log metrics to console in scientific format.

### train `def train(self, dataset, num_epochs)`
- Defined: `hamiltonian_mbl.py:1672`
- Doc: Full training loop.

### __init__ `def __init__(self, checkpoint_path, arch_config, mbl_config)`
- Defined: `hamiltonian_mbl.py:1713`

### _load_checkpoint `def _load_checkpoint(self)`
- Defined: `hamiltonian_mbl.py:1723`
- Doc: Load and migrate checkpoint.

### analyze `def analyze(self)`
- Defined: `hamiltonian_mbl.py:1763`
- Doc: Perform complete MBL analysis.

### _generate_summary `def _generate_summary(self, metrics)`
- Defined: `hamiltonian_mbl.py:1787`
- Doc: Generate executive summary.

### _print_report `def _print_report(self, results)`
- Defined: `hamiltonian_mbl.py:1807`
- Doc: Print formatted analysis report.

### __init__ `def __init__(self, arch_config, mbl_config)`
- Defined: `hamiltonian_mbl.py:1878`

### process_checkpoint `def process_checkpoint(self, checkpoint_path, output_dir)`
- Defined: `hamiltonian_mbl.py:1882`
- Doc: Process single checkpoint and save results.

### process_directory `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- Defined: `hamiltonian_mbl.py:1901`
- Doc: Process multiple checkpoints from directory.

### generate_summary `def generate_summary(self, all_results, output_dir)`
- Defined: `hamiltonian_mbl.py:1941`
- Doc: Generate aggregate summary report.

### _generate_text_report `def _generate_text_report(self, summary, output_dir)`
- Defined: `hamiltonian_mbl.py:1982`
- Doc: Generate human-readable text report.

## mining_seeds.py

### set_seed `def set_seed(seed)`
- Defined: `mining_seeds.py:88`

### setup_logger `def setup_logger(name, level)`
- Defined: `mining_seeds.py:96`

### train_with_early_glass_stop `def train_with_early_glass_stop(model, optimizer, seed, epochs)`
- Defined: `mining_seeds.py:670`
- Doc: Train model with early stopping for glass detection.

### seed_miner `def seed_miner(total_attempts)`
- Defined: `mining_seeds.py:803`
- Doc: Mine for crystal seeds by trying sequential seeds.

### main `def main()`
- Defined: `mining_seeds.py:856`

### analyze `def analyze(self, model)`
- Defined: `mining_seeds.py:111`

### compute `def compute(self, model)`
- Defined: `mining_seeds.py:117`

### __init__ `def __init__(self, grid_size)`
- Defined: `mining_seeds.py:124`

### _precompute_spectral_operators `def _precompute_spectral_operators(self)`
- Defined: `mining_seeds.py:128`

### apply `def apply(self, field)`
- Defined: `mining_seeds.py:135`

### time_evolution `def time_evolution(self, field, dt)`
- Defined: `mining_seeds.py:140`

### __init__ `def __init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)`
- Defined: `mining_seeds.py:149`

### __len__ `def __len__(self)`
- Defined: `mining_seeds.py:193`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `mining_seeds.py:196`

### get_val_batch `def get_val_batch(self)`
- Defined: `mining_seeds.py:199`

### __init__ `def __init__(self, channels, grid_size)`
- Defined: `mining_seeds.py:206`

### forward `def forward(self, x)`
- Defined: `mining_seeds.py:219`

### __init__ `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- Defined: `mining_seeds.py:258`

### forward `def forward(self, x)`
- Defined: `mining_seeds.py:279`

### compute_local_complexity `def compute_local_complexity(weights, epsilon)`
- Defined: `mining_seeds.py:295`
- Doc: Compute Local Complexity (LC) metric for weight matrix.

### compute_superposition `def compute_superposition(weights)`
- Defined: `mining_seeds.py:312`
- Doc: Compute Superposition (SP) metric for weight matrix.

### compute_kappa `def compute_kappa(model, dataloader, num_batches)`
- Defined: `mining_seeds.py:342`

### compute_discretization_margin `def compute_discretization_margin(coeffs)`
- Defined: `mining_seeds.py:378`

### compute_alpha_purity `def compute_alpha_purity(coeffs)`
- Defined: `mining_seeds.py:387`

### compute_kappa_quantum `def compute_kappa_quantum(coeffs, hbar)`
- Defined: `mining_seeds.py:394`

### compute_poynting_vector `def compute_poynting_vector(coeffs)`
- Defined: `mining_seeds.py:411`

### compute_all_metrics `def compute_all_metrics(model, dataloader)`
- Defined: `mining_seeds.py:426`

### compute_effective_temperature `def compute_effective_temperature(gradient_buffer, learning_rate)`
- Defined: `mining_seeds.py:446`

### compute_specific_heat `def compute_specific_heat(loss_history, temp_history, cv_threshold)`
- Defined: `mining_seeds.py:460`

### compute_weight_diffraction `def compute_weight_diffraction(coeffs)`
- Defined: `mining_seeds.py:472`

### _compute_spectral_entropy `def _compute_spectral_entropy(power_spectrum)`
- Defined: `mining_seeds.py:491`

### __init__ `def __init__(self, interval_minutes, max_checkpoints)`
- Defined: `mining_seeds.py:501`

### should_save_checkpoint `def should_save_checkpoint(self)`
- Defined: `mining_seeds.py:509`

### save_checkpoint `def save_checkpoint(self, model, optimizer, epoch, metrics)`
- Defined: `mining_seeds.py:514`

### __init__ `def __init__(self)`
- Defined: `mining_seeds.py:574`

### update_metrics `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat, poynting_magnitude)`
- Defined: `mining_seeds.py:594`

### __init__ `def __init__(self, patience_epochs)`
- Defined: `mining_seeds.py:612`

### should_stop `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)`
- Defined: `mining_seeds.py:616`
- Doc: Check if the system is in glass state and should stop mining.

### __init__ `def __init__(self, checkpoint_path, results_dir)`
- Defined: `mining_seeds.py:880`

### load_and_analyze_checkpoint `def load_and_analyze_checkpoint(self)`
- Defined: `mining_seeds.py:886`

### dataloader `def dataloader()`
- Defined: `mining_seeds.py:903`

## plank.py

### main `def main()`
- Defined: `plank.py:213`

### __init__ `def __init__(self, checkpoint_path, device)`
- Defined: `plank.py:29`

### calculate_all `def calculate_all(self)`
- Defined: `plank.py:54`
- Doc: Ejecuta todos los cálculos de ħ.

### print_report `def print_report(self, results)`
- Defined: `plank.py:170`
- Doc: Imprime reporte formateado.

## polos.py

### analyze_checkpoint `def analyze_checkpoint(checkpoint_path, output_dir)`
- Defined: `polos.py:1152`
- Depends on: `experiment2.py`

### analyze_multiple_checkpoints `def analyze_multiple_checkpoints(checkpoint_dir, n_latest, output_dir)`
- Defined: `polos.py:1208`
- Depends on: `experiment2.py`

### main `def main()`
- Defined: `polos.py:1240`
- Depends on: `experiment2.py`

### __init__ `def __init__(self, model, device)`
- Defined: `polos.py:50`
- Depends on: `experiment2.py`

### extract_state_space_representation `def extract_state_space_representation(self)`
- Defined: `polos.py:55`
- Depends on: `experiment2.py`

### compute_transfer_function `def compute_transfer_function(self, A, B, C, D)`
- Defined: `polos.py:105`
- Depends on: `experiment2.py`

### __init__ `def __init__(self, numerator, denominator)`
- Defined: `polos.py:144`
- Depends on: `experiment2.py`

### _compute_poles_zeros `def _compute_poles_zeros(self)`
- Defined: `polos.py:153`
- Depends on: `experiment2.py`

### analyze_stability `def analyze_stability(self)`
- Defined: `polos.py:166`
- Depends on: `experiment2.py`

### classify_poles `def classify_poles(self)`
- Defined: `polos.py:207`
- Depends on: `experiment2.py`

### compute_damping_frequency `def compute_damping_frequency(self)`
- Defined: `polos.py:238`
- Depends on: `experiment2.py`

### compute_time_constants `def compute_time_constants(self)`
- Defined: `polos.py:278`
- Depends on: `experiment2.py`

### __init__ `def __init__(self, numerator, denominator)`
- Defined: `polos.py:301`
- Depends on: `experiment2.py`

### compute_bode_plot_data `def compute_bode_plot_data(self)`
- Defined: `polos.py:309`
- Depends on: `experiment2.py`

### compute_gain_phase_margins `def compute_gain_phase_margins(self)`
- Defined: `polos.py:328`
- Depends on: `experiment2.py`

### compute_nyquist_data `def compute_nyquist_data(self)`
- Defined: `polos.py:357`
- Depends on: `experiment2.py`

### evaluate_nyquist_stability `def evaluate_nyquist_stability(self, nyquist_data)`
- Defined: `polos.py:379`
- Depends on: `experiment2.py`

### __init__ `def __init__(self, numerator, denominator)`
- Defined: `polos.py:417`
- Depends on: `experiment2.py`

### compute_step_response `def compute_step_response(self)`
- Defined: `polos.py:425`
- Depends on: `experiment2.py`

### compute_impulse_response `def compute_impulse_response(self)`
- Defined: `polos.py:441`
- Depends on: `experiment2.py`

### analyze_step_response_characteristics `def analyze_step_response_characteristics(self, step_data)`
- Defined: `polos.py:457`
- Depends on: `experiment2.py`

### __init__ `def __init__(self, poles, zeros)`
- Defined: `polos.py:518`
- Depends on: `experiment2.py`

### design_pid_controller `def design_pid_controller(self, desired_damping, desired_settling_time)`
- Defined: `polos.py:522`
- Depends on: `experiment2.py`

### design_lead_compensator `def design_lead_compensator(self, desired_phase_margin)`
- Defined: `polos.py:542`
- Depends on: `experiment2.py`

### compute_root_locus `def compute_root_locus(self, num, den)`
- Defined: `polos.py:572`
- Depends on: `experiment2.py`

### __init__ `def __init__(self, checkpoint_path, device)`
- Defined: `polos.py:601`
- Depends on: `experiment2.py`

### analyze_complete_system `def analyze_complete_system(self)`
- Defined: `polos.py:625`
- Depends on: `experiment2.py`

### _print_report `def _print_report(self, results)`
- Defined: `polos.py:715`
- Depends on: `experiment2.py`

### plot_pole_zero_map `def plot_pole_zero_map(poles, zeros, output_path)`
- Defined: `polos.py:801`
- Depends on: `experiment2.py`

### plot_bode_diagram `def plot_bode_diagram(bode_data, margins, output_path)`
- Defined: `polos.py:865`
- Depends on: `experiment2.py`

### plot_nyquist_diagram `def plot_nyquist_diagram(nyquist_data, output_path)`
- Defined: `polos.py:940`
- Depends on: `experiment2.py`

### plot_time_responses `def plot_time_responses(step_data, impulse_data, output_path)`
- Defined: `polos.py:990`
- Depends on: `experiment2.py`

### plot_root_locus `def plot_root_locus(root_locus_data, output_path)`
- Defined: `polos.py:1036`
- Depends on: `experiment2.py`

### plot_combined_analysis `def plot_combined_analysis(poles, zeros, bode_data, step_data, output_path)`
- Defined: `polos.py:1096`
- Depends on: `experiment2.py`

## precision.py

### main `def main()`
- Defined: `precision.py:506`
- Depends on: `experiment2.py`, `refinamiento.py`

### __init__ `def __init__(self, lambda_quant)`
- Defined: `precision.py:37`
- Depends on: `experiment2.py`, `refinamiento.py`

### quantization_penalty `def quantization_penalty(self, model)`
- Defined: `precision.py:42`
- Depends on: `experiment2.py`, `refinamiento.py`

### forward `def forward(self, predictions, targets, model)`
- Defined: `precision.py:54`
- Depends on: `experiment2.py`, `refinamiento.py`

### __init__ `def __init__(self, checkpoint_path, device)`
- Defined: `precision.py:73`
- Depends on: `experiment2.py`, `refinamiento.py`

### _setup_logger `def _setup_logger(self)`
- Defined: `precision.py:150`
- Depends on: `experiment2.py`, `refinamiento.py`

### _find_latest_checkpoint `def _find_latest_checkpoint(self)`
- Defined: `precision.py:162`
- Depends on: `experiment2.py`, `refinamiento.py`

### _compute_initial_metrics `def _compute_initial_metrics(self, model)`
- Defined: `precision.py:191`
- Depends on: `experiment2.py`, `refinamiento.py`

### compute_discretization_metrics `def compute_discretization_metrics(self)`
- Defined: `precision.py:208`
- Depends on: `experiment2.py`, `refinamiento.py`

### validate `def validate(self)`
- Defined: `precision.py:241`
- Depends on: `experiment2.py`, `refinamiento.py`

### train_epoch `def train_epoch(self, epoch)`
- Defined: `precision.py:250`
- Depends on: `experiment2.py`, `refinamiento.py`

### refine `def refine(self)`
- Defined: `precision.py:288`
- Depends on: `experiment2.py`, `refinamiento.py`

### _save_latest_checkpoint `def _save_latest_checkpoint(self, epoch, metrics, val_acc)`
- Defined: `precision.py:430`
- Doc: Guarda/sobrescribe latest.pth - rápido, para danger zone
- Depends on: `experiment2.py`, `refinamiento.py`

### _save_crystal_checkpoint `def _save_crystal_checkpoint(self, epoch, metrics, val_acc, final, force_save, emergency)`
- Defined: `precision.py:456`
- Depends on: `experiment2.py`, `refinamiento.py`

### _compile_results `def _compile_results(self, success, final_epoch)`
- Defined: `precision.py:490`
- Depends on: `experiment2.py`, `refinamiento.py`

## refinamiento.py

### analyze_discretization `def analyze_discretization(checkpoint_path)`
- Defined: `refinamiento.py:498`
- Doc: Análisis detallado de la discretización de un checkpoint
- Depends on: `experiment2.py`
- Imported by: `precision.py`

### main `def main()`
- Defined: `refinamiento.py:575`
- Depends on: `experiment2.py`
- Imported by: `precision.py`

### __init__ `def __init__(self, lambda_quant)`
- Defined: `refinamiento.py:62`
- Depends on: `experiment2.py`
- Imported by: `precision.py`

### quantization_penalty `def quantization_penalty(self, model)`
- Defined: `refinamiento.py:67`
- Doc: Penalización L2 de la distancia al entero más cercano
- Depends on: `experiment2.py`
- Imported by: `precision.py`

### forward `def forward(self, predictions, targets, model)`
- Defined: `refinamiento.py:81`
- Depends on: `experiment2.py`
- Imported by: `precision.py`

### __init__ `def __init__(self, thresholds)`
- Defined: `refinamiento.py:98`
- Depends on: `experiment2.py`
- Imported by: `precision.py`

### should_prune `def should_prune(self, epoch)`
- Defined: `refinamiento.py:103`
- Doc: Determina si es momento de podar (cada 500 épocas)
- Depends on: `experiment2.py`
- Imported by: `precision.py`

### prune `def prune(self, model, force_threshold)`
- Defined: `refinamiento.py:107`
- Doc: Poda pesos con |w| < threshold
- Depends on: `experiment2.py`
- Imported by: `precision.py`

### get_sparsity `def get_sparsity(self, model)`
- Defined: `refinamiento.py:131`
- Doc: Calcula porcentaje de pesos exactamente en cero
- Depends on: `experiment2.py`
- Imported by: `precision.py`

### __init__ `def __init__(self, checkpoint_path, device)`
- Defined: `refinamiento.py:148`
- Depends on: `experiment2.py`
- Imported by: `precision.py`

### _setup_logger `def _setup_logger(self)`
- Defined: `refinamiento.py:186`
- Depends on: `experiment2.py`
- Imported by: `precision.py`

### _load_checkpoint `def _load_checkpoint(self)`
- Defined: `refinamiento.py:198`
- Doc: Carga el checkpoint y retorna modelo, época y métricas
- Depends on: `experiment2.py`
- Imported by: `precision.py`

### _compute_initial_metrics `def _compute_initial_metrics(self, model)`
- Defined: `refinamiento.py:234`
- Doc: Calcula métricas iniciales si no vienen en el checkpoint
- Depends on: `experiment2.py`
- Imported by: `precision.py`

### compute_discretization_metrics `def compute_discretization_metrics(self)`
- Defined: `refinamiento.py:253`
- Doc: Calcula métricas de cristalinidad actuales
- Depends on: `experiment2.py`
- Imported by: `precision.py`

### validate `def validate(self)`
- Defined: `refinamiento.py:291`
- Doc: Valida el modelo manteniendo accuracy
- Depends on: `experiment2.py`
- Imported by: `precision.py`

### train_epoch `def train_epoch(self, epoch)`
- Defined: `refinamiento.py:302`
- Doc: Entrena una época con pérdida de cuantización
- Depends on: `experiment2.py`
- Imported by: `precision.py`

### refine `def refine(self)`
- Defined: `refinamiento.py:344`
- Doc: Ejecuta el refinamiento hasta alcanzar δ < TARGET_DELTA o MAX_EPOCHS
- Depends on: `experiment2.py`
- Imported by: `precision.py`

### _save_crystal_checkpoint `def _save_crystal_checkpoint(self, epoch, metrics, val_acc, final)`
- Defined: `refinamiento.py:459`
- Doc: Guarda checkpoint cristalino
- Depends on: `experiment2.py`
- Imported by: `precision.py`

### _compile_results `def _compile_results(self, success, final_epoch)`
- Defined: `refinamiento.py:482`
- Doc: Compila resultados finales
- Depends on: `experiment2.py`
- Imported by: `precision.py`

## test_grokkit.py

### run_quick_test `def run_quick_test()`
- Defined: `test_grokkit.py:507`
- Doc: Executes a quick validation test with minimal output.
- Depends on: `audio/main.py`

### __init__ `def __init__(self, weights_dir)`
- Defined: `test_grokkit.py:56`
- Depends on: `audio/main.py`

### load_model `def load_model(self)`
- Defined: `test_grokkit.py:64`
- Doc: Loads the trained model from checkpoint.
- Depends on: `audio/main.py`

### generate_test_dataset `def generate_test_dataset(self, num_samples)`
- Defined: `test_grokkit.py:119`
- Doc: Generates test dataset using the true Hamiltonian operator.
- Depends on: `audio/main.py`

### compute_local_complexity `def compute_local_complexity(self, model)`
- Defined: `test_grokkit.py:158`
- Doc: Computes Local Complexity (LC) metric for the model.
- Depends on: `audio/main.py`

### compute_superposition `def compute_superposition(self, model)`
- Defined: `test_grokkit.py:181`
- Doc: Computes Superposition (SP) metric for the model.
- Depends on: `audio/main.py`

### compute_operator_error `def compute_operator_error(self, model, inputs, targets)`
- Defined: `test_grokkit.py:203`
- Doc: Computes operator approximation error.
- Depends on: `audio/main.py`

### compute_spectral_gap `def compute_spectral_gap(self, model)`
- Defined: `test_grokkit.py:228`
- Doc: Estimates the spectral gap in weight singular values.
- Depends on: `audio/main.py`

### run_validation `def run_validation(self)`
- Defined: `test_grokkit.py:259`
- Doc: Executes the complete validation suite.
- Depends on: `audio/main.py`

### generate_report `def generate_report(self)`
- Defined: `test_grokkit.py:382`
- Doc: Generates a formal validation report.
- Depends on: `audio/main.py`

## verify.py

### verify_latest_checkpoints `def verify_latest_checkpoints(checkpoint_dir, n)`
- Defined: `verify.py:444`
- Doc: Verifica los N checkpoints más recientes
- Depends on: `experiment2.py`

### main `def main()`
- Defined: `verify.py:486`
- Depends on: `experiment2.py`

### __init__ `def __init__(self, checkpoint_path, device)`
- Defined: `verify.py:15`
- Depends on: `experiment2.py`

### verify_all_metrics `def verify_all_metrics(self)`
- Defined: `verify.py:50`
- Doc: Calcula TODAS las métricas desde cero y compara con las guardadas
- Depends on: `experiment2.py`

### _check_weight_integrity `def _check_weight_integrity(self)`
- Defined: `verify.py:101`
- Doc: Verifica que los pesos no tengan NaN/Inf
- Depends on: `experiment2.py`

### _compute_validation_metrics `def _compute_validation_metrics(self)`
- Defined: `verify.py:146`
- Doc: Calcula MSE y accuracy de validación desde cero
- Depends on: `experiment2.py`

### _compute_discretization_metrics `def _compute_discretization_metrics(self)`
- Defined: `verify.py:173`
- Doc: Calcula delta, alpha, purity, etc.
- Depends on: `experiment2.py`

### _compute_quantization_metrics `def _compute_quantization_metrics(self)`
- Defined: `verify.py:225`
- Doc: Calcula la penalización de cuantización
- Depends on: `experiment2.py`

### _compute_loss_metrics `def _compute_loss_metrics(self)`
- Defined: `verify.py:244`
- Doc: Reconstruye el loss total
- Depends on: `experiment2.py`

### _compare_with_stored `def _compare_with_stored(self, computed)`
- Defined: `verify.py:269`
- Doc: Compara métricas calculadas vs almacenadas en el checkpoint
- Depends on: `experiment2.py`

### _check_internal_consistency `def _check_internal_consistency(self, results)`
- Defined: `verify.py:302`
- Doc: Verifica consistencia entre métricas relacionadas
- Depends on: `experiment2.py`

### _compute_health_score `def _compute_health_score(self, results)`
- Defined: `verify.py:331`
- Doc: Calcula un score de salud del checkpoint (0-100)
- Depends on: `experiment2.py`

### _print_report `def _print_report(self, results)`
- Defined: `verify.py:374`
- Doc: Imprime reporte formateado
- Depends on: `experiment2.py`
