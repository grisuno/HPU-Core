# audio: metrics

*Community 0 | 11 files | cohesion 1.00*

## Definition

This community groups 11 file(s) rooted at `audio` with dominant language py (cohesion 1.00). Central symbols: `AudioProcessingConfig`, `AudioProcessor`, `AudioSpectrogramDatasetBuilder`, `CheckpointConfig`, `CheckpointManager`, `GrokkingValidator`, `HamiltonianAudioConfig`, `HamiltonianAudioInference`. Core file: `audio/metrics.py` (25 symbols). Documented purpose: Audio I/O and Spectral Transform Module.  Core design principle: audio is processed in the COMPLEX STFT domain, which is the natural 2D (time x frequency) compl.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `audio/audio_io.py` | py | utility | 12 | yes |
| `audio/checkpoint_manager.py` | py | utility | 6 | yes |
| `audio/config.py` | py | infrastructure | 13 | yes |
| `audio/inference.py` | py | utility | 7 | yes |
| `audio/losses.py` | py | utility | 11 | yes |
| `audio/main.py` | py | utility | 7 | yes |
| `audio/metrics.py` | py | utility | 25 | yes |
| `audio/model.py` | py | business_logic | 11 | yes |
| `audio/trainer.py` | py | utility | 11 | yes |
| `audio/visualization.py` | py | utility | 8 | yes |
| `test_grokkit.py` | py | testing | 11 | yes |

## Key Symbols

- `AudioProcessor` (class, `audio/audio_io.py:31`) `class AudioProcessor` - Audio processing pipeline centered on the complex STFT domain.
- `__init__` (method, `audio/audio_io.py:41`) `def __init__(self, config, device)`
- `load_audio` (method, `audio/audio_io.py:57`) `def load_audio(self, file_path)` - Load an audio file and convert to mono at the target sample rate.
- `waveform_to_stft_complex` (method, `audio/audio_io.py:80`) `def waveform_to_stft_complex(self, waveform)` - Compute the complex STFT of a waveform.
- `stft_complex_to_waveform` (method, `audio/audio_io.py:105`) `def stft_complex_to_waveform(self, stft_complex)` - Reconstruct waveform from complex STFT via inverse STFT.
- `stft_to_magnitude_phase` (method, `audio/audio_io.py:130`) `def stft_to_magnitude_phase(self, stft_complex)` - Decompose complex STFT into magnitude and phase.
- `magnitude_phase_to_stft` (method, `audio/audio_io.py:146`) `def magnitude_phase_to_stft(self, magnitude, phase)` - Recombine magnitude and phase into complex STFT.
- `stft_magnitude_to_model_input` (method, `audio/audio_io.py:161`) `def stft_magnitude_to_model_input(self, magnitude)` - Prepare STFT magnitude for input to the Hamiltonian network.
- `model_output_to_stft_magnitude` (method, `audio/audio_io.py:186`) `def model_output_to_stft_magnitude(self, model_output, original_magnitude)` - Convert model output (energy mask in [0, 1]) back to STFT magnitude scale.
- `waveform_to_mel_spectrogram` (method, `audio/audio_io.py:206`) `def waveform_to_mel_spectrogram(self, waveform)` - Convert waveform to normalized mel spectrogram (for visualization only).
- `save_audio` (method, `audio/audio_io.py:229`) `def save_audio(self, waveform, file_path, sample_rate)` - Save a waveform tensor to an audio file.
- `get_spectrogram_db_range` (method, `audio/audio_io.py:249`) `def get_spectrogram_db_range(self, waveform)` - Compute the dB range of a waveform's mel spectrogram.
- `CheckpointManager` (class, `audio/checkpoint_manager.py:28`) `class CheckpointManager` - Manages model checkpointing with time-based intervals
- `__init__` (method, `audio/checkpoint_manager.py:34`) `def __init__(self, config)`
- `should_save_checkpoint` (method, `audio/checkpoint_manager.py:41`) `def should_save_checkpoint(self)` - Check if enough time has elapsed since the last checkpoint.
- `save_checkpoint` (method, `audio/checkpoint_manager.py:46`) `def save_checkpoint(self, model, optimizer, scheduler, epoch, step, metrics, cur` - Save the current model state and training metadata.
- `load_checkpoint` (method, `audio/checkpoint_manager.py:98`) `def load_checkpoint(self, model, load_best)` - Load a model checkpoint and return training metadata.
- `best_loss` (method, `audio/checkpoint_manager.py:156`) `def best_loss(self)`
- `AudioProcessingConfig` (class, `audio/config.py:18`) `class AudioProcessingConfig` - Parameters governing raw audio ingestion and spectrogram computation.
- `ModelArchitectureConfig` (class, `audio/config.py:34`) `class ModelArchitectureConfig` - Parametric architecture dimensions for the Hamiltonian Neural Network.
- `validate` (method, `audio/config.py:62`) `def validate(self)` - Ensure architectural coherence.
- `TrainingConfig` (class, `audio/config.py:80`) `class TrainingConfig` - All training loop hyperparameters and scheduling constants.
- `CheckpointConfig` (class, `audio/config.py:109`) `class CheckpointConfig` - Checkpoint persistence parameters.
- `checkpoint_path` (method, `audio/config.py:121`) `def checkpoint_path(self)`
- `best_model_path` (method, `audio/config.py:127`) `def best_model_path(self)`
- `metadata_path` (method, `audio/config.py:131`) `def metadata_path(self)`
- `VisualizationConfig` (class, `audio/config.py:136`) `class VisualizationConfig` - Parameters for audio reconstruction visualization and output.
- `MetricsConfig` (class, `audio/config.py:158`) `class MetricsConfig` - Configuration for all tracked metrics during training and inference.
- `HamiltonianAudioConfig` (class, `audio/config.py:182`) `class HamiltonianAudioConfig` - Top-level configuration aggregator.
- `validate_all` (method, `audio/config.py:198`) `def validate_all(self)` - Run validation on all sub-configurations.

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 22
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- [INFERRED] shares_context community 0 <-> 1 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 0 (audio: metrics) and community 1 (root).
- [INFERRED] shares_context community 0 <-> 2 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 0 (audio: metrics) and community 2 (audio: experiment2).
- [INFERRED] shares_context community 0 <-> 3 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 0 (audio: metrics) and community 3 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- What would break if the most connected file in audio: metrics changed?
- Should audio: metrics be split, given cohesion 1.00?

## Sources

- `audio/audio_io.py`
- `audio/checkpoint_manager.py`
- `audio/config.py`
- `audio/inference.py`
- `audio/losses.py`
- `audio/main.py`
- `audio/metrics.py`
- `audio/model.py`
- `audio/trainer.py`
- `audio/visualization.py`
- `test_grokkit.py`
