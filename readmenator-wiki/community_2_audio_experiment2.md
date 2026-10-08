# audio: experiment2

*Community 2 | 2 files | cohesion 1.00*

## Definition

This community groups 2 file(s) rooted at `audio` with dominant language py (cohesion 1.00). Central symbols: `Application`, `AudioResampler`, `AudioSpectrogramConverter`, `CheckpointAnalyzer`, `CheckpointManager`, `ComprehensiveMetricCollector`, `Config`, `CrystallographyMetricsCalculator`. Core file: `audio/experiment2.py` (83 symbols). Documented purpose: Hamiltonian Perception Unit - Audio Modality A SOLID-compliant implementation demonstrating that sensory perception is an epiphenomenon of underlying Hamiltonia.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `audio/audios.py` | py | utility | 47 | yes |
| `audio/experiment2.py` | py | utility | 83 | no |

## Key Symbols

- `HamiltonianConfig` (class, `audio/audios.py:48`) `class HamiltonianConfig` - Immutable configuration container for all hyperparameters.
- `segment_samples` (method, `audio/audios.py:89`) `def segment_samples(self)` - Calculate segment length in samples.
- `freq_bins` (method, `audio/audios.py:94`) `def freq_bins(self)` - Calculate frequency bins for real FFT.
- `IAudioSource` (class, `audio/audios.py:103`) `class IAudioSource(ABC)` - Interface for audio input sources.
- `read_segment` (method, `audio/audios.py:107`) `def read_segment(self)` - Read audio segment. Returns None when exhausted.
- `get_properties` (method, `audio/audios.py:112`) `def get_properties(self)` - Return audio properties.
- `close` (method, `audio/audios.py:117`) `def close(self)` - Release resources.
- `IFieldOperator` (class, `audio/audios.py:122`) `class IFieldOperator(ABC)` - Interface for Hamiltonian field evolution operators.
- `evolve` (method, `audio/audios.py:126`) `def evolve(self, field_state)` - Evolve field state through Hamiltonian dynamics.
- `IMetricCollector` (class, `audio/audios.py:131`) `class IMetricCollector(ABC)` - Interface for training metrics collection.
- `record` (method, `audio/audios.py:135`) `def record(self, metrics)` - Record metric values.
- `get_summary` (method, `audio/audios.py:140`) `def get_summary(self)` - Return aggregated metrics.
- `AudioResampler` (class, `audio/audios.py:149`) `class AudioResampler` - Handles audio resampling using scipy.signal, avoiding librosa/numba dependencies.
- `resample` (method, `audio/audios.py:155`) `def resample(audio, orig_sr, target_sr)` - Resample audio from orig_sr to target_sr using polyphase filtering.
- `load_wav_with_resample` (method, `audio/audios.py:173`) `def load_wav_with_resample(file_path, target_sr)` - Load WAV file and resample to target sample rate.
- `WaveFileSource` (class, `audio/audios.py:204`) `class WaveFileSource(IAudioSource)` - Concrete implementation of audio source from file.
- `__init__` (method, `audio/audios.py:210`) `def __init__(self, file_path, config)`
- `_validate_and_load` (method, `audio/audios.py:220`) `def _validate_and_load(self)` - Validate file format and load with automatic resampling.
- `read_segment` (method, `audio/audios.py:244`) `def read_segment(self)` - Read next audio segment.
- `get_properties` (method, `audio/audios.py:261`) `def get_properties(self)` - Return audio file properties.
- `close` (method, `audio/audios.py:273`) `def close(self)` - Release resources.
- `ComprehensiveMetricCollector` (class, `audio/audios.py:278`) `class ComprehensiveMetricCollector(IMetricCollector)` - Collects all metrics from Hamiltonian paper, activation functions,
- `__init__` (method, `audio/audios.py:284`) `def __init__(self, config)`
- `record` (method, `audio/audios.py:289`) `def record(self, metrics)` - Record comprehensive metrics.
- `get_summary` (method, `audio/audios.py:298`) `def get_summary(self)` - Return statistical summary of all metrics.
- `export_to_json` (method, `audio/audios.py:320`) `def export_to_json(self, path)` - Export full history to JSON.
- `CheckpointManager` (class, `audio/audios.py:326`) `class CheckpointManager` - Manages periodic checkpointing with atomic writes.
- `__init__` (method, `audio/audios.py:331`) `def __init__(self, model, config, checkpoint_dir)`
- `check_and_save` (method, `audio/audios.py:344`) `def check_and_save(self, force)` - Check if checkpoint interval elapsed and save if necessary.
- `_save_checkpoint` (method, `audio/audios.py:357`) `def _save_checkpoint(self)` - Atomic checkpoint save.

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 1
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- [INFERRED] duplicates community 1 <-> 2 (strength 0.53): Inferred duplicated scope: communities 1 and 2 share 66 symbols (Jaccard 0.32), e.g. `Application`, `CheckpointAnalyzer`, `CheckpointManager`, `Config`, `CrystallographyMetricsCalculator`, `GlassStateDetector`. Candidate for consolidation.
- [INFERRED] shares_context community 0 <-> 2 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 0 (audio: metrics) and community 2 (audio: experiment2).
- [INFERRED] shares_context community 2 <-> 3 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 2 (audio: experiment2) and community 3 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 1 file(s) lack file-level docs (e.g. `audio/experiment2.py`)? What purpose do they serve?
- What would break if the most connected file in audio: experiment2 changed?
- Should audio: experiment2 be split, given cohesion 1.00?

## Sources

- `audio/audios.py`
- `audio/experiment2.py`
