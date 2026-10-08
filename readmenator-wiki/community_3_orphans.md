# orphans

*Community 3 | 9 files | cohesion 0.00*

## Definition

This community groups 9 file(s) rooted at `root` with dominant language py (cohesion 0.00). Central symbols: `ArchitectureMigrator`, `BoltzmannAnalysisProgram`, `CheckpointManager`, `CheckpointMigrator`, `Config`, `CrystallinityIndexCalculator`, `CrystallographyMetrics`, `DiscretizationDialAnalyzer`. Core file: `hamiltonian_mbl.py` (126 symbols). Documented purpose: Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: xx/xx/xxxx Licencia: GPL v3  Descripción:  Hamiltonian Grokking - .

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `app.py` | py | utility | 22 | yes |
| `diff_weights.py` | py | utility | 1 | no |
| `expand.py` | py | utility | 5 | yes |
| `experiment.py` | py | utility | 57 | yes |
| `export.py` | py | utility | 0 | no |
| `hamiltonian_mbl.py` | py | utility | 126 | yes |
| `install.sh` | sh | utility | 0 | no |
| `mining_seeds.py` | py | data_access | 57 | yes |
| `plank.py` | py | utility | 5 | yes |

## Key Symbols

- `SimpleConfig` (class, `app.py:29`) `class SimpleConfig`
- `__init__` (method, `app.py:30`) `def __init__(self, grid_size, hidden_dim, num_spectral_layers, target_accuracy,`
- `compute_local_complexity` (method, `app.py:45`) `def compute_local_complexity(weights, epsilon)` - Compute Local Complexity (LC) metric for weight matrix.
- `compute_superposition` (method, `app.py:60`) `def compute_superposition(weights)` - Compute Superposition (SP) metric for weight matrix.
- `HamiltonianOperator` (class, `app.py:88`) `class HamiltonianOperator` - True Hamiltonian operator H = -nabla^2 on torus.
- `__init__` (method, `app.py:91`) `def __init__(self, grid_size)`
- `_precompute_spectral_operators` (method, `app.py:95`) `def _precompute_spectral_operators(self)`
- `apply` (method, `app.py:102`) `def apply(self, field)`
- `time_evolution` (method, `app.py:107`) `def time_evolution(self, field, dt)`
- `FastDataset` (class, `app.py:113`) `class FastDataset(Dataset)` - Fast dataset for Hamiltonian operator learning.
- `__init__` (method, `app.py:116`) `def __init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)`
- `__len__` (method, `app.py:160`) `def __len__(self)`
- `__getitem__` (method, `app.py:163`) `def __getitem__(self, idx)`
- `get_val_batch` (method, `app.py:166`) `def get_val_batch(self)`
- `SpectralLayer` (class, `app.py:170`) `class SpectralLayer(Module)` - Spectral layer with correct complex multiplication.
- `__init__` (method, `app.py:173`) `def __init__(self, channels, grid_size)`
- `forward` (method, `app.py:186`) `def forward(self, x)`
- `SimpleHamiltonianNet` (class, `app.py:222`) `class SimpleHamiltonianNet(Module)` - Compact network for Hamiltonian operator learning.
- `__init__` (method, `app.py:225`) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `forward` (method, `app.py:246`) `def forward(self, x)`
- `train_model` (method, `app.py:260`) `def train_model(grid_size, epochs, hidden_dim, num_spectral_layers, lr)` - Train the Hamiltonian operator model.
- `main` (method, `app.py:369`) `def main()`
- `analize_checkpoint` (function, `diff_weights.py:5`) `def analize_checkpoint(path)`
- `load_config` (function, `expand.py:18`) `def load_config(toml_path)`
- `expand_spectral_weights` (function, `expand.py:23`) `def expand_spectral_weights(kernel_real, kernel_imag, target_size, source_size)` - Expand spectral kernels via zero-padding in frequency domain.
- `expand_model` (function, `expand.py:43`) `def expand_model(model, target_resolution, source_resolution)` - Create a new model with expanded spectral weights.
- `evaluate_model` (function, `expand.py:74`) `def evaluate_model(model, resolution, device)` - Evaluate expanded model on synthetic data.
- `main` (function, `expand.py:105`) `def main()`
- `Config` (class, `experiment.py:41`) `class Config`
- `set_seed` (method, `experiment.py:88`) `def set_seed(seed)`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 0
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- [INFERRED] shares_context community 0 <-> 3 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 0 (audio: metrics) and community 3 (orphans).
- [INFERRED] shares_context community 1 <-> 3 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 1 (root) and community 3 (orphans).
- [INFERRED] shares_context community 2 <-> 3 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 2 (audio: experiment2) and community 3 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 3 file(s) lack file-level docs (e.g. `diff_weights.py`)? What purpose do they serve?
- What would break if the most connected file in orphans changed?
- Should orphans be split, given cohesion 0.00?

## Sources

- `app.py`
- `diff_weights.py`
- `expand.py`
- `experiment.py`
- `export.py`
- `hamiltonian_mbl.py`
- `install.sh`
- `mining_seeds.py`
- `plank.py`
