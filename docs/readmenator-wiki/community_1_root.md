# root

*Community 1 | 10 files | cohesion 1.00*

## Definition

This community groups 10 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `Application`, `CheckpointAnalyzer`, `CheckpointManager`, `CheckpointVerifier`, `Config`, `ContinuationEngine`, `ControlConfig`, `ControlSystemAnalyzer`. Core file: `experiment2.py` (83 symbols). Documented purpose: Script de Refinamiento Cristalino para Discretización de Pesos Carga un checkpoint estable y fuerza la transición vidrio -> cristal puro.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `check_fase_berry.py` | py | utility | 0 | no |
| `dirac.py` | py | utility | 19 | no |
| `experiment2.py` | py | utility | 83 | no |
| `get_meditions.py` | py | utility | 52 | no |
| `hpu_view.py` | py | presentation | 0 | no |
| `polos.py` | py | utility | 42 | no |
| `precision.py` | py | utility | 18 | no |
| `refinamiento.py` | py | utility | 23 | yes |
| `simple_hpu_view.py` | py | presentation | 0 | no |
| `verify.py` | py | utility | 14 | no |

## Key Symbols

- `DiracConfig` (class, `dirac.py:25`) `class DiracConfig`
- `DiracDeltaAnalyzer` (class, `dirac.py:38`) `class DiracDeltaAnalyzer`
- `__init__` (method, `dirac.py:40`) `def __init__(self, checkpoint_path, device)`
- `extract_charge_distribution` (method, `dirac.py:64`) `def extract_charge_distribution(self)`
- `compute_dirac_delta_approximation` (method, `dirac.py:77`) `def compute_dirac_delta_approximation(self, charge_density)`
- `compute_electric_field` (method, `dirac.py:112`) `def compute_electric_field(self, dirac_data, eval_points)`
- `compute_electric_flux` (method, `dirac.py:157`) `def compute_electric_flux(self, electric_field, surface_points)`
- `compute_divergence` (method, `dirac.py:192`) `def compute_divergence(self, electric_field)`
- `verify_gauss_law` (method, `dirac.py:200`) `def verify_gauss_law(self, dirac_data, flux_data)`
- `analyze_all` (method, `dirac.py:223`) `def analyze_all(self)`
- `_print_report` (method, `dirac.py:279`) `def _print_report(self, results)`
- `DiracVisualizer` (class, `dirac.py:328`) `class DiracVisualizer`
- `plot_charge_distribution` (method, `dirac.py:331`) `def plot_charge_distribution(charge_density, point_positions, point_charges, out`
- `plot_electric_field` (method, `dirac.py:379`) `def plot_electric_field(electric_field, output_path)`
- `plot_divergence` (method, `dirac.py:441`) `def plot_divergence(divergence, output_path)`
- `plot_combined_analysis` (method, `dirac.py:490`) `def plot_combined_analysis(charge_density, point_positions, point_charges, elect`
- `analyze_checkpoint` (method, `dirac.py:574`) `def analyze_checkpoint(checkpoint_path, output_dir)`
- `analyze_multiple_checkpoints` (method, `dirac.py:620`) `def analyze_multiple_checkpoints(checkpoint_dir, n_latest, output_dir)`
- `main` (method, `dirac.py:652`) `def main()`
- `Config` (class, `experiment2.py:22`) `class Config`
- `SeedManager` (class, `experiment2.py:74`) `class SeedManager`
- `set_seed` (method, `experiment2.py:76`) `def set_seed(seed)`
- `LoggerFactory` (class, `experiment2.py:84`) `class LoggerFactory`
- `create_logger` (method, `experiment2.py:86`) `def create_logger(name, level)`
- `IAnalysisStrategy` (class, `experiment2.py:99`) `class IAnalysisStrategy(ABC)`
- `analyze` (method, `experiment2.py:101`) `def analyze(self, model)`
- `IMetricsCalculator` (class, `experiment2.py:105`) `class IMetricsCalculator(ABC)`
- `compute` (method, `experiment2.py:107`) `def compute(self, model)`
- `HamiltonianOperator` (class, `experiment2.py:111`) `class HamiltonianOperator`
- `__init__` (method, `experiment2.py:112`) `def __init__(self, grid_size)`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 10
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- [INFERRED] duplicates community 1 <-> 2 (strength 0.53): Inferred duplicated scope: communities 1 and 2 share 66 symbols (Jaccard 0.32), e.g. `Application`, `CheckpointAnalyzer`, `CheckpointManager`, `Config`, `CrystallographyMetricsCalculator`, `GlassStateDetector`. Candidate for consolidation.
- [INFERRED] shares_context community 0 <-> 1 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 0 (audio: metrics) and community 1 (root).
- [INFERRED] shares_context community 1 <-> 3 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 1 (root) and community 3 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 9 file(s) lack file-level docs (e.g. `check_fase_berry.py`)? What purpose do they serve?
- What would break if the most connected file in root changed?
- Should root be split, given cohesion 1.00?

## Sources

- `check_fase_berry.py`
- `dirac.py`
- `experiment2.py`
- `get_meditions.py`
- `hpu_view.py`
- `polos.py`
- `precision.py`
- `refinamiento.py`
- `simple_hpu_view.py`
- `verify.py`
