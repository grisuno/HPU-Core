# Subsystem: root (page 2 of 2)
Previous: [KB_root.md](KB_root.md)

## plank.py
- Doc: Calcula la constante de Planck efectiva (ħ) desde checkpoints de HPU.
- Layer: utility
- Language: py
- Symbols:
  - `HBarCalculator` (class, line 26) `class HBarCalculator`
  - `main` (method, line 213) `def main()`
  - `__init__` (method, line 29) `def __init__(self, checkpoint_path, device)`
  - `calculate_all` (method, line 54) `def calculate_all(self)`
  - `print_report` (method, line 170) `def print_report(self, results)`

## polos.py
- Layer: utility
- Language: py
- Symbols:
  - `ControlConfig` (class, line 27) `class ControlConfig`
  - `TransferFunctionExtractor` (class, line 48) `class TransferFunctionExtractor`
  - `PoleZeroAnalyzer` (class, line 142) `class PoleZeroAnalyzer`
  - `FrequencyResponseAnalyzer` (class, line 299) `class FrequencyResponseAnalyzer`
  - `TimeResponseAnalyzer` (class, line 415) `class TimeResponseAnalyzer`
  - `ControllerDesigner` (class, line 516) `class ControllerDesigner`
  - `ControlSystemAnalyzer` (class, line 599) `class ControlSystemAnalyzer`
  - `ControlVisualizer` (class, line 798) `class ControlVisualizer`
  - `analyze_checkpoint` (method, line 1152) `def analyze_checkpoint(checkpoint_path, output_dir)`
  - `analyze_multiple_checkpoints` (method, line 1208) `def analyze_multiple_checkpoints(checkpoint_dir, n_latest, output_dir)`
  - `main` (method, line 1240) `def main()`
  - `__init__` (method, line 50) `def __init__(self, model, device)`
  - `extract_state_space_representation` (method, line 55) `def extract_state_space_representation(self)`
  - `compute_transfer_function` (method, line 105) `def compute_transfer_function(self, A, B, C, D)`
  - `__init__` (method, line 144) `def __init__(self, numerator, denominator)`
  - `_compute_poles_zeros` (method, line 153) `def _compute_poles_zeros(self)`
  - `analyze_stability` (method, line 166) `def analyze_stability(self)`
  - `classify_poles` (method, line 207) `def classify_poles(self)`
  - `compute_damping_frequency` (method, line 238) `def compute_damping_frequency(self)`
  - `compute_time_constants` (method, line 278) `def compute_time_constants(self)`
  - `__init__` (method, line 301) `def __init__(self, numerator, denominator)`
  - `compute_bode_plot_data` (method, line 309) `def compute_bode_plot_data(self)`
  - `compute_gain_phase_margins` (method, line 328) `def compute_gain_phase_margins(self)`
  - `compute_nyquist_data` (method, line 357) `def compute_nyquist_data(self)`
  - `evaluate_nyquist_stability` (method, line 379) `def evaluate_nyquist_stability(self, nyquist_data)`
  - `__init__` (method, line 417) `def __init__(self, numerator, denominator)`
  - `compute_step_response` (method, line 425) `def compute_step_response(self)`
  - `compute_impulse_response` (method, line 441) `def compute_impulse_response(self)`
  - `analyze_step_response_characteristics` (method, line 457) `def analyze_step_response_characteristics(self, step_data)`
  - `__init__` (method, line 518) `def __init__(self, poles, zeros)`
  - `design_pid_controller` (method, line 522) `def design_pid_controller(self, desired_damping, desired_settling_time)`
  - `design_lead_compensator` (method, line 542) `def design_lead_compensator(self, desired_phase_margin)`
  - `compute_root_locus` (method, line 572) `def compute_root_locus(self, num, den)`
  - `__init__` (method, line 601) `def __init__(self, checkpoint_path, device)`
  - `analyze_complete_system` (method, line 625) `def analyze_complete_system(self)`
  - `_print_report` (method, line 715) `def _print_report(self, results)`
  - `plot_pole_zero_map` (method, line 801) `def plot_pole_zero_map(poles, zeros, output_path)`
  - `plot_bode_diagram` (method, line 865) `def plot_bode_diagram(bode_data, margins, output_path)`
  - `plot_nyquist_diagram` (method, line 940) `def plot_nyquist_diagram(nyquist_data, output_path)`
  - `plot_time_responses` (method, line 990) `def plot_time_responses(step_data, impulse_data, output_path)`
  - `plot_root_locus` (method, line 1036) `def plot_root_locus(root_locus_data, output_path)`
  - `plot_combined_analysis` (method, line 1096) `def plot_combined_analysis(poles, zeros, bode_data, step_data, output_path)`
- Depends on: `experiment2.py`

## precision.py
- Doc: _save_latest_checkpoint: Guarda/sobrescribe latest.pth - rápido, para danger zone
- Layer: utility
- Language: py
- Symbols:
  - `MassiveLambdaConfig` (class, line 26) `class MassiveLambdaConfig`
  - `CrystallizationLossMassive` (class, line 36) `class CrystallizationLossMassive(Module)`
  - `ContinuationEngine` (class, line 72) `class ContinuationEngine`
  - `main` (method, line 506) `def main()`
  - `__init__` (method, line 37) `def __init__(self, lambda_quant)`
  - `quantization_penalty` (method, line 42) `def quantization_penalty(self, model)`
  - `forward` (method, line 54) `def forward(self, predictions, targets, model)`
  - `__init__` (method, line 73) `def __init__(self, checkpoint_path, device)`
  - `_setup_logger` (method, line 150) `def _setup_logger(self)`
  - `_find_latest_checkpoint` (method, line 162) `def _find_latest_checkpoint(self)`
  - `_compute_initial_metrics` (method, line 191) `def _compute_initial_metrics(self, model)`
  - `compute_discretization_metrics` (method, line 208) `def compute_discretization_metrics(self)`
  - `validate` (method, line 241) `def validate(self)`
  - `train_epoch` (method, line 250) `def train_epoch(self, epoch)`
  - `refine` (method, line 288) `def refine(self)`
  - `_save_latest_checkpoint` (method, line 430) `def _save_latest_checkpoint(self, epoch, metrics, val_acc)`
  - `_save_crystal_checkpoint` (method, line 456) `def _save_crystal_checkpoint(self, epoch, metrics, val_acc, final, force_save, emergency)`
  - `_compile_results` (method, line 490) `def _compile_results(self, success, final_epoch)`
- Depends on: `experiment2.py`, `refinamiento.py`

## refinamiento.py
- Doc: Script de Refinamiento Cristalino para Discretización de Pesos Carga un checkpoint estable y...
- Layer: utility
- Language: py
- Symbols:
  - `CrystallizationConfig` (class, line 33) `class CrystallizationConfig`
  - `CrystallizationLoss` (class, line 57) `class CrystallizationLoss(Module)`
  - `StructuralPruner` (class, line 96) `class StructuralPruner`
  - `CrystallizationEngine` (class, line 144) `class CrystallizationEngine`
  - `analyze_discretization` (method, line 498) `def analyze_discretization(checkpoint_path)`
  - `main` (method, line 575) `def main()`
  - `__init__` (method, line 62) `def __init__(self, lambda_quant)`
  - `quantization_penalty` (method, line 67) `def quantization_penalty(self, model)`
  - `forward` (method, line 81) `def forward(self, predictions, targets, model)`
  - `__init__` (method, line 98) `def __init__(self, thresholds)`
  - `should_prune` (method, line 103) `def should_prune(self, epoch)`
  - `prune` (method, line 107) `def prune(self, model, force_threshold)`
  - `get_sparsity` (method, line 131) `def get_sparsity(self, model)`
  - `__init__` (method, line 148) `def __init__(self, checkpoint_path, device)`
  - `_setup_logger` (method, line 186) `def _setup_logger(self)`
  - `_load_checkpoint` (method, line 198) `def _load_checkpoint(self)`
  - `_compute_initial_metrics` (method, line 234) `def _compute_initial_metrics(self, model)`
  - `compute_discretization_metrics` (method, line 253) `def compute_discretization_metrics(self)`
  - `validate` (method, line 291) `def validate(self)`
  - `train_epoch` (method, line 302) `def train_epoch(self, epoch)`
  - `refine` (method, line 344) `def refine(self)`
  - `_save_crystal_checkpoint` (method, line 459) `def _save_crystal_checkpoint(self, epoch, metrics, val_acc, final)`
  - `_compile_results` (method, line 482) `def _compile_results(self, success, final_epoch)`
- Depends on: `experiment2.py`
- Imported by: `precision.py`

## simple_hpu_view.py
- Layer: presentation
- Language: py
- Depends on: `experiment2.py`

## test_grokkit.py
- Doc: Test Suite for Hamiltonian Grokking Experiment.
- Layer: testing
- Language: py
- Symbols:
  - `GrokkingValidator` (class, line 41) `class GrokkingValidator`
  - `run_quick_test` (method, line 507) `def run_quick_test()`
  - `__init__` (method, line 56) `def __init__(self, weights_dir)`
  - `load_model` (method, line 64) `def load_model(self)`
  - `generate_test_dataset` (method, line 119) `def generate_test_dataset(self, num_samples)`
  - `compute_local_complexity` (method, line 158) `def compute_local_complexity(self, model)`
  - `compute_superposition` (method, line 181) `def compute_superposition(self, model)`
  - `compute_operator_error` (method, line 203) `def compute_operator_error(self, model, inputs, targets)`
  - `compute_spectral_gap` (method, line 228) `def compute_spectral_gap(self, model)`
  - `run_validation` (method, line 259) `def run_validation(self)`
  - `generate_report` (method, line 382) `def generate_report(self)`
- Depends on: `audio/main.py`

## verify.py
- Doc: verify_latest_checkpoints: Verifica los N checkpoints más recientes
- Layer: utility
- Language: py
- Symbols:
  - `CheckpointVerifier` (class, line 14) `class CheckpointVerifier`
  - `verify_latest_checkpoints` (method, line 444) `def verify_latest_checkpoints(checkpoint_dir, n)`
  - `main` (method, line 486) `def main()`
  - `__init__` (method, line 15) `def __init__(self, checkpoint_path, device)`
  - `verify_all_metrics` (method, line 50) `def verify_all_metrics(self)`
  - `_check_weight_integrity` (method, line 101) `def _check_weight_integrity(self)`
  - `_compute_validation_metrics` (method, line 146) `def _compute_validation_metrics(self)`
  - `_compute_discretization_metrics` (method, line 173) `def _compute_discretization_metrics(self)`
  - `_compute_quantization_metrics` (method, line 225) `def _compute_quantization_metrics(self)`
  - `_compute_loss_metrics` (method, line 244) `def _compute_loss_metrics(self)`
  - `_compare_with_stored` (method, line 269) `def _compare_with_stored(self, computed)`
  - `_check_internal_consistency` (method, line 302) `def _check_internal_consistency(self, results)`
  - `_compute_health_score` (method, line 331) `def _compute_health_score(self, results)`
  - `_print_report` (method, line 374) `def _print_report(self, results)`
- Depends on: `experiment2.py`

