# API (page 2 of 2)
Previous: [API.md](API.md)

## polos.py
Depends on: `experiment2.py`
- `TransferFunctionExtractor.__init__` (method) `polos.py:50` `def __init__(self, model, device)`
- `TransferFunctionExtractor.extract_state_space_representation` (method) `polos.py:55` `def extract_state_space_representation(self)`
- `TransferFunctionExtractor.compute_transfer_function` (method) `polos.py:105` `def compute_transfer_function(self, A, B, C, D)`
- `PoleZeroAnalyzer.__init__` (method) `polos.py:144` `def __init__(self, numerator, denominator)`
- `PoleZeroAnalyzer.analyze_stability` (method) `polos.py:166` `def analyze_stability(self)`
- `PoleZeroAnalyzer.classify_poles` (method) `polos.py:207` `def classify_poles(self)`
- `PoleZeroAnalyzer.compute_damping_frequency` (method) `polos.py:238` `def compute_damping_frequency(self)`
- `PoleZeroAnalyzer.compute_time_constants` (method) `polos.py:278` `def compute_time_constants(self)`
- `FrequencyResponseAnalyzer.__init__` (method) `polos.py:301` `def __init__(self, numerator, denominator)`
- `FrequencyResponseAnalyzer.compute_bode_plot_data` (method) `polos.py:309` `def compute_bode_plot_data(self)`
- `FrequencyResponseAnalyzer.compute_gain_phase_margins` (method) `polos.py:328` `def compute_gain_phase_margins(self)`
- `FrequencyResponseAnalyzer.compute_nyquist_data` (method) `polos.py:357` `def compute_nyquist_data(self)`
- `FrequencyResponseAnalyzer.evaluate_nyquist_stability` (method) `polos.py:379` `def evaluate_nyquist_stability(self, nyquist_data)`
- `TimeResponseAnalyzer.__init__` (method) `polos.py:417` `def __init__(self, numerator, denominator)`
- `TimeResponseAnalyzer.compute_step_response` (method) `polos.py:425` `def compute_step_response(self)`
- `TimeResponseAnalyzer.compute_impulse_response` (method) `polos.py:441` `def compute_impulse_response(self)`
- `TimeResponseAnalyzer.analyze_step_response_characteristics` (method) `polos.py:457` `def analyze_step_response_characteristics(self, step_data)`
- `ControllerDesigner.__init__` (method) `polos.py:518` `def __init__(self, poles, zeros)`
- `ControllerDesigner.design_pid_controller` (method) `polos.py:522` `def design_pid_controller(self, desired_damping, desired_settling_time)`
- `ControllerDesigner.design_lead_compensator` (method) `polos.py:542` `def design_lead_compensator(self, desired_phase_margin)`
- `ControllerDesigner.compute_root_locus` (method) `polos.py:572` `def compute_root_locus(self, num, den)`
- `ControlSystemAnalyzer.__init__` (method) `polos.py:601` `def __init__(self, checkpoint_path, device)`
- `ControlSystemAnalyzer.analyze_complete_system` (method) `polos.py:625` `def analyze_complete_system(self)`
- `ControlVisualizer.plot_pole_zero_map` (method) `polos.py:801` `def plot_pole_zero_map(poles, zeros, output_path)`
- `ControlVisualizer.plot_bode_diagram` (method) `polos.py:865` `def plot_bode_diagram(bode_data, margins, output_path)`
- `ControlVisualizer.plot_nyquist_diagram` (method) `polos.py:940` `def plot_nyquist_diagram(nyquist_data, output_path)`
- `ControlVisualizer.plot_time_responses` (method) `polos.py:990` `def plot_time_responses(step_data, impulse_data, output_path)`
- `ControlVisualizer.plot_root_locus` (method) `polos.py:1036` `def plot_root_locus(root_locus_data, output_path)`
- `ControlVisualizer.plot_combined_analysis` (method) `polos.py:1096` `def plot_combined_analysis(poles, zeros, bode_data, step_data, output_path)`
- `ControlVisualizer.analyze_checkpoint` (method) `polos.py:1152` `def analyze_checkpoint(checkpoint_path, output_dir)`
- `ControlVisualizer.analyze_multiple_checkpoints` (method) `polos.py:1208` `def analyze_multiple_checkpoints(checkpoint_dir, n_latest, output_dir)`
- `ControlVisualizer.main` (method) `polos.py:1240` `def main()`

## precision.py
Depends on: `experiment2.py`, `refinamiento.py`
- `CrystallizationLossMassive.__init__` (method) `precision.py:37` `def __init__(self, lambda_quant)`
- `CrystallizationLossMassive.quantization_penalty` (method) `precision.py:42` `def quantization_penalty(self, model)`
- `CrystallizationLossMassive.forward` (method) `precision.py:54` `def forward(self, predictions, targets, model)`
- `ContinuationEngine.__init__` (method) `precision.py:73` `def __init__(self, checkpoint_path, device)`
- `ContinuationEngine.compute_discretization_metrics` (method) `precision.py:208` `def compute_discretization_metrics(self)`
- `ContinuationEngine.validate` (method) `precision.py:241` `def validate(self)`
- `ContinuationEngine.train_epoch` (method) `precision.py:250` `def train_epoch(self, epoch)`
- `ContinuationEngine.refine` (method) `precision.py:288` `def refine(self)`
- `ContinuationEngine.main` (method) `precision.py:506` `def main()`

## refinamiento.py
Depends on: `experiment2.py`
Imported by: `precision.py`
- `CrystallizationLoss.__init__` (method) `refinamiento.py:62` `def __init__(self, lambda_quant)`
- `CrystallizationLoss.quantization_penalty` (method) `refinamiento.py:67` `def quantization_penalty(self, model)` -- Penalización L2 de la distancia al entero más cercano
- `CrystallizationLoss.forward` (method) `refinamiento.py:81` `def forward(self, predictions, targets, model)`
- `StructuralPruner.__init__` (method) `refinamiento.py:98` `def __init__(self, thresholds)`
- `StructuralPruner.should_prune` (method) `refinamiento.py:103` `def should_prune(self, epoch)` -- Determina si es momento de podar (cada 500 épocas)
- `StructuralPruner.prune` (method) `refinamiento.py:107` `def prune(self, model, force_threshold)` -- Poda pesos con |w| < threshold Retorna número de parámetros podados
- `StructuralPruner.get_sparsity` (method) `refinamiento.py:131` `def get_sparsity(self, model)` -- Calcula porcentaje de pesos exactamente en cero
- `CrystallizationEngine.__init__` (method) `refinamiento.py:148` `def __init__(self, checkpoint_path, device)`
- `CrystallizationEngine.compute_discretization_metrics` (method) `refinamiento.py:253` `def compute_discretization_metrics(self)` -- Calcula métricas de cristalinidad actuales
- `CrystallizationEngine.validate` (method) `refinamiento.py:291` `def validate(self)` -- Valida el modelo manteniendo accuracy
- `CrystallizationEngine.train_epoch` (method) `refinamiento.py:302` `def train_epoch(self, epoch)` -- Entrena una época con pérdida de cuantización
- `CrystallizationEngine.refine` (method) `refinamiento.py:344` `def refine(self)` -- Ejecuta el refinamiento hasta alcanzar δ < TARGET_DELTA o MAX_EPOCHS
- `CrystallizationEngine.analyze_discretization` (method) `refinamiento.py:498` `def analyze_discretization(checkpoint_path)` -- Análisis detallado de la discretización de un checkpoint
- `CrystallizationEngine.main` (method) `refinamiento.py:575` `def main()`

## verify.py
Depends on: `experiment2.py`
- `CheckpointVerifier.__init__` (method) `verify.py:15` `def __init__(self, checkpoint_path, device)`
- `CheckpointVerifier.verify_all_metrics` (method) `verify.py:50` `def verify_all_metrics(self)` -- Calcula TODAS las métricas desde cero y compara con las guardadas
- `CheckpointVerifier.verify_latest_checkpoints` (method) `verify.py:444` `def verify_latest_checkpoints(checkpoint_dir, n)` -- Verifica los N checkpoints más recientes
- `CheckpointVerifier.main` (method) `verify.py:486` `def main()`

