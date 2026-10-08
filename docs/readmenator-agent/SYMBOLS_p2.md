# Symbols (page 2 of 2)
Previous: [SYMBOLS.md](SYMBOLS.md)

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
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
| `update_metrics` | method | `mining_seeds.py:594` | `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat...` |
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

