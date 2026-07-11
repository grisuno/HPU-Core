# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis.

**Total Files Parsed:** 32 | **Total Symbols Extracted:** 776 | **Total Imports:** 345

## Structural Knowledge Map
```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray: 5 5,color:#aaa;
    experiment_py["experiment.py (py)"]
    class experiment_py mod;
    experiment_py_Config["Config"]
    class experiment_py_Config cls;
    experiment_py --> experiment_py_Config
    experiment_py_set_seed["set_seed"]
    class experiment_py_set_seed fn;
    experiment_py --> experiment_py_set_seed
    experiment_py_setup_logger["setup_logger"]
    class experiment_py_setup_logger fn;
    experiment_py --> experiment_py_setup_logger
    experiment_py_IAnalysisStrategy["IAnalysisStrategy"]
    class experiment_py_IAnalysisStrategy cls;
    experiment_py --> experiment_py_IAnalysisStrategy
    experiment_py_IMetricsCalculator["IMetricsCalculator"]
    class experiment_py_IMetricsCalculator cls;
    experiment_py --> experiment_py_IMetricsCalculator
    mining_seeds_py["mining_seeds.py (py)"]
    class mining_seeds_py mod;
    mining_seeds_py_Config["Config"]
    class mining_seeds_py_Config cls;
    mining_seeds_py --> mining_seeds_py_Config
    mining_seeds_py_set_seed["set_seed"]
    class mining_seeds_py_set_seed fn;
    mining_seeds_py --> mining_seeds_py_set_seed
    mining_seeds_py_setup_logger["setup_logger"]
    class mining_seeds_py_setup_logger fn;
    mining_seeds_py --> mining_seeds_py_setup_logger
    mining_seeds_py_IAnalysisStrategy["IAnalysisStrategy"]
    class mining_seeds_py_IAnalysisStrategy cls;
    mining_seeds_py --> mining_seeds_py_IAnalysisStrategy
    mining_seeds_py_IMetricsCalculator["IMetricsCalculator"]
    class mining_seeds_py_IMetricsCalculator cls;
    mining_seeds_py --> mining_seeds_py_IMetricsCalculator
    hamiltonian_mbl_py["hamiltonian_mbl.py (py)"]
    class hamiltonian_mbl_py mod;
    hamiltonian_mbl_py_HamiltonianArchitectureConfig["HamiltonianArchitectureConfig"]
    class hamiltonian_mbl_py_HamiltonianArchitectureConfig cls;
    hamiltonian_mbl_py --> hamiltonian_mbl_py_HamiltonianArchitectureConfig
    hamiltonian_mbl_py_MBLAnalysisConfig["MBLAnalysisConfig"]
    class hamiltonian_mbl_py_MBLAnalysisConfig cls;
    hamiltonian_mbl_py --> hamiltonian_mbl_py_MBLAnalysisConfig
    hamiltonian_mbl_py_TrainingConfig["TrainingConfig"]
    class hamiltonian_mbl_py_TrainingConfig cls;
    hamiltonian_mbl_py --> hamiltonian_mbl_py_TrainingConfig
    hamiltonian_mbl_py_IModel["IModel"]
    class hamiltonian_mbl_py_IModel cls;
    hamiltonian_mbl_py --> hamiltonian_mbl_py_IModel
    hamiltonian_mbl_py_ILevelSpacingCalculator["ILevelSpacingCalculator"]
    class hamiltonian_mbl_py_ILevelSpacingCalculator cls;
    hamiltonian_mbl_py --> hamiltonian_mbl_py_ILevelSpacingCalculator
    get_meditions_py["get_meditions.py (py)"]
    class get_meditions_py mod;
    get_meditions_py_ThermodynamicConfig["ThermodynamicConfig"]
    class get_meditions_py_ThermodynamicConfig cls;
    get_meditions_py --> get_meditions_py_ThermodynamicConfig
    get_meditions_py_setup_logger["setup_logger"]
    class get_meditions_py_setup_logger fn;
    get_meditions_py --> get_meditions_py_setup_logger
    get_meditions_py_ThermodynamicPotential["ThermodynamicPotential"]
    class get_meditions_py_ThermodynamicPotential cls;
    get_meditions_py --> get_meditions_py_ThermodynamicPotential
    get_meditions_py_CrystallographyMetrics["CrystallographyMetrics"]
    class get_meditions_py_CrystallographyMetrics cls;
    get_meditions_py --> get_meditions_py_CrystallographyMetrics
    get_meditions_py_ThermodynamicMetrics["ThermodynamicMetrics"]
    class get_meditions_py_ThermodynamicMetrics cls;
    get_meditions_py --> get_meditions_py_ThermodynamicMetrics
    polos_py["polos.py (py)"]
    class polos_py mod;
    polos_py_ControlConfig["ControlConfig"]
    class polos_py_ControlConfig cls;
    polos_py --> polos_py_ControlConfig
    polos_py_TransferFunctionExtractor["TransferFunctionExtractor"]
    class polos_py_TransferFunctionExtractor cls;
    polos_py --> polos_py_TransferFunctionExtractor
    polos_py_PoleZeroAnalyzer["PoleZeroAnalyzer"]
    class polos_py_PoleZeroAnalyzer cls;
    polos_py --> polos_py_PoleZeroAnalyzer
    polos_py_FrequencyResponseAnalyzer["FrequencyResponseAnalyzer"]
    class polos_py_FrequencyResponseAnalyzer cls;
    polos_py --> polos_py_FrequencyResponseAnalyzer
    polos_py_TimeResponseAnalyzer["TimeResponseAnalyzer"]
    class polos_py_TimeResponseAnalyzer cls;
    polos_py --> polos_py_TimeResponseAnalyzer
    audio_audios_py["audios.py (py)"]
    class audio_audios_py mod;
    audio_audios_py_HamiltonianConfig["HamiltonianConfig"]
    class audio_audios_py_HamiltonianConfig cls;
    audio_audios_py --> audio_audios_py_HamiltonianConfig
    audio_audios_py_IAudioSource["IAudioSource"]
    class audio_audios_py_IAudioSource cls;
    audio_audios_py --> audio_audios_py_IAudioSource
    audio_audios_py_IFieldOperator["IFieldOperator"]
    class audio_audios_py_IFieldOperator cls;
    audio_audios_py --> audio_audios_py_IFieldOperator
    audio_audios_py_IMetricCollector["IMetricCollector"]
    class audio_audios_py_IMetricCollector cls;
    audio_audios_py --> audio_audios_py_IMetricCollector
    audio_audios_py_AudioResampler["AudioResampler"]
    class audio_audios_py_AudioResampler cls;
    audio_audios_py --> audio_audios_py_AudioResampler
    audio_experiment2_py["experiment2.py (py)"]
    class audio_experiment2_py mod;
    audio_experiment2_py_Config["Config"]
    class audio_experiment2_py_Config cls;
    audio_experiment2_py --> audio_experiment2_py_Config
    audio_experiment2_py_SeedManager["SeedManager"]
    class audio_experiment2_py_SeedManager cls;
    audio_experiment2_py --> audio_experiment2_py_SeedManager
    audio_experiment2_py_LoggerFactory["LoggerFactory"]
    class audio_experiment2_py_LoggerFactory cls;
    audio_experiment2_py --> audio_experiment2_py_LoggerFactory
    audio_experiment2_py_IAnalysisStrategy["IAnalysisStrategy"]
    class audio_experiment2_py_IAnalysisStrategy cls;
    audio_experiment2_py --> audio_experiment2_py_IAnalysisStrategy
    audio_experiment2_py_IMetricsCalculator["IMetricsCalculator"]
    class audio_experiment2_py_IMetricsCalculator cls;
    audio_experiment2_py --> audio_experiment2_py_IMetricsCalculator
    experiment2_py["experiment2.py (py)"]
    class experiment2_py mod;
    experiment2_py_Config["Config"]
    class experiment2_py_Config cls;
    experiment2_py --> experiment2_py_Config
    experiment2_py_SeedManager["SeedManager"]
    class experiment2_py_SeedManager cls;
    experiment2_py --> experiment2_py_SeedManager
    experiment2_py_LoggerFactory["LoggerFactory"]
    class experiment2_py_LoggerFactory cls;
    experiment2_py --> experiment2_py_LoggerFactory
    experiment2_py_IAnalysisStrategy["IAnalysisStrategy"]
    class experiment2_py_IAnalysisStrategy cls;
    experiment2_py --> experiment2_py_IAnalysisStrategy
    experiment2_py_IMetricsCalculator["IMetricsCalculator"]
    class experiment2_py_IMetricsCalculator cls;
    experiment2_py --> experiment2_py_IMetricsCalculator
    dirac_py["dirac.py (py)"]
    class dirac_py mod;
    dirac_py_DiracConfig["DiracConfig"]
    class dirac_py_DiracConfig cls;
    dirac_py --> dirac_py_DiracConfig
    dirac_py_DiracDeltaAnalyzer["DiracDeltaAnalyzer"]
    class dirac_py_DiracDeltaAnalyzer cls;
    dirac_py --> dirac_py_DiracDeltaAnalyzer
    dirac_py_DiracVisualizer["DiracVisualizer"]
    class dirac_py_DiracVisualizer cls;
    dirac_py --> dirac_py_DiracVisualizer
    dirac_py_analyze_checkpoint["analyze_checkpoint"]
    class dirac_py_analyze_checkpoint fn;
    dirac_py --> dirac_py_analyze_checkpoint
    dirac_py_analyze_multiple_checkpoints["analyze_multiple_checkpoints"]
    class dirac_py_analyze_multiple_checkpoints fn;
    dirac_py --> dirac_py_analyze_multiple_checkpoints
    refinamiento_py["refinamiento.py (py)"]
    class refinamiento_py mod;
    refinamiento_py_CrystallizationConfig["CrystallizationConfig"]
    class refinamiento_py_CrystallizationConfig cls;
    refinamiento_py --> refinamiento_py_CrystallizationConfig
    refinamiento_py_CrystallizationLoss["CrystallizationLoss"]
    class refinamiento_py_CrystallizationLoss cls;
    refinamiento_py --> refinamiento_py_CrystallizationLoss
    refinamiento_py_StructuralPruner["StructuralPruner"]
    class refinamiento_py_StructuralPruner cls;
    refinamiento_py --> refinamiento_py_StructuralPruner
    refinamiento_py_CrystallizationEngine["CrystallizationEngine"]
    class refinamiento_py_CrystallizationEngine cls;
    refinamiento_py --> refinamiento_py_CrystallizationEngine
    refinamiento_py_analyze_discretization["analyze_discretization"]
    class refinamiento_py_analyze_discretization fn;
    refinamiento_py --> refinamiento_py_analyze_discretization
    precision_py["precision.py (py)"]
    class precision_py mod;
    precision_py_MassiveLambdaConfig["MassiveLambdaConfig"]
    class precision_py_MassiveLambdaConfig cls;
    precision_py --> precision_py_MassiveLambdaConfig
    precision_py_CrystallizationLossMassive["CrystallizationLossMassive"]
    class precision_py_CrystallizationLossMassive cls;
    precision_py --> precision_py_CrystallizationLossMassive
    precision_py_ContinuationEngine["ContinuationEngine"]
    class precision_py_ContinuationEngine cls;
    precision_py --> precision_py_ContinuationEngine
    precision_py_main["main"]
    class precision_py_main fn;
    precision_py --> precision_py_main
    precision_py___init__["__init__"]
    class precision_py___init__ fn;
    precision_py --> precision_py___init__
    audio_trainer_py["trainer.py (py)"]
    class audio_trainer_py mod;
    audio_trainer_py_AudioSpectrogramDatasetBuilder["AudioSpectrogramDatasetBuilder"]
    class audio_trainer_py_AudioSpectrogramDatasetBuilder cls;
    audio_trainer_py --> audio_trainer_py_AudioSpectrogramDatasetBuilder
    audio_trainer_py_HamiltonianAudioTrainer["HamiltonianAudioTrainer"]
    class audio_trainer_py_HamiltonianAudioTrainer cls;
    audio_trainer_py --> audio_trainer_py_HamiltonianAudioTrainer
    audio_trainer_py___init__["__init__"]
    class audio_trainer_py___init__ fn;
    audio_trainer_py --> audio_trainer_py___init__
    audio_trainer_py_build_dataset["build_dataset"]
    class audio_trainer_py_build_dataset fn;
    audio_trainer_py --> audio_trainer_py_build_dataset
    audio_trainer_py___init__["__init__"]
    class audio_trainer_py___init__ fn;
    audio_trainer_py --> audio_trainer_py___init__
    app_py["app.py (py)"]
    class app_py mod;
    app_py_SimpleConfig["SimpleConfig"]
    class app_py_SimpleConfig cls;
    app_py --> app_py_SimpleConfig
    app_py_compute_local_complexity["compute_local_complexity"]
    class app_py_compute_local_complexity fn;
    app_py --> app_py_compute_local_complexity
    app_py_compute_superposition["compute_superposition"]
    class app_py_compute_superposition fn;
    app_py --> app_py_compute_superposition
    app_py_HamiltonianOperator["HamiltonianOperator"]
    class app_py_HamiltonianOperator cls;
    app_py --> app_py_HamiltonianOperator
    app_py_FastDataset["FastDataset"]
    class app_py_FastDataset cls;
    app_py --> app_py_FastDataset
    verify_py["verify.py (py)"]
    class verify_py mod;
    verify_py_CheckpointVerifier["CheckpointVerifier"]
    class verify_py_CheckpointVerifier cls;
    verify_py --> verify_py_CheckpointVerifier
    verify_py_verify_latest_checkpoints["verify_latest_checkpoints"]
    class verify_py_verify_latest_checkpoints fn;
    verify_py --> verify_py_verify_latest_checkpoints
    verify_py_main["main"]
    class verify_py_main fn;
    verify_py --> verify_py_main
    verify_py___init__["__init__"]
    class verify_py___init__ fn;
    verify_py --> verify_py___init__
    verify_py_verify_all_metrics["verify_all_metrics"]
    class verify_py_verify_all_metrics fn;
    verify_py --> verify_py_verify_all_metrics
    expand_py["expand.py (py)"]
    class expand_py mod;
    expand_py_load_config["load_config"]
    class expand_py_load_config fn;
    expand_py --> expand_py_load_config
    expand_py_expand_spectral_weights["expand_spectral_weights"]
    class expand_py_expand_spectral_weights fn;
    expand_py --> expand_py_expand_spectral_weights
    expand_py_expand_model["expand_model"]
    class expand_py_expand_model fn;
    expand_py --> expand_py_expand_model
    expand_py_evaluate_model["evaluate_model"]
    class expand_py_evaluate_model fn;
    expand_py --> expand_py_evaluate_model
    expand_py_main["main"]
    class expand_py_main fn;
    expand_py --> expand_py_main
    test_grokkit_py["test_grokkit.py (py)"]
    class test_grokkit_py mod;
    test_grokkit_py_GrokkingValidator["GrokkingValidator"]
    class test_grokkit_py_GrokkingValidator cls;
    test_grokkit_py --> test_grokkit_py_GrokkingValidator
    test_grokkit_py_run_quick_test["run_quick_test"]
    class test_grokkit_py_run_quick_test fn;
    test_grokkit_py --> test_grokkit_py_run_quick_test
    test_grokkit_py___init__["__init__"]
    class test_grokkit_py___init__ fn;
    test_grokkit_py --> test_grokkit_py___init__
    test_grokkit_py_load_model["load_model"]
    class test_grokkit_py_load_model fn;
    test_grokkit_py --> test_grokkit_py_load_model
    test_grokkit_py_generate_test_dataset["generate_test_dataset"]
    class test_grokkit_py_generate_test_dataset fn;
    test_grokkit_py --> test_grokkit_py_generate_test_dataset
    audio_inference_py["inference.py (py)"]
    class audio_inference_py mod;
    audio_inference_py_HamiltonianAudioInference["HamiltonianAudioInference"]
    class audio_inference_py_HamiltonianAudioInference cls;
    audio_inference_py --> audio_inference_py_HamiltonianAudioInference
    audio_inference_py___init__["__init__"]
    class audio_inference_py___init__ fn;
    audio_inference_py --> audio_inference_py___init__
    audio_inference_py_analyze_audio["analyze_audio"]
    class audio_inference_py_analyze_audio fn;
    audio_inference_py --> audio_inference_py_analyze_audio
    audio_inference_py__compute_energy_mask_patched["_compute_energy_mask_patched"]
    class audio_inference_py__compute_energy_mask_patched fn;
    audio_inference_py --> audio_inference_py__compute_energy_mask_patched
    audio_inference_py__extract_hamiltonian_fields_patched["_extract_hamiltonian_fields_patched"]
    class audio_inference_py__extract_hamiltonian_fields_patched fn;
    audio_inference_py --> audio_inference_py__extract_hamiltonian_fields_patched
    audio_visualization_py["visualization.py (py)"]
    class audio_visualization_py mod;
    audio_visualization_py_HamiltonianAudioVisualizer["HamiltonianAudioVisualizer"]
    class audio_visualization_py_HamiltonianAudioVisualizer cls;
    audio_visualization_py --> audio_visualization_py_HamiltonianAudioVisualizer
    audio_visualization_py___init__["__init__"]
    class audio_visualization_py___init__ fn;
    audio_visualization_py --> audio_visualization_py___init__
    audio_visualization_py_render_complete_analysis["render_complete_analysis"]
    class audio_visualization_py_render_complete_analysis fn;
    audio_visualization_py --> audio_visualization_py_render_complete_analysis
    audio_visualization_py__render_hamiltonian_fields["_render_hamiltonian_fields"]
    class audio_visualization_py__render_hamiltonian_fields fn;
    audio_visualization_py --> audio_visualization_py__render_hamiltonian_fields
    audio_visualization_py__render_spectrogram_comparison["_render_spectrogram_comparison"]
    class audio_visualization_py__render_spectrogram_comparison fn;
    audio_visualization_py --> audio_visualization_py__render_spectrogram_comparison
    audio_checkpoint_manager_py["checkpoint_manager.py (py)"]
    class audio_checkpoint_manager_py mod;
    audio_checkpoint_manager_py_CheckpointManager["CheckpointManager"]
    class audio_checkpoint_manager_py_CheckpointManager cls;
    audio_checkpoint_manager_py --> audio_checkpoint_manager_py_CheckpointManager
    audio_checkpoint_manager_py___init__["__init__"]
    class audio_checkpoint_manager_py___init__ fn;
    audio_checkpoint_manager_py --> audio_checkpoint_manager_py___init__
    audio_checkpoint_manager_py_should_save_checkpoint["should_save_checkpoint"]
    class audio_checkpoint_manager_py_should_save_checkpoint fn;
    audio_checkpoint_manager_py --> audio_checkpoint_manager_py_should_save_checkpoint
    audio_checkpoint_manager_py_save_checkpoint["save_checkpoint"]
    class audio_checkpoint_manager_py_save_checkpoint fn;
    audio_checkpoint_manager_py --> audio_checkpoint_manager_py_save_checkpoint
    audio_checkpoint_manager_py_load_checkpoint["load_checkpoint"]
    class audio_checkpoint_manager_py_load_checkpoint fn;
    audio_checkpoint_manager_py --> audio_checkpoint_manager_py_load_checkpoint
    plank_py["plank.py (py)"]
    class plank_py mod;
    plank_py_HBarCalculator["HBarCalculator"]
    class plank_py_HBarCalculator cls;
    plank_py --> plank_py_HBarCalculator
    plank_py_main["main"]
    class plank_py_main fn;
    plank_py --> plank_py_main
    plank_py___init__["__init__"]
    class plank_py___init__ fn;
    plank_py --> plank_py___init__
    plank_py_calculate_all["calculate_all"]
    class plank_py_calculate_all fn;
    plank_py --> plank_py_calculate_all
    plank_py_print_report["print_report"]
    class plank_py_print_report fn;
    plank_py --> plank_py_print_report
    check_fase_berry_py["check_fase_berry.py (py)"]
    class check_fase_berry_py mod;
    hpu_view_py["hpu_view.py (py)"]
    class hpu_view_py mod;
    simple_hpu_view_py["simple_hpu_view.py (py)"]
    class simple_hpu_view_py mod;
    audio_main_py["main.py (py)"]
    class audio_main_py mod;
    audio_main_py_build_argument_parser["build_argument_parser"]
    class audio_main_py_build_argument_parser fn;
    audio_main_py --> audio_main_py_build_argument_parser
    audio_main_py_build_config_from_args["build_config_from_args"]
    class audio_main_py_build_config_from_args fn;
    audio_main_py --> audio_main_py_build_config_from_args
    audio_main_py_validate_audio_file["validate_audio_file"]
    class audio_main_py_validate_audio_file fn;
    audio_main_py --> audio_main_py_validate_audio_file
    audio_main_py_print_configuration_banner["print_configuration_banner"]
    class audio_main_py_print_configuration_banner fn;
    audio_main_py --> audio_main_py_print_configuration_banner
    audio_main_py_run_training["run_training"]
    class audio_main_py_run_training fn;
    audio_main_py --> audio_main_py_run_training
    audio_metrics_py["metrics.py (py)"]
    class audio_metrics_py mod;
    audio_metrics_py_HamiltonianMetricsTracker["HamiltonianMetricsTracker"]
    class audio_metrics_py_HamiltonianMetricsTracker cls;
    audio_metrics_py --> audio_metrics_py_HamiltonianMetricsTracker
    audio_metrics_py___init__["__init__"]
    class audio_metrics_py___init__ fn;
    audio_metrics_py --> audio_metrics_py___init__
    audio_metrics_py__initialize_history_buffers["_initialize_history_buffers"]
    class audio_metrics_py__initialize_history_buffers fn;
    audio_metrics_py --> audio_metrics_py__initialize_history_buffers
    audio_metrics_py_compute_hamiltonian_energy["compute_hamiltonian_energy"]
    class audio_metrics_py_compute_hamiltonian_energy fn;
    audio_metrics_py --> audio_metrics_py_compute_hamiltonian_energy
    audio_metrics_py_compute_symplectic_form["compute_symplectic_form"]
    class audio_metrics_py_compute_symplectic_form fn;
    audio_metrics_py --> audio_metrics_py_compute_symplectic_form
    audio_audio_io_py["audio_io.py (py)"]
    class audio_audio_io_py mod;
    audio_audio_io_py_AudioProcessor["AudioProcessor"]
    class audio_audio_io_py_AudioProcessor cls;
    audio_audio_io_py --> audio_audio_io_py_AudioProcessor
    audio_audio_io_py___init__["__init__"]
    class audio_audio_io_py___init__ fn;
    audio_audio_io_py --> audio_audio_io_py___init__
    audio_audio_io_py_load_audio["load_audio"]
    class audio_audio_io_py_load_audio fn;
    audio_audio_io_py --> audio_audio_io_py_load_audio
    audio_audio_io_py_waveform_to_stft_complex["waveform_to_stft_complex"]
    class audio_audio_io_py_waveform_to_stft_complex fn;
    audio_audio_io_py --> audio_audio_io_py_waveform_to_stft_complex
    audio_audio_io_py_stft_complex_to_waveform["stft_complex_to_waveform"]
    class audio_audio_io_py_stft_complex_to_waveform fn;
    audio_audio_io_py --> audio_audio_io_py_stft_complex_to_waveform
    audio_losses_py["losses.py (py)"]
    class audio_losses_py mod;
    audio_losses_py_HamiltonianLossComputer["HamiltonianLossComputer"]
    class audio_losses_py_HamiltonianLossComputer cls;
    audio_losses_py --> audio_losses_py_HamiltonianLossComputer
    audio_losses_py___init__["__init__"]
    class audio_losses_py___init__ fn;
    audio_losses_py --> audio_losses_py___init__
    audio_losses_py_compute_total_loss["compute_total_loss"]
    class audio_losses_py_compute_total_loss fn;
    audio_losses_py --> audio_losses_py_compute_total_loss
    audio_losses_py__compute_reconstruction_loss["_compute_reconstruction_loss"]
    class audio_losses_py__compute_reconstruction_loss fn;
    audio_losses_py --> audio_losses_py__compute_reconstruction_loss
    audio_losses_py__compute_energy_conservation_loss["_compute_energy_conservation_loss"]
    class audio_losses_py__compute_energy_conservation_loss fn;
    audio_losses_py --> audio_losses_py__compute_energy_conservation_loss
    audio_model_py["model.py (py)"]
    class audio_model_py mod;
    audio_model_py_SpectralEvolutionLayer["SpectralEvolutionLayer"]
    class audio_model_py_SpectralEvolutionLayer cls;
    audio_model_py --> audio_model_py_SpectralEvolutionLayer
    audio_model_py_HamiltonianNeuralNetwork["HamiltonianNeuralNetwork"]
    class audio_model_py_HamiltonianNeuralNetwork cls;
    audio_model_py --> audio_model_py_HamiltonianNeuralNetwork
    audio_model_py___init__["__init__"]
    class audio_model_py___init__ fn;
    audio_model_py --> audio_model_py___init__
    audio_model_py_forward["forward"]
    class audio_model_py_forward fn;
    audio_model_py --> audio_model_py_forward
    audio_model_py_evolve_complex["evolve_complex"]
    class audio_model_py_evolve_complex fn;
    audio_model_py --> audio_model_py_evolve_complex
    audio_config_py["config.py (py)"]
    class audio_config_py mod;
    audio_config_py_AudioProcessingConfig["AudioProcessingConfig"]
    class audio_config_py_AudioProcessingConfig cls;
    audio_config_py --> audio_config_py_AudioProcessingConfig
    audio_config_py_ModelArchitectureConfig["ModelArchitectureConfig"]
    class audio_config_py_ModelArchitectureConfig cls;
    audio_config_py --> audio_config_py_ModelArchitectureConfig
    audio_config_py_TrainingConfig["TrainingConfig"]
    class audio_config_py_TrainingConfig cls;
    audio_config_py --> audio_config_py_TrainingConfig
    audio_config_py_CheckpointConfig["CheckpointConfig"]
    class audio_config_py_CheckpointConfig cls;
    audio_config_py --> audio_config_py_CheckpointConfig
    audio_config_py_VisualizationConfig["VisualizationConfig"]
    class audio_config_py_VisualizationConfig cls;
    audio_config_py --> audio_config_py_VisualizationConfig
    diff_weights_py["diff_weights.py (py)"]
    class diff_weights_py mod;
    diff_weights_py_analize_checkpoint["analize_checkpoint"]
    class diff_weights_py_analize_checkpoint fn;
    diff_weights_py --> diff_weights_py_analize_checkpoint
    export_py["export.py (py)"]
    class export_py mod;
    install_sh["install.sh (sh)"]
    class install_sh mod;
    ext_torch["torch"]
    class ext_torch ext;
    app_py -.->|imports| ext_torch
    ext_torch_nn["torch.nn"]
    class ext_torch_nn ext;
    app_py -.->|imports| ext_torch_nn
    ext_torch_nn_functional["torch.nn.functional"]
    class ext_torch_nn_functional ext;
    app_py -.->|imports| ext_torch_nn_functional
    ext_torch_optim["torch.optim"]
    class ext_torch_optim ext;
    app_py -.->|imports| ext_torch_optim
    ext_torch_utils_data["torch.utils.data"]
    class ext_torch_utils_data ext;
    app_py -.->|imports| ext_torch_utils_data
    ext_numpy["numpy"]
    class ext_numpy ext;
    app_py -.->|imports| ext_numpy
    ext_os["os"]
    class ext_os ext;
    app_py -.->|imports| ext_os
    ext_time["time"]
    class ext_time ext;
    app_py -.->|imports| ext_time
    ext_typing["typing"]
    class ext_typing ext;
    app_py -.->|imports| ext_typing
    ext_argparse["argparse"]
    class ext_argparse ext;
    app_py -.->|imports| ext_argparse
    audio_audio_io_py -.->|imports| ext_torch
    ext_torchaudio["torchaudio"]
    class ext_torchaudio ext;
    audio_audio_io_py -.->|imports| ext_torchaudio
    ext_torchaudio_transforms["torchaudio.transforms"]
    class ext_torchaudio_transforms ext;
    audio_audio_io_py -.->|imports| ext_torchaudio_transforms
    audio_audio_io_py -.->|imports| ext_typing
    ext_config["config"]
    class ext_config ext;
    audio_audio_io_py -.->|imports| ext_config
    audio_audios_py -.->|imports| ext_torch
    audio_audios_py -.->|imports| ext_torch_nn
    audio_audios_py -.->|imports| ext_torch_nn_functional
    audio_audios_py -.->|imports| ext_numpy
    audio_audios_py -.->|imports| ext_argparse
    audio_audios_py -.->|imports| ext_os
    audio_audios_py -.->|imports| ext_time
    ext_json["json"]
    class ext_json ext;
    audio_audios_py -.->|imports| ext_json
    ext_sys["sys"]
    class ext_sys ext;
    audio_audios_py -.->|imports| ext_sys
    ext_abc["abc"]
    class ext_abc ext;
    audio_audios_py -.->|imports| ext_abc
    ext_dataclasses["dataclasses"]
    class ext_dataclasses ext;
    audio_audios_py -.->|imports| ext_dataclasses
    audio_audios_py -.->|imports| ext_typing
    ext_pathlib["pathlib"]
    class ext_pathlib ext;
    audio_audios_py -.->|imports| ext_pathlib
    ext_safetensors_torch["safetensors.torch"]
    class ext_safetensors_torch ext;
    audio_audios_py -.->|imports| ext_safetensors_torch
    ext_experiment2["experiment2"]
    class ext_experiment2 ext;
    audio_audios_py -.->|imports| ext_experiment2
    ext_scipy_io["scipy.io"]
    class ext_scipy_io ext;
    audio_audios_py -.->|imports| ext_scipy_io
    ext_scipy["scipy"]
    class ext_scipy ext;
    audio_audios_py -.->|imports| ext_scipy
    ext_cv2["cv2"]
    class ext_cv2 ext;
    audio_audios_py -.->|imports| ext_cv2
    audio_checkpoint_manager_py -.->|imports| ext_os
    audio_checkpoint_manager_py -.->|imports| ext_json
    audio_checkpoint_manager_py -.->|imports| ext_time
    audio_checkpoint_manager_py -.->|imports| ext_torch
    audio_checkpoint_manager_py -.->|imports| ext_torch_nn
    audio_checkpoint_manager_py -.->|imports| ext_typing
    audio_checkpoint_manager_py -.->|imports| ext_safetensors_torch
    audio_checkpoint_manager_py -.->|imports| ext_config
    audio_config_py -.->|imports| ext_dataclasses
    audio_config_py -.->|imports| ext_typing
    audio_config_py -.->|imports| ext_os
    audio_experiment2_py -.->|imports| ext_argparse
    audio_experiment2_py -.->|imports| ext_torch
    audio_experiment2_py -.->|imports| ext_torch_nn
    audio_experiment2_py -.->|imports| ext_torch_nn_functional
    audio_experiment2_py -.->|imports| ext_torch_optim
    audio_experiment2_py -.->|imports| ext_torch_utils_data
    audio_experiment2_py -.->|imports| ext_numpy
    audio_experiment2_py -.->|imports| ext_os
    audio_experiment2_py -.->|imports| ext_time
    audio_experiment2_py -.->|imports| ext_json
    ext_datetime["datetime"]
    class ext_datetime ext;
    audio_experiment2_py -.->|imports| ext_datetime
    audio_experiment2_py -.->|imports| ext_typing
    audio_experiment2_py -.->|imports| ext_abc
    audio_experiment2_py -.->|imports| ext_dataclasses
    ext_collections["collections"]
    class ext_collections ext;
    audio_experiment2_py -.->|imports| ext_collections
    ext_logging["logging"]
    class ext_logging ext;
    audio_experiment2_py -.->|imports| ext_logging
    ext_traceback["traceback"]
    class ext_traceback ext;
    audio_experiment2_py -.->|imports| ext_traceback
    audio_inference_py -.->|imports| ext_os
    audio_inference_py -.->|imports| ext_torch
    audio_inference_py -.->|imports| ext_typing
    audio_inference_py -.->|imports| ext_config
    ext_model["model"]
    class ext_model ext;
    audio_inference_py -.->|imports| ext_model
    ext_audio_io["audio_io"]
    class ext_audio_io ext;
    audio_inference_py -.->|imports| ext_audio_io
    ext_visualization["visualization"]
    class ext_visualization ext;
    audio_inference_py -.->|imports| ext_visualization
    ext_metrics["metrics"]
    class ext_metrics ext;
    audio_inference_py -.->|imports| ext_metrics
    ext_checkpoint_manager["checkpoint_manager"]
    class ext_checkpoint_manager ext;
    audio_inference_py -.->|imports| ext_checkpoint_manager
    audio_losses_py -.->|imports| ext_torch
    audio_losses_py -.->|imports| ext_torch_nn
    audio_losses_py -.->|imports| ext_torch_nn_functional
    audio_losses_py -.->|imports| ext_typing
    audio_losses_py -.->|imports| ext_config
    audio_main_py -.->|imports| ext_argparse
    audio_main_py -.->|imports| ext_sys
    audio_main_py -.->|imports| ext_os
    audio_main_py -.->|imports| ext_config
    ext_trainer["trainer"]
    class ext_trainer ext;
    audio_main_py -.->|imports| ext_trainer
    ext_inference["inference"]
    class ext_inference ext;
    audio_main_py -.->|imports| ext_inference
    audio_metrics_py -.->|imports| ext_torch
    audio_metrics_py -.->|imports| ext_torch_nn_functional
    audio_metrics_py -.->|imports| ext_typing
    audio_metrics_py -.->|imports| ext_collections
    audio_metrics_py -.->|imports| ext_config
    audio_model_py -.->|imports| ext_torch
    audio_model_py -.->|imports| ext_torch_nn
    audio_model_py -.->|imports| ext_torch_nn_functional
    audio_model_py -.->|imports| ext_typing
    audio_model_py -.->|imports| ext_config
    audio_trainer_py -.->|imports| ext_os
    audio_trainer_py -.->|imports| ext_time
    audio_trainer_py -.->|imports| ext_torch
    audio_trainer_py -.->|imports| ext_torch_nn
    audio_trainer_py -.->|imports| ext_torch_optim
    audio_trainer_py -.->|imports| ext_torch_utils_data
    ext_tqdm["tqdm"]
    class ext_tqdm ext;
    audio_trainer_py -.->|imports| ext_tqdm
    audio_trainer_py -.->|imports| ext_typing
    audio_trainer_py -.->|imports| ext_config
    audio_trainer_py -.->|imports| ext_model
    audio_trainer_py -.->|imports| ext_audio_io
    ext_losses["losses"]
    class ext_losses ext;
    audio_trainer_py -.->|imports| ext_losses
    audio_trainer_py -.->|imports| ext_metrics
    audio_trainer_py -.->|imports| ext_checkpoint_manager
    audio_visualization_py -.->|imports| ext_os
    audio_visualization_py -.->|imports| ext_torch
    audio_visualization_py -.->|imports| ext_numpy
    ext_matplotlib["matplotlib"]
    class ext_matplotlib ext;
    audio_visualization_py -.->|imports| ext_matplotlib
    ext_matplotlib_pyplot["matplotlib.pyplot"]
    class ext_matplotlib_pyplot ext;
    audio_visualization_py -.->|imports| ext_matplotlib_pyplot
    ext_matplotlib_gridspec["matplotlib.gridspec"]
    class ext_matplotlib_gridspec ext;
    audio_visualization_py -.->|imports| ext_matplotlib_gridspec
    audio_visualization_py -.->|imports| ext_typing
    audio_visualization_py -.->|imports| ext_config
    check_fase_berry_py -.->|imports| ext_torch
    check_fase_berry_py -.->|imports| ext_cv2
    check_fase_berry_py -.->|imports| ext_numpy
    ext_mss["mss"]
    class ext_mss ext;
    check_fase_berry_py -.->|imports| ext_mss
    check_fase_berry_py -.->|imports| ext_torch_nn_functional
    check_fase_berry_py -.->|imports| ext_safetensors_torch
    check_fase_berry_py -.->|imports| ext_experiment2
    diff_weights_py -.->|imports| ext_torch
    diff_weights_py -.->|imports| ext_numpy
    diff_weights_py -.->|imports| ext_argparse
    dirac_py -.->|imports| ext_torch
    dirac_py -.->|imports| ext_torch_nn_functional
    dirac_py -.->|imports| ext_numpy
    dirac_py -.->|imports| ext_json
    dirac_py -.->|imports| ext_os
    dirac_py -.->|imports| ext_argparse
    dirac_py -.->|imports| ext_datetime
    dirac_py -.->|imports| ext_typing
    dirac_py -.->|imports| ext_pathlib
    ext_glob["glob"]
    class ext_glob ext;
    dirac_py -.->|imports| ext_glob
    dirac_py -.->|imports| ext_dataclasses
    ext_warnings["warnings"]
    class ext_warnings ext;
    dirac_py -.->|imports| ext_warnings
    dirac_py -.->|imports| ext_matplotlib_pyplot
    ext_matplotlib_colors["matplotlib.colors"]
    class ext_matplotlib_colors ext;
    dirac_py -.->|imports| ext_matplotlib_colors
    ext_matplotlib_cm["matplotlib.cm"]
    class ext_matplotlib_cm ext;
    dirac_py -.->|imports| ext_matplotlib_cm
    ext_mpl_toolkits_mplot3d["mpl_toolkits.mplot3d"]
    class ext_mpl_toolkits_mplot3d ext;
    dirac_py -.->|imports| ext_mpl_toolkits_mplot3d
    dirac_py -.->|imports| ext_experiment2
    expand_py -.->|imports| ext_torch
    expand_py -.->|imports| ext_torch_nn_functional
    expand_py -.->|imports| ext_numpy
    expand_py -.->|imports| ext_os
    ext_toml["toml"]
    class ext_toml ext;
    expand_py -.->|imports| ext_toml
    expand_py -.->|imports| ext_typing
    ext_main_fast["main_fast"]
    class ext_main_fast ext;
    expand_py -.->|imports| ext_main_fast
    expand_py -.->|imports| ext_typing
    expand_py -.->|imports| ext_time
    expand_py -.->|imports| ext_json
    experiment_py -.->|imports| ext_argparse
    experiment_py -.->|imports| ext_torch
    experiment_py -.->|imports| ext_torch_nn
    experiment_py -.->|imports| ext_torch_nn_functional
    experiment_py -.->|imports| ext_torch_optim
    experiment_py -.->|imports| ext_torch_utils_data
    experiment_py -.->|imports| ext_numpy
    experiment_py -.->|imports| ext_os
    experiment_py -.->|imports| ext_time
    experiment_py -.->|imports| ext_json
    ext_threading["threading"]
    class ext_threading ext;
    experiment_py -.->|imports| ext_threading
    experiment_py -.->|imports| ext_datetime
    experiment_py -.->|imports| ext_typing
    experiment_py -.->|imports| ext_abc
    experiment_py -.->|imports| ext_dataclasses
    experiment_py -.->|imports| ext_collections
    experiment_py -.->|imports| ext_logging
    experiment_py -.->|imports| ext_matplotlib_pyplot
    ext_seaborn["seaborn"]
    class ext_seaborn ext;
    experiment_py -.->|imports| ext_seaborn
    ext_scipy_stats["scipy.stats"]
    class ext_scipy_stats ext;
    experiment_py -.->|imports| ext_scipy_stats
    ext_scipy_linalg["scipy.linalg"]
    class ext_scipy_linalg ext;
    experiment_py -.->|imports| ext_scipy_linalg
    ext_scipy_optimize["scipy.optimize"]
    class ext_scipy_optimize ext;
    experiment_py -.->|imports| ext_scipy_optimize
    ext_sklearn_decomposition["sklearn.decomposition"]
    class ext_sklearn_decomposition ext;
    experiment_py -.->|imports| ext_sklearn_decomposition
    experiment_py -.->|imports| ext_pathlib
    experiment2_py -.->|imports| ext_argparse
    experiment2_py -.->|imports| ext_torch
    experiment2_py -.->|imports| ext_torch_nn
    experiment2_py -.->|imports| ext_torch_nn_functional
    experiment2_py -.->|imports| ext_torch_optim
    experiment2_py -.->|imports| ext_torch_utils_data
    experiment2_py -.->|imports| ext_numpy
    experiment2_py -.->|imports| ext_os
    experiment2_py -.->|imports| ext_time
    experiment2_py -.->|imports| ext_json
    experiment2_py -.->|imports| ext_datetime
    experiment2_py -.->|imports| ext_typing
    experiment2_py -.->|imports| ext_abc
    experiment2_py -.->|imports| ext_dataclasses
    experiment2_py -.->|imports| ext_collections
    experiment2_py -.->|imports| ext_logging
    experiment2_py -.->|imports| ext_traceback
    export_py -.->|imports| ext_torch
    export_py -.->|imports| ext_safetensors_torch
    get_meditions_py -.->|imports| ext_torch
    get_meditions_py -.->|imports| ext_torch_nn_functional
    get_meditions_py -.->|imports| ext_numpy
    get_meditions_py -.->|imports| ext_json
    get_meditions_py -.->|imports| ext_os
    get_meditions_py -.->|imports| ext_argparse
    get_meditions_py -.->|imports| ext_datetime
    get_meditions_py -.->|imports| ext_typing
    get_meditions_py -.->|imports| ext_pathlib
    get_meditions_py -.->|imports| ext_glob
    get_meditions_py -.->|imports| ext_dataclasses
    get_meditions_py -.->|imports| ext_collections
    get_meditions_py -.->|imports| ext_warnings
    get_meditions_py -.->|imports| ext_logging
    get_meditions_py -.->|imports| ext_scipy_stats
    get_meditions_py -.->|imports| ext_scipy_linalg
    get_meditions_py -.->|imports| ext_scipy_optimize
    get_meditions_py -.->|imports| ext_sklearn_decomposition
    get_meditions_py -.->|imports| ext_experiment2
    hamiltonian_mbl_py -.->|imports| ext_torch
    hamiltonian_mbl_py -.->|imports| ext_torch_nn
    hamiltonian_mbl_py -.->|imports| ext_torch_nn_functional
    hamiltonian_mbl_py -.->|imports| ext_numpy
    hamiltonian_mbl_py -.->|imports| ext_json
    hamiltonian_mbl_py -.->|imports| ext_os
    hamiltonian_mbl_py -.->|imports| ext_argparse
    hamiltonian_mbl_py -.->|imports| ext_time
    hamiltonian_mbl_py -.->|imports| ext_warnings
    hamiltonian_mbl_py -.->|imports| ext_datetime
    hamiltonian_mbl_py -.->|imports| ext_typing
    hamiltonian_mbl_py -.->|imports| ext_pathlib
    hamiltonian_mbl_py -.->|imports| ext_glob
    hamiltonian_mbl_py -.->|imports| ext_dataclasses
    hamiltonian_mbl_py -.->|imports| ext_scipy_stats
    hamiltonian_mbl_py -.->|imports| ext_scipy_linalg
    ext_scipy_sparse["scipy.sparse"]
    class ext_scipy_sparse ext;
    hamiltonian_mbl_py -.->|imports| ext_scipy_sparse
    ext_scipy_sparse_linalg["scipy.sparse.linalg"]
    class ext_scipy_sparse_linalg ext;
    hamiltonian_mbl_py -.->|imports| ext_scipy_sparse_linalg
    hamiltonian_mbl_py -.->|imports| ext_matplotlib_pyplot
    ext_gc["gc"]
    class ext_gc ext;
    hamiltonian_mbl_py -.->|imports| ext_gc
    hpu_view_py -.->|imports| ext_torch
    hpu_view_py -.->|imports| ext_cv2
    hpu_view_py -.->|imports| ext_numpy
    hpu_view_py -.->|imports| ext_mss
    hpu_view_py -.->|imports| ext_torch_nn_functional
    hpu_view_py -.->|imports| ext_safetensors_torch
    hpu_view_py -.->|imports| ext_experiment2
    mining_seeds_py -.->|imports| ext_argparse
    mining_seeds_py -.->|imports| ext_torch
    mining_seeds_py -.->|imports| ext_torch_nn
    mining_seeds_py -.->|imports| ext_torch_nn_functional
    mining_seeds_py -.->|imports| ext_torch_optim
    mining_seeds_py -.->|imports| ext_torch_utils_data
    mining_seeds_py -.->|imports| ext_numpy
    mining_seeds_py -.->|imports| ext_os
    mining_seeds_py -.->|imports| ext_time
    mining_seeds_py -.->|imports| ext_json
    mining_seeds_py -.->|imports| ext_threading
    mining_seeds_py -.->|imports| ext_datetime
    mining_seeds_py -.->|imports| ext_typing
    mining_seeds_py -.->|imports| ext_abc
    mining_seeds_py -.->|imports| ext_dataclasses
    mining_seeds_py -.->|imports| ext_collections
    mining_seeds_py -.->|imports| ext_logging
    mining_seeds_py -.->|imports| ext_matplotlib_pyplot
    mining_seeds_py -.->|imports| ext_seaborn
    mining_seeds_py -.->|imports| ext_scipy_stats
    mining_seeds_py -.->|imports| ext_scipy_linalg
    mining_seeds_py -.->|imports| ext_scipy_optimize
    mining_seeds_py -.->|imports| ext_sklearn_decomposition
    mining_seeds_py -.->|imports| ext_pathlib
    plank_py -.->|imports| ext_torch
    plank_py -.->|imports| ext_numpy
    plank_py -.->|imports| ext_json
    plank_py -.->|imports| ext_os
    plank_py -.->|imports| ext_argparse
    plank_py -.->|imports| ext_datetime
    plank_py -.->|imports| ext_typing
    polos_py -.->|imports| ext_torch
    polos_py -.->|imports| ext_torch_nn_functional
    polos_py -.->|imports| ext_numpy
    polos_py -.->|imports| ext_json
    polos_py -.->|imports| ext_os
    polos_py -.->|imports| ext_argparse
    polos_py -.->|imports| ext_datetime
    polos_py -.->|imports| ext_typing
    polos_py -.->|imports| ext_pathlib
    polos_py -.->|imports| ext_glob
    polos_py -.->|imports| ext_dataclasses
    polos_py -.->|imports| ext_warnings
    polos_py -.->|imports| ext_scipy
    polos_py -.->|imports| ext_scipy_linalg
    polos_py -.->|imports| ext_scipy_optimize
    polos_py -.->|imports| ext_matplotlib_pyplot
    ext_matplotlib_patches["matplotlib.patches"]
    class ext_matplotlib_patches ext;
    polos_py -.->|imports| ext_matplotlib_patches
    polos_py -.->|imports| ext_mpl_toolkits_mplot3d
    polos_py -.->|imports| ext_experiment2
    precision_py -.->|imports| ext_torch
    precision_py -.->|imports| ext_torch_nn
    precision_py -.->|imports| ext_torch_nn_functional
    precision_py -.->|imports| ext_torch_optim
    precision_py -.->|imports| ext_torch_utils_data
    precision_py -.->|imports| ext_numpy
    precision_py -.->|imports| ext_os
    precision_py -.->|imports| ext_json
    precision_py -.->|imports| ext_datetime
    precision_py -.->|imports| ext_typing
    precision_py -.->|imports| ext_logging
    precision_py -.->|imports| ext_glob
    precision_py -.->|imports| ext_experiment2
    ext_refinamiento["refinamiento"]
    class ext_refinamiento ext;
    precision_py -.->|imports| ext_refinamiento
    precision_py -.->|imports| ext_argparse
    refinamiento_py -.->|imports| ext_torch
    refinamiento_py -.->|imports| ext_torch_nn
    refinamiento_py -.->|imports| ext_torch_nn_functional
    refinamiento_py -.->|imports| ext_torch_optim
    refinamiento_py -.->|imports| ext_torch_utils_data
    refinamiento_py -.->|imports| ext_numpy
    refinamiento_py -.->|imports| ext_os
    refinamiento_py -.->|imports| ext_time
    refinamiento_py -.->|imports| ext_json
    refinamiento_py -.->|imports| ext_datetime
    refinamiento_py -.->|imports| ext_typing
    refinamiento_py -.->|imports| ext_logging
    refinamiento_py -.->|imports| ext_collections
    refinamiento_py -.->|imports| ext_experiment2
    refinamiento_py -.->|imports| ext_argparse
    simple_hpu_view_py -.->|imports| ext_torch
    simple_hpu_view_py -.->|imports| ext_cv2
    simple_hpu_view_py -.->|imports| ext_numpy
    simple_hpu_view_py -.->|imports| ext_mss
    simple_hpu_view_py -.->|imports| ext_torch_nn_functional
    simple_hpu_view_py -.->|imports| ext_safetensors_torch
    simple_hpu_view_py -.->|imports| ext_experiment2
    test_grokkit_py -.->|imports| ext_torch
    test_grokkit_py -.->|imports| ext_torch_nn
    test_grokkit_py -.->|imports| ext_numpy
    test_grokkit_py -.->|imports| ext_pathlib
    test_grokkit_py -.->|imports| ext_sys
    test_grokkit_py -.->|imports| ext_os
    ext_main["main"]
    class ext_main ext;
    test_grokkit_py -.->|imports| ext_main
    test_grokkit_py -.->|imports| ext_main_fast
    test_grokkit_py -.->|imports| ext_main_fast
    verify_py -.->|imports| ext_torch
    verify_py -.->|imports| ext_torch_nn_functional
    verify_py -.->|imports| ext_numpy
    verify_py -.->|imports| ext_json
    verify_py -.->|imports| ext_os
    verify_py -.->|imports| ext_argparse
    verify_py -.->|imports| ext_datetime
    verify_py -.->|imports| ext_typing
    verify_py -.->|imports| ext_glob
    verify_py -.->|imports| ext_experiment2
```

---

## Architecture Reference

### PY (31 files)

#### `app.py`
**Path:** `app.py`

**Classs:**
- `SimpleConfig` (line 29)
- `HamiltonianOperator` (line 88) - *True Hamiltonian operator H = -nabla^2 on torus.*
- `FastDataset` (line 113) - *Fast dataset for Hamiltonian operator learning.*
- `SpectralLayer` (line 170) - *Spectral layer with correct complex multiplication.*
- `SimpleHamiltonianNet` (line 222) - *Compact network for Hamiltonian operator learning.*

**Functions:**
- `compute_local_complexity` (line 45) - *Compute Local Complexity (LC) metric for weight matrix.*
- `compute_superposition` (line 60) - *Compute Superposition (SP) metric for weight matrix.*
- `train_model` (line 260) - *Train the Hamiltonian operator model.*
- `main` (line 369)
- `__init__` (line 30)
- `__init__` (line 91)
- `_precompute_spectral_operators` (line 95)
- `apply` (line 102)
- `time_evolution` (line 107)
- `__init__` (line 116)
- `__len__` (line 160)
- `__getitem__` (line 163)
- `get_val_batch` (line 166)
- `__init__` (line 173)
- `forward` (line 186)
- `__init__` (line 225)
- `forward` (line 246)

#### `audio_io.py`
**Path:** `audio/audio_io.py`

**Classs:**
- `AudioProcessor` (line 31) - *Audio processing pipeline centered on the complex STFT domain.

The complex STFT is the audio equivalent of a grayscale image:
a 2D field where one axis is time, the other is frequency, and
each point carries a complex value (magnitude + phase). This is
the correct domain for applying the Hamiltonian spectral evolution.*

**Functions:**
- `__init__` (line 41)
- `load_audio` (line 57) - *Load an audio file and convert to mono at the target sample rate.

Args:
    file_path: Path to the audio file.

Returns:
    Tuple of (waveform tensor [1, T], sample_rate).*
- `waveform_to_stft_complex` (line 80) - *Compute the complex STFT of a waveform.

This is the primary transform: converts 1D audio into a 2D complex
field (freq_bins x time_frames) that the Hamiltonian network
can process identically to how it processes images.

Args:
    waveform: Audio waveform [1, T] or [T].

Returns:
    Complex STFT tensor [freq_bins, time_frames].*
- `stft_complex_to_waveform` (line 105) - *Reconstruct waveform from complex STFT via inverse STFT.

Unlike Griffin-Lim (which estimates phase), ISTFT uses the
EXACT phase from the complex STFT, producing a faithful
reconstruction when the magnitude/phase have been coherently
modified by the Hamiltonian evolution.

Args:
    stft_complex: Complex STFT tensor [freq_bins, time_frames].

Returns:
    Reconstructed waveform [1, T].*
- `stft_to_magnitude_phase` (line 130) - *Decompose complex STFT into magnitude and phase.

Args:
    stft_complex: Complex STFT [freq_bins, time_frames].

Returns:
    Tuple of (magnitude, phase) each [freq_bins, time_frames].*
- `magnitude_phase_to_stft` (line 146) - *Recombine magnitude and phase into complex STFT.

Args:
    magnitude: Magnitude spectrum [freq_bins, time_frames].
    phase: Phase spectrum [freq_bins, time_frames].

Returns:
    Complex STFT [freq_bins, time_frames].*
- `stft_magnitude_to_model_input` (line 161) - *Prepare STFT magnitude for input to the Hamiltonian network.

Normalizes magnitude to [0, 1] range and shapes as [1, 1, H, W],
matching the expected input format (analogous to a grayscale image).

Args:
    magnitude: STFT magnitude [freq_bins, time_frames].

Returns:
    Normalized tensor [1, 1, freq_bins, time_frames].*
- `model_output_to_stft_magnitude` (line 186) - *Convert model output (energy mask in [0, 1]) back to STFT magnitude scale.

The model output represents the Hamiltonian energy structure --
which regions of the time-frequency plane carry coherent energy.
This is used to modulate the original magnitude.

Args:
    model_output: Energy mask [1, 1, freq_bins, time_frames] in [0, 1].
    original_magnitude: Original STFT magnitude [freq_bins, time_frames].

Returns:
    Reconstructed magnitude [freq_bins, time_frames].*
- `waveform_to_mel_spectrogram` (line 206) - *Convert waveform to normalized mel spectrogram (for visualization only).

Args:
    waveform: Audio waveform tensor [1, T] or [B, 1, T].

Returns:
    Normalized mel spectrogram [B, 1, n_mels, time_frames].*
- `save_audio` (line 229) - *Save a waveform tensor to an audio file.

Args:
    waveform: Audio tensor [1, T] or [T].
    file_path: Output file path.
    sample_rate: Sample rate (defaults to config sample rate).*
- `get_spectrogram_db_range` (line 249) - *Compute the dB range of a waveform's mel spectrogram.

Args:
    waveform: Audio waveform [1, T].

Returns:
    Tuple of (db_min, db_max).*

#### `audios.py`
**Path:** `audio/audios.py`

**Classs:**
- `HamiltonianConfig` (line 48) - *Immutable configuration container for all hyperparameters.
Eliminates magic numbers and provides single point of control.*
- `IAudioSource` (line 103) - *Interface for audio input sources.*
- `IFieldOperator` (line 122) - *Interface for Hamiltonian field evolution operators.*
- `IMetricCollector` (line 131) - *Interface for training metrics collection.*
- `AudioResampler` (line 149) - *Handles audio resampling using scipy.signal, avoiding librosa/numba dependencies.*
- `WaveFileSource` (line 204) - *Concrete implementation of audio source from file.
Supports automatic resampling to target sample rate using scipy.*
- `ComprehensiveMetricCollector` (line 278) - *Collects all metrics from Hamiltonian paper, activation functions,
and architectural diagnostics for informed decision-making.*
- `CheckpointManager` (line 326) - *Manages periodic checkpointing with atomic writes.*
- `AudioSpectrogramConverter` (line 387) - *Converts between audio waveforms and 2D field representations.
Adaptado para la arquitectura de experiment2 (grid_size=16).*
- `HamiltonianAudioProcessor` (line 468) - *Main orchestrator for Hamiltonian audio processing.
Demonstrates that auditory perception is epiphenomenon of Hamiltonian dynamics.
Usa la arquitectura exacta de experiment2.*

**Functions:**
- `main` (line 742) - *Entry point with argument parsing.*
- `segment_samples` (line 89) - *Calculate segment length in samples.*
- `freq_bins` (line 94) - *Calculate frequency bins for real FFT.*
- `read_segment` (line 107) - *Read audio segment. Returns None when exhausted.*
- `get_properties` (line 112) - *Return audio properties.*
- `close` (line 117) - *Release resources.*
- `evolve` (line 126) - *Evolve field state through Hamiltonian dynamics.*
- `record` (line 135) - *Record metric values.*
- `get_summary` (line 140) - *Return aggregated metrics.*
- `resample` (line 155) - *Resample audio from orig_sr to target_sr using polyphase filtering.*
- `load_wav_with_resample` (line 173) - *Load WAV file and resample to target sample rate.
Returns (audio_data, original_sample_rate).*
- `__init__` (line 210)
- `_validate_and_load` (line 220) - *Validate file format and load with automatic resampling.*
- `read_segment` (line 244) - *Read next audio segment.*
- `get_properties` (line 261) - *Return audio file properties.*
- `close` (line 273) - *Release resources.*
- `__init__` (line 284)
- `record` (line 289) - *Record comprehensive metrics.*
- `get_summary` (line 298) - *Return statistical summary of all metrics.*
- `export_to_json` (line 320) - *Export full history to JSON.*
- `__init__` (line 331)
- `check_and_save` (line 344) - *Check if checkpoint interval elapsed and save if necessary.
Returns path if saved, None otherwise.*
- `_save_checkpoint` (line 357) - *Atomic checkpoint save.*
- `__init__` (line 393)
- `waveform_to_field` (line 396) - *Convert 1D audio to 2D field representation via STFT.
Returns (1, 1, grid_size, grid_size) tensor compatible con experiment2.*
- `field_to_waveform` (line 432) - *Reconstruct waveform from 2D field representation.*
- `_forward_spectrogram` (line 458) - *Compute magnitude spectrogram.*
- `_inverse_spectrogram` (line 463) - *Griffin-Lim inverse.*
- `__init__` (line 475)
- `load_model_weights` (line 504) - *Load pretrained Hamiltonian operator desde safetensors.*
- `attach_source` (line 513) - *Attach audio source via dependency injection.*
- `process_stream` (line 517) - *Process audio stream through Hamiltonian perception.
Generates three epiphenomenal representations:
1. Energy Density (Resonance)
2. Topological Phase (Vortices)
3. Action Map (Perceptual Clarity)*
- `_process_single_segment` (line 592) - *Process single audio segment and return metrics.*
- `_calculate_phase_entropy` (line 684) - *Calculate topological entropy from phase distribution.*
- `_render_epiphenomena` (line 692) - *Render three epiphenomenal visualizations.*
- `export_metrics` (line 729) - *Export comprehensive metrics to file.*
- `force_checkpoint` (line 733) - *Force immediate checkpoint save.*

#### `checkpoint_manager.py`
**Path:** `audio/checkpoint_manager.py`

**Classs:**
- `CheckpointManager` (line 28) - *Manages model checkpointing with time-based intervals
and best-model tracking.*

**Functions:**
- `__init__` (line 34)
- `should_save_checkpoint` (line 41) - *Check if enough time has elapsed since the last checkpoint.*
- `save_checkpoint` (line 46) - *Save the current model state and training metadata.

Saves to a single 'latest.safetensors' file plus a JSON
metadata file containing optimizer state, epoch, and metrics.

Args:
    model: The model to checkpoint.
    optimizer: Current optimizer state.
    scheduler: Current LR scheduler state.
    epoch: Current epoch number.
    step: Current global step.
    metrics: Dictionary of current metric values.
    current_loss: Current total loss value.*
- `load_checkpoint` (line 98) - *Load a model checkpoint and return training metadata.

Uses the exact safetensors loading pattern:
    load_model(model, checkpoint_path)

Path resolution priority:
1. If CheckpointConfig.checkpoint_file_path is set, use that exact path
   (overrides load_best flag).
2. If load_best is True, use checkpoint_directory/best.safetensors.
3. Otherwise, use checkpoint_directory/latest.safetensors.

Args:
    model: The model to load weights into.
    load_best: If True and no explicit path set, load best model.

Returns:
    Metadata dictionary if available, None otherwise.*
- `best_loss` (line 156)

#### `config.py`
**Path:** `audio/config.py`

**Classs:**
- `AudioProcessingConfig` (line 18) - *Parameters governing raw audio ingestion and spectrogram computation.*
- `ModelArchitectureConfig` (line 34) - *Parametric architecture dimensions for the Hamiltonian Neural Network.

All hidden dimensions, matrix sizes, expansion factors, and layer counts
are configurable from this single source of truth.*
- `TrainingConfig` (line 80) - *All training loop hyperparameters and scheduling constants.*
- `CheckpointConfig` (line 109) - *Checkpoint persistence parameters.*
- `VisualizationConfig` (line 136) - *Parameters for audio reconstruction visualization and output.*
- `MetricsConfig` (line 158) - *Configuration for all tracked metrics during training and inference.*
- `HamiltonianAudioConfig` (line 182) - *Top-level configuration aggregator.

Composes all sub-configurations into a single injectable dependency,
following the Dependency Inversion Principle.*

**Functions:**
- `validate` (line 62) - *Ensure architectural coherence.*
- `checkpoint_path` (line 121)
- `best_model_path` (line 127)
- `metadata_path` (line 131)
- `validate_all` (line 198) - *Run validation on all sub-configurations.*
- `ensure_directories` (line 206) - *Create required output directories if they do not exist.*

#### `experiment2.py`
**Path:** `audio/experiment2.py`

**Classs:**
- `Config` (line 22)
- `SeedManager` (line 74)
- `LoggerFactory` (line 84)
- `IAnalysisStrategy` (line 99)
- `IMetricsCalculator` (line 105)
- `HamiltonianOperator` (line 111)
- `HamiltonianDataset` (line 133)
- `SpectralLayer` (line 183)
- `HamiltonianNeuralNetwork` (line 227)
- `LocalComplexityAnalyzer` (line 257)
- `SuperpositionAnalyzer` (line 273)
- `CrystallographyMetricsCalculator` (line 302)
- `ThermodynamicMetricsCalculator` (line 737)
- `SpectroscopyMetricsCalculator` (line 770)
- `CheckpointManager` (line 804)
- `TrainingMetricsMonitor` (line 874)
- `GlassStateDetector` (line 912)
- `TrainingEngine` (line 973)
- `SeedMiningSystem` (line 1113)
- `SingleExperimentRunner` (line 1164)
- `CheckpointAnalyzer` (line 1223)
- `Application` (line 1275)

**Functions:**
- `main` (line 1334)
- `set_seed` (line 76)
- `create_logger` (line 86)
- `analyze` (line 101)
- `compute` (line 107)
- `__init__` (line 112)
- `_precompute_spectral_operators` (line 116)
- `apply` (line 122)
- `time_evolution` (line 127)
- `__init__` (line 134)
- `__len__` (line 173)
- `__getitem__` (line 176)
- `get_validation_batch` (line 179)
- `__init__` (line 184)
- `forward` (line 195)
- `__init__` (line 228)
- `forward` (line 243)
- `compute_local_complexity` (line 259)
- `compute_superposition` (line 275)
- `compute` (line 303) - *Implementación de interfaz IMetricsCalculator.
Delega a compute_all_metrics con los argumentos correctos.*
- `compute_gradient_covariance_kappa` (line 311)
- `compute_discretization_margin_from_state_dict` (line 348) - *Calcula el margen de discretización desde los parámetros del modelo.
Versión estática que no requiere diccionario externo.*
- `compute_discretization_margin` (line 361) - *Calcula el margen de discretización desde un diccionario de coeficientes.*
- `compute_alpha_purity_from_model` (line 373) - *Calcula el índice de pureza alpha directamente desde el modelo.*
- `compute_alpha_purity` (line 383) - *Calcula el índice de pureza alpha desde un diccionario de coeficientes.*
- `compute_kappa` (line 393) - *Número de condición de la matriz de covarianza de gradientes.*
- `compute_kappa_quantum` (line 464) - *Versión del cálculo cuántico de kappa que opera directamente sobre el modelo.*
- `compute_kappa_quantum_from_coeffs` (line 492) - *Versión del cálculo cuántico de kappa desde diccionario de coeficientes.*
- `_compute_crystallography_metrics` (line 511) - *Métricas cristalográficas con aislamiento completo de errores.*
- `_check_weight_integrity` (line 539) - *Verifica integridad de pesos: NaN, Inf, y estadísticas básicas.*
- `compute_poynting_vector` (line 603) - *Vector de Poynting: flujo de energía en el espacio de parámetros.
Análogo electromagnético para redes neuronales.*
- `compute_all_metrics` (line 679) - *Calcula todas las métricas cristalográficas con manejo de errores.*
- `compute` (line 738)
- `compute_effective_temperature` (line 747)
- `compute_specific_heat` (line 760)
- `compute` (line 771)
- `compute_weight_diffraction` (line 776)
- `_compute_spectral_entropy` (line 795)
- `__init__` (line 805)
- `should_save_checkpoint` (line 813)
- `save_checkpoint` (line 818)
- `__init__` (line 875)
- `update_metrics` (line 895)
- `__init__` (line 913)
- `should_stop` (line 918)
- `is_crystal_formed` (line 963)
- `__init__` (line 974)
- `train_epoch` (line 997)
- `validate` (line 1027)
- `compute_weight_metrics` (line 1040)
- `execute_training` (line 1056)
- `__init__` (line 1114)
- `mine` (line 1118)
- `__init__` (line 1165)
- `run` (line 1175)
- `__init__` (line 1224)
- `analyze` (line 1230)
- `__init__` (line 1276)
- `_create_argument_parser` (line 1280)
- `run` (line 1294)
- `safe_compute` (line 694)

#### `inference.py`
**Path:** `audio/inference.py`

**Classs:**
- `HamiltonianAudioInference` (line 37) - *Performs complete Hamiltonian audio analysis on a given audio file.*

**Functions:**
- `__init__` (line 42)
- `analyze_audio` (line 73) - *Perform complete Hamiltonian analysis on an audio file.

Args:
    audio_file_path: Path to the audio file to analyze.
    output_prefix: Optional prefix for output filenames.*
- `_compute_energy_mask_patched` (line 158) - *Compute energy mask over the full STFT magnitude, processing
in patches along the time axis if the input exceeds patch width.

Args:
    model_input: Normalized STFT magnitude [1, 1, freq_bins, time_frames].

Returns:
    Energy mask [1, 1, freq_bins, time_frames] in [0, 1].*
- `_extract_hamiltonian_fields_patched` (line 209) - *Extract Hamiltonian fields over full STFT magnitude with patching.

Args:
    model_input: Normalized STFT magnitude [1, 1, freq_bins, time_frames].

Returns:
    Tuple of (amplitude_map, phase_map, action_map).*
- `_compute_inference_metrics` (line 270) - *Compute all inference-time metrics on the STFT domain.*
- `_print_inference_metrics` (line 288) - *Print all computed inference metrics.*

#### `losses.py`
**Path:** `audio/losses.py`

**Classs:**
- `HamiltonianLossComputer` (line 28) - *Computes the composite Hamiltonian loss function with all
physics-based regularization terms.

Each loss component is independently weighted via TrainingConfig,
enabling fine-grained control over the training objective.*

**Functions:**
- `__init__` (line 37)
- `compute_total_loss` (line 41) - *Compute the complete weighted loss with all Hamiltonian terms.

Args:
    prediction: Model output [B, 1, H, W].
    target: Ground truth [B, 1, H, W].
    intermediates: List of intermediate hidden states from forward pass.
    model: The model (for parameter access in regularization).

Returns:
    Tuple of (total_loss tensor, dict of individual loss values).*
- `_compute_reconstruction_loss` (line 94) - *MSE reconstruction loss between predicted and target spectrograms.*
- `_compute_energy_conservation_loss` (line 100) - *Penalize energy drift across layers.

The Hamiltonian energy E = 0.5 * ||phi||^2 should remain
approximately constant through the evolution layers.*
- `_compute_symplectic_loss` (line 122) - *Penalize violation of symplectic structure.

For pairs of consecutive states (q_i, q_{i+1}), we interpret
q as position and dq = q_{i+1} - q_i as a proxy for momentum.
The symplectic form dq ^ dp should be preserved.*
- `_compute_spectral_consistency_loss` (line 145) - *Penalize spectral divergence in frequency domain.

||FFT(prediction) - FFT(target)||_F / ||FFT(target)||_F*
- `_compute_phase_coherence_loss` (line 161) - *Penalize phase misalignment between prediction and target.

1 - |mean(exp(i * (angle(FFT(pred)) - angle(FFT(target)))))|*
- `_compute_action_minimization_loss` (line 177) - *Principle of least action: minimize the total action
S = sum(|phi_{i+1} - phi_i|) along the trajectory.*
- `_compute_liouville_loss` (line 192) - *Liouville theorem: phase space volume should be preserved.

We approximate this by checking that the variance of hidden
states remains approximately constant through evolution.*
- `_compute_hamiltonian_constraint_loss` (line 214) - *Hamilton's equations: dq/dt = dH/dp, dp/dt = -dH/dq.

Approximated by checking time-reversal symmetry:
the forward evolution followed by reverse should return
to the initial state.*

#### `main.py`
**Path:** `audio/main.py`

**Functions:**
- `build_argument_parser` (line 34) - *Construct the complete argument parser with all configurable parameters.*
- `build_config_from_args` (line 103) - *Construct the full configuration from parsed CLI arguments.*
- `validate_audio_file` (line 163) - *Validate that the audio file exists and has a supported extension.*
- `print_configuration_banner` (line 175) - *Print a formatted configuration summary.*
- `run_training` (line 215) - *Execute the training pipeline.*
- `run_inference` (line 226) - *Execute the inference pipeline.*
- `main` (line 236) - *Main entry point.*

#### `metrics.py`
**Path:** `audio/metrics.py`

**Classs:**
- `HamiltonianMetricsTracker` (line 26) - *Tracks and computes all Hamiltonian mechanics metrics during
training and inference.

Each metric method is a pure computation with no side effects
beyond updating internal accumulators, following the
Interface Segregation Principle by exposing granular metric methods.*

**Functions:**
- `__init__` (line 36)
- `_initialize_history_buffers` (line 43) - *Pre-allocate deque buffers for each tracked metric.*
- `compute_hamiltonian_energy` (line 74) - *Compute the Hamiltonian H(q, p) = T(p) + V(q).

T(p) = 0.5 * ||p||^2 (kinetic energy)
V(q) = 0.5 * ||q||^2 (potential energy in harmonic approximation)

Args:
    q: Generalized coordinates tensor (position in phase space).
    p: Conjugate momenta tensor.

Returns:
    Scalar Hamiltonian energy value.*
- `compute_symplectic_form` (line 97) - *Compute the symplectic 2-form omega(dq, dp) = sum(dq_i ^ dp_i).

Measures preservation of the canonical symplectic structure
under Hamiltonian flow. Should remain invariant for symplectic
integrators.

Args:
    q: Generalized coordinates.
    p: Conjugate momenta.
    dq: Variation in coordinates.
    dp: Variation in momenta.

Returns:
    Scalar symplectic form magnitude.*
- `compute_liouville_measure` (line 125) - *Compute Liouville measure |det(J)| for the flow map Jacobian.

By Liouville's theorem, Hamiltonian flow preserves phase space
volume, so det(J) should equal 1 for exact symplectic evolution.

Args:
    jacobian: The Jacobian matrix of the phase space transformation.

Returns:
    Absolute determinant of the Jacobian.*
- `compute_phase_space_volume` (line 153) - *Estimate phase space volume occupied by the state (q, p).

Uses the covariance ellipsoid approximation:
V ~ sqrt(det(Cov([q, p])))

Args:
    q: Generalized coordinates (flattened).
    p: Conjugate momenta (flattened).

Returns:
    Estimated phase space volume.*
- `compute_action_integral` (line 181) - *Compute the action integral S = integral(L dt) along a trajectory.

L = T - V = 0.5*||p||^2 - 0.5*||q||^2 (Lagrangian)

Args:
    q_trajectory: Sequence of coordinate states [T, ...].
    p_trajectory: Sequence of momentum states [T, ...].
    dt: Time step between trajectory points.

Returns:
    Total action along the trajectory.*
- `compute_poisson_bracket` (line 208) - *Estimate the Poisson bracket {f, g} = sum(df/dq * dg/dp - df/dp * dg/dq).

Uses finite differences on the discretized phase space.

Args:
    f_values: Observable f evaluated on phase space grid.
    g_values: Observable g evaluated on phase space grid.
    q: Coordinate grid.
    p: Momentum grid.

Returns:
    Estimated Poisson bracket scalar.*
- `compute_spectral_entropy` (line 238) - *Compute spectral entropy H = -sum(p_i * log(p_i)).

Measures the disorder/uniformity of the spectral distribution.
Maximum entropy indicates uniform spectrum (white noise),
minimum indicates pure tone (single frequency).

Args:
    spectrum: Power spectrum tensor (non-negative).

Returns:
    Scalar spectral entropy.*
- `compute_reconstruction_snr` (line 259) - *Compute Signal-to-Noise Ratio in dB.

SNR = 10 * log10(||original||^2 / ||original - reconstructed||^2)

Args:
    original: Ground truth signal.
    reconstructed: Reconstructed signal.

Returns:
    SNR in decibels.*
- `compute_spectral_convergence` (line 281) - *Compute spectral convergence metric.

SC = ||S_orig - S_recon||_F / ||S_orig||_F

Lower values indicate better spectral fidelity.

Args:
    original_spectrum: Original frequency domain representation.
    reconstructed_spectrum: Reconstructed frequency domain representation.

Returns:
    Spectral convergence ratio.*
- `compute_phase_coherence` (line 305) - *Compute phase coherence between original and reconstructed signals.

PC = |mean(exp(i * (phi_orig - phi_recon)))|

Value of 1.0 indicates perfect phase alignment.

Args:
    phase_original: Phase spectrum of original signal.
    phase_reconstructed: Phase spectrum of reconstructed signal.

Returns:
    Phase coherence in [0, 1].*
- `compute_energy_drift` (line 328) - *Compute relative energy drift from initial state.

drift = |E_current - E_initial| / (|E_initial| + epsilon)

Args:
    energy_initial: Hamiltonian energy at t=0.
    energy_current: Hamiltonian energy at current time.

Returns:
    Relative energy drift.*
- `record_gradient_norm` (line 348) - *Compute and record the total gradient norm across all parameters.*
- `record_parameter_norm` (line 359) - *Compute and record the total parameter norm.*
- `record_learning_rate` (line 369) - *Record current learning rate.*
- `record_loss_component` (line 374) - *Record an individual loss component value.*
- `_record` (line 379) - *Store a metric value in history and current snapshot.*
- `get_current_metrics` (line 388) - *Return a snapshot of all current metric values.*
- `get_moving_averages` (line 392) - *Compute moving averages for all tracked metrics.*
- `get_formatted_metrics_string` (line 400) - *Format all current metrics into a human-readable string for progress bars.*
- `increment_step` (line 413) - *Advance the global step counter.*
- `step_count` (line 418)
- `should_log` (line 421) - *Determine if metrics should be logged at this step.*

#### `model.py`
**Path:** `audio/model.py`

**Classs:**
- `SpectralEvolutionLayer` (line 27) - *Single Hamiltonian spectral evolution layer.

Performs frequency-domain evolution using learnable complex kernels.
Kernel shape: [hidden_dim, hidden_dim, kernel_base_height, kernel_base_width]
matching the original experiment2 architecture.*
- `HamiltonianNeuralNetwork` (line 162) - *Complete Hamiltonian Neural Network with parametric architecture.

Architecture (matching experiment2):
    1. Input projection: Conv2d(1, hidden_dim, kernel, pad)
    2. N spectral evolution layers with learnable complex kernels
    3. Output projection: Conv2d(hidden_dim, 1, kernel, pad)*

**Functions:**
- `__init__` (line 36)
- `forward` (line 51) - *Apply one step of Hamiltonian spectral evolution via RFFT2.

Args:
    x: Input tensor [B, C, H, W] in spatial domain.

Returns:
    Evolved tensor [B, C, H, W] in spatial domain.*
- `evolve_complex` (line 85) - *Full complex FFT evolution for amplitude and phase extraction.

Uses full FFT2 (not RFFT2) to preserve complete complex structure.

Args:
    x: Input tensor [B, C, H, W].
    target_height: Output spatial height.
    target_width: Output spatial width.

Returns:
    Complex-valued evolved field in spatial domain.*
- `evolve_real` (line 124) - *Real FFT evolution for action map computation.

Args:
    x: Input tensor [B, C, H, W].
    target_height: Output spatial height.
    target_width: Output spatial width.

Returns:
    Real-valued evolved field in spatial domain.*
- `__init__` (line 172)
- `forward` (line 200) - *Full forward pass: project -> evolve -> reconstruct.

Args:
    x: Input tensor [B, 1, H, W].

Returns:
    Reconstructed tensor [B, 1, H, W].*
- `forward_with_intermediates` (line 216) - *Forward pass returning intermediate hidden states for analysis.

Args:
    x: Input tensor [B, 1, H, W].

Returns:
    Tuple of (output, list of intermediate states).*
- `extract_hamiltonian_fields` (line 237) - *Extract the three Hamiltonian field representations:
1. Amplitude map (energy density / resonance)
2. Phase map (topological structure / vortices)
3. Action map (constructive interference = clear vision)

Mirrors the visual processing logic from the original code.

Args:
    x: Input tensor [B, 1, H, W].

Returns:
    Tuple of (amplitude_map, phase_map, action_map) each [H, W].*
- `compute_energy_mask` (line 270) - *Compute the Hamiltonian energy mask for spectral reconstruction.

This method implements the constructive interference principle:
the complex FFT evolution reveals amplitude (resonance) and
phase (topology). Their constructive sum amplitude * cos(phase)
identifies WHERE in the time-frequency plane the model detects
coherent energy structure.

The real FFT evolution provides a complementary view (action),
which highlights WHERE the model sees change/structure.

Both are combined and normalized to [0, 1] as a mask that
can be applied to the original STFT magnitude to produce
the reconstructed audio.

This is the audio equivalent of the "clear vision" (action map)
in the visual domain.

Args:
    x: STFT magnitude input [B, 1, freq_bins, time_frames],
       normalized to [0, 1].

Returns:
    Energy mask [B, 1, freq_bins, time_frames] in [0, 1].*

#### `trainer.py`
**Path:** `audio/trainer.py`

**Classs:**
- `AudioSpectrogramDatasetBuilder` (line 38) - *Builds a TensorDataset of spectrogram patches from an audio file.

Segments the full mel spectrogram into overlapping patches
of size (n_mels, matrix_size_width) for training.*
- `HamiltonianAudioTrainer` (line 79) - *Complete training pipeline for the Hamiltonian Audio Network.

Manages:
- Model initialization and checkpoint recovery
- Optimizer and scheduler configuration
- Training and validation loops
- Full metric reporting at every step
- Time-based checkpointing
- Early stopping*

**Functions:**
- `__init__` (line 46)
- `build_dataset` (line 49) - *Segment a mel spectrogram into training patches.

Args:
    mel_spectrogram: Full spectrogram [1, 1, n_mels, time_frames].

Returns:
    TensorDataset of (input_patch, target_patch) pairs.*
- `__init__` (line 92)
- `_attempt_checkpoint_recovery` (line 129) - *Load existing checkpoint if available.*
- `train` (line 151) - *Execute the full training pipeline on an audio file.

Args:
    audio_file_path: Path to the input audio file.*
- `_train_one_epoch` (line 255) - *Execute one training epoch with full metric tracking.*
- `_validate` (line 344) - *Run validation pass and return metrics.*
- `model` (line 377)
- `audio_processor` (line 381)

#### `visualization.py`
**Path:** `audio/visualization.py`

**Classs:**
- `HamiltonianAudioVisualizer` (line 29) - *Generates all scientific visualizations for Hamiltonian audio analysis.*

**Functions:**
- `__init__` (line 34)
- `render_complete_analysis` (line 43) - *Generate the complete suite of Hamiltonian analysis visualizations.

Args:
    amplitude_map: Energy density field [H, W].
    phase_map: Phase topology field [H, W].
    action_map: Action density field [H, W].
    original_spectrogram: Original mel spectrogram [1, 1, H, W].
    reconstructed_spectrogram: Reconstructed mel spectrogram [1, 1, H, W].
    original_waveform: Original audio waveform [1, T].
    reconstructed_waveform: Reconstructed audio waveform [1, T].
    output_prefix: Filename prefix for all outputs.*
- `_render_hamiltonian_fields` (line 91) - *Render the three Hamiltonian field visualizations.*
- `_render_spectrogram_comparison` (line 140) - *Render original vs reconstructed spectrogram comparison.*
- `_render_phase_portrait` (line 185) - *Render 2D phase portrait (amplitude vs phase histogram).*
- `_render_energy_landscape` (line 218) - *Render energy landscape as a 3D surface plot.*
- `_render_waveform_comparison` (line 270) - *Render original vs reconstructed waveform comparison.*

#### `check_fase_berry.py`
**Path:** `check_fase_berry.py`

*No symbols extracted*

#### `diff_weights.py`
**Path:** `diff_weights.py`

**Functions:**
- `analize_checkpoint` (line 5)

#### `dirac.py`
**Path:** `dirac.py`

**Classs:**
- `DiracConfig` (line 25)
- `DiracDeltaAnalyzer` (line 38)
- `DiracVisualizer` (line 328)

**Functions:**
- `analyze_checkpoint` (line 574)
- `analyze_multiple_checkpoints` (line 620)
- `main` (line 652)
- `__init__` (line 40)
- `extract_charge_distribution` (line 64)
- `compute_dirac_delta_approximation` (line 77)
- `compute_electric_field` (line 112)
- `compute_electric_flux` (line 157)
- `compute_divergence` (line 192)
- `verify_gauss_law` (line 200)
- `analyze_all` (line 223)
- `_print_report` (line 279)
- `plot_charge_distribution` (line 331)
- `plot_electric_field` (line 379)
- `plot_divergence` (line 441)
- `plot_combined_analysis` (line 490)

#### `expand.py`
**Path:** `expand.py`

**Functions:**
- `load_config` (line 18)
- `expand_spectral_weights` (line 23) - *Expand spectral kernels via zero-padding in frequency domain.*
- `expand_model` (line 43) - *Create a new model with expanded spectral weights.*
- `evaluate_model` (line 74) - *Evaluate expanded model on synthetic data.*
- `main` (line 105)

#### `experiment.py`
**Path:** `experiment.py`

**Classs:**
- `Config` (line 41)
- `IAnalysisStrategy` (line 109)
- `IMetricsCalculator` (line 115)
- `HamiltonianOperator` (line 121) - *True Hamiltonian operator H = -nabla^2 on torus.*
- `FastDataset` (line 146) - *Fast dataset for Hamiltonian operator learning.*
- `SpectralLayer` (line 203) - *Spectral layer with correct complex multiplication.*
- `SimpleHamiltonianNet` (line 255) - *Compact network for Hamiltonian operator learning.*
- `LocalComplexityAnalyzer` (line 293)
- `SuperpositionAnalyzer` (line 310)
- `CrystallographyMetrics` (line 340)
- `ThermodynamicMetrics` (line 444)
- `SpectroscopyMetrics` (line 470)
- `CheckpointManager` (line 500)
- `TrainingMonitor` (line 573)
- `GlassStopper` (line 611)
- `BoltzmannAnalysisProgram` (line 879)

**Functions:**
- `set_seed` (line 88)
- `setup_logger` (line 96)
- `train_with_early_glass_stop` (line 670) - *Train model with early stopping for glass detection.*
- `seed_miner` (line 803) - *Mine for crystal seeds by trying sequential seeds.*
- `main` (line 856)
- `analyze` (line 111)
- `compute` (line 117)
- `__init__` (line 124)
- `_precompute_spectral_operators` (line 128)
- `apply` (line 135)
- `time_evolution` (line 140)
- `__init__` (line 149)
- `__len__` (line 193)
- `__getitem__` (line 196)
- `get_val_batch` (line 199)
- `__init__` (line 206)
- `forward` (line 219)
- `__init__` (line 258)
- `forward` (line 279)
- `compute_local_complexity` (line 295) - *Compute Local Complexity (LC) metric for weight matrix.*
- `compute_superposition` (line 312) - *Compute Superposition (SP) metric for weight matrix.*
- `compute_kappa` (line 342)
- `compute_discretization_margin` (line 378)
- `compute_alpha_purity` (line 387)
- `compute_kappa_quantum` (line 394)
- `compute_poynting_vector` (line 411)
- `compute_all_metrics` (line 426)
- `compute_effective_temperature` (line 446)
- `compute_specific_heat` (line 460)
- `compute_weight_diffraction` (line 472)
- `_compute_spectral_entropy` (line 491)
- `__init__` (line 501)
- `should_save_checkpoint` (line 509)
- `save_checkpoint` (line 514)
- `__init__` (line 574)
- `update_metrics` (line 594)
- `__init__` (line 612)
- `should_stop` (line 616) - *Check if the system is in glass state and should stop mining.*
- `__init__` (line 880)
- `load_and_analyze_checkpoint` (line 886)
- `dataloader` (line 903)

#### `experiment2.py`
**Path:** `experiment2.py`

**Classs:**
- `Config` (line 22)
- `SeedManager` (line 74)
- `LoggerFactory` (line 84)
- `IAnalysisStrategy` (line 99)
- `IMetricsCalculator` (line 105)
- `HamiltonianOperator` (line 111)
- `HamiltonianDataset` (line 133)
- `SpectralLayer` (line 183)
- `HamiltonianNeuralNetwork` (line 227)
- `LocalComplexityAnalyzer` (line 257)
- `SuperpositionAnalyzer` (line 273)
- `CrystallographyMetricsCalculator` (line 302)
- `ThermodynamicMetricsCalculator` (line 737)
- `SpectroscopyMetricsCalculator` (line 770)
- `CheckpointManager` (line 804)
- `TrainingMetricsMonitor` (line 874)
- `GlassStateDetector` (line 912)
- `TrainingEngine` (line 973)
- `SeedMiningSystem` (line 1113)
- `SingleExperimentRunner` (line 1164)
- `CheckpointAnalyzer` (line 1223)
- `Application` (line 1275)

**Functions:**
- `main` (line 1334)
- `set_seed` (line 76)
- `create_logger` (line 86)
- `analyze` (line 101)
- `compute` (line 107)
- `__init__` (line 112)
- `_precompute_spectral_operators` (line 116)
- `apply` (line 122)
- `time_evolution` (line 127)
- `__init__` (line 134)
- `__len__` (line 173)
- `__getitem__` (line 176)
- `get_validation_batch` (line 179)
- `__init__` (line 184)
- `forward` (line 195)
- `__init__` (line 228)
- `forward` (line 243)
- `compute_local_complexity` (line 259)
- `compute_superposition` (line 275)
- `compute` (line 303) - *Implementación de interfaz IMetricsCalculator.
Delega a compute_all_metrics con los argumentos correctos.*
- `compute_gradient_covariance_kappa` (line 311)
- `compute_discretization_margin_from_state_dict` (line 348) - *Calcula el margen de discretización desde los parámetros del modelo.
Versión estática que no requiere diccionario externo.*
- `compute_discretization_margin` (line 361) - *Calcula el margen de discretización desde un diccionario de coeficientes.*
- `compute_alpha_purity_from_model` (line 373) - *Calcula el índice de pureza alpha directamente desde el modelo.*
- `compute_alpha_purity` (line 383) - *Calcula el índice de pureza alpha desde un diccionario de coeficientes.*
- `compute_kappa` (line 393) - *Número de condición de la matriz de covarianza de gradientes.*
- `compute_kappa_quantum` (line 464) - *Versión del cálculo cuántico de kappa que opera directamente sobre el modelo.*
- `compute_kappa_quantum_from_coeffs` (line 492) - *Versión del cálculo cuántico de kappa desde diccionario de coeficientes.*
- `_compute_crystallography_metrics` (line 511) - *Métricas cristalográficas con aislamiento completo de errores.*
- `_check_weight_integrity` (line 539) - *Verifica integridad de pesos: NaN, Inf, y estadísticas básicas.*
- `compute_poynting_vector` (line 603) - *Vector de Poynting: flujo de energía en el espacio de parámetros.
Análogo electromagnético para redes neuronales.*
- `compute_all_metrics` (line 679) - *Calcula todas las métricas cristalográficas con manejo de errores.*
- `compute` (line 738)
- `compute_effective_temperature` (line 747)
- `compute_specific_heat` (line 760)
- `compute` (line 771)
- `compute_weight_diffraction` (line 776)
- `_compute_spectral_entropy` (line 795)
- `__init__` (line 805)
- `should_save_checkpoint` (line 813)
- `save_checkpoint` (line 818)
- `__init__` (line 875)
- `update_metrics` (line 895)
- `__init__` (line 913)
- `should_stop` (line 918)
- `is_crystal_formed` (line 963)
- `__init__` (line 974)
- `train_epoch` (line 997)
- `validate` (line 1027)
- `compute_weight_metrics` (line 1040)
- `execute_training` (line 1056)
- `__init__` (line 1114)
- `mine` (line 1118)
- `__init__` (line 1165)
- `run` (line 1175)
- `__init__` (line 1224)
- `analyze` (line 1230)
- `__init__` (line 1276)
- `_create_argument_parser` (line 1280)
- `run` (line 1294)
- `safe_compute` (line 694)

#### `export.py`
**Path:** `export.py`

*No symbols extracted*

#### `get_meditions.py`
**Path:** `get_meditions.py`

**Classs:**
- `ThermodynamicConfig` (line 28) - *Configuración termodinámica para análisis de HPU Core*
- `ThermodynamicPotential` (line 72) - *Potencial de Helmholtz: F = U - T*S + μ*N + α_term*C*
- `CrystallographyMetrics` (line 98) - *Métricas de cristalografía para redes neuronales Hamiltonianas.
Mide la "pureza" estructural de los pesos aprendidos.*
- `ThermodynamicMetrics` (line 394) - *Análisis termodinámico del proceso de entrenamiento.
Temperatura efectiva, calor específico, transiciones de fase.*
- `SpectroscopyMetrics` (line 634) - *Análisis espectroscópico de pesos: difracción, descomposición, parámetros de red.*
- `CheckpointVerifier` (line 740)
- `SpectralCoefficients` (line 105) - *Contenedor para coeficientes espectrales de HPU Core*

**Functions:**
- `setup_logger` (line 51)
- `verify_latest_checkpoints` (line 1439) - *Verifica los N checkpoints más recientes con análisis completo*
- `main` (line 1500)
- `helmholtz_free_energy` (line 81) - *F = U - T*S (a μ y N constantes)*
- `gibbs_free_energy` (line 85) - *G = F + μ*N + P*V (presión algorítmica)*
- `is_stable` (line 90) - *Criterio de estabilidad: dG < 0*
- `compute_kappa` (line 127) - *Número de condición de la matriz de covarianza de gradientes.
FIX: Limita tamaño de gradientes y usa método iterativo si es necesario.*
- `compute_discretization_margin` (line 187) - *δ = max |w - round(w)| sobre todos los parámetros.
Mide qué tan cerca están los pesos de valores enteros.*
- `compute_alpha_purity` (line 203) - *α = -log(δ). Pureza cristalina.
α > 7 indica estructura cristalina perfecta.*
- `compute_local_complexity` (line 214) - *Fracción de parámetros "activos" (no cerca de cero).*
- `compute_kappa_quantum` (line 226) - *κ cuántico: número de condición con regularización cuántica.
FIX: Usa método iterativo de potencia en lugar de matriz densa.*
- `_compute_kappa_iterative` (line 255) - *Método de potencia para estimar κ sin construir matriz.
Estima λ_max y λ_min de la matriz de covarianza regularizada.*
- `compute_poynting_vector` (line 312) - *Vector de Poynting: flujo de energía en el espacio de parámetros.
Análogo electromagnético para redes neuronales.*
- `compute_all_metrics` (line 366) - *Calcula todas las métricas cristalográficas.*
- `compute_effective_temperature` (line 401) - *T_eff = (lr/2) * Var(∇L). Temperatura de fluctuaciones.*
- `compute_specific_heat` (line 418) - *C_v = Var(U) / T^2. Detecta transiciones de fase (picos en C_v).*
- `compute_critical_exponents` (line 436) - *Exponentes críticos cerca de transiciones de fase.*
- `compute_equation_of_state` (line 504) - *Ecuación de estado: T_c(α) = T_0 * exp(-c*α)
Relación constitutiva cristal-vidrio.*
- `compute_mutual_information` (line 539) - *Información mutua pesos-gradientes.*
- `estimate_hbar_algorithmic` (line 561) - *ħ algorítmico efectivo.*
- `compute_fisher_information_matrix` (line 572) - *Matriz de información de Fisher.*
- `compute_ricci_curvature` (line 593) - *Curvatura de Ricci escalar.*
- `calculate_carnot_efficiency` (line 608) - *Eficiencia de Carnot del proceso de aprendizaje.*
- `compute_weight_diffraction` (line 640) - *Patrón de difracción de pesos (FFT).
Detecta periodicidad cristalina (picos de Bragg).*
- `_compute_spectral_entropy` (line 672) - *Entropía espectral de Shannon.*
- `extract_lattice_parameters` (line 680) - *Extrae parámetros de red vía SVD.*
- `compute_gibbs_free_energy` (line 732) - *Energía libre de Gibbs.*
- `__init__` (line 741)
- `verify_all_metrics` (line 782) - *Calcula TODAS las métricas desde cero y compara con las guardadas*
- `_check_weight_integrity` (line 849) - *Verifica que los pesos no tengan NaN/Inf.
FIX: Evita calcular std en tensores con 1 elemento.*
- `_compute_validation_metrics` (line 915) - *Calcula MSE y accuracy de validación desde cero*
- `_compute_discretization_metrics` (line 942) - *Calcula delta, alpha, purity, etc.*
- `_compute_quantization_metrics` (line 986) - *Calcula la penalización de cuantización*
- `_compute_loss_metrics` (line 1005) - *Reconstruye el loss total*
- `_compute_crystallography_metrics` (line 1031) - *Métricas cristalográficas completas*
- `_compute_thermodynamic_metrics` (line 1038) - *Métricas termodinámicas*
- `_approximate_ricci_curvature` (line 1113) - *Aproximación de curvatura de Ricci para HPU Core*
- `_compute_spectroscopy` (line 1134) - *Análisis espectroscópico*
- `_compute_thermodynamic_potential` (line 1164) - *Calcula potencial termodinámico completo*
- `_compare_with_stored` (line 1188) - *Compara métricas calculadas vs almacenadas en el checkpoint*
- `_check_internal_consistency` (line 1226) - *Verifica consistencia entre métricas relacionadas*
- `_compute_health_score` (line 1267) - *Calcula un score de salud del checkpoint (0-100)*
- `_assign_crystallographic_grade` (line 1336) - *Asigna grado cristalográfico*
- `_print_report` (line 1349) - *Imprime reporte formateado con todas las métricas nuevas*
- `from_model` (line 112) - *Extrae coeficientes del modelo HPU Core*

#### `hamiltonian_mbl.py`
**Path:** `hamiltonian_mbl.py`

**Classs:**
- `HamiltonianArchitectureConfig` (line 37) - *Configuration for Hamiltonian Neural Network architecture.
All architectural hyperparameters are centralized here.*
- `MBLAnalysisConfig` (line 67) - *Comprehensive configuration for MBL analysis of Hamiltonian NN crystallization.
All analysis parameters are centralized following SOLID principles.*
- `TrainingConfig` (line 141) - *Configuration for training process.*
- `IModel` (line 160) - *Protocol for models compatible with MBL analysis.*
- `ILevelSpacingCalculator` (line 167) - *Protocol for level spacing ratio calculation.*
- `IParticipationRatioCalculator` (line 173) - *Protocol for participation ratio calculation.*
- `ISyntheticPlanckCalculator` (line 179) - *Protocol for synthetic Planck's constant calculation.*
- `IDiscretizationDialAnalyzer` (line 185) - *Protocol for discretization dial analysis.*
- `ICheckpointManager` (line 191) - *Protocol for checkpoint management.*
- `ITrainingMetricsCollector` (line 199) - *Protocol for collecting all training metrics.*
- `ArchitectureMigrator` (line 209) - *Migra pesos de SimpleHamiltonianNet (Conv2d) a HamiltonianNeuralNetwork (Linear).*
- `SpectralHamiltonianLayer` (line 346) - *Spectral layer implementing Hamiltonian dynamics in Fourier space.
Preserves energy conservation through symplectic integration.*
- `HamiltonianNeuralNetwork` (line 412) - *Complete Hamiltonian Neural Network for learning dynamical systems.
Uses spectral layers to ensure energy conservation and symplectic structure.*
- `HamiltonianDataset` (line 555) - *Generates physics-informed training data for Hamiltonian NN.
Creates trajectories from known dynamical systems.*
- `LevelSpacingRatioCalculator` (line 608) - *Calculates the level spacing ratio r for MBL phase detection.

The ratio r_n = min(delta_n, delta_{n+1}) / max(delta_n, delta_{n+1})
where delta_n = E_{n+1} - E_n (energy level spacing).*
- `ParticipationRatioCalculator` (line 719) - *Calculates Inverse Participation Ratio (IPR) for localization analysis.
IPR = sum_i |c_i|^4 where c_i are coefficients in the chosen basis.*
- `SyntheticPlanckConstantCalculator` (line 801) - *Calculates effective synthetic Planck's constant (hbar_eff) from model properties.
Based on the relation: hbar_eff ∝ 1 / sqrt(PR * Energy_Gap)*
- `DiscretizationDialAnalyzer` (line 844) - *Analyzes the discretization parameter delta as a phase transition control.*
- `PurityIndexCalculator` (line 947) - *Calculates the 'crystallinity' of the weight distribution.*
- `EffectiveTemperatureCalculator` (line 1004) - *Calculates effective temperature from loss history.*
- `KrylovComplexityCalculator` (line 1048) - *Calculates Krylov complexity as a measure of operator growth and scrambling.
Based on the spread of operators in Krylov space.*
- `CrystallinityIndexCalculator` (line 1089) - *Calculates crystallinity index through spectral analysis of weight matrices.
Analogous to X-ray diffraction for physical crystals.*
- `ResilienceSpectrometer` (line 1144) - *Measures algorithmic resilience through controlled perturbations.
Tests stability across different subspaces and noise levels.*
- `PhaseClassifier` (line 1243) - *Classifies the crystallization phase based on alpha and temperature.*
- `CheckpointMigrator` (line 1270) - *Handles migration between different checkpoint formats.*
- `MBLCheckpointManager` (line 1310) - *Manages checkpoint saving with 5-minute intervals and latest file maintenance.*
- `HamiltonianMBLMetricsCollector` (line 1386) - *Collects all MBL metrics for comprehensive training monitoring.
Includes all metrics from the crystallography paper.*
- `HamiltonianTrainer` (line 1540) - *Training system for Hamiltonian Neural Networks with integrated MBL monitoring.*
- `HamiltonianCheckpointAnalyzer` (line 1710) - *Comprehensive analyzer for Hamiltonian NN checkpoints with migration support.*
- `HamiltonianMBLPipeline` (line 1875) - *Main pipeline for processing checkpoints and generating reports.*

**Functions:**
- `main` (line 2031)
- `get_input_dim` (line 55) - *Calculate input dimension from grid size.*
- `get_total_parameters` (line 59) - *Estimate total parameter count.*
- `get_reduced_dimension` (line 135) - *Calculate reduced dimension for analysis.*
- `get_coefficients` (line 162)
- `forward` (line 163)
- `calculate` (line 169)
- `calculate` (line 175)
- `calculate` (line 181)
- `analyze_robustness` (line 187)
- `save_checkpoint` (line 193)
- `load_checkpoint` (line 195)
- `collect` (line 201)
- `__init__` (line 214)
- `migrate_state_dict` (line 218) - *Migra estado de SimpleHamiltonianNet a HamiltonianNeuralNetwork.

SimpleHamiltonianNet tiene:
- input_proj: Conv2d(1, hidden_dim, 1) -> weight [hidden_dim, 1, 1, 1]
- spectral_layers.{i}.kernel_real: [hidden_dim, hidden_dim, grid//2+1, grid]
- output_proj: Conv2d(hidden_dim, 1, 1) -> weight [1, hidden_dim, 1, 1]

HamiltonianNeuralNetwork necesita:
- q_projection, p_projection: Linear(input_dim, hidden_dim)
- spectral_layers.{i}.spectral_weights: [hidden_dim, spectral_modes]
- q_output, p_output: Linear(hidden_dim, input_dim)*
- `_create_default_parameter` (line 320) - *Crea parámetro por defecto.*
- `__init__` (line 352)
- `_initialize_spectral_parameters` (line 366) - *Initialize with physics-informed priors.*
- `forward` (line 373) - *Symplectic Euler integration of Hamilton's equations.
dq/dt = dH/dp, dp/dt = -dH/dq*
- `get_hamiltonian` (line 401) - *Compute Hamiltonian H = T + V in spectral space.*
- `__init__` (line 418)
- `_initialize_weights` (line 438) - *Orthogonal initialization for Hamiltonian structure preservation.*
- `forward` (line 443) - *Forward pass through Hamiltonian dynamics.*
- `time_evolution` (line 459) - *Generate trajectory through time evolution.*
- `get_hamiltonian` (line 474) - *Compute total Hamiltonian.*
- `get_coefficients` (line 486)
- `get_flat_parameters` (line 496) - *Returns all parameters flattened for Hamiltonian construction.*
- `construct_hessian_approximation` (line 503) - *MÉTODO CORREGIDO - No usa 65GB de RAM.*
- `__init__` (line 561)
- `generate_harmonic_oscillator` (line 567) - *Generate harmonic oscillator initial conditions.*
- `generate_double_well` (line 591) - *Generate double-well potential trajectories.*
- `__init__` (line 616)
- `calculate` (line 619) - *Calculate level spacing statistics from model weights.*
- `_construct_hessian_from_weights` (line 651) - *Alternative Hessian construction for generic models.*
- `_compute_eigenvalues` (line 666) - *Compute sorted eigenvalues of the Hamiltonian.*
- `_calculate_spacing_ratios` (line 671) - *Calculate adjacent gap ratios r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}).*
- `_classify_phase` (line 684) - *Classify the quantum phase based on level spacing ratio.*
- `_estimate_brody_parameter` (line 699) - *Estimate Brody parameter for intermediate statistics.
0 = Poisson (integrable), 1 = Wigner-Dyson (chaotic)*
- `__init__` (line 725)
- `calculate` (line 728) - *Calculate participation ratios for all weight layers.*
- `_calculate_ipr` (line 771) - *Calculate standard Inverse Participation Ratio.*
- `_calculate_renyi_ipr` (line 782) - *Calculate q-th order Rényi IPR.*
- `_calculate_fractal_dimension` (line 793) - *Calculate fractal dimension D_q from IPR.*
- `__init__` (line 807)
- `calculate` (line 810) - *Calculate synthetic Planck's constant.*
- `calculate_from_model` (line 820) - *Comprehensive calculation from model and previous analyses.*
- `__init__` (line 849)
- `calculate_base_discretization` (line 853) - *Calculate the base discretization level from weight rounding error.*
- `analyze_robustness` (line 877) - *Test robustness by applying noise and measuring gap collapse.*
- `_perturb_and_measure` (line 921) - *Apply noise to model and measure resulting metrics.*
- `_delta_to_alpha` (line 940) - *Convert discretization error to purity alpha.*
- `__init__` (line 950)
- `calculate` (line 953)
- `_compute_layer_purity` (line 982)
- `_delta_to_alpha` (line 988)
- `_assess_purity_quality` (line 993)
- `__init__` (line 1007)
- `calculate` (line 1010)
- `__init__` (line 1054)
- `calculate` (line 1057) - *Calculate Krylov complexity from model dynamics.*
- `__init__` (line 1095)
- `calculate` (line 1098) - *Calculate crystallinity index from weight spectra.*
- `__init__` (line 1150)
- `measure` (line 1153) - *Comprehensive resilience measurement.*
- `_measure_base_performance` (line 1176) - *Measure baseline performance metrics.*
- `_test_perturbation` (line 1195) - *Test resilience to specific perturbation.*
- `_aggregate_by_dimension` (line 1220) - *Aggregate resilience scores by perturbation dimension.*
- `_aggregate_by_noise` (line 1231) - *Aggregate resilience scores by noise level.*
- `__init__` (line 1246)
- `classify` (line 1249)
- `__init__` (line 1273)
- `migrate` (line 1277)
- `_migrate_if_needed` (line 1290) - *Detecta el formato y aplica migración si es necesario.*
- `__init__` (line 1315)
- `should_save_checkpoint` (line 1322) - *Check if 5 minutes have elapsed since last checkpoint.*
- `save_checkpoint` (line 1328) - *Save checkpoint with all MBL metrics.*
- `load_checkpoint` (line 1361) - *Load checkpoint with automatic device placement and migration.*
- `__init__` (line 1392)
- `collect` (line 1405) - *Collect core metrics for the current training state.*
- `collect_comprehensive` (line 1493) - *Collect comprehensive metrics including expensive calculations.*
- `_classify_quantum_phase` (line 1520) - *Classify combined quantum phase.*
- `__init__` (line 1545)
- `train_step` (line 1571) - *Single training step with Hamiltonian loss.*
- `train_epoch` (line 1603) - *Train for one epoch with MBL monitoring.*
- `_log_metrics` (line 1658) - *Log metrics to console in scientific format.*
- `train` (line 1672) - *Full training loop.*
- `__init__` (line 1713)
- `_load_checkpoint` (line 1723) - *Load and migrate checkpoint.*
- `analyze` (line 1763) - *Perform complete MBL analysis.*
- `_generate_summary` (line 1787) - *Generate executive summary.*
- `_print_report` (line 1807) - *Print formatted analysis report.*
- `__init__` (line 1878)
- `process_checkpoint` (line 1882) - *Process single checkpoint and save results.*
- `process_directory` (line 1901) - *Process multiple checkpoints from directory.*
- `generate_summary` (line 1941) - *Generate aggregate summary report.*
- `_generate_text_report` (line 1982) - *Generate human-readable text report.*

#### `hpu_view.py`
**Path:** `hpu_view.py`

*No symbols extracted*

#### `mining_seeds.py`
**Path:** `mining_seeds.py`

**Classs:**
- `Config` (line 41)
- `IAnalysisStrategy` (line 109)
- `IMetricsCalculator` (line 115)
- `HamiltonianOperator` (line 121) - *True Hamiltonian operator H = -nabla^2 on torus.*
- `FastDataset` (line 146) - *Fast dataset for Hamiltonian operator learning.*
- `SpectralLayer` (line 203) - *Spectral layer with correct complex multiplication.*
- `SimpleHamiltonianNet` (line 255) - *Compact network for Hamiltonian operator learning.*
- `LocalComplexityAnalyzer` (line 293)
- `SuperpositionAnalyzer` (line 310)
- `CrystallographyMetrics` (line 340)
- `ThermodynamicMetrics` (line 444)
- `SpectroscopyMetrics` (line 470)
- `CheckpointManager` (line 500)
- `TrainingMonitor` (line 573)
- `GlassStopper` (line 611)
- `BoltzmannAnalysisProgram` (line 879)

**Functions:**
- `set_seed` (line 88)
- `setup_logger` (line 96)
- `train_with_early_glass_stop` (line 670) - *Train model with early stopping for glass detection.*
- `seed_miner` (line 803) - *Mine for crystal seeds by trying sequential seeds.*
- `main` (line 856)
- `analyze` (line 111)
- `compute` (line 117)
- `__init__` (line 124)
- `_precompute_spectral_operators` (line 128)
- `apply` (line 135)
- `time_evolution` (line 140)
- `__init__` (line 149)
- `__len__` (line 193)
- `__getitem__` (line 196)
- `get_val_batch` (line 199)
- `__init__` (line 206)
- `forward` (line 219)
- `__init__` (line 258)
- `forward` (line 279)
- `compute_local_complexity` (line 295) - *Compute Local Complexity (LC) metric for weight matrix.*
- `compute_superposition` (line 312) - *Compute Superposition (SP) metric for weight matrix.*
- `compute_kappa` (line 342)
- `compute_discretization_margin` (line 378)
- `compute_alpha_purity` (line 387)
- `compute_kappa_quantum` (line 394)
- `compute_poynting_vector` (line 411)
- `compute_all_metrics` (line 426)
- `compute_effective_temperature` (line 446)
- `compute_specific_heat` (line 460)
- `compute_weight_diffraction` (line 472)
- `_compute_spectral_entropy` (line 491)
- `__init__` (line 501)
- `should_save_checkpoint` (line 509)
- `save_checkpoint` (line 514)
- `__init__` (line 574)
- `update_metrics` (line 594)
- `__init__` (line 612)
- `should_stop` (line 616) - *Check if the system is in glass state and should stop mining.*
- `__init__` (line 880)
- `load_and_analyze_checkpoint` (line 886)
- `dataloader` (line 903)

#### `plank.py`
**Path:** `plank.py`

**Classs:**
- `HBarCalculator` (line 26) - *Calcula ħ efectiva desde checkpoint HPU usando física realista.*

**Functions:**
- `main` (line 213)
- `__init__` (line 29)
- `calculate_all` (line 54) - *Ejecuta todos los cálculos de ħ.*
- `print_report` (line 170) - *Imprime reporte formateado.*

#### `polos.py`
**Path:** `polos.py`

**Classs:**
- `ControlConfig` (line 27)
- `TransferFunctionExtractor` (line 48)
- `PoleZeroAnalyzer` (line 142)
- `FrequencyResponseAnalyzer` (line 299)
- `TimeResponseAnalyzer` (line 415)
- `ControllerDesigner` (line 516)
- `ControlSystemAnalyzer` (line 599)
- `ControlVisualizer` (line 798)

**Functions:**
- `analyze_checkpoint` (line 1152)
- `analyze_multiple_checkpoints` (line 1208)
- `main` (line 1240)
- `__init__` (line 50)
- `extract_state_space_representation` (line 55)
- `compute_transfer_function` (line 105)
- `__init__` (line 144)
- `_compute_poles_zeros` (line 153)
- `analyze_stability` (line 166)
- `classify_poles` (line 207)
- `compute_damping_frequency` (line 238)
- `compute_time_constants` (line 278)
- `__init__` (line 301)
- `compute_bode_plot_data` (line 309)
- `compute_gain_phase_margins` (line 328)
- `compute_nyquist_data` (line 357)
- `evaluate_nyquist_stability` (line 379)
- `__init__` (line 417)
- `compute_step_response` (line 425)
- `compute_impulse_response` (line 441)
- `analyze_step_response_characteristics` (line 457)
- `__init__` (line 518)
- `design_pid_controller` (line 522)
- `design_lead_compensator` (line 542)
- `compute_root_locus` (line 572)
- `__init__` (line 601)
- `analyze_complete_system` (line 625)
- `_print_report` (line 715)
- `plot_pole_zero_map` (line 801)
- `plot_bode_diagram` (line 865)
- `plot_nyquist_diagram` (line 940)
- `plot_time_responses` (line 990)
- `plot_root_locus` (line 1036)
- `plot_combined_analysis` (line 1096)

#### `precision.py`
**Path:** `precision.py`

**Classs:**
- `MassiveLambdaConfig` (line 26)
- `CrystallizationLossMassive` (line 36)
- `ContinuationEngine` (line 72)

**Functions:**
- `main` (line 506)
- `__init__` (line 37)
- `quantization_penalty` (line 42)
- `forward` (line 54)
- `__init__` (line 73)
- `_setup_logger` (line 150)
- `_find_latest_checkpoint` (line 162)
- `_compute_initial_metrics` (line 191)
- `compute_discretization_metrics` (line 208)
- `validate` (line 241)
- `train_epoch` (line 250)
- `refine` (line 288)
- `_save_latest_checkpoint` (line 430) - *Guarda/sobrescribe latest.pth - rápido, para danger zone*
- `_save_crystal_checkpoint` (line 456)
- `_compile_results` (line 490)

#### `refinamiento.py`
**Path:** `refinamiento.py`

**Classs:**
- `CrystallizationConfig` (line 33) - *Configuración agresiva para forzar discretización*
- `CrystallizationLoss` (line 57) - *Pérdida combinada: MSE + penalización de cuantización
Fuerza los pesos a caer en {-1, 0, 1}*
- `StructuralPruner` (line 96) - *Implementa poda progresiva de pesos pequeños*
- `CrystallizationEngine` (line 144) - *Motor de refinamiento que carga un checkpoint y fuerza discretización*

**Functions:**
- `analyze_discretization` (line 498) - *Análisis detallado de la discretización de un checkpoint*
- `main` (line 575)
- `__init__` (line 62)
- `quantization_penalty` (line 67) - *Penalización L2 de la distancia al entero más cercano*
- `forward` (line 81)
- `__init__` (line 98)
- `should_prune` (line 103) - *Determina si es momento de podar (cada 500 épocas)*
- `prune` (line 107) - *Poda pesos con |w| < threshold
Retorna número de parámetros podados*
- `get_sparsity` (line 131) - *Calcula porcentaje de pesos exactamente en cero*
- `__init__` (line 148)
- `_setup_logger` (line 186)
- `_load_checkpoint` (line 198) - *Carga el checkpoint y retorna modelo, época y métricas*
- `_compute_initial_metrics` (line 234) - *Calcula métricas iniciales si no vienen en el checkpoint*
- `compute_discretization_metrics` (line 253) - *Calcula métricas de cristalinidad actuales*
- `validate` (line 291) - *Valida el modelo manteniendo accuracy*
- `train_epoch` (line 302) - *Entrena una época con pérdida de cuantización*
- `refine` (line 344) - *Ejecuta el refinamiento hasta alcanzar δ < TARGET_DELTA o MAX_EPOCHS*
- `_save_crystal_checkpoint` (line 459) - *Guarda checkpoint cristalino*
- `_compile_results` (line 482) - *Compila resultados finales*

#### `simple_hpu_view.py`
**Path:** `simple_hpu_view.py`

*No symbols extracted*

#### `test_grokkit.py`
**Path:** `test_grokkit.py`

**Classs:**
- `GrokkingValidator` (line 41) - *Validates grokking phenomenon in Hamiltonian operator learning.

Implements Theorem 1.1 requirements:
1. Spectral convergence to true H operator
2. Operator kernel representation in weights
3. Phase transition from memorization to generalization

This class implements a battery of tests to confirm that the trained
model has successfully transitioned from the memorization phase to
the generalization phase, exhibiting the characteristic properties
of spectral convergence as predicted by Theorem 1.1.*

**Functions:**
- `run_quick_test` (line 507) - *Executes a quick validation test with minimal output.

This function provides a streamlined testing interface for
rapid verification of model performance.*
- `__init__` (line 56)
- `load_model` (line 64) - *Loads the trained model from checkpoint.

Returns:
    Tuple of (model, checkpoint)
    
Raises:
    FileNotFoundError: If no checkpoint exists in weights directory.*
- `generate_test_dataset` (line 119) - *Generates test dataset using the true Hamiltonian operator.

Creates random initial fields and evolves them under H to produce
ground truth targets for validation.

Args:
    num_samples: Number of test samples
    
Returns:
    Tuple of (inputs, targets) tensors*
- `compute_local_complexity` (line 158) - *Computes Local Complexity (LC) metric for the model.

LC measures the effective dimensionality of the model's
learned representations. High LC indicates diverse, independent
feature utilization - a key indicator of operator learning.

Args:
    model: Neural network model
    
Returns:
    LC value in [0, 1] range*
- `compute_superposition` (line 181) - *Computes Superposition (SP) metric for the model.

SP measures the correlation between weight vectors.
Low SP indicates orthogonal, non-redundant representations.

Args:
    model: Neural network model
    
Returns:
    SP value in [0, 1] range*
- `compute_operator_error` (line 203) - *Computes operator approximation error.

Measures how well the learned model approximates the true
Hamiltonian operator on held-out test data.

Args:
    model: Trained model
    inputs: Test input fields
    targets: True evolved fields under H
    
Returns:
    Mean squared error between prediction and target*
- `compute_spectral_gap` (line 228) - *Estimates the spectral gap in weight singular values.

The spectral gap provides insight into the model's capacity
utilization and the degree of weight superposition.

Args:
    model: Neural network model
    
Returns:
    Ratio of largest to smallest non-zero singular value*
- `run_validation` (line 259) - *Executes the complete validation suite.

Runs all tests and aggregates results into a comprehensive
report documenting the grokking phenomenon characteristics
as predicted by Theorem 1.1.

Returns:
    Dictionary containing all test results and metrics*
- `generate_report` (line 382) - *Generates a formal validation report.

Returns:
    Markdown formatted validation report*

#### `verify.py`
**Path:** `verify.py`

**Classs:**
- `CheckpointVerifier` (line 14)

**Functions:**
- `verify_latest_checkpoints` (line 444) - *Verifica los N checkpoints más recientes*
- `main` (line 486)
- `__init__` (line 15)
- `verify_all_metrics` (line 50) - *Calcula TODAS las métricas desde cero y compara con las guardadas*
- `_check_weight_integrity` (line 101) - *Verifica que los pesos no tengan NaN/Inf*
- `_compute_validation_metrics` (line 146) - *Calcula MSE y accuracy de validación desde cero*
- `_compute_discretization_metrics` (line 173) - *Calcula delta, alpha, purity, etc.*
- `_compute_quantization_metrics` (line 225) - *Calcula la penalización de cuantización*
- `_compute_loss_metrics` (line 244) - *Reconstruye el loss total*
- `_compare_with_stored` (line 269) - *Compara métricas calculadas vs almacenadas en el checkpoint*
- `_check_internal_consistency` (line 302) - *Verifica consistencia entre métricas relacionadas*
- `_compute_health_score` (line 331) - *Calcula un score de salud del checkpoint (0-100)*
- `_print_report` (line 374) - *Imprime reporte formateado*

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
