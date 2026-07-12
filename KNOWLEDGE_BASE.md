# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis. See more [here](https://github.com/grisuno/ReadMenator)

**Total Files Parsed:** 32 | **Total Symbols Extracted:** 776 | **Total Imports:** 345

## Structural Knowledge Map
```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray:5 5,color:#aaa;
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

**Classes:**
- `SimpleConfig` (line 29) `class SimpleConfig`
- `HamiltonianOperator` (line 88) `class HamiltonianOperator` - *True Hamiltonian operator H = -nabla^2 on torus.*
- `FastDataset` (line 113) `class FastDataset(Dataset)` - *Fast dataset for Hamiltonian operator learning.*
- `SpectralLayer` (line 170) `class SpectralLayer` - *Spectral layer with correct complex multiplication.*
- `SimpleHamiltonianNet` (line 222) `class SimpleHamiltonianNet` - *Compact network for Hamiltonian operator learning.*

**Functions:**
- `compute_local_complexity` (line 45) `def compute_local_complexity(weights, epsilon)` - *Compute Local Complexity (LC) metric for weight matrix.*
- `compute_superposition` (line 60) `def compute_superposition(weights)` - *Compute Superposition (SP) metric for weight matrix.*
- `train_model` (line 260) `def train_model(grid_size, epochs, hidden_dim, num_spectral_layers, lr)` - *Train the Hamiltonian operator model.*
- `main` (line 369) `def main()`
- `__init__` (line 30) `def __init__(self, grid_size, hidden_dim, num_spectral_layers, target_accuracy, learning_rate)`
- `__init__` (line 91) `def __init__(self, grid_size)`
- `_precompute_spectral_operators` (line 95) `def _precompute_spectral_operators(self)`
- `apply` (line 102) `def apply(self, field)`
- `time_evolution` (line 107) `def time_evolution(self, field, dt)`
- `__init__` (line 116) `def __init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)`
- `__len__` (line 160) `def __len__(self)`
- `__getitem__` (line 163) `def __getitem__(self, idx)`
- `get_val_batch` (line 166) `def get_val_batch(self)`
- `__init__` (line 173) `def __init__(self, channels, grid_size)`
- `forward` (line 186) `def forward(self, x)`
- `__init__` (line 225) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `forward` (line 246) `def forward(self, x)`

#### `audio_io.py`
**Path:** `audio/audio_io.py`

**Classes:**
- `AudioProcessor` (line 31) `class AudioProcessor` - *Audio processing pipeline centered on the complex STFT domain.

The complex STFT is the audio equivalent of a grayscale image:
a 2D field where one axis is time, the other is frequency, and
each point carries a complex value (magnitude + phase). This is
the correct domain for applying the Hamiltonian spectral evolution.*

**Functions:**
- `__init__` (line 41) `def __init__(self, config, device)`
- `load_audio` (line 57) `def load_audio(self, file_path)` - *Load an audio file and convert to mono at the target sample rate.

Args:
    file_path: Path to the audio file.

Returns:
    Tuple of (waveform tensor [1, T], sample_rate).*
- `waveform_to_stft_complex` (line 80) `def waveform_to_stft_complex(self, waveform)` - *Compute the complex STFT of a waveform.

This is the primary transform: converts 1D audio into a 2D complex
field (freq_bins x time_frames) that the Hamiltonian network
can process identically to how it processes images.

Args:
    waveform: Audio waveform [1, T] or [T].

Returns:
    Complex STFT tensor [freq_bins, time_frames].*
- `stft_complex_to_waveform` (line 105) `def stft_complex_to_waveform(self, stft_complex)` - *Reconstruct waveform from complex STFT via inverse STFT.

Unlike Griffin-Lim (which estimates phase), ISTFT uses the
EXACT phase from the complex STFT, producing a faithful
reconstruction when the magnitude/phase have been coherently
modified by the Hamiltonian evolution.

Args:
    stft_complex: Complex STFT tensor [freq_bins, time_frames].

Returns:
    Reconstructed waveform [1, T].*
- `stft_to_magnitude_phase` (line 130) `def stft_to_magnitude_phase(self, stft_complex)` - *Decompose complex STFT into magnitude and phase.

Args:
    stft_complex: Complex STFT [freq_bins, time_frames].

Returns:
    Tuple of (magnitude, phase) each [freq_bins, time_frames].*
- `magnitude_phase_to_stft` (line 146) `def magnitude_phase_to_stft(self, magnitude, phase)` - *Recombine magnitude and phase into complex STFT.

Args:
    magnitude: Magnitude spectrum [freq_bins, time_frames].
    phase: Phase spectrum [freq_bins, time_frames].

Returns:
    Complex STFT [freq_bins, time_frames].*
- `stft_magnitude_to_model_input` (line 161) `def stft_magnitude_to_model_input(self, magnitude)` - *Prepare STFT magnitude for input to the Hamiltonian network.

Normalizes magnitude to [0, 1] range and shapes as [1, 1, H, W],
matching the expected input format (analogous to a grayscale image).

Args:
    magnitude: STFT magnitude [freq_bins, time_frames].

Returns:
    Normalized tensor [1, 1, freq_bins, time_frames].*
- `model_output_to_stft_magnitude` (line 186) `def model_output_to_stft_magnitude(self, model_output, original_magnitude)` - *Convert model output (energy mask in [0, 1]) back to STFT magnitude scale.

The model output represents the Hamiltonian energy structure --
which regions of the time-frequency plane carry coherent energy.
This is used to modulate the original magnitude.

Args:
    model_output: Energy mask [1, 1, freq_bins, time_frames] in [0, 1].
    original_magnitude: Original STFT magnitude [freq_bins, time_frames].

Returns:
    Reconstructed magnitude [freq_bins, time_frames].*
- `waveform_to_mel_spectrogram` (line 206) `def waveform_to_mel_spectrogram(self, waveform)` - *Convert waveform to normalized mel spectrogram (for visualization only).

Args:
    waveform: Audio waveform tensor [1, T] or [B, 1, T].

Returns:
    Normalized mel spectrogram [B, 1, n_mels, time_frames].*
- `save_audio` (line 229) `def save_audio(self, waveform, file_path, sample_rate)` - *Save a waveform tensor to an audio file.

Args:
    waveform: Audio tensor [1, T] or [T].
    file_path: Output file path.
    sample_rate: Sample rate (defaults to config sample rate).*
- `get_spectrogram_db_range` (line 249) `def get_spectrogram_db_range(self, waveform)` - *Compute the dB range of a waveform's mel spectrogram.

Args:
    waveform: Audio waveform [1, T].

Returns:
    Tuple of (db_min, db_max).*

#### `audios.py`
**Path:** `audio/audios.py`

**Classes:**
- `HamiltonianConfig` (line 48) `class HamiltonianConfig` - *Immutable configuration container for all hyperparameters.
Eliminates magic numbers and provides single point of control.*
- `IAudioSource` (line 103) `class IAudioSource(ABC)` - *Interface for audio input sources.*
- `IFieldOperator` (line 122) `class IFieldOperator(ABC)` - *Interface for Hamiltonian field evolution operators.*
- `IMetricCollector` (line 131) `class IMetricCollector(ABC)` - *Interface for training metrics collection.*
- `AudioResampler` (line 149) `class AudioResampler` - *Handles audio resampling using scipy.signal, avoiding librosa/numba dependencies.*
- `WaveFileSource` (line 204) `class WaveFileSource(IAudioSource)` - *Concrete implementation of audio source from file.
Supports automatic resampling to target sample rate using scipy.*
- `ComprehensiveMetricCollector` (line 278) `class ComprehensiveMetricCollector(IMetricCollector)` - *Collects all metrics from Hamiltonian paper, activation functions,
and architectural diagnostics for informed decision-making.*
- `CheckpointManager` (line 326) `class CheckpointManager` - *Manages periodic checkpointing with atomic writes.*
- `AudioSpectrogramConverter` (line 387) `class AudioSpectrogramConverter` - *Converts between audio waveforms and 2D field representations.
Adaptado para la arquitectura de experiment2 (grid_size=16).*
- `HamiltonianAudioProcessor` (line 468) `class HamiltonianAudioProcessor` - *Main orchestrator for Hamiltonian audio processing.
Demonstrates that auditory perception is epiphenomenon of Hamiltonian dynamics.
Usa la arquitectura exacta de experiment2.*

**Functions:**
- `main` (line 742) `def main()` - *Entry point with argument parsing.*
- `segment_samples` (line 89) `def segment_samples(self)` - *Calculate segment length in samples.*
- `freq_bins` (line 94) `def freq_bins(self)` - *Calculate frequency bins for real FFT.*
- `read_segment` (line 107) `def read_segment(self)` - *Read audio segment. Returns None when exhausted.*
- `get_properties` (line 112) `def get_properties(self)` - *Return audio properties.*
- `close` (line 117) `def close(self)` - *Release resources.*
- `evolve` (line 126) `def evolve(self, field_state)` - *Evolve field state through Hamiltonian dynamics.*
- `record` (line 135) `def record(self, metrics)` - *Record metric values.*
- `get_summary` (line 140) `def get_summary(self)` - *Return aggregated metrics.*
- `resample` (line 155) `def resample(audio, orig_sr, target_sr)` - *Resample audio from orig_sr to target_sr using polyphase filtering.*
- `load_wav_with_resample` (line 173) `def load_wav_with_resample(file_path, target_sr)` - *Load WAV file and resample to target sample rate.
Returns (audio_data, original_sample_rate).*
- `__init__` (line 210) `def __init__(self, file_path, config)`
- `_validate_and_load` (line 220) `def _validate_and_load(self)` - *Validate file format and load with automatic resampling.*
- `read_segment` (line 244) `def read_segment(self)` - *Read next audio segment.*
- `get_properties` (line 261) `def get_properties(self)` - *Return audio file properties.*
- `close` (line 273) `def close(self)` - *Release resources.*
- `__init__` (line 284) `def __init__(self, config)`
- `record` (line 289) `def record(self, metrics)` - *Record comprehensive metrics.*
- `get_summary` (line 298) `def get_summary(self)` - *Return statistical summary of all metrics.*
- `export_to_json` (line 320) `def export_to_json(self, path)` - *Export full history to JSON.*
- `__init__` (line 331) `def __init__(self, model, config, checkpoint_dir)`
- `check_and_save` (line 344) `def check_and_save(self, force)` - *Check if checkpoint interval elapsed and save if necessary.
Returns path if saved, None otherwise.*
- `_save_checkpoint` (line 357) `def _save_checkpoint(self)` - *Atomic checkpoint save.*
- `__init__` (line 393) `def __init__(self, config)`
- `waveform_to_field` (line 396) `def waveform_to_field(self, waveform)` - *Convert 1D audio to 2D field representation via STFT.
Returns (1, 1, grid_size, grid_size) tensor compatible con experiment2.*
- `field_to_waveform` (line 432) `def field_to_waveform(self, field, original_length)` - *Reconstruct waveform from 2D field representation.*
- `_forward_spectrogram` (line 458) `def _forward_spectrogram(self, x)` - *Compute magnitude spectrogram.*
- `_inverse_spectrogram` (line 463) `def _inverse_spectrogram(self, spectrogram)` - *Griffin-Lim inverse.*
- `__init__` (line 475) `def __init__(self, config, model, source)`
- `load_model_weights` (line 504) `def load_model_weights(self, path)` - *Load pretrained Hamiltonian operator desde safetensors.*
- `attach_source` (line 513) `def attach_source(self, source)` - *Attach audio source via dependency injection.*
- `process_stream` (line 517) `def process_stream(self)` - *Process audio stream through Hamiltonian perception.
Generates three epiphenomenal representations:
1. Energy Density (Resonance)
2. Topological Phase (Vortices)
3. Action Map (Perceptual Clarity)*
- `_process_single_segment` (line 592) `def _process_single_segment(self, waveform, index)` - *Process single audio segment and return metrics.*
- `_calculate_phase_entropy` (line 684) `def _calculate_phase_entropy(self, phase_map)` - *Calculate topological entropy from phase distribution.*
- `_render_epiphenomena` (line 692) `def _render_epiphenomena(self, amplitude, phase, action)` - *Render three epiphenomenal visualizations.*
- `export_metrics` (line 729) `def export_metrics(self, path)` - *Export comprehensive metrics to file.*
- `force_checkpoint` (line 733) `def force_checkpoint(self)` - *Force immediate checkpoint save.*

#### `checkpoint_manager.py`
**Path:** `audio/checkpoint_manager.py`

**Classes:**
- `CheckpointManager` (line 28) `class CheckpointManager` - *Manages model checkpointing with time-based intervals
and best-model tracking.*

**Functions:**
- `__init__` (line 34) `def __init__(self, config)`
- `should_save_checkpoint` (line 41) `def should_save_checkpoint(self)` - *Check if enough time has elapsed since the last checkpoint.*
- `save_checkpoint` (line 46) `def save_checkpoint(self, model, optimizer, scheduler, epoch, step, metrics, current_loss)` - *Save the current model state and training metadata.

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
- `load_checkpoint` (line 98) `def load_checkpoint(self, model, load_best)` - *Load a model checkpoint and return training metadata.

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
- `best_loss` (line 156) `def best_loss(self)`

#### `config.py`
**Path:** `audio/config.py`

**Classes:**
- `AudioProcessingConfig` (line 18) `class AudioProcessingConfig` - *Parameters governing raw audio ingestion and spectrogram computation.*
- `ModelArchitectureConfig` (line 34) `class ModelArchitectureConfig` - *Parametric architecture dimensions for the Hamiltonian Neural Network.

All hidden dimensions, matrix sizes, expansion factors, and layer counts
are configurable from this single source of truth.*
- `TrainingConfig` (line 80) `class TrainingConfig` - *All training loop hyperparameters and scheduling constants.*
- `CheckpointConfig` (line 109) `class CheckpointConfig` - *Checkpoint persistence parameters.*
- `VisualizationConfig` (line 136) `class VisualizationConfig` - *Parameters for audio reconstruction visualization and output.*
- `MetricsConfig` (line 158) `class MetricsConfig` - *Configuration for all tracked metrics during training and inference.*
- `HamiltonianAudioConfig` (line 182) `class HamiltonianAudioConfig` - *Top-level configuration aggregator.

Composes all sub-configurations into a single injectable dependency,
following the Dependency Inversion Principle.*

**Functions:**
- `validate` (line 62) `def validate(self)` - *Ensure architectural coherence.*
- `checkpoint_path` (line 121) `def checkpoint_path(self)`
- `best_model_path` (line 127) `def best_model_path(self)`
- `metadata_path` (line 131) `def metadata_path(self)`
- `validate_all` (line 198) `def validate_all(self)` - *Run validation on all sub-configurations.*
- `ensure_directories` (line 206) `def ensure_directories(self)` - *Create required output directories if they do not exist.*

#### `experiment2.py`
**Path:** `audio/experiment2.py`

**Classes:**
- `Config` (line 22) `class Config`
- `SeedManager` (line 74) `class SeedManager`
- `LoggerFactory` (line 84) `class LoggerFactory`
- `IAnalysisStrategy` (line 99) `class IAnalysisStrategy(ABC)`
- `IMetricsCalculator` (line 105) `class IMetricsCalculator(ABC)`
- `HamiltonianOperator` (line 111) `class HamiltonianOperator`
- `HamiltonianDataset` (line 133) `class HamiltonianDataset(Dataset)`
- `SpectralLayer` (line 183) `class SpectralLayer`
- `HamiltonianNeuralNetwork` (line 227) `class HamiltonianNeuralNetwork`
- `LocalComplexityAnalyzer` (line 257) `class LocalComplexityAnalyzer`
- `SuperpositionAnalyzer` (line 273) `class SuperpositionAnalyzer`
- `CrystallographyMetricsCalculator` (line 302) `class CrystallographyMetricsCalculator(IMetricsCalculator)`
- `ThermodynamicMetricsCalculator` (line 737) `class ThermodynamicMetricsCalculator(IMetricsCalculator)`
- `SpectroscopyMetricsCalculator` (line 770) `class SpectroscopyMetricsCalculator(IMetricsCalculator)`
- `CheckpointManager` (line 804) `class CheckpointManager`
- `TrainingMetricsMonitor` (line 874) `class TrainingMetricsMonitor`
- `GlassStateDetector` (line 912) `class GlassStateDetector`
- `TrainingEngine` (line 973) `class TrainingEngine`
- `SeedMiningSystem` (line 1113) `class SeedMiningSystem`
- `SingleExperimentRunner` (line 1164) `class SingleExperimentRunner`
- `CheckpointAnalyzer` (line 1223) `class CheckpointAnalyzer`
- `Application` (line 1275) `class Application`

**Functions:**
- `main` (line 1334) `def main()`
- `set_seed` (line 76) `def set_seed(seed)`
- `create_logger` (line 86) `def create_logger(name, level)`
- `analyze` (line 101) `def analyze(self, model)`
- `compute` (line 107) `def compute(self, model)`
- `__init__` (line 112) `def __init__(self, grid_size)`
- `_precompute_spectral_operators` (line 116) `def _precompute_spectral_operators(self)`
- `apply` (line 122) `def apply(self, field)`
- `time_evolution` (line 127) `def time_evolution(self, field, dt)`
- `__init__` (line 134) `def __init__(self, num_samples, grid_size, time_steps, dt, train_ratio)`
- `__len__` (line 173) `def __len__(self)`
- `__getitem__` (line 176) `def __getitem__(self, idx)`
- `get_validation_batch` (line 179) `def get_validation_batch(self)`
- `__init__` (line 184) `def __init__(self, channels, grid_size)`
- `forward` (line 195) `def forward(self, x)`
- `__init__` (line 228) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `forward` (line 243) `def forward(self, x)`
- `compute_local_complexity` (line 259) `def compute_local_complexity(weights, epsilon)`
- `compute_superposition` (line 275) `def compute_superposition(weights)`
- `compute` (line 303) `def compute(self, model, val_x, val_y)` - *Implementación de interfaz IMetricsCalculator.
Delega a compute_all_metrics con los argumentos correctos.*
- `compute_gradient_covariance_kappa` (line 311) `def compute_gradient_covariance_kappa(model, dataloader, num_batches)`
- `compute_discretization_margin_from_state_dict` (line 348) `def compute_discretization_margin_from_state_dict(model)` - *Calcula el margen de discretización desde los parámetros del modelo.
Versión estática que no requiere diccionario externo.*
- `compute_discretization_margin` (line 361) `def compute_discretization_margin(coeffs)` - *Calcula el margen de discretización desde un diccionario de coeficientes.*
- `compute_alpha_purity_from_model` (line 373) `def compute_alpha_purity_from_model(model)` - *Calcula el índice de pureza alpha directamente desde el modelo.*
- `compute_alpha_purity` (line 383) `def compute_alpha_purity(coeffs)` - *Calcula el índice de pureza alpha desde un diccionario de coeficientes.*
- `compute_kappa` (line 393) `def compute_kappa(model, val_x, val_y, num_batches)` - *Número de condición de la matriz de covarianza de gradientes.*
- `compute_kappa_quantum` (line 464) `def compute_kappa_quantum(model, hbar)` - *Versión del cálculo cuántico de kappa que opera directamente sobre el modelo.*
- `compute_kappa_quantum_from_coeffs` (line 492) `def compute_kappa_quantum_from_coeffs(coeffs, hbar)` - *Versión del cálculo cuántico de kappa desde diccionario de coeficientes.*
- `_compute_crystallography_metrics` (line 511) `def _compute_crystallography_metrics(self, model, val_x, val_y)` - *Métricas cristalográficas con aislamiento completo de errores.*
- `_check_weight_integrity` (line 539) `def _check_weight_integrity(self, model)` - *Verifica integridad de pesos: NaN, Inf, y estadísticas básicas.*
- `compute_poynting_vector` (line 603) `def compute_poynting_vector(model)` - *Vector de Poynting: flujo de energía en el espacio de parámetros.
Análogo electromagnético para redes neuronales.*
- `compute_all_metrics` (line 679) `def compute_all_metrics(model, val_x, val_y)` - *Calcula todas las métricas cristalográficas con manejo de errores.*
- `compute` (line 738) `def compute(self, model, gradient_buffer, learning_rate, loss_history, temp_history)`
- `compute_effective_temperature` (line 747) `def compute_effective_temperature(gradient_buffer, learning_rate)`
- `compute_specific_heat` (line 760) `def compute_specific_heat(loss_history, temp_history, cv_threshold)`
- `compute` (line 771) `def compute(self, model)`
- `compute_weight_diffraction` (line 776) `def compute_weight_diffraction(coeffs)`
- `_compute_spectral_entropy` (line 795) `def _compute_spectral_entropy(power_spectrum)`
- `__init__` (line 805) `def __init__(self, interval_minutes, max_checkpoints)`
- `should_save_checkpoint` (line 813) `def should_save_checkpoint(self)`
- `save_checkpoint` (line 818) `def save_checkpoint(self, model, optimizer, epoch, metrics)`
- `__init__` (line 875) `def __init__(self)`
- `update_metrics` (line 895) `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat, poynting_magnitude)`
- `__init__` (line 913) `def __init__(self, patience_epochs)`
- `should_stop` (line 918) `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)`
- `is_crystal_formed` (line 963) `def is_crystal_formed(self, lc, sp, kappa, delta, temp, cv)`
- `__init__` (line 974) `def __init__(self, model, optimizer, device, logger)`
- `train_epoch` (line 997) `def train_epoch(self, dataloader, epoch)`
- `validate` (line 1027) `def validate(self, val_x, val_y)`
- `compute_weight_metrics` (line 1040) `def compute_weight_metrics(self)`
- `execute_training` (line 1056) `def execute_training(self, dataloader, val_x, val_y, epochs, seed, early_stopping)`
- `__init__` (line 1114) `def __init__(self, max_attempts)`
- `mine` (line 1118) `def mine(self)`
- `__init__` (line 1165) `def __init__(self, seed, epochs, grid_size, hidden_dim, num_spectral_layers, learning_rate)`
- `run` (line 1175) `def run(self)`
- `__init__` (line 1224) `def __init__(self, checkpoint_path, results_dir)`
- `analyze` (line 1230) `def analyze(self)`
- `__init__` (line 1276) `def __init__(self)`
- `_create_argument_parser` (line 1280) `def _create_argument_parser(self)`
- `run` (line 1294) `def run(self)`
- `safe_compute` (line 694) `def safe_compute(func)`

#### `inference.py`
**Path:** `audio/inference.py`

**Classes:**
- `HamiltonianAudioInference` (line 37) `class HamiltonianAudioInference` - *Performs complete Hamiltonian audio analysis on a given audio file.*

**Functions:**
- `__init__` (line 42) `def __init__(self, config, load_best)`
- `analyze_audio` (line 73) `def analyze_audio(self, audio_file_path, output_prefix)` - *Perform complete Hamiltonian analysis on an audio file.

Args:
    audio_file_path: Path to the audio file to analyze.
    output_prefix: Optional prefix for output filenames.*
- `_compute_energy_mask_patched` (line 158) `def _compute_energy_mask_patched(self, model_input)` - *Compute energy mask over the full STFT magnitude, processing
in patches along the time axis if the input exceeds patch width.

Args:
    model_input: Normalized STFT magnitude [1, 1, freq_bins, time_frames].

Returns:
    Energy mask [1, 1, freq_bins, time_frames] in [0, 1].*
- `_extract_hamiltonian_fields_patched` (line 209) `def _extract_hamiltonian_fields_patched(self, model_input)` - *Extract Hamiltonian fields over full STFT magnitude with patching.

Args:
    model_input: Normalized STFT magnitude [1, 1, freq_bins, time_frames].

Returns:
    Tuple of (amplitude_map, phase_map, action_map).*
- `_compute_inference_metrics` (line 270) `def _compute_inference_metrics(self, original_magnitude, reconstructed_magnitude, original_stft)` - *Compute all inference-time metrics on the STFT domain.*
- `_print_inference_metrics` (line 288) `def _print_inference_metrics(self)` - *Print all computed inference metrics.*

#### `losses.py`
**Path:** `audio/losses.py`

**Classes:**
- `HamiltonianLossComputer` (line 28) `class HamiltonianLossComputer` - *Computes the composite Hamiltonian loss function with all
physics-based regularization terms.

Each loss component is independently weighted via TrainingConfig,
enabling fine-grained control over the training objective.*

**Functions:**
- `__init__` (line 37) `def __init__(self, config)`
- `compute_total_loss` (line 41) `def compute_total_loss(self, prediction, target, intermediates, model)` - *Compute the complete weighted loss with all Hamiltonian terms.

Args:
    prediction: Model output [B, 1, H, W].
    target: Ground truth [B, 1, H, W].
    intermediates: List of intermediate hidden states from forward pass.
    model: The model (for parameter access in regularization).

Returns:
    Tuple of (total_loss tensor, dict of individual loss values).*
- `_compute_reconstruction_loss` (line 94) `def _compute_reconstruction_loss(self, prediction, target)` - *MSE reconstruction loss between predicted and target spectrograms.*
- `_compute_energy_conservation_loss` (line 100) `def _compute_energy_conservation_loss(self, intermediates)` - *Penalize energy drift across layers.

The Hamiltonian energy E = 0.5 * ||phi||^2 should remain
approximately constant through the evolution layers.*
- `_compute_symplectic_loss` (line 122) `def _compute_symplectic_loss(self, intermediates)` - *Penalize violation of symplectic structure.

For pairs of consecutive states (q_i, q_{i+1}), we interpret
q as position and dq = q_{i+1} - q_i as a proxy for momentum.
The symplectic form dq ^ dp should be preserved.*
- `_compute_spectral_consistency_loss` (line 145) `def _compute_spectral_consistency_loss(self, prediction, target)` - *Penalize spectral divergence in frequency domain.

||FFT(prediction) - FFT(target)||_F / ||FFT(target)||_F*
- `_compute_phase_coherence_loss` (line 161) `def _compute_phase_coherence_loss(self, prediction, target)` - *Penalize phase misalignment between prediction and target.

1 - |mean(exp(i * (angle(FFT(pred)) - angle(FFT(target)))))|*
- `_compute_action_minimization_loss` (line 177) `def _compute_action_minimization_loss(self, intermediates)` - *Principle of least action: minimize the total action
S = sum(|phi_{i+1} - phi_i|) along the trajectory.*
- `_compute_liouville_loss` (line 192) `def _compute_liouville_loss(self, intermediates)` - *Liouville theorem: phase space volume should be preserved.

We approximate this by checking that the variance of hidden
states remains approximately constant through evolution.*
- `_compute_hamiltonian_constraint_loss` (line 214) `def _compute_hamiltonian_constraint_loss(self, intermediates)` - *Hamilton's equations: dq/dt = dH/dp, dp/dt = -dH/dq.

Approximated by checking time-reversal symmetry:
the forward evolution followed by reverse should return
to the initial state.*

#### `main.py`
**Path:** `audio/main.py`

**Functions:**
- `build_argument_parser` (line 34) `def build_argument_parser()` - *Construct the complete argument parser with all configurable parameters.*
- `build_config_from_args` (line 103) `def build_config_from_args(args)` - *Construct the full configuration from parsed CLI arguments.*
- `validate_audio_file` (line 163) `def validate_audio_file(file_path)` - *Validate that the audio file exists and has a supported extension.*
- `print_configuration_banner` (line 175) `def print_configuration_banner(config, mode, audio_path)` - *Print a formatted configuration summary.*
- `run_training` (line 215) `def run_training(args)` - *Execute the training pipeline.*
- `run_inference` (line 226) `def run_inference(args)` - *Execute the inference pipeline.*
- `main` (line 236) `def main()` - *Main entry point.*

#### `metrics.py`
**Path:** `audio/metrics.py`

**Classes:**
- `HamiltonianMetricsTracker` (line 26) `class HamiltonianMetricsTracker` - *Tracks and computes all Hamiltonian mechanics metrics during
training and inference.

Each metric method is a pure computation with no side effects
beyond updating internal accumulators, following the
Interface Segregation Principle by exposing granular metric methods.*

**Functions:**
- `__init__` (line 36) `def __init__(self, config)`
- `_initialize_history_buffers` (line 43) `def _initialize_history_buffers(self)` - *Pre-allocate deque buffers for each tracked metric.*
- `compute_hamiltonian_energy` (line 74) `def compute_hamiltonian_energy(self, q, p)` - *Compute the Hamiltonian H(q, p) = T(p) + V(q).

T(p) = 0.5 * ||p||^2 (kinetic energy)
V(q) = 0.5 * ||q||^2 (potential energy in harmonic approximation)

Args:
    q: Generalized coordinates tensor (position in phase space).
    p: Conjugate momenta tensor.

Returns:
    Scalar Hamiltonian energy value.*
- `compute_symplectic_form` (line 97) `def compute_symplectic_form(self, q, p, dq, dp)` - *Compute the symplectic 2-form omega(dq, dp) = sum(dq_i ^ dp_i).

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
- `compute_liouville_measure` (line 125) `def compute_liouville_measure(self, jacobian)` - *Compute Liouville measure |det(J)| for the flow map Jacobian.

By Liouville's theorem, Hamiltonian flow preserves phase space
volume, so det(J) should equal 1 for exact symplectic evolution.

Args:
    jacobian: The Jacobian matrix of the phase space transformation.

Returns:
    Absolute determinant of the Jacobian.*
- `compute_phase_space_volume` (line 153) `def compute_phase_space_volume(self, q, p)` - *Estimate phase space volume occupied by the state (q, p).

Uses the covariance ellipsoid approximation:
V ~ sqrt(det(Cov([q, p])))

Args:
    q: Generalized coordinates (flattened).
    p: Conjugate momenta (flattened).

Returns:
    Estimated phase space volume.*
- `compute_action_integral` (line 181) `def compute_action_integral(self, q_trajectory, p_trajectory, dt)` - *Compute the action integral S = integral(L dt) along a trajectory.

L = T - V = 0.5*||p||^2 - 0.5*||q||^2 (Lagrangian)

Args:
    q_trajectory: Sequence of coordinate states [T, ...].
    p_trajectory: Sequence of momentum states [T, ...].
    dt: Time step between trajectory points.

Returns:
    Total action along the trajectory.*
- `compute_poisson_bracket` (line 208) `def compute_poisson_bracket(self, f_values, g_values, q, p)` - *Estimate the Poisson bracket {f, g} = sum(df/dq * dg/dp - df/dp * dg/dq).

Uses finite differences on the discretized phase space.

Args:
    f_values: Observable f evaluated on phase space grid.
    g_values: Observable g evaluated on phase space grid.
    q: Coordinate grid.
    p: Momentum grid.

Returns:
    Estimated Poisson bracket scalar.*
- `compute_spectral_entropy` (line 238) `def compute_spectral_entropy(self, spectrum)` - *Compute spectral entropy H = -sum(p_i * log(p_i)).

Measures the disorder/uniformity of the spectral distribution.
Maximum entropy indicates uniform spectrum (white noise),
minimum indicates pure tone (single frequency).

Args:
    spectrum: Power spectrum tensor (non-negative).

Returns:
    Scalar spectral entropy.*
- `compute_reconstruction_snr` (line 259) `def compute_reconstruction_snr(self, original, reconstructed)` - *Compute Signal-to-Noise Ratio in dB.

SNR = 10 * log10(||original||^2 / ||original - reconstructed||^2)

Args:
    original: Ground truth signal.
    reconstructed: Reconstructed signal.

Returns:
    SNR in decibels.*
- `compute_spectral_convergence` (line 281) `def compute_spectral_convergence(self, original_spectrum, reconstructed_spectrum)` - *Compute spectral convergence metric.

SC = ||S_orig - S_recon||_F / ||S_orig||_F

Lower values indicate better spectral fidelity.

Args:
    original_spectrum: Original frequency domain representation.
    reconstructed_spectrum: Reconstructed frequency domain representation.

Returns:
    Spectral convergence ratio.*
- `compute_phase_coherence` (line 305) `def compute_phase_coherence(self, phase_original, phase_reconstructed)` - *Compute phase coherence between original and reconstructed signals.

PC = |mean(exp(i * (phi_orig - phi_recon)))|

Value of 1.0 indicates perfect phase alignment.

Args:
    phase_original: Phase spectrum of original signal.
    phase_reconstructed: Phase spectrum of reconstructed signal.

Returns:
    Phase coherence in [0, 1].*
- `compute_energy_drift` (line 328) `def compute_energy_drift(self, energy_initial, energy_current)` - *Compute relative energy drift from initial state.

drift = |E_current - E_initial| / (|E_initial| + epsilon)

Args:
    energy_initial: Hamiltonian energy at t=0.
    energy_current: Hamiltonian energy at current time.

Returns:
    Relative energy drift.*
- `record_gradient_norm` (line 348) `def record_gradient_norm(self, model_parameters)` - *Compute and record the total gradient norm across all parameters.*
- `record_parameter_norm` (line 359) `def record_parameter_norm(self, model_parameters)` - *Compute and record the total parameter norm.*
- `record_learning_rate` (line 369) `def record_learning_rate(self, lr)` - *Record current learning rate.*
- `record_loss_component` (line 374) `def record_loss_component(self, name, value)` - *Record an individual loss component value.*
- `_record` (line 379) `def _record(self, metric_name, value)` - *Store a metric value in history and current snapshot.*
- `get_current_metrics` (line 388) `def get_current_metrics(self)` - *Return a snapshot of all current metric values.*
- `get_moving_averages` (line 392) `def get_moving_averages(self)` - *Compute moving averages for all tracked metrics.*
- `get_formatted_metrics_string` (line 400) `def get_formatted_metrics_string(self)` - *Format all current metrics into a human-readable string for progress bars.*
- `increment_step` (line 413) `def increment_step(self)` - *Advance the global step counter.*
- `step_count` (line 418) `def step_count(self)`
- `should_log` (line 421) `def should_log(self)` - *Determine if metrics should be logged at this step.*

#### `model.py`
**Path:** `audio/model.py`

**Classes:**
- `SpectralEvolutionLayer` (line 27) `class SpectralEvolutionLayer` - *Single Hamiltonian spectral evolution layer.

Performs frequency-domain evolution using learnable complex kernels.
Kernel shape: [hidden_dim, hidden_dim, kernel_base_height, kernel_base_width]
matching the original experiment2 architecture.*
- `HamiltonianNeuralNetwork` (line 162) `class HamiltonianNeuralNetwork` - *Complete Hamiltonian Neural Network with parametric architecture.

Architecture (matching experiment2):
    1. Input projection: Conv2d(1, hidden_dim, kernel, pad)
    2. N spectral evolution layers with learnable complex kernels
    3. Output projection: Conv2d(hidden_dim, 1, kernel, pad)*

**Functions:**
- `__init__` (line 36) `def __init__(self, hidden_dim, kernel_base_height, kernel_base_width, init_std)`
- `forward` (line 51) `def forward(self, x)` - *Apply one step of Hamiltonian spectral evolution via RFFT2.

Args:
    x: Input tensor [B, C, H, W] in spatial domain.

Returns:
    Evolved tensor [B, C, H, W] in spatial domain.*
- `evolve_complex` (line 85) `def evolve_complex(self, x, target_height, target_width)` - *Full complex FFT evolution for amplitude and phase extraction.

Uses full FFT2 (not RFFT2) to preserve complete complex structure.

Args:
    x: Input tensor [B, C, H, W].
    target_height: Output spatial height.
    target_width: Output spatial width.

Returns:
    Complex-valued evolved field in spatial domain.*
- `evolve_real` (line 124) `def evolve_real(self, x, target_height, target_width)` - *Real FFT evolution for action map computation.

Args:
    x: Input tensor [B, C, H, W].
    target_height: Output spatial height.
    target_width: Output spatial width.

Returns:
    Real-valued evolved field in spatial domain.*
- `__init__` (line 172) `def __init__(self, config)`
- `forward` (line 200) `def forward(self, x)` - *Full forward pass: project -> evolve -> reconstruct.

Args:
    x: Input tensor [B, 1, H, W].

Returns:
    Reconstructed tensor [B, 1, H, W].*
- `forward_with_intermediates` (line 216) `def forward_with_intermediates(self, x)` - *Forward pass returning intermediate hidden states for analysis.

Args:
    x: Input tensor [B, 1, H, W].

Returns:
    Tuple of (output, list of intermediate states).*
- `extract_hamiltonian_fields` (line 237) `def extract_hamiltonian_fields(self, x)` - *Extract the three Hamiltonian field representations:
1. Amplitude map (energy density / resonance)
2. Phase map (topological structure / vortices)
3. Action map (constructive interference = clear vision)

Mirrors the visual processing logic from the original code.

Args:
    x: Input tensor [B, 1, H, W].

Returns:
    Tuple of (amplitude_map, phase_map, action_map) each [H, W].*
- `compute_energy_mask` (line 270) `def compute_energy_mask(self, x)` - *Compute the Hamiltonian energy mask for spectral reconstruction.

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

**Classes:**
- `AudioSpectrogramDatasetBuilder` (line 38) `class AudioSpectrogramDatasetBuilder` - *Builds a TensorDataset of spectrogram patches from an audio file.

Segments the full mel spectrogram into overlapping patches
of size (n_mels, matrix_size_width) for training.*
- `HamiltonianAudioTrainer` (line 79) `class HamiltonianAudioTrainer` - *Complete training pipeline for the Hamiltonian Audio Network.

Manages:
- Model initialization and checkpoint recovery
- Optimizer and scheduler configuration
- Training and validation loops
- Full metric reporting at every step
- Time-based checkpointing
- Early stopping*

**Functions:**
- `__init__` (line 46) `def __init__(self, config)`
- `build_dataset` (line 49) `def build_dataset(self, mel_spectrogram)` - *Segment a mel spectrogram into training patches.

Args:
    mel_spectrogram: Full spectrogram [1, 1, n_mels, time_frames].

Returns:
    TensorDataset of (input_patch, target_patch) pairs.*
- `__init__` (line 92) `def __init__(self, config)`
- `_attempt_checkpoint_recovery` (line 129) `def _attempt_checkpoint_recovery(self)` - *Load existing checkpoint if available.*
- `train` (line 151) `def train(self, audio_file_path)` - *Execute the full training pipeline on an audio file.

Args:
    audio_file_path: Path to the input audio file.*
- `_train_one_epoch` (line 255) `def _train_one_epoch(self, train_loader, epoch)` - *Execute one training epoch with full metric tracking.*
- `_validate` (line 344) `def _validate(self, val_loader, epoch)` - *Run validation pass and return metrics.*
- `model` (line 377) `def model(self)`
- `audio_processor` (line 381) `def audio_processor(self)`

#### `visualization.py`
**Path:** `audio/visualization.py`

**Classes:**
- `HamiltonianAudioVisualizer` (line 29) `class HamiltonianAudioVisualizer` - *Generates all scientific visualizations for Hamiltonian audio analysis.*

**Functions:**
- `__init__` (line 34) `def __init__(self, vis_config, audio_config)`
- `render_complete_analysis` (line 43) `def render_complete_analysis(self, amplitude_map, phase_map, action_map, original_spectrogram, reconstructed_spectrogram, original_waveform, reconstructed_waveform, output_prefix)` - *Generate the complete suite of Hamiltonian analysis visualizations.

Args:
    amplitude_map: Energy density field [H, W].
    phase_map: Phase topology field [H, W].
    action_map: Action density field [H, W].
    original_spectrogram: Original mel spectrogram [1, 1, H, W].
    reconstructed_spectrogram: Reconstructed mel spectrogram [1, 1, H, W].
    original_waveform: Original audio waveform [1, T].
    reconstructed_waveform: Reconstructed audio waveform [1, T].
    output_prefix: Filename prefix for all outputs.*
- `_render_hamiltonian_fields` (line 91) `def _render_hamiltonian_fields(self, amplitude_map, phase_map, action_map, output_prefix)` - *Render the three Hamiltonian field visualizations.*
- `_render_spectrogram_comparison` (line 140) `def _render_spectrogram_comparison(self, original, reconstructed, output_prefix)` - *Render original vs reconstructed spectrogram comparison.*
- `_render_phase_portrait` (line 185) `def _render_phase_portrait(self, amplitude_map, phase_map, output_prefix)` - *Render 2D phase portrait (amplitude vs phase histogram).*
- `_render_energy_landscape` (line 218) `def _render_energy_landscape(self, amplitude_map, action_map, output_prefix)` - *Render energy landscape as a 3D surface plot.*
- `_render_waveform_comparison` (line 270) `def _render_waveform_comparison(self, original_waveform, reconstructed_waveform, output_prefix)` - *Render original vs reconstructed waveform comparison.*

#### `check_fase_berry.py`
**Path:** `check_fase_berry.py`

*No symbols extracted*

#### `diff_weights.py`
**Path:** `diff_weights.py`

**Functions:**
- `analize_checkpoint` (line 5) `def analize_checkpoint(path)`

#### `dirac.py`
**Path:** `dirac.py`

**Classes:**
- `DiracConfig` (line 25) `class DiracConfig`
- `DiracDeltaAnalyzer` (line 38) `class DiracDeltaAnalyzer`
- `DiracVisualizer` (line 328) `class DiracVisualizer`

**Functions:**
- `analyze_checkpoint` (line 574) `def analyze_checkpoint(checkpoint_path, output_dir)`
- `analyze_multiple_checkpoints` (line 620) `def analyze_multiple_checkpoints(checkpoint_dir, n_latest, output_dir)`
- `main` (line 652) `def main()`
- `__init__` (line 40) `def __init__(self, checkpoint_path, device)`
- `extract_charge_distribution` (line 64) `def extract_charge_distribution(self)`
- `compute_dirac_delta_approximation` (line 77) `def compute_dirac_delta_approximation(self, charge_density)`
- `compute_electric_field` (line 112) `def compute_electric_field(self, dirac_data, eval_points)`
- `compute_electric_flux` (line 157) `def compute_electric_flux(self, electric_field, surface_points)`
- `compute_divergence` (line 192) `def compute_divergence(self, electric_field)`
- `verify_gauss_law` (line 200) `def verify_gauss_law(self, dirac_data, flux_data)`
- `analyze_all` (line 223) `def analyze_all(self)`
- `_print_report` (line 279) `def _print_report(self, results)`
- `plot_charge_distribution` (line 331) `def plot_charge_distribution(charge_density, point_positions, point_charges, output_path)`
- `plot_electric_field` (line 379) `def plot_electric_field(electric_field, output_path)`
- `plot_divergence` (line 441) `def plot_divergence(divergence, output_path)`
- `plot_combined_analysis` (line 490) `def plot_combined_analysis(charge_density, point_positions, point_charges, electric_field, divergence, output_path)`

#### `expand.py`
**Path:** `expand.py`

**Functions:**
- `load_config` (line 18) `def load_config(toml_path)`
- `expand_spectral_weights` (line 23) `def expand_spectral_weights(kernel_real, kernel_imag, target_size, source_size)` - *Expand spectral kernels via zero-padding in frequency domain.*
- `expand_model` (line 43) `def expand_model(model, target_resolution, source_resolution)` - *Create a new model with expanded spectral weights.*
- `evaluate_model` (line 74) `def evaluate_model(model, resolution, device)` - *Evaluate expanded model on synthetic data.*
- `main` (line 105) `def main()`

#### `experiment.py`
**Path:** `experiment.py`

**Classes:**
- `Config` (line 41) `class Config`
- `IAnalysisStrategy` (line 109) `class IAnalysisStrategy(ABC)`
- `IMetricsCalculator` (line 115) `class IMetricsCalculator(ABC)`
- `HamiltonianOperator` (line 121) `class HamiltonianOperator` - *True Hamiltonian operator H = -nabla^2 on torus.*
- `FastDataset` (line 146) `class FastDataset(Dataset)` - *Fast dataset for Hamiltonian operator learning.*
- `SpectralLayer` (line 203) `class SpectralLayer` - *Spectral layer with correct complex multiplication.*
- `SimpleHamiltonianNet` (line 255) `class SimpleHamiltonianNet` - *Compact network for Hamiltonian operator learning.*
- `LocalComplexityAnalyzer` (line 293) `class LocalComplexityAnalyzer`
- `SuperpositionAnalyzer` (line 310) `class SuperpositionAnalyzer`
- `CrystallographyMetrics` (line 340) `class CrystallographyMetrics`
- `ThermodynamicMetrics` (line 444) `class ThermodynamicMetrics`
- `SpectroscopyMetrics` (line 470) `class SpectroscopyMetrics`
- `CheckpointManager` (line 500) `class CheckpointManager`
- `TrainingMonitor` (line 573) `class TrainingMonitor`
- `GlassStopper` (line 611) `class GlassStopper`
- `BoltzmannAnalysisProgram` (line 879) `class BoltzmannAnalysisProgram`

**Functions:**
- `set_seed` (line 88) `def set_seed(seed)`
- `setup_logger` (line 96) `def setup_logger(name, level)`
- `train_with_early_glass_stop` (line 670) `def train_with_early_glass_stop(model, optimizer, seed, epochs)` - *Train model with early stopping for glass detection.*
- `seed_miner` (line 803) `def seed_miner(total_attempts)` - *Mine for crystal seeds by trying sequential seeds.*
- `main` (line 856) `def main()`
- `analyze` (line 111) `def analyze(self, model)`
- `compute` (line 117) `def compute(self, model)`
- `__init__` (line 124) `def __init__(self, grid_size)`
- `_precompute_spectral_operators` (line 128) `def _precompute_spectral_operators(self)`
- `apply` (line 135) `def apply(self, field)`
- `time_evolution` (line 140) `def time_evolution(self, field, dt)`
- `__init__` (line 149) `def __init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)`
- `__len__` (line 193) `def __len__(self)`
- `__getitem__` (line 196) `def __getitem__(self, idx)`
- `get_val_batch` (line 199) `def get_val_batch(self)`
- `__init__` (line 206) `def __init__(self, channels, grid_size)`
- `forward` (line 219) `def forward(self, x)`
- `__init__` (line 258) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `forward` (line 279) `def forward(self, x)`
- `compute_local_complexity` (line 295) `def compute_local_complexity(weights, epsilon)` - *Compute Local Complexity (LC) metric for weight matrix.*
- `compute_superposition` (line 312) `def compute_superposition(weights)` - *Compute Superposition (SP) metric for weight matrix.*
- `compute_kappa` (line 342) `def compute_kappa(model, dataloader, num_batches)`
- `compute_discretization_margin` (line 378) `def compute_discretization_margin(coeffs)`
- `compute_alpha_purity` (line 387) `def compute_alpha_purity(coeffs)`
- `compute_kappa_quantum` (line 394) `def compute_kappa_quantum(coeffs, hbar)`
- `compute_poynting_vector` (line 411) `def compute_poynting_vector(coeffs)`
- `compute_all_metrics` (line 426) `def compute_all_metrics(model, dataloader)`
- `compute_effective_temperature` (line 446) `def compute_effective_temperature(gradient_buffer, learning_rate)`
- `compute_specific_heat` (line 460) `def compute_specific_heat(loss_history, temp_history, cv_threshold)`
- `compute_weight_diffraction` (line 472) `def compute_weight_diffraction(coeffs)`
- `_compute_spectral_entropy` (line 491) `def _compute_spectral_entropy(power_spectrum)`
- `__init__` (line 501) `def __init__(self, interval_minutes, max_checkpoints)`
- `should_save_checkpoint` (line 509) `def should_save_checkpoint(self)`
- `save_checkpoint` (line 514) `def save_checkpoint(self, model, optimizer, epoch, metrics)`
- `__init__` (line 574) `def __init__(self)`
- `update_metrics` (line 594) `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat, poynting_magnitude)`
- `__init__` (line 612) `def __init__(self, patience_epochs)`
- `should_stop` (line 616) `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)` - *Check if the system is in glass state and should stop mining.*
- `__init__` (line 880) `def __init__(self, checkpoint_path, results_dir)`
- `load_and_analyze_checkpoint` (line 886) `def load_and_analyze_checkpoint(self)`
- `dataloader` (line 903) `def dataloader()`

#### `experiment2.py`
**Path:** `experiment2.py`

**Classes:**
- `Config` (line 22) `class Config`
- `SeedManager` (line 74) `class SeedManager`
- `LoggerFactory` (line 84) `class LoggerFactory`
- `IAnalysisStrategy` (line 99) `class IAnalysisStrategy(ABC)`
- `IMetricsCalculator` (line 105) `class IMetricsCalculator(ABC)`
- `HamiltonianOperator` (line 111) `class HamiltonianOperator`
- `HamiltonianDataset` (line 133) `class HamiltonianDataset(Dataset)`
- `SpectralLayer` (line 183) `class SpectralLayer`
- `HamiltonianNeuralNetwork` (line 227) `class HamiltonianNeuralNetwork`
- `LocalComplexityAnalyzer` (line 257) `class LocalComplexityAnalyzer`
- `SuperpositionAnalyzer` (line 273) `class SuperpositionAnalyzer`
- `CrystallographyMetricsCalculator` (line 302) `class CrystallographyMetricsCalculator(IMetricsCalculator)`
- `ThermodynamicMetricsCalculator` (line 737) `class ThermodynamicMetricsCalculator(IMetricsCalculator)`
- `SpectroscopyMetricsCalculator` (line 770) `class SpectroscopyMetricsCalculator(IMetricsCalculator)`
- `CheckpointManager` (line 804) `class CheckpointManager`
- `TrainingMetricsMonitor` (line 874) `class TrainingMetricsMonitor`
- `GlassStateDetector` (line 912) `class GlassStateDetector`
- `TrainingEngine` (line 973) `class TrainingEngine`
- `SeedMiningSystem` (line 1113) `class SeedMiningSystem`
- `SingleExperimentRunner` (line 1164) `class SingleExperimentRunner`
- `CheckpointAnalyzer` (line 1223) `class CheckpointAnalyzer`
- `Application` (line 1275) `class Application`

**Functions:**
- `main` (line 1334) `def main()`
- `set_seed` (line 76) `def set_seed(seed)`
- `create_logger` (line 86) `def create_logger(name, level)`
- `analyze` (line 101) `def analyze(self, model)`
- `compute` (line 107) `def compute(self, model)`
- `__init__` (line 112) `def __init__(self, grid_size)`
- `_precompute_spectral_operators` (line 116) `def _precompute_spectral_operators(self)`
- `apply` (line 122) `def apply(self, field)`
- `time_evolution` (line 127) `def time_evolution(self, field, dt)`
- `__init__` (line 134) `def __init__(self, num_samples, grid_size, time_steps, dt, train_ratio)`
- `__len__` (line 173) `def __len__(self)`
- `__getitem__` (line 176) `def __getitem__(self, idx)`
- `get_validation_batch` (line 179) `def get_validation_batch(self)`
- `__init__` (line 184) `def __init__(self, channels, grid_size)`
- `forward` (line 195) `def forward(self, x)`
- `__init__` (line 228) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `forward` (line 243) `def forward(self, x)`
- `compute_local_complexity` (line 259) `def compute_local_complexity(weights, epsilon)`
- `compute_superposition` (line 275) `def compute_superposition(weights)`
- `compute` (line 303) `def compute(self, model, val_x, val_y)` - *Implementación de interfaz IMetricsCalculator.
Delega a compute_all_metrics con los argumentos correctos.*
- `compute_gradient_covariance_kappa` (line 311) `def compute_gradient_covariance_kappa(model, dataloader, num_batches)`
- `compute_discretization_margin_from_state_dict` (line 348) `def compute_discretization_margin_from_state_dict(model)` - *Calcula el margen de discretización desde los parámetros del modelo.
Versión estática que no requiere diccionario externo.*
- `compute_discretization_margin` (line 361) `def compute_discretization_margin(coeffs)` - *Calcula el margen de discretización desde un diccionario de coeficientes.*
- `compute_alpha_purity_from_model` (line 373) `def compute_alpha_purity_from_model(model)` - *Calcula el índice de pureza alpha directamente desde el modelo.*
- `compute_alpha_purity` (line 383) `def compute_alpha_purity(coeffs)` - *Calcula el índice de pureza alpha desde un diccionario de coeficientes.*
- `compute_kappa` (line 393) `def compute_kappa(model, val_x, val_y, num_batches)` - *Número de condición de la matriz de covarianza de gradientes.*
- `compute_kappa_quantum` (line 464) `def compute_kappa_quantum(model, hbar)` - *Versión del cálculo cuántico de kappa que opera directamente sobre el modelo.*
- `compute_kappa_quantum_from_coeffs` (line 492) `def compute_kappa_quantum_from_coeffs(coeffs, hbar)` - *Versión del cálculo cuántico de kappa desde diccionario de coeficientes.*
- `_compute_crystallography_metrics` (line 511) `def _compute_crystallography_metrics(self, model, val_x, val_y)` - *Métricas cristalográficas con aislamiento completo de errores.*
- `_check_weight_integrity` (line 539) `def _check_weight_integrity(self, model)` - *Verifica integridad de pesos: NaN, Inf, y estadísticas básicas.*
- `compute_poynting_vector` (line 603) `def compute_poynting_vector(model)` - *Vector de Poynting: flujo de energía en el espacio de parámetros.
Análogo electromagnético para redes neuronales.*
- `compute_all_metrics` (line 679) `def compute_all_metrics(model, val_x, val_y)` - *Calcula todas las métricas cristalográficas con manejo de errores.*
- `compute` (line 738) `def compute(self, model, gradient_buffer, learning_rate, loss_history, temp_history)`
- `compute_effective_temperature` (line 747) `def compute_effective_temperature(gradient_buffer, learning_rate)`
- `compute_specific_heat` (line 760) `def compute_specific_heat(loss_history, temp_history, cv_threshold)`
- `compute` (line 771) `def compute(self, model)`
- `compute_weight_diffraction` (line 776) `def compute_weight_diffraction(coeffs)`
- `_compute_spectral_entropy` (line 795) `def _compute_spectral_entropy(power_spectrum)`
- `__init__` (line 805) `def __init__(self, interval_minutes, max_checkpoints)`
- `should_save_checkpoint` (line 813) `def should_save_checkpoint(self)`
- `save_checkpoint` (line 818) `def save_checkpoint(self, model, optimizer, epoch, metrics)`
- `__init__` (line 875) `def __init__(self)`
- `update_metrics` (line 895) `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat, poynting_magnitude)`
- `__init__` (line 913) `def __init__(self, patience_epochs)`
- `should_stop` (line 918) `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)`
- `is_crystal_formed` (line 963) `def is_crystal_formed(self, lc, sp, kappa, delta, temp, cv)`
- `__init__` (line 974) `def __init__(self, model, optimizer, device, logger)`
- `train_epoch` (line 997) `def train_epoch(self, dataloader, epoch)`
- `validate` (line 1027) `def validate(self, val_x, val_y)`
- `compute_weight_metrics` (line 1040) `def compute_weight_metrics(self)`
- `execute_training` (line 1056) `def execute_training(self, dataloader, val_x, val_y, epochs, seed, early_stopping)`
- `__init__` (line 1114) `def __init__(self, max_attempts)`
- `mine` (line 1118) `def mine(self)`
- `__init__` (line 1165) `def __init__(self, seed, epochs, grid_size, hidden_dim, num_spectral_layers, learning_rate)`
- `run` (line 1175) `def run(self)`
- `__init__` (line 1224) `def __init__(self, checkpoint_path, results_dir)`
- `analyze` (line 1230) `def analyze(self)`
- `__init__` (line 1276) `def __init__(self)`
- `_create_argument_parser` (line 1280) `def _create_argument_parser(self)`
- `run` (line 1294) `def run(self)`
- `safe_compute` (line 694) `def safe_compute(func)`

#### `export.py`
**Path:** `export.py`

*No symbols extracted*

#### `get_meditions.py`
**Path:** `get_meditions.py`

**Classes:**
- `ThermodynamicConfig` (line 28) `class ThermodynamicConfig` - *Configuración termodinámica para análisis de HPU Core*
- `ThermodynamicPotential` (line 72) `class ThermodynamicPotential` - *Potencial de Helmholtz: F = U - T*S + μ*N + α_term*C*
- `CrystallographyMetrics` (line 98) `class CrystallographyMetrics` - *Métricas de cristalografía para redes neuronales Hamiltonianas.
Mide la "pureza" estructural de los pesos aprendidos.*
- `ThermodynamicMetrics` (line 394) `class ThermodynamicMetrics` - *Análisis termodinámico del proceso de entrenamiento.
Temperatura efectiva, calor específico, transiciones de fase.*
- `SpectroscopyMetrics` (line 634) `class SpectroscopyMetrics` - *Análisis espectroscópico de pesos: difracción, descomposición, parámetros de red.*
- `CheckpointVerifier` (line 740) `class CheckpointVerifier`
- `SpectralCoefficients` (line 105) `class SpectralCoefficients` - *Contenedor para coeficientes espectrales de HPU Core*

**Functions:**
- `setup_logger` (line 51) `def setup_logger(name, level)`
- `verify_latest_checkpoints` (line 1439) `def verify_latest_checkpoints(checkpoint_dir, n)` - *Verifica los N checkpoints más recientes con análisis completo*
- `main` (line 1500) `def main()`
- `helmholtz_free_energy` (line 81) `def helmholtz_free_energy(self)` - *F = U - T*S (a μ y N constantes)*
- `gibbs_free_energy` (line 85) `def gibbs_free_energy(self)` - *G = F + μ*N + P*V (presión algorítmica)*
- `is_stable` (line 90) `def is_stable(self)` - *Criterio de estabilidad: dG < 0*
- `compute_kappa` (line 127) `def compute_kappa(model, val_x, val_y, num_batches)` - *Número de condición de la matriz de covarianza de gradientes.
FIX: Limita tamaño de gradientes y usa método iterativo si es necesario.*
- `compute_discretization_margin` (line 187) `def compute_discretization_margin(model)` - *δ = max |w - round(w)| sobre todos los parámetros.
Mide qué tan cerca están los pesos de valores enteros.*
- `compute_alpha_purity` (line 203) `def compute_alpha_purity(model)` - *α = -log(δ). Pureza cristalina.
α > 7 indica estructura cristalina perfecta.*
- `compute_local_complexity` (line 214) `def compute_local_complexity(model)` - *Fracción de parámetros "activos" (no cerca de cero).*
- `compute_kappa_quantum` (line 226) `def compute_kappa_quantum(model, hbar)` - *κ cuántico: número de condición con regularización cuántica.
FIX: Usa método iterativo de potencia en lugar de matriz densa.*
- `_compute_kappa_iterative` (line 255) `def _compute_kappa_iterative(params, hbar, n, max_iters, tol)` - *Método de potencia para estimar κ sin construir matriz.
Estima λ_max y λ_min de la matriz de covarianza regularizada.*
- `compute_poynting_vector` (line 312) `def compute_poynting_vector(model)` - *Vector de Poynting: flujo de energía en el espacio de parámetros.
Análogo electromagnético para redes neuronales.*
- `compute_all_metrics` (line 366) `def compute_all_metrics(model, val_x, val_y)` - *Calcula todas las métricas cristalográficas.*
- `compute_effective_temperature` (line 401) `def compute_effective_temperature(gradient_buffer, learning_rate)` - *T_eff = (lr/2) * Var(∇L). Temperatura de fluctuaciones.*
- `compute_specific_heat` (line 418) `def compute_specific_heat(loss_history, temp_history, cv_threshold)` - *C_v = Var(U) / T^2. Detecta transiciones de fase (picos en C_v).*
- `compute_critical_exponents` (line 436) `def compute_critical_exponents(temp_history, cv_history, alpha_history)` - *Exponentes críticos cerca de transiciones de fase.*
- `compute_equation_of_state` (line 504) `def compute_equation_of_state(temp_eff, alpha, kappa)` - *Ecuación de estado: T_c(α) = T_0 * exp(-c*α)
Relación constitutiva cristal-vidrio.*
- `compute_mutual_information` (line 539) `def compute_mutual_information(weights, gradients)` - *Información mutua pesos-gradientes.*
- `estimate_hbar_algorithmic` (line 561) `def estimate_hbar_algorithmic(model_complexity, weight_dim, mutual_information)` - *ħ algorítmico efectivo.*
- `compute_fisher_information_matrix` (line 572) `def compute_fisher_information_matrix(model, samples)` - *Matriz de información de Fisher.*
- `compute_ricci_curvature` (line 593) `def compute_ricci_curvature(fisher_matrix)` - *Curvatura de Ricci escalar.*
- `calculate_carnot_efficiency` (line 608) `def calculate_carnot_efficiency(delta_alpha, total_flops, initial_alpha)` - *Eficiencia de Carnot del proceso de aprendizaje.*
- `compute_weight_diffraction` (line 640) `def compute_weight_diffraction(model)` - *Patrón de difracción de pesos (FFT).
Detecta periodicidad cristalina (picos de Bragg).*
- `_compute_spectral_entropy` (line 672) `def _compute_spectral_entropy(power_spectrum)` - *Entropía espectral de Shannon.*
- `extract_lattice_parameters` (line 680) `def extract_lattice_parameters(weight_tensor, rank)` - *Extrae parámetros de red vía SVD.*
- `compute_gibbs_free_energy` (line 732) `def compute_gibbs_free_energy(loss, temp, entropy)` - *Energía libre de Gibbs.*
- `__init__` (line 741) `def __init__(self, checkpoint_path, device)`
- `verify_all_metrics` (line 782) `def verify_all_metrics(self)` - *Calcula TODAS las métricas desde cero y compara con las guardadas*
- `_check_weight_integrity` (line 849) `def _check_weight_integrity(self)` - *Verifica que los pesos no tengan NaN/Inf.
FIX: Evita calcular std en tensores con 1 elemento.*
- `_compute_validation_metrics` (line 915) `def _compute_validation_metrics(self)` - *Calcula MSE y accuracy de validación desde cero*
- `_compute_discretization_metrics` (line 942) `def _compute_discretization_metrics(self)` - *Calcula delta, alpha, purity, etc.*
- `_compute_quantization_metrics` (line 986) `def _compute_quantization_metrics(self)` - *Calcula la penalización de cuantización*
- `_compute_loss_metrics` (line 1005) `def _compute_loss_metrics(self)` - *Reconstruye el loss total*
- `_compute_crystallography_metrics` (line 1031) `def _compute_crystallography_metrics(self)` - *Métricas cristalográficas completas*
- `_compute_thermodynamic_metrics` (line 1038) `def _compute_thermodynamic_metrics(self)` - *Métricas termodinámicas*
- `_approximate_ricci_curvature` (line 1113) `def _approximate_ricci_curvature(self)` - *Aproximación de curvatura de Ricci para HPU Core*
- `_compute_spectroscopy` (line 1134) `def _compute_spectroscopy(self)` - *Análisis espectroscópico*
- `_compute_thermodynamic_potential` (line 1164) `def _compute_thermodynamic_potential(self, results)` - *Calcula potencial termodinámico completo*
- `_compare_with_stored` (line 1188) `def _compare_with_stored(self, computed)` - *Compara métricas calculadas vs almacenadas en el checkpoint*
- `_check_internal_consistency` (line 1226) `def _check_internal_consistency(self, results)` - *Verifica consistencia entre métricas relacionadas*
- `_compute_health_score` (line 1267) `def _compute_health_score(self, results)` - *Calcula un score de salud del checkpoint (0-100)*
- `_assign_crystallographic_grade` (line 1336) `def _assign_crystallographic_grade(self, delta, alpha)` - *Asigna grado cristalográfico*
- `_print_report` (line 1349) `def _print_report(self, results)` - *Imprime reporte formateado con todas las métricas nuevas*
- `from_model` (line 112) `def from_model(cls, model)` - *Extrae coeficientes del modelo HPU Core*

#### `hamiltonian_mbl.py`
**Path:** `hamiltonian_mbl.py`

**Classes:**
- `HamiltonianArchitectureConfig` (line 37) `class HamiltonianArchitectureConfig` - *Configuration for Hamiltonian Neural Network architecture.
All architectural hyperparameters are centralized here.*
- `MBLAnalysisConfig` (line 67) `class MBLAnalysisConfig` - *Comprehensive configuration for MBL analysis of Hamiltonian NN crystallization.
All analysis parameters are centralized following SOLID principles.*
- `TrainingConfig` (line 141) `class TrainingConfig` - *Configuration for training process.*
- `IModel` (line 160) `class IModel(Protocol)` - *Protocol for models compatible with MBL analysis.*
- `ILevelSpacingCalculator` (line 167) `class ILevelSpacingCalculator(Protocol)` - *Protocol for level spacing ratio calculation.*
- `IParticipationRatioCalculator` (line 173) `class IParticipationRatioCalculator(Protocol)` - *Protocol for participation ratio calculation.*
- `ISyntheticPlanckCalculator` (line 179) `class ISyntheticPlanckCalculator(Protocol)` - *Protocol for synthetic Planck's constant calculation.*
- `IDiscretizationDialAnalyzer` (line 185) `class IDiscretizationDialAnalyzer(Protocol)` - *Protocol for discretization dial analysis.*
- `ICheckpointManager` (line 191) `class ICheckpointManager(Protocol)` - *Protocol for checkpoint management.*
- `ITrainingMetricsCollector` (line 199) `class ITrainingMetricsCollector(Protocol)` - *Protocol for collecting all training metrics.*
- `ArchitectureMigrator` (line 209) `class ArchitectureMigrator` - *Migra pesos de SimpleHamiltonianNet (Conv2d) a HamiltonianNeuralNetwork (Linear).*
- `SpectralHamiltonianLayer` (line 346) `class SpectralHamiltonianLayer` - *Spectral layer implementing Hamiltonian dynamics in Fourier space.
Preserves energy conservation through symplectic integration.*
- `HamiltonianNeuralNetwork` (line 412) `class HamiltonianNeuralNetwork` - *Complete Hamiltonian Neural Network for learning dynamical systems.
Uses spectral layers to ensure energy conservation and symplectic structure.*
- `HamiltonianDataset` (line 555) `class HamiltonianDataset` - *Generates physics-informed training data for Hamiltonian NN.
Creates trajectories from known dynamical systems.*
- `LevelSpacingRatioCalculator` (line 608) `class LevelSpacingRatioCalculator` - *Calculates the level spacing ratio r for MBL phase detection.

The ratio r_n = min(delta_n, delta_{n+1}) / max(delta_n, delta_{n+1})
where delta_n = E_{n+1} - E_n (energy level spacing).*
- `ParticipationRatioCalculator` (line 719) `class ParticipationRatioCalculator` - *Calculates Inverse Participation Ratio (IPR) for localization analysis.
IPR = sum_i |c_i|^4 where c_i are coefficients in the chosen basis.*
- `SyntheticPlanckConstantCalculator` (line 801) `class SyntheticPlanckConstantCalculator` - *Calculates effective synthetic Planck's constant (hbar_eff) from model properties.
Based on the relation: hbar_eff ∝ 1 / sqrt(PR * Energy_Gap)*
- `DiscretizationDialAnalyzer` (line 844) `class DiscretizationDialAnalyzer` - *Analyzes the discretization parameter delta as a phase transition control.*
- `PurityIndexCalculator` (line 947) `class PurityIndexCalculator` - *Calculates the 'crystallinity' of the weight distribution.*
- `EffectiveTemperatureCalculator` (line 1004) `class EffectiveTemperatureCalculator` - *Calculates effective temperature from loss history.*
- `KrylovComplexityCalculator` (line 1048) `class KrylovComplexityCalculator` - *Calculates Krylov complexity as a measure of operator growth and scrambling.
Based on the spread of operators in Krylov space.*
- `CrystallinityIndexCalculator` (line 1089) `class CrystallinityIndexCalculator` - *Calculates crystallinity index through spectral analysis of weight matrices.
Analogous to X-ray diffraction for physical crystals.*
- `ResilienceSpectrometer` (line 1144) `class ResilienceSpectrometer` - *Measures algorithmic resilience through controlled perturbations.
Tests stability across different subspaces and noise levels.*
- `PhaseClassifier` (line 1243) `class PhaseClassifier` - *Classifies the crystallization phase based on alpha and temperature.*
- `CheckpointMigrator` (line 1270) `class CheckpointMigrator` - *Handles migration between different checkpoint formats.*
- `MBLCheckpointManager` (line 1310) `class MBLCheckpointManager` - *Manages checkpoint saving with 5-minute intervals and latest file maintenance.*
- `HamiltonianMBLMetricsCollector` (line 1386) `class HamiltonianMBLMetricsCollector` - *Collects all MBL metrics for comprehensive training monitoring.
Includes all metrics from the crystallography paper.*
- `HamiltonianTrainer` (line 1540) `class HamiltonianTrainer` - *Training system for Hamiltonian Neural Networks with integrated MBL monitoring.*
- `HamiltonianCheckpointAnalyzer` (line 1710) `class HamiltonianCheckpointAnalyzer` - *Comprehensive analyzer for Hamiltonian NN checkpoints with migration support.*
- `HamiltonianMBLPipeline` (line 1875) `class HamiltonianMBLPipeline` - *Main pipeline for processing checkpoints and generating reports.*

**Functions:**
- `main` (line 2031) `def main()`
- `get_input_dim` (line 55) `def get_input_dim(self)` - *Calculate input dimension from grid size.*
- `get_total_parameters` (line 59) `def get_total_parameters(self)` - *Estimate total parameter count.*
- `get_reduced_dimension` (line 135) `def get_reduced_dimension(self)` - *Calculate reduced dimension for analysis.*
- `get_coefficients` (line 162) `def get_coefficients(self)`
- `forward` (line 163) `def forward(self)`
- `calculate` (line 169) `def calculate(self, model)`
- `calculate` (line 175) `def calculate(self, model)`
- `calculate` (line 181) `def calculate(self, participation_ratio, energy_gap)`
- `analyze_robustness` (line 187) `def analyze_robustness(self, model, noise_levels)`
- `save_checkpoint` (line 193) `def save_checkpoint(self, model, epoch, metrics, loss_history, path)`
- `load_checkpoint` (line 195) `def load_checkpoint(self, path)`
- `collect` (line 201) `def collect(self, model, loss, epoch, loss_history)`
- `__init__` (line 214) `def __init__(self, source_config, target_config)`
- `migrate_state_dict` (line 218) `def migrate_state_dict(self, source_state)` - *Migra estado de SimpleHamiltonianNet a HamiltonianNeuralNetwork.

SimpleHamiltonianNet tiene:
- input_proj: Conv2d(1, hidden_dim, 1) -> weight [hidden_dim, 1, 1, 1]
- spectral_layers.{i}.kernel_real: [hidden_dim, hidden_dim, grid//2+1, grid]
- output_proj: Conv2d(hidden_dim, 1, 1) -> weight [1, hidden_dim, 1, 1]

HamiltonianNeuralNetwork necesita:
- q_projection, p_projection: Linear(input_dim, hidden_dim)
- spectral_layers.{i}.spectral_weights: [hidden_dim, spectral_modes]
- q_output, p_output: Linear(hidden_dim, input_dim)*
- `_create_default_parameter` (line 320) `def _create_default_parameter(self, key)` - *Crea parámetro por defecto.*
- `__init__` (line 352) `def __init__(self, config)`
- `_initialize_spectral_parameters` (line 366) `def _initialize_spectral_parameters(self)` - *Initialize with physics-informed priors.*
- `forward` (line 373) `def forward(self, q, p, dt)` - *Symplectic Euler integration of Hamilton's equations.
dq/dt = dH/dp, dp/dt = -dH/dq*
- `get_hamiltonian` (line 401) `def get_hamiltonian(self, q, p)` - *Compute Hamiltonian H = T + V in spectral space.*
- `__init__` (line 418) `def __init__(self, config)`
- `_initialize_weights` (line 438) `def _initialize_weights(self)` - *Orthogonal initialization for Hamiltonian structure preservation.*
- `forward` (line 443) `def forward(self, q, p, dt)` - *Forward pass through Hamiltonian dynamics.*
- `time_evolution` (line 459) `def time_evolution(self, q_initial, p_initial, num_steps, dt)` - *Generate trajectory through time evolution.*
- `get_hamiltonian` (line 474) `def get_hamiltonian(self, q, p)` - *Compute total Hamiltonian.*
- `get_coefficients` (line 486) `def get_coefficients(self)`
- `get_flat_parameters` (line 496) `def get_flat_parameters(self)` - *Returns all parameters flattened for Hamiltonian construction.*
- `construct_hessian_approximation` (line 503) `def construct_hessian_approximation(self, max_dim, method)` - *MÉTODO CORREGIDO - No usa 65GB de RAM.*
- `__init__` (line 561) `def __init__(self, grid_size, num_samples, device)`
- `generate_harmonic_oscillator` (line 567) `def generate_harmonic_oscillator(self, omega)` - *Generate harmonic oscillator initial conditions.*
- `generate_double_well` (line 591) `def generate_double_well(self, barrier_height)` - *Generate double-well potential trajectories.*
- `__init__` (line 616) `def __init__(self, config)`
- `calculate` (line 619) `def calculate(self, model)` - *Calculate level spacing statistics from model weights.*
- `_construct_hessian_from_weights` (line 651) `def _construct_hessian_from_weights(self, model)` - *Alternative Hessian construction for generic models.*
- `_compute_eigenvalues` (line 666) `def _compute_eigenvalues(self, hessian)` - *Compute sorted eigenvalues of the Hamiltonian.*
- `_calculate_spacing_ratios` (line 671) `def _calculate_spacing_ratios(self, spacings)` - *Calculate adjacent gap ratios r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}).*
- `_classify_phase` (line 684) `def _classify_phase(self, mean_ratio)` - *Classify the quantum phase based on level spacing ratio.*
- `_estimate_brody_parameter` (line 699) `def _estimate_brody_parameter(self, ratios)` - *Estimate Brody parameter for intermediate statistics.
0 = Poisson (integrable), 1 = Wigner-Dyson (chaotic)*
- `__init__` (line 725) `def __init__(self, config)`
- `calculate` (line 728) `def calculate(self, model)` - *Calculate participation ratios for all weight layers.*
- `_calculate_ipr` (line 771) `def _calculate_ipr(self, coefficients)` - *Calculate standard Inverse Participation Ratio.*
- `_calculate_renyi_ipr` (line 782) `def _calculate_renyi_ipr(self, coefficients, q)` - *Calculate q-th order Rényi IPR.*
- `_calculate_fractal_dimension` (line 793) `def _calculate_fractal_dimension(self, ipr, n)` - *Calculate fractal dimension D_q from IPR.*
- `__init__` (line 807) `def __init__(self, config)`
- `calculate` (line 810) `def calculate(self, participation_ratio, energy_gap)` - *Calculate synthetic Planck's constant.*
- `calculate_from_model` (line 820) `def calculate_from_model(self, model, level_spacing_results, pr_results)` - *Comprehensive calculation from model and previous analyses.*
- `__init__` (line 849) `def __init__(self, config)`
- `calculate_base_discretization` (line 853) `def calculate_base_discretization(self, model)` - *Calculate the base discretization level from weight rounding error.*
- `analyze_robustness` (line 877) `def analyze_robustness(self, model, noise_levels)` - *Test robustness by applying noise and measuring gap collapse.*
- `_perturb_and_measure` (line 921) `def _perturb_and_measure(self, model, noise_level)` - *Apply noise to model and measure resulting metrics.*
- `_delta_to_alpha` (line 940) `def _delta_to_alpha(self, delta)` - *Convert discretization error to purity alpha.*
- `__init__` (line 950) `def __init__(self, config)`
- `calculate` (line 953) `def calculate(self, model)`
- `_compute_layer_purity` (line 982) `def _compute_layer_purity(self, weights)`
- `_delta_to_alpha` (line 988) `def _delta_to_alpha(self, delta)`
- `_assess_purity_quality` (line 993) `def _assess_purity_quality(self, alpha, variance)`
- `__init__` (line 1007) `def __init__(self, config)`
- `calculate` (line 1010) `def calculate(self, loss_history)`
- `__init__` (line 1054) `def __init__(self, config)`
- `calculate` (line 1057) `def calculate(self, model)` - *Calculate Krylov complexity from model dynamics.*
- `__init__` (line 1095) `def __init__(self, config)`
- `calculate` (line 1098) `def calculate(self, model)` - *Calculate crystallinity index from weight spectra.*
- `__init__` (line 1150) `def __init__(self, config)`
- `measure` (line 1153) `def measure(self, model)` - *Comprehensive resilience measurement.*
- `_measure_base_performance` (line 1176) `def _measure_base_performance(self, model)` - *Measure baseline performance metrics.*
- `_test_perturbation` (line 1195) `def _test_perturbation(self, model, dimension, noise_level)` - *Test resilience to specific perturbation.*
- `_aggregate_by_dimension` (line 1220) `def _aggregate_by_dimension(self, results)` - *Aggregate resilience scores by perturbation dimension.*
- `_aggregate_by_noise` (line 1231) `def _aggregate_by_noise(self, results)` - *Aggregate resilience scores by noise level.*
- `__init__` (line 1246) `def __init__(self, config)`
- `classify` (line 1249) `def classify(self, alpha, temperature)`
- `__init__` (line 1273) `def __init__(self, arch_config)`
- `migrate` (line 1277) `def migrate(self, raw_data, device)`
- `_migrate_if_needed` (line 1290) `def _migrate_if_needed(self, state_dict, device)` - *Detecta el formato y aplica migración si es necesario.*
- `__init__` (line 1315) `def __init__(self, config, arch_config)`
- `should_save_checkpoint` (line 1322) `def should_save_checkpoint(self)` - *Check if 5 minutes have elapsed since last checkpoint.*
- `save_checkpoint` (line 1328) `def save_checkpoint(self, model, epoch, metrics, loss_history, checkpoint_dir)` - *Save checkpoint with all MBL metrics.*
- `load_checkpoint` (line 1361) `def load_checkpoint(self, path)` - *Load checkpoint with automatic device placement and migration.*
- `__init__` (line 1392) `def __init__(self, config)`
- `collect` (line 1405) `def collect(self, model, loss, epoch, loss_history, step)` - *Collect core metrics for the current training state.*
- `collect_comprehensive` (line 1493) `def collect_comprehensive(self, model, loss, epoch, loss_history, step)` - *Collect comprehensive metrics including expensive calculations.*
- `_classify_quantum_phase` (line 1520) `def _classify_quantum_phase(self, level_spacing, hbar_results)` - *Classify combined quantum phase.*
- `__init__` (line 1545) `def __init__(self, model, arch_config, mbl_config, train_config)`
- `train_step` (line 1571) `def train_step(self, q_batch, p_batch, q_target, p_target)` - *Single training step with Hamiltonian loss.*
- `train_epoch` (line 1603) `def train_epoch(self, dataset, epoch)` - *Train for one epoch with MBL monitoring.*
- `_log_metrics` (line 1658) `def _log_metrics(self, metrics)` - *Log metrics to console in scientific format.*
- `train` (line 1672) `def train(self, dataset, num_epochs)` - *Full training loop.*
- `__init__` (line 1713) `def __init__(self, checkpoint_path, arch_config, mbl_config)`
- `_load_checkpoint` (line 1723) `def _load_checkpoint(self)` - *Load and migrate checkpoint.*
- `analyze` (line 1763) `def analyze(self)` - *Perform complete MBL analysis.*
- `_generate_summary` (line 1787) `def _generate_summary(self, metrics)` - *Generate executive summary.*
- `_print_report` (line 1807) `def _print_report(self, results)` - *Print formatted analysis report.*
- `__init__` (line 1878) `def __init__(self, arch_config, mbl_config)`
- `process_checkpoint` (line 1882) `def process_checkpoint(self, checkpoint_path, output_dir)` - *Process single checkpoint and save results.*
- `process_directory` (line 1901) `def process_directory(self, checkpoint_dir, n_latest, output_dir)` - *Process multiple checkpoints from directory.*
- `generate_summary` (line 1941) `def generate_summary(self, all_results, output_dir)` - *Generate aggregate summary report.*
- `_generate_text_report` (line 1982) `def _generate_text_report(self, summary, output_dir)` - *Generate human-readable text report.*

#### `hpu_view.py`
**Path:** `hpu_view.py`

*No symbols extracted*

#### `mining_seeds.py`
**Path:** `mining_seeds.py`

**Classes:**
- `Config` (line 41) `class Config`
- `IAnalysisStrategy` (line 109) `class IAnalysisStrategy(ABC)`
- `IMetricsCalculator` (line 115) `class IMetricsCalculator(ABC)`
- `HamiltonianOperator` (line 121) `class HamiltonianOperator` - *True Hamiltonian operator H = -nabla^2 on torus.*
- `FastDataset` (line 146) `class FastDataset(Dataset)` - *Fast dataset for Hamiltonian operator learning.*
- `SpectralLayer` (line 203) `class SpectralLayer` - *Spectral layer with correct complex multiplication.*
- `SimpleHamiltonianNet` (line 255) `class SimpleHamiltonianNet` - *Compact network for Hamiltonian operator learning.*
- `LocalComplexityAnalyzer` (line 293) `class LocalComplexityAnalyzer`
- `SuperpositionAnalyzer` (line 310) `class SuperpositionAnalyzer`
- `CrystallographyMetrics` (line 340) `class CrystallographyMetrics`
- `ThermodynamicMetrics` (line 444) `class ThermodynamicMetrics`
- `SpectroscopyMetrics` (line 470) `class SpectroscopyMetrics`
- `CheckpointManager` (line 500) `class CheckpointManager`
- `TrainingMonitor` (line 573) `class TrainingMonitor`
- `GlassStopper` (line 611) `class GlassStopper`
- `BoltzmannAnalysisProgram` (line 879) `class BoltzmannAnalysisProgram`

**Functions:**
- `set_seed` (line 88) `def set_seed(seed)`
- `setup_logger` (line 96) `def setup_logger(name, level)`
- `train_with_early_glass_stop` (line 670) `def train_with_early_glass_stop(model, optimizer, seed, epochs)` - *Train model with early stopping for glass detection.*
- `seed_miner` (line 803) `def seed_miner(total_attempts)` - *Mine for crystal seeds by trying sequential seeds.*
- `main` (line 856) `def main()`
- `analyze` (line 111) `def analyze(self, model)`
- `compute` (line 117) `def compute(self, model)`
- `__init__` (line 124) `def __init__(self, grid_size)`
- `_precompute_spectral_operators` (line 128) `def _precompute_spectral_operators(self)`
- `apply` (line 135) `def apply(self, field)`
- `time_evolution` (line 140) `def time_evolution(self, field, dt)`
- `__init__` (line 149) `def __init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)`
- `__len__` (line 193) `def __len__(self)`
- `__getitem__` (line 196) `def __getitem__(self, idx)`
- `get_val_batch` (line 199) `def get_val_batch(self)`
- `__init__` (line 206) `def __init__(self, channels, grid_size)`
- `forward` (line 219) `def forward(self, x)`
- `__init__` (line 258) `def __init__(self, grid_size, hidden_dim, num_spectral_layers)`
- `forward` (line 279) `def forward(self, x)`
- `compute_local_complexity` (line 295) `def compute_local_complexity(weights, epsilon)` - *Compute Local Complexity (LC) metric for weight matrix.*
- `compute_superposition` (line 312) `def compute_superposition(weights)` - *Compute Superposition (SP) metric for weight matrix.*
- `compute_kappa` (line 342) `def compute_kappa(model, dataloader, num_batches)`
- `compute_discretization_margin` (line 378) `def compute_discretization_margin(coeffs)`
- `compute_alpha_purity` (line 387) `def compute_alpha_purity(coeffs)`
- `compute_kappa_quantum` (line 394) `def compute_kappa_quantum(coeffs, hbar)`
- `compute_poynting_vector` (line 411) `def compute_poynting_vector(coeffs)`
- `compute_all_metrics` (line 426) `def compute_all_metrics(model, dataloader)`
- `compute_effective_temperature` (line 446) `def compute_effective_temperature(gradient_buffer, learning_rate)`
- `compute_specific_heat` (line 460) `def compute_specific_heat(loss_history, temp_history, cv_threshold)`
- `compute_weight_diffraction` (line 472) `def compute_weight_diffraction(coeffs)`
- `_compute_spectral_entropy` (line 491) `def _compute_spectral_entropy(power_spectrum)`
- `__init__` (line 501) `def __init__(self, interval_minutes, max_checkpoints)`
- `should_save_checkpoint` (line 509) `def should_save_checkpoint(self)`
- `save_checkpoint` (line 514) `def save_checkpoint(self, model, optimizer, epoch, metrics)`
- `__init__` (line 574) `def __init__(self)`
- `update_metrics` (line 594) `def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat, poynting_magnitude)`
- `__init__` (line 612) `def __init__(self, patience_epochs)`
- `should_stop` (line 616) `def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)` - *Check if the system is in glass state and should stop mining.*
- `__init__` (line 880) `def __init__(self, checkpoint_path, results_dir)`
- `load_and_analyze_checkpoint` (line 886) `def load_and_analyze_checkpoint(self)`
- `dataloader` (line 903) `def dataloader()`

#### `plank.py`
**Path:** `plank.py`

**Classes:**
- `HBarCalculator` (line 26) `class HBarCalculator` - *Calcula ħ efectiva desde checkpoint HPU usando física realista.*

**Functions:**
- `main` (line 213) `def main()`
- `__init__` (line 29) `def __init__(self, checkpoint_path, device)`
- `calculate_all` (line 54) `def calculate_all(self)` - *Ejecuta todos los cálculos de ħ.*
- `print_report` (line 170) `def print_report(self, results)` - *Imprime reporte formateado.*

#### `polos.py`
**Path:** `polos.py`

**Classes:**
- `ControlConfig` (line 27) `class ControlConfig`
- `TransferFunctionExtractor` (line 48) `class TransferFunctionExtractor`
- `PoleZeroAnalyzer` (line 142) `class PoleZeroAnalyzer`
- `FrequencyResponseAnalyzer` (line 299) `class FrequencyResponseAnalyzer`
- `TimeResponseAnalyzer` (line 415) `class TimeResponseAnalyzer`
- `ControllerDesigner` (line 516) `class ControllerDesigner`
- `ControlSystemAnalyzer` (line 599) `class ControlSystemAnalyzer`
- `ControlVisualizer` (line 798) `class ControlVisualizer`

**Functions:**
- `analyze_checkpoint` (line 1152) `def analyze_checkpoint(checkpoint_path, output_dir)`
- `analyze_multiple_checkpoints` (line 1208) `def analyze_multiple_checkpoints(checkpoint_dir, n_latest, output_dir)`
- `main` (line 1240) `def main()`
- `__init__` (line 50) `def __init__(self, model, device)`
- `extract_state_space_representation` (line 55) `def extract_state_space_representation(self)`
- `compute_transfer_function` (line 105) `def compute_transfer_function(self, A, B, C, D)`
- `__init__` (line 144) `def __init__(self, numerator, denominator)`
- `_compute_poles_zeros` (line 153) `def _compute_poles_zeros(self)`
- `analyze_stability` (line 166) `def analyze_stability(self)`
- `classify_poles` (line 207) `def classify_poles(self)`
- `compute_damping_frequency` (line 238) `def compute_damping_frequency(self)`
- `compute_time_constants` (line 278) `def compute_time_constants(self)`
- `__init__` (line 301) `def __init__(self, numerator, denominator)`
- `compute_bode_plot_data` (line 309) `def compute_bode_plot_data(self)`
- `compute_gain_phase_margins` (line 328) `def compute_gain_phase_margins(self)`
- `compute_nyquist_data` (line 357) `def compute_nyquist_data(self)`
- `evaluate_nyquist_stability` (line 379) `def evaluate_nyquist_stability(self, nyquist_data)`
- `__init__` (line 417) `def __init__(self, numerator, denominator)`
- `compute_step_response` (line 425) `def compute_step_response(self)`
- `compute_impulse_response` (line 441) `def compute_impulse_response(self)`
- `analyze_step_response_characteristics` (line 457) `def analyze_step_response_characteristics(self, step_data)`
- `__init__` (line 518) `def __init__(self, poles, zeros)`
- `design_pid_controller` (line 522) `def design_pid_controller(self, desired_damping, desired_settling_time)`
- `design_lead_compensator` (line 542) `def design_lead_compensator(self, desired_phase_margin)`
- `compute_root_locus` (line 572) `def compute_root_locus(self, num, den)`
- `__init__` (line 601) `def __init__(self, checkpoint_path, device)`
- `analyze_complete_system` (line 625) `def analyze_complete_system(self)`
- `_print_report` (line 715) `def _print_report(self, results)`
- `plot_pole_zero_map` (line 801) `def plot_pole_zero_map(poles, zeros, output_path)`
- `plot_bode_diagram` (line 865) `def plot_bode_diagram(bode_data, margins, output_path)`
- `plot_nyquist_diagram` (line 940) `def plot_nyquist_diagram(nyquist_data, output_path)`
- `plot_time_responses` (line 990) `def plot_time_responses(step_data, impulse_data, output_path)`
- `plot_root_locus` (line 1036) `def plot_root_locus(root_locus_data, output_path)`
- `plot_combined_analysis` (line 1096) `def plot_combined_analysis(poles, zeros, bode_data, step_data, output_path)`

#### `precision.py`
**Path:** `precision.py`

**Classes:**
- `MassiveLambdaConfig` (line 26) `class MassiveLambdaConfig`
- `CrystallizationLossMassive` (line 36) `class CrystallizationLossMassive`
- `ContinuationEngine` (line 72) `class ContinuationEngine`

**Functions:**
- `main` (line 506) `def main()`
- `__init__` (line 37) `def __init__(self, lambda_quant)`
- `quantization_penalty` (line 42) `def quantization_penalty(self, model)`
- `forward` (line 54) `def forward(self, predictions, targets, model)`
- `__init__` (line 73) `def __init__(self, checkpoint_path, device)`
- `_setup_logger` (line 150) `def _setup_logger(self)`
- `_find_latest_checkpoint` (line 162) `def _find_latest_checkpoint(self)`
- `_compute_initial_metrics` (line 191) `def _compute_initial_metrics(self, model)`
- `compute_discretization_metrics` (line 208) `def compute_discretization_metrics(self)`
- `validate` (line 241) `def validate(self)`
- `train_epoch` (line 250) `def train_epoch(self, epoch)`
- `refine` (line 288) `def refine(self)`
- `_save_latest_checkpoint` (line 430) `def _save_latest_checkpoint(self, epoch, metrics, val_acc)` - *Guarda/sobrescribe latest.pth - rápido, para danger zone*
- `_save_crystal_checkpoint` (line 456) `def _save_crystal_checkpoint(self, epoch, metrics, val_acc, final, force_save, emergency)`
- `_compile_results` (line 490) `def _compile_results(self, success, final_epoch)`

#### `refinamiento.py`
**Path:** `refinamiento.py`

**Classes:**
- `CrystallizationConfig` (line 33) `class CrystallizationConfig` - *Configuración agresiva para forzar discretización*
- `CrystallizationLoss` (line 57) `class CrystallizationLoss` - *Pérdida combinada: MSE + penalización de cuantización
Fuerza los pesos a caer en {-1, 0, 1}*
- `StructuralPruner` (line 96) `class StructuralPruner` - *Implementa poda progresiva de pesos pequeños*
- `CrystallizationEngine` (line 144) `class CrystallizationEngine` - *Motor de refinamiento que carga un checkpoint y fuerza discretización*

**Functions:**
- `analyze_discretization` (line 498) `def analyze_discretization(checkpoint_path)` - *Análisis detallado de la discretización de un checkpoint*
- `main` (line 575) `def main()`
- `__init__` (line 62) `def __init__(self, lambda_quant)`
- `quantization_penalty` (line 67) `def quantization_penalty(self, model)` - *Penalización L2 de la distancia al entero más cercano*
- `forward` (line 81) `def forward(self, predictions, targets, model)`
- `__init__` (line 98) `def __init__(self, thresholds)`
- `should_prune` (line 103) `def should_prune(self, epoch)` - *Determina si es momento de podar (cada 500 épocas)*
- `prune` (line 107) `def prune(self, model, force_threshold)` - *Poda pesos con |w| < threshold
Retorna número de parámetros podados*
- `get_sparsity` (line 131) `def get_sparsity(self, model)` - *Calcula porcentaje de pesos exactamente en cero*
- `__init__` (line 148) `def __init__(self, checkpoint_path, device)`
- `_setup_logger` (line 186) `def _setup_logger(self)`
- `_load_checkpoint` (line 198) `def _load_checkpoint(self)` - *Carga el checkpoint y retorna modelo, época y métricas*
- `_compute_initial_metrics` (line 234) `def _compute_initial_metrics(self, model)` - *Calcula métricas iniciales si no vienen en el checkpoint*
- `compute_discretization_metrics` (line 253) `def compute_discretization_metrics(self)` - *Calcula métricas de cristalinidad actuales*
- `validate` (line 291) `def validate(self)` - *Valida el modelo manteniendo accuracy*
- `train_epoch` (line 302) `def train_epoch(self, epoch)` - *Entrena una época con pérdida de cuantización*
- `refine` (line 344) `def refine(self)` - *Ejecuta el refinamiento hasta alcanzar δ < TARGET_DELTA o MAX_EPOCHS*
- `_save_crystal_checkpoint` (line 459) `def _save_crystal_checkpoint(self, epoch, metrics, val_acc, final)` - *Guarda checkpoint cristalino*
- `_compile_results` (line 482) `def _compile_results(self, success, final_epoch)` - *Compila resultados finales*

#### `simple_hpu_view.py`
**Path:** `simple_hpu_view.py`

*No symbols extracted*

#### `test_grokkit.py`
**Path:** `test_grokkit.py`

**Classes:**
- `GrokkingValidator` (line 41) `class GrokkingValidator` - *Validates grokking phenomenon in Hamiltonian operator learning.

Implements Theorem 1.1 requirements:
1. Spectral convergence to true H operator
2. Operator kernel representation in weights
3. Phase transition from memorization to generalization

This class implements a battery of tests to confirm that the trained
model has successfully transitioned from the memorization phase to
the generalization phase, exhibiting the characteristic properties
of spectral convergence as predicted by Theorem 1.1.*

**Functions:**
- `run_quick_test` (line 507) `def run_quick_test()` - *Executes a quick validation test with minimal output.

This function provides a streamlined testing interface for
rapid verification of model performance.*
- `__init__` (line 56) `def __init__(self, weights_dir)`
- `load_model` (line 64) `def load_model(self)` - *Loads the trained model from checkpoint.

Returns:
    Tuple of (model, checkpoint)
    
Raises:
    FileNotFoundError: If no checkpoint exists in weights directory.*
- `generate_test_dataset` (line 119) `def generate_test_dataset(self, num_samples)` - *Generates test dataset using the true Hamiltonian operator.

Creates random initial fields and evolves them under H to produce
ground truth targets for validation.

Args:
    num_samples: Number of test samples
    
Returns:
    Tuple of (inputs, targets) tensors*
- `compute_local_complexity` (line 158) `def compute_local_complexity(self, model)` - *Computes Local Complexity (LC) metric for the model.

LC measures the effective dimensionality of the model's
learned representations. High LC indicates diverse, independent
feature utilization - a key indicator of operator learning.

Args:
    model: Neural network model
    
Returns:
    LC value in [0, 1] range*
- `compute_superposition` (line 181) `def compute_superposition(self, model)` - *Computes Superposition (SP) metric for the model.

SP measures the correlation between weight vectors.
Low SP indicates orthogonal, non-redundant representations.

Args:
    model: Neural network model
    
Returns:
    SP value in [0, 1] range*
- `compute_operator_error` (line 203) `def compute_operator_error(self, model, inputs, targets)` - *Computes operator approximation error.

Measures how well the learned model approximates the true
Hamiltonian operator on held-out test data.

Args:
    model: Trained model
    inputs: Test input fields
    targets: True evolved fields under H
    
Returns:
    Mean squared error between prediction and target*
- `compute_spectral_gap` (line 228) `def compute_spectral_gap(self, model)` - *Estimates the spectral gap in weight singular values.

The spectral gap provides insight into the model's capacity
utilization and the degree of weight superposition.

Args:
    model: Neural network model
    
Returns:
    Ratio of largest to smallest non-zero singular value*
- `run_validation` (line 259) `def run_validation(self)` - *Executes the complete validation suite.

Runs all tests and aggregates results into a comprehensive
report documenting the grokking phenomenon characteristics
as predicted by Theorem 1.1.

Returns:
    Dictionary containing all test results and metrics*
- `generate_report` (line 382) `def generate_report(self)` - *Generates a formal validation report.

Returns:
    Markdown formatted validation report*

#### `verify.py`
**Path:** `verify.py`

**Classes:**
- `CheckpointVerifier` (line 14) `class CheckpointVerifier`

**Functions:**
- `verify_latest_checkpoints` (line 444) `def verify_latest_checkpoints(checkpoint_dir, n)` - *Verifica los N checkpoints más recientes*
- `main` (line 486) `def main()`
- `__init__` (line 15) `def __init__(self, checkpoint_path, device)`
- `verify_all_metrics` (line 50) `def verify_all_metrics(self)` - *Calcula TODAS las métricas desde cero y compara con las guardadas*
- `_check_weight_integrity` (line 101) `def _check_weight_integrity(self)` - *Verifica que los pesos no tengan NaN/Inf*
- `_compute_validation_metrics` (line 146) `def _compute_validation_metrics(self)` - *Calcula MSE y accuracy de validación desde cero*
- `_compute_discretization_metrics` (line 173) `def _compute_discretization_metrics(self)` - *Calcula delta, alpha, purity, etc.*
- `_compute_quantization_metrics` (line 225) `def _compute_quantization_metrics(self)` - *Calcula la penalización de cuantización*
- `_compute_loss_metrics` (line 244) `def _compute_loss_metrics(self)` - *Reconstruye el loss total*
- `_compare_with_stored` (line 269) `def _compare_with_stored(self, computed)` - *Compara métricas calculadas vs almacenadas en el checkpoint*
- `_check_internal_consistency` (line 302) `def _check_internal_consistency(self, results)` - *Verifica consistencia entre métricas relacionadas*
- `_compute_health_score` (line 331) `def _compute_health_score(self, results)` - *Calcula un score de salud del checkpoint (0-100)*
- `_print_report` (line 374) `def _print_report(self, results)` - *Imprime reporte formateado*

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
