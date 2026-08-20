# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM, Ruby, Swift, Kotlin, Scala, Lua, Elixir.
> No LLMs. No tokens. Pure static analysis. See more [here](https://github.com/grisuno/ReadMenator)

**Total Files Parsed:** 32 | **Total Symbols Extracted:** 776 | **Total Imports:** 345
 | **Resolved Imports:** 33

<!-- ranking_model: v1.0 | weights: {ppr:0.45,auth:0.2,test:0.15,doc:0.1,fresh:0.1} | alpha:0.85 | commit:75d209c | date:2026-07-18 -->


## Table of Contents

1. [Statistics Dashboard](#statistics-dashboard)
2. [Architectural Layers](#architectural-layers)
3. [Ranked Context](#ranked-context)
4. [God Nodes](#god-nodes)
5. [Community Analysis](#community-analysis)
6. [Suggested Questions](#suggested-questions)
7. [Hotspot Analysis](#hotspot-analysis)
8. [Change Impact Analysis](#change-impact-analysis)
9. [Suggested Linting Rules](#suggested-linting-rules)
10. [Orphans](#orphans)
11. [Query Recipes](#query-recipes)
12. [Structural Knowledge Map](#structural-knowledge-map)
13. [UML Class Diagram](#uml-class-diagram)
14. [Code Property Graph](#code-property-graph)
15. [Architecture Reference](#architecture-reference)
    - [PY (31 files)](#py-31-files)
    - [SH (1 files)](#sh-1-files)

---

## Statistics Dashboard

| Metric | Value |
|--------|-------|
| Total Files | 32 |
| Total Symbols | 776 |
| Total Imports | 345 |
| Call Edges | 6360 |
| Inheritance Edges | 47 |
| Languages | 2 |
| Avg Symbols/File | 24.2 |
| Avg Imports/File | 10.8 |
| Resolved Imports | 33 |

### Top Files by Import Count (Fan-Out)

| File | Imports | Symbols | Language |
|------|---------|---------|----------|
| `experiment.py` | 24 | 57 | py |
| `mining_seeds.py` | 24 | 57 | py |
| `hamiltonian_mbl.py` | 20 | 126 | py |
| `get_meditions.py` | 19 | 52 | py |
| `polos.py` | 19 | 42 | py |
| `audios.py` | 18 | 47 | py |
| `experiment2.py` | 17 | 83 | py |
| `dirac.py` | 17 | 19 | py |
| `experiment2.py` | 17 | 83 | py |
| `precision.py` | 15 | 18 | py |

---

## Architectural Layers

Auto-detected from path patterns, naming conventions, and imported frameworks.

| Layer | Files |
|-------|-------|
| infrastructure | 14 |
| utility | 13 |
| presentation | 2 |
| business_logic | 1 |
| data_access | 1 |
| testing | 1 |

### utility

- `app.py` (py, 22 symbols)
- `check_fase_berry.py` (py, 0 symbols)
- `expand.py` (py, 5 symbols)
- `experiment.py` (py, 57 symbols)
- `experiment2.py` (py, 83 symbols)
- `export.py` (py, 0 symbols)
- `hamiltonian_mbl.py` (py, 126 symbols)
- `install.sh` (sh, 0 symbols)
- `plank.py` (py, 5 symbols)
- `polos.py` (py, 42 symbols)
- `precision.py` (py, 18 symbols)
- `refinamiento.py` (py, 23 symbols)
- `verify.py` (py, 14 symbols)

### infrastructure

- `audio_io.py` (py, 12 symbols)
- `audios.py` (py, 47 symbols)
- `checkpoint_manager.py` (py, 6 symbols)
- `config.py` (py, 13 symbols)
- `experiment2.py` (py, 83 symbols)
- `inference.py` (py, 7 symbols)
- `losses.py` (py, 11 symbols)
- `main.py` (py, 7 symbols)
- `metrics.py` (py, 25 symbols)
- `trainer.py` (py, 11 symbols)
- `visualization.py` (py, 8 symbols)
- `diff_weights.py` (py, 1 symbols)
- `dirac.py` (py, 19 symbols)
- `get_meditions.py` (py, 52 symbols)

### business_logic

- `model.py` (py, 11 symbols)

### presentation

- `hpu_view.py` (py, 0 symbols)
- `simple_hpu_view.py` (py, 0 symbols)

### data_access

- `mining_seeds.py` (py, 57 symbols)

### testing

- `test_grokkit.py` (py, 11 symbols)

---

## Ranked Context

Files ranked by composite score for the current query context. The ranking combines Personalized PageRank (query relevance), global authority, test coverage, documentation coverage, and code freshness. Model: v1.0.

| Rank | File | Composite | PPR | Authority | Test | Doc |
|------|------|-----------|-----|-----------|------|-----|
| 1 | `config.py` | 0.2754 | 0.3509 | 0.2030 | 0.00 | 0.77 |
| 2 | `experiment2.py` | 0.1897 | 0.2982 | 0.1960 | 0.01 | 0.14 |
| 3 | `polos.py` | 0.1660 | 0.3509 | 0.0228 | 0.02 | 0.00 |
| 4 | `main.py` | 0.1299 | 0.0000 | 0.0422 | 0.14 | 1.00 |
| 5 | `get_meditions.py` | 0.0998 | 0.0000 | 0.0228 | 0.02 | 0.92 |
| 6 | `metrics.py` | 0.0985 | 0.0000 | 0.0327 | 0.00 | 0.92 |
| 7 | `audio_io.py` | 0.0982 | 0.0000 | 0.0327 | 0.00 | 0.92 |
| 8 | `audios.py` | 0.0971 | 0.0000 | 0.0228 | 0.02 | 0.89 |
| 9 | `losses.py` | 0.0965 | 0.0000 | 0.0278 | 0.00 | 0.91 |
| 10 | `test_grokkit.py` | 0.0955 | 0.0000 | 0.0228 | 0.00 | 0.91 |

**Query anchors:** audio/config.py, hamiltonian_mbl.py, polos.py

**Top result justification paths:**

  `config.py`

---

## God Nodes

Most architecturally central files ranked by combined import/export degree and symbol richness.

| File | Score | Connections | PageRank |
|------|-------|-------------|----------|
| `experiment2.py` | 26.3 | | 0.1960 |
| `config.py` | 19.3 | | 0.2030 |
| `trainer.py` | 15.1 | | 0.0000 |
| `inference.py` | 14.7 | | 0.0000 |
| `hamiltonian_mbl.py` | 12.6 | | 0.0000 |
| `experiment2.py` | 10.3 | | 0.0000 |
| `main.py` | 8.7 | | 0.0422 |
| `metrics.py` | 8.5 | | 0.0327 |
| `audio_io.py` | 7.2 | | 0.0327 |
| `get_meditions.py` | 7.2 | | 0.0228 |

---

## Community Analysis

Files grouped by import-based community detection. Cohesion measures how tightly connected each community is internally.

### audio (Cohesion: 1.00)

**11 files** in this community:

- `audio_io.py` (py, 12 symbols)
- `checkpoint_manager.py` (py, 6 symbols)
- `config.py` (py, 13 symbols)
- `inference.py` (py, 7 symbols)
- `losses.py` (py, 11 symbols)
- `main.py` (py, 7 symbols)
- `metrics.py` (py, 25 symbols)
- `model.py` (py, 11 symbols)
- `trainer.py` (py, 11 symbols)
- `visualization.py` (py, 8 symbols)
- `test_grokkit.py` (py, 11 symbols)

### audio (Cohesion: 1.00)

**2 files** in this community:

- `audios.py` (py, 47 symbols)
- `experiment2.py` (py, 83 symbols)

### root (Cohesion: 1.00)

**10 files** in this community:

- `check_fase_berry.py` (py, 0 symbols)
- `dirac.py` (py, 19 symbols)
- `experiment2.py` (py, 83 symbols)
- `get_meditions.py` (py, 52 symbols)
- `hpu_view.py` (py, 0 symbols)
- `polos.py` (py, 42 symbols)
- `precision.py` (py, 18 symbols)
- `refinamiento.py` (py, 23 symbols)
- `simple_hpu_view.py` (py, 0 symbols)
- `verify.py` (py, 14 symbols)

---

## Suggested Questions

Auto-generated exploration prompts based on graph structure:

- What does experiment2.py depend on, and what depends on it? (9 connections)
- What does config.py depend on, and what depends on it? (9 connections)
- What does trainer.py depend on, and what depends on it? (7 connections)
- How are the 11 files in 'audio' related to each other?
- What is SimpleConfig in app.py and how is it used?

---

## Hotspot Analysis

Files ranked by combined complexity (symbol count) and centrality (connection count). High-scoring files are architecturally critical and may need refactoring attention.

| File | Complexity | Centrality | Combined | Symbols | Connections |
|------|-----------|------------|----------|---------|-------------|
| `config.py` | 0.103 | 0.462 | 0.318 | 13 | 12 |
| `experiment2.py` | 0.659 | 1.000 | 0.864 | 83 | 26 |
| `polos.py` | 0.333 | 0.769 | 0.595 | 42 | 20 |
| `main.py` | 0.056 | 0.385 | 0.253 | 7 | 10 |
| `get_meditions.py` | 0.413 | 0.769 | 0.627 | 52 | 20 |
| `metrics.py` | 0.198 | 0.308 | 0.264 | 25 | 8 |
| `audio_io.py` | 0.095 | 0.308 | 0.223 | 12 | 8 |
| `audios.py` | 0.373 | 0.731 | 0.588 | 47 | 19 |
| `losses.py` | 0.087 | 0.269 | 0.197 | 11 | 7 |
| `test_grokkit.py` | 0.087 | 0.385 | 0.266 | 11 | 10 |
| `hamiltonian_mbl.py` | 1.000 | 0.769 | 0.862 | 126 | 20 |
| `experiment.py` | 0.452 | 0.923 | 0.735 | 57 | 24 |
| `mining_seeds.py` | 0.452 | 0.923 | 0.735 | 57 | 24 |
| `experiment2.py` | 0.659 | 0.692 | 0.679 | 83 | 18 |
| `trainer.py` | 0.087 | 0.808 | 0.519 | 11 | 21 |

---

## Change Impact Analysis

Files sorted by how many other files would be affected if they changed. High-impact files should be changed with caution.

| File | Direct Dependents | Transitive Dependents | Total Impact |
|------|------------------|----------------------|--------------|
| `config.py` | 9 | 1 | 10 |
| `experiment2.py` | 9 | 0 | 9 |
| `audio_io.py` | 2 | 2 | 4 |
| `checkpoint_manager.py` | 2 | 2 | 4 |
| `metrics.py` | 2 | 2 | 4 |
| `model.py` | 2 | 2 | 4 |
| `losses.py` | 1 | 2 | 3 |
| `visualization.py` | 1 | 2 | 3 |
| `inference.py` | 1 | 1 | 2 |
| `trainer.py` | 1 | 1 | 2 |
| `experiment2.py` | 1 | 0 | 1 |
| `main.py` | 1 | 0 | 1 |
| `refinamiento.py` | 1 | 0 | 1 |
| `app.py` | 0 | 0 | 0 |
| `audios.py` | 0 | 0 | 0 |

---

## Suggested Linting Rules

Automatically suggested linting and security rules based on patterns detected in the codebase. These can be exported as Semgrep rules using the `--export-rules` flag.

| Rule ID | Severity | Description | Language | Matches |
|---------|----------|-------------|----------|---------|
| `RM002` | warning | Bare except clause catches all exceptions including SystemExit | python | 10 |
| `RM001` | info | Large number of functions in py: 610 total | py | 610 |
| `RM003` | info | Print statement found (consider logging instead) | python | 613 |

---

## Orphans

Files with no documentation or low connectivity. These are candidates for documentation investment or cleanup.

- `polos.py` (42 symbols, no doc)
- `check_fase_berry.py` (0 symbols, no doc)
- `diff_weights.py` (1 symbols, no doc)
- `dirac.py` (19 symbols, no doc)
- `export.py` (0 symbols, no doc)
- `hpu_view.py` (0 symbols, no doc)
- `install.sh` (0 symbols, no doc)
- `simple_hpu_view.py` (0 symbols, no doc)

---

## Query Recipes

Example queries you can run against this knowledge base using the ranking engine:

```
# Find files most relevant to a concept
readmenator query "Where is the import resolver implemented?"

# Rank files by relevance to a topic
readmenator query "How does documentation generation work?"

# Explain why a file ranks highly
readmenator query "explain readmenator/_documentation.py"

# Trace dependency paths with ranked context
readmenator query "path from CLI to exporter"
```

The ranking model uses the following signals:

- **Personalized PageRank** (45% weight): query-specific relevance via seed propagation
- **Global Authority** (20% weight): structural importance via standard PageRank
- **Test Coverage** (15% weight): fraction of symbols referenced in test files
- **Doc Coverage** (10% weight): presence of docstrings and file-level docs
- **Freshness** (10% weight): recent modification activity

Results include score decomposition and justification paths for each ranked item.

---

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
    hamiltonian_mbl_py["hamiltonian_mbl.py (py)"]
    class hamiltonian_mbl_py mod;
    subgraph community_2 ["root"]
    get_meditions_py["get_meditions.py (py)"]
    class get_meditions_py mod;
    polos_py["polos.py (py)"]
    class polos_py mod;
    end
    subgraph community_0 ["audio"]
    audio_trainer_py["trainer.py (py)"]
    class audio_trainer_py mod;
    end
    subgraph community_1 ["audio"]
    audio_audios_py["audios.py (py)"]
    class audio_audios_py mod;
    dirac_py["dirac.py (py)"]
    class dirac_py mod;
    audio_experiment2_py["experiment2.py (py)"]
    class audio_experiment2_py mod;
    experiment2_py["experiment2.py (py)"]
    class experiment2_py mod;
    precision_py["precision.py (py)"]
    class precision_py mod;
    refinamiento_py["refinamiento.py (py)"]
    class refinamiento_py mod;
    audio_inference_py["inference.py (py)"]
    class audio_inference_py mod;
    verify_py["verify.py (py)"]
    class verify_py mod;
    app_py["app.py (py)"]
    class app_py mod;
    test_grokkit_py["test_grokkit.py (py)"]
    class test_grokkit_py mod;
    expand_py["expand.py (py)"]
    class expand_py mod;
    audio_visualization_py["visualization.py (py)"]
    class audio_visualization_py mod;
    audio_main_py["main.py (py)"]
    class audio_main_py mod;
    audio_checkpoint_manager_py["checkpoint_manager.py (py)"]
    class audio_checkpoint_manager_py mod;
    check_fase_berry_py["check_fase_berry.py (py)"]
    class check_fase_berry_py mod;
    hpu_view_py["hpu_view.py (py)"]
    class hpu_view_py mod;
    simple_hpu_view_py["simple_hpu_view.py (py)"]
    class simple_hpu_view_py mod;
    plank_py["plank.py (py)"]
    class plank_py mod;
    audio_metrics_py["metrics.py (py)"]
    class audio_metrics_py mod;
    audio_audio_io_py["audio_io.py (py)"]
    class audio_audio_io_py mod;
    audio_losses_py["losses.py (py)"]
    class audio_losses_py mod;
    audio_model_py["model.py (py)"]
    class audio_model_py mod;
    audio_config_py["config.py (py)"]
    class audio_config_py mod;
    diff_weights_py["diff_weights.py (py)"]
    class diff_weights_py mod;
    export_py["export.py (py)"]
    class export_py mod;
    install_sh["install.sh (sh)"]
    class install_sh mod;
    end
    audio_audio_io_py -- resolved_imports --> audio_config_py
    audio_audios_py -- resolved_imports --> audio_experiment2_py
    audio_checkpoint_manager_py -- resolved_imports --> audio_config_py
    audio_inference_py -- resolved_imports --> audio_config_py
    audio_inference_py -- resolved_imports --> audio_model_py
    audio_inference_py -- resolved_imports --> audio_audio_io_py
    audio_inference_py -- resolved_imports --> audio_visualization_py
    audio_inference_py -- resolved_imports --> audio_metrics_py
    audio_inference_py -- resolved_imports --> audio_checkpoint_manager_py
    audio_losses_py -- resolved_imports --> audio_config_py
    audio_main_py -- resolved_imports --> audio_config_py
    audio_main_py -- resolved_imports --> audio_trainer_py
    audio_main_py -- resolved_imports --> audio_inference_py
    audio_metrics_py -- resolved_imports --> audio_config_py
    audio_model_py -- resolved_imports --> audio_config_py
    audio_trainer_py -- resolved_imports --> audio_config_py
    audio_trainer_py -- resolved_imports --> audio_model_py
    audio_trainer_py -- resolved_imports --> audio_audio_io_py
    audio_trainer_py -- resolved_imports --> audio_losses_py
    audio_trainer_py -- resolved_imports --> audio_metrics_py
    audio_trainer_py -- resolved_imports --> audio_checkpoint_manager_py
    audio_visualization_py -- resolved_imports --> audio_config_py
    check_fase_berry_py -- resolved_imports --> experiment2_py
    dirac_py -- resolved_imports --> experiment2_py
    get_meditions_py -- resolved_imports --> experiment2_py
    hpu_view_py -- resolved_imports --> experiment2_py
    polos_py -- resolved_imports --> experiment2_py
    precision_py -- resolved_imports --> experiment2_py
    precision_py -- resolved_imports --> refinamiento_py
    refinamiento_py -- resolved_imports --> experiment2_py
    simple_hpu_view_py -- resolved_imports --> experiment2_py
    test_grokkit_py -- resolved_imports --> audio_main_py
    verify_py -- resolved_imports --> experiment2_py
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

## UML Class Diagram

Auto-generated Mermaid class diagram from parsed class-level symbols. Shows classes, structs, interfaces, traits, and their methods with inheritance and dependency relationships.

```mermaid
classDiagram
  class app_py_SimpleConfig {
    <<class>>
    +compute_local_complexity(weights, epsilon)
    +compute_superposition(weights)
    +train_model(grid_size, epochs, hidden_dim, num_spectral_layers, lr)
    +main()
    +__init__(self, grid_size, hidden_dim, num_spectral_layers, target_accuracy, learning_rate)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)
  }
  class app_py_HamiltonianOperator {
    <<class>>
    +compute_local_complexity(weights, epsilon)
    +compute_superposition(weights)
    +train_model(grid_size, epochs, hidden_dim, num_spectral_layers, lr)
    +main()
    +__init__(self, grid_size, hidden_dim, num_spectral_layers, target_accuracy, learning_rate)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)
  }
  class app_py_FastDataset {
    <<class>>
    +compute_local_complexity(weights, epsilon)
    +compute_superposition(weights)
    +train_model(grid_size, epochs, hidden_dim, num_spectral_layers, lr)
    +main()
    +__init__(self, grid_size, hidden_dim, num_spectral_layers, target_accuracy, learning_rate)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)
  }
  class app_py_SpectralLayer {
    <<class>>
    +compute_local_complexity(weights, epsilon)
    +compute_superposition(weights)
    +train_model(grid_size, epochs, hidden_dim, num_spectral_layers, lr)
    +main()
    +__init__(self, grid_size, hidden_dim, num_spectral_layers, target_accuracy, learning_rate)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)
  }
  class app_py_SimpleHamiltonianNet {
    <<class>>
    +compute_local_complexity(weights, epsilon)
    +compute_superposition(weights)
    +train_model(grid_size, epochs, hidden_dim, num_spectral_layers, lr)
    +main()
    +__init__(self, grid_size, hidden_dim, num_spectral_layers, target_accuracy, learning_rate)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)
  }
  class audio_io_py_AudioProcessor {
    <<class>>
    +__init__(self, config, device)
    +load_audio(self, file_path)
    +waveform_to_stft_complex(self, waveform)
    +stft_complex_to_waveform(self, stft_complex)
    +stft_to_magnitude_phase(self, stft_complex)
    +magnitude_phase_to_stft(self, magnitude, phase)
    +stft_magnitude_to_model_input(self, magnitude)
    +model_output_to_stft_magnitude(self, model_output, original_magnitude)
    +waveform_to_mel_spectrogram(self, waveform)
    +save_audio(self, waveform, file_path, sample_rate)
  }
  class audios_py_HamiltonianConfig {
    <<class>>
    +main()
    +segment_samples(self)
    +freq_bins(self)
    +read_segment(self)
    +get_properties(self)
    +close(self)
    +evolve(self, field_state)
    +record(self, metrics)
    +get_summary(self)
    +resample(audio, orig_sr, target_sr)
  }
  class audios_py_IAudioSource {
    <<class>>
    +main()
    +segment_samples(self)
    +freq_bins(self)
    +read_segment(self)
    +get_properties(self)
    +close(self)
    +evolve(self, field_state)
    +record(self, metrics)
    +get_summary(self)
    +resample(audio, orig_sr, target_sr)
  }
  class audios_py_IFieldOperator {
    <<class>>
    +main()
    +segment_samples(self)
    +freq_bins(self)
    +read_segment(self)
    +get_properties(self)
    +close(self)
    +evolve(self, field_state)
    +record(self, metrics)
    +get_summary(self)
    +resample(audio, orig_sr, target_sr)
  }
  class audios_py_IMetricCollector {
    <<class>>
    +main()
    +segment_samples(self)
    +freq_bins(self)
    +read_segment(self)
    +get_properties(self)
    +close(self)
    +evolve(self, field_state)
    +record(self, metrics)
    +get_summary(self)
    +resample(audio, orig_sr, target_sr)
  }
  class audios_py_AudioResampler {
    <<class>>
    +main()
    +segment_samples(self)
    +freq_bins(self)
    +read_segment(self)
    +get_properties(self)
    +close(self)
    +evolve(self, field_state)
    +record(self, metrics)
    +get_summary(self)
    +resample(audio, orig_sr, target_sr)
  }
  class audios_py_WaveFileSource {
    <<class>>
    +main()
    +segment_samples(self)
    +freq_bins(self)
    +read_segment(self)
    +get_properties(self)
    +close(self)
    +evolve(self, field_state)
    +record(self, metrics)
    +get_summary(self)
    +resample(audio, orig_sr, target_sr)
  }
  class audios_py_ComprehensiveMetricCollector {
    <<class>>
    +main()
    +segment_samples(self)
    +freq_bins(self)
    +read_segment(self)
    +get_properties(self)
    +close(self)
    +evolve(self, field_state)
    +record(self, metrics)
    +get_summary(self)
    +resample(audio, orig_sr, target_sr)
  }
  class audios_py_CheckpointManager {
    <<class>>
    +main()
    +segment_samples(self)
    +freq_bins(self)
    +read_segment(self)
    +get_properties(self)
    +close(self)
    +evolve(self, field_state)
    +record(self, metrics)
    +get_summary(self)
    +resample(audio, orig_sr, target_sr)
  }
  class audios_py_AudioSpectrogramConverter {
    <<class>>
    +main()
    +segment_samples(self)
    +freq_bins(self)
    +read_segment(self)
    +get_properties(self)
    +close(self)
    +evolve(self, field_state)
    +record(self, metrics)
    +get_summary(self)
    +resample(audio, orig_sr, target_sr)
  }
  class audios_py_HamiltonianAudioProcessor {
    <<class>>
    +main()
    +segment_samples(self)
    +freq_bins(self)
    +read_segment(self)
    +get_properties(self)
    +close(self)
    +evolve(self, field_state)
    +record(self, metrics)
    +get_summary(self)
    +resample(audio, orig_sr, target_sr)
  }
  class checkpoint_manager_py_CheckpointManager {
    <<class>>
    +__init__(self, config)
    +should_save_checkpoint(self)
    +save_checkpoint(self, model, optimizer, scheduler, epoch, step, metrics, current_loss)
    +load_checkpoint(self, model, load_best)
    +best_loss(self)
  }
  class config_py_AudioProcessingConfig {
    <<class>>
    +validate(self)
    +checkpoint_path(self)
    +best_model_path(self)
    +metadata_path(self)
    +validate_all(self)
    +ensure_directories(self)
  }
  class config_py_ModelArchitectureConfig {
    <<class>>
    +validate(self)
    +checkpoint_path(self)
    +best_model_path(self)
    +metadata_path(self)
    +validate_all(self)
    +ensure_directories(self)
  }
  class config_py_TrainingConfig {
    <<class>>
    +validate(self)
    +checkpoint_path(self)
    +best_model_path(self)
    +metadata_path(self)
    +validate_all(self)
    +ensure_directories(self)
  }
  class config_py_CheckpointConfig {
    <<class>>
    +validate(self)
    +checkpoint_path(self)
    +best_model_path(self)
    +metadata_path(self)
    +validate_all(self)
    +ensure_directories(self)
  }
  class config_py_VisualizationConfig {
    <<class>>
    +validate(self)
    +checkpoint_path(self)
    +best_model_path(self)
    +metadata_path(self)
    +validate_all(self)
    +ensure_directories(self)
  }
  class config_py_MetricsConfig {
    <<class>>
    +validate(self)
    +checkpoint_path(self)
    +best_model_path(self)
    +metadata_path(self)
    +validate_all(self)
    +ensure_directories(self)
  }
  class config_py_HamiltonianAudioConfig {
    <<class>>
    +validate(self)
    +checkpoint_path(self)
    +best_model_path(self)
    +metadata_path(self)
    +validate_all(self)
    +ensure_directories(self)
  }
  class experiment2_py_Config {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_SeedManager {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_LoggerFactory {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_IAnalysisStrategy {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_IMetricsCalculator {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_HamiltonianOperator {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_HamiltonianDataset {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_SpectralLayer {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_HamiltonianNeuralNetwork {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_LocalComplexityAnalyzer {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_SuperpositionAnalyzer {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_CrystallographyMetricsCalculator {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_ThermodynamicMetricsCalculator {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_SpectroscopyMetricsCalculator {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_CheckpointManager {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_TrainingMetricsMonitor {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_GlassStateDetector {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_TrainingEngine {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_SeedMiningSystem {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_SingleExperimentRunner {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_CheckpointAnalyzer {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class experiment2_py_Application {
    <<class>>
    +main()
    +set_seed(seed)
    +create_logger(name, level)
    +analyze(self, model)
    +compute(self, model)
    +__init__(self, grid_size)
    +_precompute_spectral_operators(self)
    +apply(self, field)
    +time_evolution(self, field, dt)
    +__init__(self, num_samples, grid_size, time_steps, dt, train_ratio)
  }
  class inference_py_HamiltonianAudioInference {
    <<class>>
    +__init__(self, config, load_best)
    +analyze_audio(self, audio_file_path, output_prefix)
    +_compute_energy_mask_patched(self, model_input)
    +_extract_hamiltonian_fields_patched(self, model_input)
    +_compute_inference_metrics(self, original_magnitude, reconstructed_magnitude, original_stft)
    +_print_inference_metrics(self)
  }
  class losses_py_HamiltonianLossComputer {
    <<class>>
    +__init__(self, config)
    +compute_total_loss(self, prediction, target, intermediates, model)
    +_compute_reconstruction_loss(self, prediction, target)
    +_compute_energy_conservation_loss(self, intermediates)
    +_compute_symplectic_loss(self, intermediates)
    +_compute_spectral_consistency_loss(self, prediction, target)
    +_compute_phase_coherence_loss(self, prediction, target)
    +_compute_action_minimization_loss(self, intermediates)
    +_compute_liouville_loss(self, intermediates)
    +_compute_hamiltonian_constraint_loss(self, intermediates)
  }
  class metrics_py_HamiltonianMetricsTracker {
    <<class>>
    +__init__(self, config)
    +_initialize_history_buffers(self)
    +compute_hamiltonian_energy(self, q, p)
    +compute_symplectic_form(self, q, p, dq, dp)
    +compute_liouville_measure(self, jacobian)
    +compute_phase_space_volume(self, q, p)
    +compute_action_integral(self, q_trajectory, p_trajectory, dt)
    +compute_poisson_bracket(self, f_values, g_values, q, p)
    +compute_spectral_entropy(self, spectrum)
    +compute_reconstruction_snr(self, original, reconstructed)
  }
  class model_py_SpectralEvolutionLayer {
    <<class>>
    +__init__(self, hidden_dim, kernel_base_height, kernel_base_width, init_std)
    +forward(self, x)
    +evolve_complex(self, x, target_height, target_width)
    +evolve_real(self, x, target_height, target_width)
    +__init__(self, config)
    +forward(self, x)
    +forward_with_intermediates(self, x)
    +extract_hamiltonian_fields(self, x)
    +compute_energy_mask(self, x)
  }
```

---

## Code Property Graph

Machine-readable Code Property Graph (CPG) in JSON-LD format. This block allows AI agents to parse the full structural graph without additional file reads. Compatible with GraphRAG pipelines.

```json
{"@context": "https://schema.org", "analysis": {"communities": [{"cohesion": 1.0, "id": 0, "label": "audio", "size": 11}, {"cohesion": 1.0, "id": 1, "label": "audio", "size": 2}, {"cohesion": 1.0, "id": 2, "label": "root", "size": 10}], "god_nodes": [{"node_id": "experiment2.py", "score": 26.3}, {"node_id": "audio/config.py", "score": 19.3}, {"node_id": "audio/trainer.py", "score": 15.1}, {"node_id": "audio/inference.py", "score": 14.7}, {"node_id": "hamiltonian_mbl.py", "score": 12.6}, {"node_id": "audio/experiment2.py", "score": 10.3}, {"node_id": "audio/main.py", "score": 8.7}, {"node_id": "audio/metrics.py", "score": 8.5}, {"node_id": "audio/audio_io.py", "score": 7.2}, {"node_id": "get_meditions.py", "score": 7.2}], "surprising_connections": []}, "edges": [{"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torch.utils.data"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audio_io.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audio_io.py", "target": "torchaudio"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audio_io.py", "target": "torchaudio.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audio_io.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audio_io.py", "target": "config"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audios.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audios.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audios.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audios.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audios.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audios.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audios.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audios.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audios.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audios.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audios.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audios.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audios.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audios.py", "target": "safetensors.torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audios.py", "target": "experiment2"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audios.py", "target": "scipy.io"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audios.py", "target": "scipy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/audios.py", "target": "cv2"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/checkpoint_manager.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/checkpoint_manager.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/checkpoint_manager.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/checkpoint_manager.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/checkpoint_manager.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/checkpoint_manager.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/checkpoint_manager.py", "target": "safetensors.torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/checkpoint_manager.py", "target": "config"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/config.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/config.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/config.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/experiment2.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/experiment2.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/experiment2.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/experiment2.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/experiment2.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/experiment2.py", "target": "torch.utils.data"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/experiment2.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/experiment2.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/experiment2.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/experiment2.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/experiment2.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/experiment2.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/experiment2.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/experiment2.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/experiment2.py", "target": "collections"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/experiment2.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/experiment2.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/inference.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/inference.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/inference.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/inference.py", "target": "config"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/inference.py", "target": "model"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/inference.py", "target": "audio_io"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/inference.py", "target": "visualization"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/inference.py", "target": "metrics"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/inference.py", "target": "checkpoint_manager"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/losses.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/losses.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/losses.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/losses.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/losses.py", "target": "config"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/main.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/main.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/main.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/main.py", "target": "config"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/main.py", "target": "trainer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/main.py", "target": "inference"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/metrics.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/metrics.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/metrics.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/metrics.py", "target": "collections"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/metrics.py", "target": "config"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/model.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/model.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/model.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/model.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/model.py", "target": "config"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/trainer.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/trainer.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/trainer.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/trainer.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/trainer.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/trainer.py", "target": "torch.utils.data"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/trainer.py", "target": "tqdm"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/trainer.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/trainer.py", "target": "config"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/trainer.py", "target": "model"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/trainer.py", "target": "audio_io"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/trainer.py", "target": "losses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/trainer.py", "target": "metrics"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/trainer.py", "target": "checkpoint_manager"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/visualization.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/visualization.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/visualization.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/visualization.py", "target": "matplotlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/visualization.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/visualization.py", "target": "matplotlib.gridspec"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/visualization.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "audio/visualization.py", "target": "config"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "check_fase_berry.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "check_fase_berry.py", "target": "cv2"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "check_fase_berry.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "check_fase_berry.py", "target": "mss"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "check_fase_berry.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "check_fase_berry.py", "target": "safetensors.torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "check_fase_berry.py", "target": "experiment2"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "diff_weights.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "diff_weights.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "diff_weights.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac.py", "target": "glob"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac.py", "target": "matplotlib.colors"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac.py", "target": "matplotlib.cm"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac.py", "target": "mpl_toolkits.mplot3d"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac.py", "target": "experiment2"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "expand.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "expand.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "expand.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "expand.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "expand.py", "target": "toml"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "expand.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "expand.py", "target": "main_fast"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "expand.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "expand.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "expand.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "torch.utils.data"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "threading"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "collections"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "seaborn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "scipy.stats"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "scipy.optimize"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "sklearn.decomposition"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment2.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment2.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment2.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment2.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment2.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment2.py", "target": "torch.utils.data"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment2.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment2.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment2.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment2.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment2.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment2.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment2.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment2.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment2.py", "target": "collections"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment2.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiment2.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "export.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "export.py", "target": "safetensors.torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "get_meditions.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "get_meditions.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "get_meditions.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "get_meditions.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "get_meditions.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "get_meditions.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "get_meditions.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "get_meditions.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "get_meditions.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "get_meditions.py", "target": "glob"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "get_meditions.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "get_meditions.py", "target": "collections"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "get_meditions.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "get_meditions.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "get_meditions.py", "target": "scipy.stats"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "get_meditions.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "get_meditions.py", "target": "scipy.optimize"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "get_meditions.py", "target": "sklearn.decomposition"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "get_meditions.py", "target": "experiment2"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hamiltonian_mbl.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hamiltonian_mbl.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hamiltonian_mbl.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hamiltonian_mbl.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hamiltonian_mbl.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hamiltonian_mbl.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hamiltonian_mbl.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hamiltonian_mbl.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hamiltonian_mbl.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hamiltonian_mbl.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hamiltonian_mbl.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hamiltonian_mbl.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hamiltonian_mbl.py", "target": "glob"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hamiltonian_mbl.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hamiltonian_mbl.py", "target": "scipy.stats"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hamiltonian_mbl.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hamiltonian_mbl.py", "target": "scipy.sparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hamiltonian_mbl.py", "target": "scipy.sparse.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hamiltonian_mbl.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hamiltonian_mbl.py", "target": "gc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hpu_view.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hpu_view.py", "target": "cv2"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hpu_view.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hpu_view.py", "target": "mss"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hpu_view.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hpu_view.py", "target": "safetensors.torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hpu_view.py", "target": "experiment2"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "torch.utils.data"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "threading"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "collections"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "seaborn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "scipy.stats"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "scipy.optimize"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "sklearn.decomposition"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mining_seeds.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polos.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polos.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polos.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polos.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polos.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polos.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polos.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polos.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polos.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polos.py", "target": "glob"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polos.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polos.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polos.py", "target": "scipy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polos.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polos.py", "target": "scipy.optimize"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polos.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polos.py", "target": "matplotlib.patches"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polos.py", "target": "mpl_toolkits.mplot3d"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "polos.py", "target": "experiment2"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "precision.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "precision.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "precision.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "precision.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "precision.py", "target": "torch.utils.data"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "precision.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "precision.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "precision.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "precision.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "precision.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "precision.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "precision.py", "target": "glob"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "precision.py", "target": "experiment2"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "precision.py", "target": "refinamiento"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "precision.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "refinamiento.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "refinamiento.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "refinamiento.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "refinamiento.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "refinamiento.py", "target": "torch.utils.data"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "refinamiento.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "refinamiento.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "refinamiento.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "refinamiento.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "refinamiento.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "refinamiento.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "refinamiento.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "refinamiento.py", "target": "collections"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "refinamiento.py", "target": "experiment2"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "refinamiento.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "simple_hpu_view.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "simple_hpu_view.py", "target": "cv2"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "simple_hpu_view.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "simple_hpu_view.py", "target": "mss"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "simple_hpu_view.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "simple_hpu_view.py", "target": "safetensors.torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "simple_hpu_view.py", "target": "experiment2"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_grokkit.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_grokkit.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_grokkit.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_grokkit.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_grokkit.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_grokkit.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_grokkit.py", "target": "main"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_grokkit.py", "target": "main_fast"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_grokkit.py", "target": "main_fast"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "verify.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "verify.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "verify.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "verify.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "verify.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "verify.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "verify.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "verify.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "verify.py", "target": "glob"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "verify.py", "target": "experiment2"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/audio_io.py", "target": "audio/config.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/audios.py", "target": "audio/experiment2.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/checkpoint_manager.py", "target": "audio/config.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/inference.py", "target": "audio/config.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/inference.py", "target": "audio/model.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/inference.py", "target": "audio/audio_io.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/inference.py", "target": "audio/visualization.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/inference.py", "target": "audio/metrics.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/inference.py", "target": "audio/checkpoint_manager.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/losses.py", "target": "audio/config.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/main.py", "target": "audio/config.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/main.py", "target": "audio/trainer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/main.py", "target": "audio/inference.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/metrics.py", "target": "audio/config.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/model.py", "target": "audio/config.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/trainer.py", "target": "audio/config.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/trainer.py", "target": "audio/model.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/trainer.py", "target": "audio/audio_io.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/trainer.py", "target": "audio/losses.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/trainer.py", "target": "audio/metrics.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/trainer.py", "target": "audio/checkpoint_manager.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "audio/visualization.py", "target": "audio/config.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "check_fase_berry.py", "target": "experiment2.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "dirac.py", "target": "experiment2.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "get_meditions.py", "target": "experiment2.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "hpu_view.py", "target": "experiment2.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "polos.py", "target": "experiment2.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "precision.py", "target": "experiment2.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "precision.py", "target": "refinamiento.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "refinamiento.py", "target": "experiment2.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "simple_hpu_view.py", "target": "experiment2.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_grokkit.py", "target": "audio/main.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "verify.py", "target": "experiment2.py"}], "generator": "readmenator", "metadata": {"edge_count": 6785, "file_count": 32, "language_count": 2, "symbol_count": 776}, "nodes": [{"doc": "_*_ coding: utf8 _*_", "id": "app.py", "kind": "module", "label": "app.py", "language": "py", "sha256": "7d7de62ae6a739b7", "symbol_count": 22, "symbols": [{"kind": "class", "line": 29, "name": "SimpleConfig", "signature": "class SimpleConfig"}, {"doc": "Compute Local Complexity (LC) metric for weight matrix.", "kind": "method", "line": 45, "name": "compute_local_complexity", "signature": "def compute_local_complexity(weights, epsilon)"}, {"doc": "Compute Superposition (SP) metric for weight matrix.", "kind": "method", "line": 60, "name": "compute_superposition", "signature": "def compute_superposition(weights)"}, {"doc": "True Hamiltonian operator H = -nabla^2 on torus.", "kind": "class", "line": 88, "name": "HamiltonianOperator", "signature": "class HamiltonianOperator"}, {"doc": "Fast dataset for Hamiltonian operator learning.", "kind": "class", "line": 113, "name": "FastDataset", "signature": "class FastDataset(Dataset)"}, {"doc": "Spectral layer with correct complex multiplication.", "kind": "class", "line": 170, "name": "SpectralLayer", "signature": "class SpectralLayer(Module)"}, {"doc": "Compact network for Hamiltonian operator learning.", "kind": "class", "line": 222, "name": "SimpleHamiltonianNet", "signature": "class SimpleHamiltonianNet(Module)"}, {"doc": "Train the Hamiltonian operator model.", "kind": "method", "line": 260, "name": "train_model", "signature": "def train_model(grid_size, epochs, hidden_dim, num_spectral_layers, lr)"}, {"kind": "method", "line": 369, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 30, "name": "__init__", "signature": "def __init__(self, grid_size, hidden_dim, num_spectral_layers, target_accuracy, learning_rate)"}, {"kind": "method", "line": 91, "name": "__init__", "signature": "def __init__(self, grid_size)"}, {"kind": "method", "line": 95, "name": "_precompute_spectral_operators", "signature": "def _precompute_spectral_operators(self)"}, {"kind": "method", "line": 102, "name": "apply", "signature": "def apply(self, field)"}, {"kind": "method", "line": 107, "name": "time_evolution", "signature": "def time_evolution(self, field, dt)"}, {"kind": "method", "line": 116, "name": "__init__", "signature": "def __init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)"}, {"kind": "method", "line": 160, "name": "__len__", "signature": "def __len__(self)"}, {"kind": "method", "line": 163, "name": "__getitem__", "signature": "def __getitem__(self, idx)"}, {"kind": "method", "line": 166, "name": "get_val_batch", "signature": "def get_val_batch(self)"}, {"kind": "method", "line": 173, "name": "__init__", "signature": "def __init__(self, channels, grid_size)"}, {"kind": "method", "line": 186, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 225, "name": "__init__", "signature": "def __init__(self, grid_size, hidden_dim, num_spectral_layers)"}, {"kind": "method", "line": 246, "name": "forward", "signature": "def forward(self, x)"}]}, {"id": "audio/audio_io.py", "kind": "module", "label": "audio_io.py", "language": "py", "sha256": "fe8f6fb5a24eb3c5", "symbol_count": 12, "symbols": [{"doc": "Audio processing pipeline centered on the complex STFT domain.\n\nThe complex STFT is the audio equivalent of a grayscale image:\na 2D field where one axis is time, the other is frequency, and\neach point carries a complex value (magnitude + phase). This is\nthe correct domain for applying the Hamiltonian spectral evolution.", "kind": "class", "line": 31, "name": "AudioProcessor", "signature": "class AudioProcessor"}, {"kind": "method", "line": 41, "name": "__init__", "signature": "def __init__(self, config, device)"}, {"doc": "Load an audio file and convert to mono at the target sample rate.\n\nArgs:\n    file_path: Path to the audio file.\n\nReturns:\n    Tuple of (waveform tensor [1, T], sample_rate).", "kind": "method", "line": 57, "name": "load_audio", "signature": "def load_audio(self, file_path)"}, {"doc": "Compute the complex STFT of a waveform.\n\nThis is the primary transform: converts 1D audio into a 2D complex\nfield (freq_bins x time_frames) that the Hamiltonian network\ncan process identically to how it processes images.\n\nArgs:\n    waveform: Audio waveform [1, T] or [T].\n\nReturns:\n    Complex STFT tensor [freq_bins, time_frames].", "kind": "method", "line": 80, "name": "waveform_to_stft_complex", "signature": "def waveform_to_stft_complex(self, waveform)"}, {"doc": "Reconstruct waveform from complex STFT via inverse STFT.\n\nUnlike Griffin-Lim (which estimates phase), ISTFT uses the\nEXACT phase from the complex STFT, producing a faithful\nreconstruction when the magnitude/phase have been coherently\nmodified by the Hamiltonian evolution.\n\nArgs:\n    stft_complex: Complex STFT tensor [freq_bins, time_frames].\n\nReturns:\n    Reconstructed waveform [1, T].", "kind": "method", "line": 105, "name": "stft_complex_to_waveform", "signature": "def stft_complex_to_waveform(self, stft_complex)"}, {"doc": "Decompose complex STFT into magnitude and phase.\n\nArgs:\n    stft_complex: Complex STFT [freq_bins, time_frames].\n\nReturns:\n    Tuple of (magnitude, phase) each [freq_bins, time_frames].", "kind": "method", "line": 130, "name": "stft_to_magnitude_phase", "signature": "def stft_to_magnitude_phase(self, stft_complex)"}, {"doc": "Recombine magnitude and phase into complex STFT.\n\nArgs:\n    magnitude: Magnitude spectrum [freq_bins, time_frames].\n    phase: Phase spectrum [freq_bins, time_frames].\n\nReturns:\n    Complex STFT [freq_bins, time_frames].", "kind": "method", "line": 146, "name": "magnitude_phase_to_stft", "signature": "def magnitude_phase_to_stft(self, magnitude, phase)"}, {"doc": "Prepare STFT magnitude for input to the Hamiltonian network.\n\nNormalizes magnitude to [0, 1] range and shapes as [1, 1, H, W],\nmatching the expected input format (analogous to a grayscale image).\n\nArgs:\n    magnitude: STFT magnitude [freq_bins, time_frames].\n\nReturns:\n    Normalized tensor [1, 1, freq_bins, time_frames].", "kind": "method", "line": 161, "name": "stft_magnitude_to_model_input", "signature": "def stft_magnitude_to_model_input(self, magnitude)"}, {"doc": "Convert model output (energy mask in [0, 1]) back to STFT magnitude scale.\n\nThe model output represents the Hamiltonian energy structure --\nwhich regions of the time-frequency plane carry coherent energy.\nThis is used to modulate the original magnitude.\n\nArgs:\n    model_output: Energy mask [1, 1, freq_bins, time_frames] in [0, 1].\n    original_magnitude: Original STFT magnitude [freq_bins, time_frames].\n\nReturns:\n    Reconstructed magnitude [freq_bins, time_frames].", "kind": "method", "line": 186, "name": "model_output_to_stft_magnitude", "signature": "def model_output_to_stft_magnitude(self, model_output, original_magnitude)"}, {"doc": "Convert waveform to normalized mel spectrogram (for visualization only).\n\nArgs:\n    waveform: Audio waveform tensor [1, T] or [B, 1, T].\n\nReturns:\n    Normalized mel spectrogram [B, 1, n_mels, time_frames].", "kind": "method", "line": 206, "name": "waveform_to_mel_spectrogram", "signature": "def waveform_to_mel_spectrogram(self, waveform)"}, {"doc": "Save a waveform tensor to an audio file.\n\nArgs:\n    waveform: Audio tensor [1, T] or [T].\n    file_path: Output file path.\n    sample_rate: Sample rate (defaults to config sample rate).", "kind": "method", "line": 229, "name": "save_audio", "signature": "def save_audio(self, waveform, file_path, sample_rate)"}, {"doc": "Compute the dB range of a waveform's mel spectrogram.\n\nArgs:\n    waveform: Audio waveform [1, T].\n\nReturns:\n    Tuple of (db_min, db_max).", "kind": "method", "line": 249, "name": "get_spectrogram_db_range", "signature": "def get_spectrogram_db_range(self, waveform)"}]}, {"id": "audio/audios.py", "kind": "module", "label": "audios.py", "language": "py", "sha256": "8419a6c094eb3fad", "symbol_count": 47, "symbols": [{"doc": "Immutable configuration container for all hyperparameters.\nEliminates magic numbers and provides single point of control.", "kind": "class", "line": 48, "name": "HamiltonianConfig", "signature": "class HamiltonianConfig"}, {"doc": "Interface for audio input sources.", "kind": "class", "line": 103, "name": "IAudioSource", "signature": "class IAudioSource(ABC)"}, {"doc": "Interface for Hamiltonian field evolution operators.", "kind": "class", "line": 122, "name": "IFieldOperator", "signature": "class IFieldOperator(ABC)"}, {"doc": "Interface for training metrics collection.", "kind": "class", "line": 131, "name": "IMetricCollector", "signature": "class IMetricCollector(ABC)"}, {"doc": "Handles audio resampling using scipy.signal, avoiding librosa/numba dependencies.", "kind": "class", "line": 149, "name": "AudioResampler", "signature": "class AudioResampler"}, {"doc": "Concrete implementation of audio source from file.\nSupports automatic resampling to target sample rate using scipy.", "kind": "class", "line": 204, "name": "WaveFileSource", "signature": "class WaveFileSource(IAudioSource)"}, {"doc": "Collects all metrics from Hamiltonian paper, activation functions,\nand architectural diagnostics for informed decision-making.", "kind": "class", "line": 278, "name": "ComprehensiveMetricCollector", "signature": "class ComprehensiveMetricCollector(IMetricCollector)"}, {"doc": "Manages periodic checkpointing with atomic writes.", "kind": "class", "line": 326, "name": "CheckpointManager", "signature": "class CheckpointManager"}, {"doc": "Converts between audio waveforms and 2D field representations.\nAdaptado para la arquitectura de experiment2 (grid_size=16).", "kind": "class", "line": 387, "name": "AudioSpectrogramConverter", "signature": "class AudioSpectrogramConverter"}, {"doc": "Main orchestrator for Hamiltonian audio processing.\nDemonstrates that auditory perception is epiphenomenon of Hamiltonian dynamics.\nUsa la arquitectura exacta de experiment2.", "kind": "class", "line": 468, "name": "HamiltonianAudioProcessor", "signature": "class HamiltonianAudioProcessor"}, {"doc": "Entry point with argument parsing.", "kind": "method", "line": 742, "name": "main", "signature": "def main()"}, {"doc": "Calculate segment length in samples.", "kind": "method", "line": 89, "name": "segment_samples", "signature": "def segment_samples(self)"}, {"doc": "Calculate frequency bins for real FFT.", "kind": "method", "line": 94, "name": "freq_bins", "signature": "def freq_bins(self)"}, {"doc": "Read audio segment. Returns None when exhausted.", "kind": "method", "line": 107, "name": "read_segment", "signature": "def read_segment(self)"}, {"doc": "Return audio properties.", "kind": "method", "line": 112, "name": "get_properties", "signature": "def get_properties(self)"}, {"doc": "Release resources.", "kind": "method", "line": 117, "name": "close", "signature": "def close(self)"}, {"doc": "Evolve field state through Hamiltonian dynamics.", "kind": "method", "line": 126, "name": "evolve", "signature": "def evolve(self, field_state)"}, {"doc": "Record metric values.", "kind": "method", "line": 135, "name": "record", "signature": "def record(self, metrics)"}, {"doc": "Return aggregated metrics.", "kind": "method", "line": 140, "name": "get_summary", "signature": "def get_summary(self)"}, {"doc": "Resample audio from orig_sr to target_sr using polyphase filtering.", "kind": "method", "line": 155, "name": "resample", "signature": "def resample(audio, orig_sr, target_sr)"}, {"doc": "Load WAV file and resample to target sample rate.\nReturns (audio_data, original_sample_rate).", "kind": "method", "line": 173, "name": "load_wav_with_resample", "signature": "def load_wav_with_resample(file_path, target_sr)"}, {"kind": "method", "line": 210, "name": "__init__", "signature": "def __init__(self, file_path, config)"}, {"doc": "Validate file format and load with automatic resampling.", "kind": "method", "line": 220, "name": "_validate_and_load", "signature": "def _validate_and_load(self)"}, {"doc": "Read next audio segment.", "kind": "method", "line": 244, "name": "read_segment", "signature": "def read_segment(self)"}, {"doc": "Return audio file properties.", "kind": "method", "line": 261, "name": "get_properties", "signature": "def get_properties(self)"}, {"doc": "Release resources.", "kind": "method", "line": 273, "name": "close", "signature": "def close(self)"}, {"kind": "method", "line": 284, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Record comprehensive metrics.", "kind": "method", "line": 289, "name": "record", "signature": "def record(self, metrics)"}, {"doc": "Return statistical summary of all metrics.", "kind": "method", "line": 298, "name": "get_summary", "signature": "def get_summary(self)"}, {"doc": "Export full history to JSON.", "kind": "method", "line": 320, "name": "export_to_json", "signature": "def export_to_json(self, path)"}, {"kind": "method", "line": 331, "name": "__init__", "signature": "def __init__(self, model, config, checkpoint_dir)"}, {"doc": "Check if checkpoint interval elapsed and save if necessary.\nReturns path if saved, None otherwise.", "kind": "method", "line": 344, "name": "check_and_save", "signature": "def check_and_save(self, force)"}, {"doc": "Atomic checkpoint save.", "kind": "method", "line": 357, "name": "_save_checkpoint", "signature": "def _save_checkpoint(self)"}, {"kind": "method", "line": 393, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Convert 1D audio to 2D field representation via STFT.\nReturns (1, 1, grid_size, grid_size) tensor compatible con experiment2.", "kind": "method", "line": 396, "name": "waveform_to_field", "signature": "def waveform_to_field(self, waveform)"}, {"doc": "Reconstruct waveform from 2D field representation.", "kind": "method", "line": 432, "name": "field_to_waveform", "signature": "def field_to_waveform(self, field, original_length)"}, {"doc": "Compute magnitude spectrogram.", "kind": "method", "line": 458, "name": "_forward_spectrogram", "signature": "def _forward_spectrogram(self, x)"}, {"doc": "Griffin-Lim inverse.", "kind": "method", "line": 463, "name": "_inverse_spectrogram", "signature": "def _inverse_spectrogram(self, spectrogram)"}, {"kind": "method", "line": 475, "name": "__init__", "signature": "def __init__(self, config, model, source)"}, {"doc": "Load pretrained Hamiltonian operator desde safetensors.", "kind": "method", "line": 504, "name": "load_model_weights", "signature": "def load_model_weights(self, path)"}, {"doc": "Attach audio source via dependency injection.", "kind": "method", "line": 513, "name": "attach_source", "signature": "def attach_source(self, source)"}, {"doc": "Process audio stream through Hamiltonian perception.\nGenerates three epiphenomenal representations:\n1. Energy Density (Resonance)\n2. Topological Phase (Vortices)\n3. Action Map (Perceptual Clarity)", "kind": "method", "line": 517, "name": "process_stream", "signature": "def process_stream(self)"}, {"doc": "Process single audio segment and return metrics.", "kind": "method", "line": 592, "name": "_process_single_segment", "signature": "def _process_single_segment(self, waveform, index)"}, {"doc": "Calculate topological entropy from phase distribution.", "kind": "method", "line": 684, "name": "_calculate_phase_entropy", "signature": "def _calculate_phase_entropy(self, phase_map)"}, {"doc": "Render three epiphenomenal visualizations.", "kind": "method", "line": 692, "name": "_render_epiphenomena", "signature": "def _render_epiphenomena(self, amplitude, phase, action)"}, {"doc": "Export comprehensive metrics to file.", "kind": "method", "line": 729, "name": "export_metrics", "signature": "def export_metrics(self, path)"}, {"doc": "Force immediate checkpoint save.", "kind": "method", "line": 733, "name": "force_checkpoint", "signature": "def force_checkpoint(self)"}]}, {"id": "audio/checkpoint_manager.py", "kind": "module", "label": "checkpoint_manager.py", "language": "py", "sha256": "489815fa845106b1", "symbol_count": 6, "symbols": [{"doc": "Manages model checkpointing with time-based intervals\nand best-model tracking.", "kind": "class", "line": 28, "name": "CheckpointManager", "signature": "class CheckpointManager"}, {"kind": "method", "line": 34, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Check if enough time has elapsed since the last checkpoint.", "kind": "method", "line": 41, "name": "should_save_checkpoint", "signature": "def should_save_checkpoint(self)"}, {"doc": "Save the current model state and training metadata.\n\nSaves to a single 'latest.safetensors' file plus a JSON\nmetadata file containing optimizer state, epoch, and metrics.\n\nArgs:\n    model: The model to checkpoint.\n    optimizer: Current optimizer state.\n    scheduler: Current LR scheduler state.\n    epoch: Current epoch number.\n    step: Current global step.\n    metrics: Dictionary of current metric values.\n    current_loss: Current total loss value.", "kind": "method", "line": 46, "name": "save_checkpoint", "signature": "def save_checkpoint(self, model, optimizer, scheduler, epoch, step, metrics, current_loss)"}, {"doc": "Load a model checkpoint and return training metadata.\n\nUses the exact safetensors loading pattern:\n    load_model(model, checkpoint_path)\n\nPath resolution priority:\n1. If CheckpointConfig.checkpoint_file_path is set, use that exact path\n   (overrides load_best flag).\n2. If load_best is True, use checkpoint_directory/best.safetensors.\n3. Otherwise, use checkpoint_directory/latest.safetensors.\n\nArgs:\n    model: The model to load weights into.\n    load_best: If True and no explicit path set, load best model.\n\nReturns:\n    Metadata dictionary if available, None otherwise.", "kind": "method", "line": 98, "name": "load_checkpoint", "signature": "def load_checkpoint(self, model, load_best)"}, {"kind": "method", "line": 156, "name": "best_loss", "signature": "def best_loss(self)"}]}, {"id": "audio/config.py", "kind": "module", "label": "config.py", "language": "py", "sha256": "959ddfef08d82a66", "symbol_count": 13, "symbols": [{"doc": "Parameters governing raw audio ingestion and spectrogram computation.", "kind": "class", "line": 18, "name": "AudioProcessingConfig", "signature": "class AudioProcessingConfig"}, {"doc": "Parametric architecture dimensions for the Hamiltonian Neural Network.\n\nAll hidden dimensions, matrix sizes, expansion factors, and layer counts\nare configurable from this single source of truth.", "kind": "class", "line": 34, "name": "ModelArchitectureConfig", "signature": "class ModelArchitectureConfig"}, {"doc": "All training loop hyperparameters and scheduling constants.", "kind": "class", "line": 80, "name": "TrainingConfig", "signature": "class TrainingConfig"}, {"doc": "Checkpoint persistence parameters.", "kind": "class", "line": 109, "name": "CheckpointConfig", "signature": "class CheckpointConfig"}, {"doc": "Parameters for audio reconstruction visualization and output.", "kind": "class", "line": 136, "name": "VisualizationConfig", "signature": "class VisualizationConfig"}, {"doc": "Configuration for all tracked metrics during training and inference.", "kind": "class", "line": 158, "name": "MetricsConfig", "signature": "class MetricsConfig"}, {"doc": "Top-level configuration aggregator.\n\nComposes all sub-configurations into a single injectable dependency,\nfollowing the Dependency Inversion Principle.", "kind": "class", "line": 182, "name": "HamiltonianAudioConfig", "signature": "class HamiltonianAudioConfig"}, {"doc": "Ensure architectural coherence.", "kind": "method", "line": 62, "name": "validate", "signature": "def validate(self)"}, {"kind": "method", "line": 121, "name": "checkpoint_path", "signature": "def checkpoint_path(self)"}, {"kind": "method", "line": 127, "name": "best_model_path", "signature": "def best_model_path(self)"}, {"kind": "method", "line": 131, "name": "metadata_path", "signature": "def metadata_path(self)"}, {"doc": "Run validation on all sub-configurations.", "kind": "method", "line": 198, "name": "validate_all", "signature": "def validate_all(self)"}, {"doc": "Create required output directories if they do not exist.", "kind": "method", "line": 206, "name": "ensure_directories", "signature": "def ensure_directories(self)"}]}, {"id": "audio/experiment2.py", "kind": "module", "label": "experiment2.py", "language": "py", "sha256": "cbbe995fe02ec9be", "symbol_count": 83, "symbols": [{"kind": "class", "line": 22, "name": "Config", "signature": "class Config"}, {"kind": "class", "line": 74, "name": "SeedManager", "signature": "class SeedManager"}, {"kind": "class", "line": 84, "name": "LoggerFactory", "signature": "class LoggerFactory"}, {"kind": "class", "line": 99, "name": "IAnalysisStrategy", "signature": "class IAnalysisStrategy(ABC)"}, {"kind": "class", "line": 105, "name": "IMetricsCalculator", "signature": "class IMetricsCalculator(ABC)"}, {"kind": "class", "line": 111, "name": "HamiltonianOperator", "signature": "class HamiltonianOperator"}, {"kind": "class", "line": 133, "name": "HamiltonianDataset", "signature": "class HamiltonianDataset(Dataset)"}, {"kind": "class", "line": 183, "name": "SpectralLayer", "signature": "class SpectralLayer(Module)"}, {"kind": "class", "line": 227, "name": "HamiltonianNeuralNetwork", "signature": "class HamiltonianNeuralNetwork(Module)"}, {"kind": "class", "line": 257, "name": "LocalComplexityAnalyzer", "signature": "class LocalComplexityAnalyzer"}, {"kind": "class", "line": 273, "name": "SuperpositionAnalyzer", "signature": "class SuperpositionAnalyzer"}, {"kind": "class", "line": 302, "name": "CrystallographyMetricsCalculator", "signature": "class CrystallographyMetricsCalculator(IMetricsCalculator)"}, {"kind": "class", "line": 737, "name": "ThermodynamicMetricsCalculator", "signature": "class ThermodynamicMetricsCalculator(IMetricsCalculator)"}, {"kind": "class", "line": 770, "name": "SpectroscopyMetricsCalculator", "signature": "class SpectroscopyMetricsCalculator(IMetricsCalculator)"}, {"kind": "class", "line": 804, "name": "CheckpointManager", "signature": "class CheckpointManager"}, {"kind": "class", "line": 874, "name": "TrainingMetricsMonitor", "signature": "class TrainingMetricsMonitor"}, {"kind": "class", "line": 912, "name": "GlassStateDetector", "signature": "class GlassStateDetector"}, {"kind": "class", "line": 973, "name": "TrainingEngine", "signature": "class TrainingEngine"}, {"kind": "class", "line": 1113, "name": "SeedMiningSystem", "signature": "class SeedMiningSystem"}, {"kind": "class", "line": 1164, "name": "SingleExperimentRunner", "signature": "class SingleExperimentRunner"}, {"kind": "class", "line": 1223, "name": "CheckpointAnalyzer", "signature": "class CheckpointAnalyzer"}, {"kind": "class", "line": 1275, "name": "Application", "signature": "class Application"}, {"kind": "method", "line": 1334, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 76, "name": "set_seed", "signature": "def set_seed(seed)"}, {"kind": "method", "line": 86, "name": "create_logger", "signature": "def create_logger(name, level)"}, {"kind": "method", "line": 101, "name": "analyze", "signature": "def analyze(self, model)"}, {"kind": "method", "line": 107, "name": "compute", "signature": "def compute(self, model)"}, {"kind": "method", "line": 112, "name": "__init__", "signature": "def __init__(self, grid_size)"}, {"kind": "method", "line": 116, "name": "_precompute_spectral_operators", "signature": "def _precompute_spectral_operators(self)"}, {"kind": "method", "line": 122, "name": "apply", "signature": "def apply(self, field)"}, {"kind": "method", "line": 127, "name": "time_evolution", "signature": "def time_evolution(self, field, dt)"}, {"kind": "method", "line": 134, "name": "__init__", "signature": "def __init__(self, num_samples, grid_size, time_steps, dt, train_ratio)"}, {"kind": "method", "line": 173, "name": "__len__", "signature": "def __len__(self)"}, {"kind": "method", "line": 176, "name": "__getitem__", "signature": "def __getitem__(self, idx)"}, {"kind": "method", "line": 179, "name": "get_validation_batch", "signature": "def get_validation_batch(self)"}, {"kind": "method", "line": 184, "name": "__init__", "signature": "def __init__(self, channels, grid_size)"}, {"kind": "method", "line": 195, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 228, "name": "__init__", "signature": "def __init__(self, grid_size, hidden_dim, num_spectral_layers)"}, {"kind": "method", "line": 243, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 259, "name": "compute_local_complexity", "signature": "def compute_local_complexity(weights, epsilon)"}, {"kind": "method", "line": 275, "name": "compute_superposition", "signature": "def compute_superposition(weights)"}, {"doc": "Implementación de interfaz IMetricsCalculator.\nDelega a compute_all_metrics con los argumentos correctos.", "kind": "method", "line": 303, "name": "compute", "signature": "def compute(self, model, val_x, val_y)"}, {"kind": "method", "line": 311, "name": "compute_gradient_covariance_kappa", "signature": "def compute_gradient_covariance_kappa(model, dataloader, num_batches)"}, {"doc": "Calcula el margen de discretización desde los parámetros del modelo.\nVersión estática que no requiere diccionario externo.", "kind": "method", "line": 348, "name": "compute_discretization_margin_from_state_dict", "signature": "def compute_discretization_margin_from_state_dict(model)"}, {"doc": "Calcula el margen de discretización desde un diccionario de coeficientes.", "kind": "method", "line": 361, "name": "compute_discretization_margin", "signature": "def compute_discretization_margin(coeffs)"}, {"doc": "Calcula el índice de pureza alpha directamente desde el modelo.", "kind": "method", "line": 373, "name": "compute_alpha_purity_from_model", "signature": "def compute_alpha_purity_from_model(model)"}, {"doc": "Calcula el índice de pureza alpha desde un diccionario de coeficientes.", "kind": "method", "line": 383, "name": "compute_alpha_purity", "signature": "def compute_alpha_purity(coeffs)"}, {"doc": "Número de condición de la matriz de covarianza de gradientes.", "kind": "method", "line": 393, "name": "compute_kappa", "signature": "def compute_kappa(model, val_x, val_y, num_batches)"}, {"doc": "Versión del cálculo cuántico de kappa que opera directamente sobre el modelo.", "kind": "method", "line": 464, "name": "compute_kappa_quantum", "signature": "def compute_kappa_quantum(model, hbar)"}, {"doc": "Versión del cálculo cuántico de kappa desde diccionario de coeficientes.", "kind": "method", "line": 492, "name": "compute_kappa_quantum_from_coeffs", "signature": "def compute_kappa_quantum_from_coeffs(coeffs, hbar)"}, {"doc": "Métricas cristalográficas con aislamiento completo de errores.", "kind": "method", "line": 511, "name": "_compute_crystallography_metrics", "signature": "def _compute_crystallography_metrics(self, model, val_x, val_y)"}, {"doc": "Verifica integridad de pesos: NaN, Inf, y estadísticas básicas.", "kind": "method", "line": 539, "name": "_check_weight_integrity", "signature": "def _check_weight_integrity(self, model)"}, {"doc": "Vector de Poynting: flujo de energía en el espacio de parámetros.\nAnálogo electromagnético para redes neuronales.", "kind": "method", "line": 603, "name": "compute_poynting_vector", "signature": "def compute_poynting_vector(model)"}, {"doc": "Calcula todas las métricas cristalográficas con manejo de errores.", "kind": "method", "line": 679, "name": "compute_all_metrics", "signature": "def compute_all_metrics(model, val_x, val_y)"}, {"kind": "method", "line": 738, "name": "compute", "signature": "def compute(self, model, gradient_buffer, learning_rate, loss_history, temp_history)"}, {"kind": "method", "line": 747, "name": "compute_effective_temperature", "signature": "def compute_effective_temperature(gradient_buffer, learning_rate)"}, {"kind": "method", "line": 760, "name": "compute_specific_heat", "signature": "def compute_specific_heat(loss_history, temp_history, cv_threshold)"}, {"kind": "method", "line": 771, "name": "compute", "signature": "def compute(self, model)"}, {"kind": "method", "line": 776, "name": "compute_weight_diffraction", "signature": "def compute_weight_diffraction(coeffs)"}, {"kind": "method", "line": 795, "name": "_compute_spectral_entropy", "signature": "def _compute_spectral_entropy(power_spectrum)"}, {"kind": "method", "line": 805, "name": "__init__", "signature": "def __init__(self, interval_minutes, max_checkpoints)"}, {"kind": "method", "line": 813, "name": "should_save_checkpoint", "signature": "def should_save_checkpoint(self)"}, {"kind": "method", "line": 818, "name": "save_checkpoint", "signature": "def save_checkpoint(self, model, optimizer, epoch, metrics)"}, {"kind": "method", "line": 875, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 895, "name": "update_metrics", "signature": "def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat, poynting_magnitude)"}, {"kind": "method", "line": 913, "name": "__init__", "signature": "def __init__(self, patience_epochs)"}, {"kind": "method", "line": 918, "name": "should_stop", "signature": "def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)"}, {"kind": "method", "line": 963, "name": "is_crystal_formed", "signature": "def is_crystal_formed(self, lc, sp, kappa, delta, temp, cv)"}, {"kind": "method", "line": 974, "name": "__init__", "signature": "def __init__(self, model, optimizer, device, logger)"}, {"kind": "method", "line": 997, "name": "train_epoch", "signature": "def train_epoch(self, dataloader, epoch)"}, {"kind": "method", "line": 1027, "name": "validate", "signature": "def validate(self, val_x, val_y)"}, {"kind": "method", "line": 1040, "name": "compute_weight_metrics", "signature": "def compute_weight_metrics(self)"}, {"kind": "method", "line": 1056, "name": "execute_training", "signature": "def execute_training(self, dataloader, val_x, val_y, epochs, seed, early_stopping)"}, {"kind": "method", "line": 1114, "name": "__init__", "signature": "def __init__(self, max_attempts)"}, {"kind": "method", "line": 1118, "name": "mine", "signature": "def mine(self)"}, {"kind": "method", "line": 1165, "name": "__init__", "signature": "def __init__(self, seed, epochs, grid_size, hidden_dim, num_spectral_layers, learning_rate)"}, {"kind": "method", "line": 1175, "name": "run", "signature": "def run(self)"}, {"kind": "method", "line": 1224, "name": "__init__", "signature": "def __init__(self, checkpoint_path, results_dir)"}, {"kind": "method", "line": 1230, "name": "analyze", "signature": "def analyze(self)"}, {"kind": "method", "line": 1276, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 1280, "name": "_create_argument_parser", "signature": "def _create_argument_parser(self)"}, {"kind": "method", "line": 1294, "name": "run", "signature": "def run(self)"}, {"kind": "method", "line": 694, "name": "safe_compute", "signature": "def safe_compute(func)"}]}, {"id": "audio/inference.py", "kind": "module", "label": "inference.py", "language": "py", "sha256": "ab5d1979c7b47c7e", "symbol_count": 7, "symbols": [{"doc": "Performs complete Hamiltonian audio analysis on a given audio file.", "kind": "class", "line": 37, "name": "HamiltonianAudioInference", "signature": "class HamiltonianAudioInference"}, {"kind": "method", "line": 42, "name": "__init__", "signature": "def __init__(self, config, load_best)"}, {"doc": "Perform complete Hamiltonian analysis on an audio file.\n\nArgs:\n    audio_file_path: Path to the audio file to analyze.\n    output_prefix: Optional prefix for output filenames.", "kind": "method", "line": 73, "name": "analyze_audio", "signature": "def analyze_audio(self, audio_file_path, output_prefix)"}, {"doc": "Compute energy mask over the full STFT magnitude, processing\nin patches along the time axis if the input exceeds patch width.\n\nArgs:\n    model_input: Normalized STFT magnitude [1, 1, freq_bins, time_frames].\n\nReturns:\n    Energy mask [1, 1, freq_bins, time_frames] in [0, 1].", "kind": "method", "line": 158, "name": "_compute_energy_mask_patched", "signature": "def _compute_energy_mask_patched(self, model_input)"}, {"doc": "Extract Hamiltonian fields over full STFT magnitude with patching.\n\nArgs:\n    model_input: Normalized STFT magnitude [1, 1, freq_bins, time_frames].\n\nReturns:\n    Tuple of (amplitude_map, phase_map, action_map).", "kind": "method", "line": 209, "name": "_extract_hamiltonian_fields_patched", "signature": "def _extract_hamiltonian_fields_patched(self, model_input)"}, {"doc": "Compute all inference-time metrics on the STFT domain.", "kind": "method", "line": 270, "name": "_compute_inference_metrics", "signature": "def _compute_inference_metrics(self, original_magnitude, reconstructed_magnitude, original_stft)"}, {"doc": "Print all computed inference metrics.", "kind": "method", "line": 288, "name": "_print_inference_metrics", "signature": "def _print_inference_metrics(self)"}]}, {"id": "audio/losses.py", "kind": "module", "label": "losses.py", "language": "py", "sha256": "8cc9c99cd69d528a", "symbol_count": 11, "symbols": [{"doc": "Computes the composite Hamiltonian loss function with all\nphysics-based regularization terms.\n\nEach loss component is independently weighted via TrainingConfig,\nenabling fine-grained control over the training objective.", "kind": "class", "line": 28, "name": "HamiltonianLossComputer", "signature": "class HamiltonianLossComputer"}, {"kind": "method", "line": 37, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Compute the complete weighted loss with all Hamiltonian terms.\n\nArgs:\n    prediction: Model output [B, 1, H, W].\n    target: Ground truth [B, 1, H, W].\n    intermediates: List of intermediate hidden states from forward pass.\n    model: The model (for parameter access in regularization).\n\nReturns:\n    Tuple of (total_loss tensor, dict of individual loss values).", "kind": "method", "line": 41, "name": "compute_total_loss", "signature": "def compute_total_loss(self, prediction, target, intermediates, model)"}, {"doc": "MSE reconstruction loss between predicted and target spectrograms.", "kind": "method", "line": 94, "name": "_compute_reconstruction_loss", "signature": "def _compute_reconstruction_loss(self, prediction, target)"}, {"doc": "Penalize energy drift across layers.\n\nThe Hamiltonian energy E = 0.5 * ||phi||^2 should remain\napproximately constant through the evolution layers.", "kind": "method", "line": 100, "name": "_compute_energy_conservation_loss", "signature": "def _compute_energy_conservation_loss(self, intermediates)"}, {"doc": "Penalize violation of symplectic structure.\n\nFor pairs of consecutive states (q_i, q_{i+1}), we interpret\nq as position and dq = q_{i+1} - q_i as a proxy for momentum.\nThe symplectic form dq ^ dp should be preserved.", "kind": "method", "line": 122, "name": "_compute_symplectic_loss", "signature": "def _compute_symplectic_loss(self, intermediates)"}, {"doc": "Penalize spectral divergence in frequency domain.\n\n||FFT(prediction) - FFT(target)||_F / ||FFT(target)||_F", "kind": "method", "line": 145, "name": "_compute_spectral_consistency_loss", "signature": "def _compute_spectral_consistency_loss(self, prediction, target)"}, {"doc": "Penalize phase misalignment between prediction and target.\n\n1 - |mean(exp(i * (angle(FFT(pred)) - angle(FFT(target)))))|", "kind": "method", "line": 161, "name": "_compute_phase_coherence_loss", "signature": "def _compute_phase_coherence_loss(self, prediction, target)"}, {"doc": "Principle of least action: minimize the total action\nS = sum(|phi_{i+1} - phi_i|) along the trajectory.", "kind": "method", "line": 177, "name": "_compute_action_minimization_loss", "signature": "def _compute_action_minimization_loss(self, intermediates)"}, {"doc": "Liouville theorem: phase space volume should be preserved.\n\nWe approximate this by checking that the variance of hidden\nstates remains approximately constant through evolution.", "kind": "method", "line": 192, "name": "_compute_liouville_loss", "signature": "def _compute_liouville_loss(self, intermediates)"}, {"doc": "Hamilton's equations: dq/dt = dH/dp, dp/dt = -dH/dq.\n\nApproximated by checking time-reversal symmetry:\nthe forward evolution followed by reverse should return\nto the initial state.", "kind": "method", "line": 214, "name": "_compute_hamiltonian_constraint_loss", "signature": "def _compute_hamiltonian_constraint_loss(self, intermediates)"}]}, {"id": "audio/main.py", "kind": "module", "label": "main.py", "language": "py", "sha256": "7280cde134cfdccc", "symbol_count": 7, "symbols": [{"doc": "Construct the complete argument parser with all configurable parameters.", "kind": "function", "line": 34, "name": "build_argument_parser", "signature": "def build_argument_parser()"}, {"doc": "Construct the full configuration from parsed CLI arguments.", "kind": "function", "line": 103, "name": "build_config_from_args", "signature": "def build_config_from_args(args)"}, {"doc": "Validate that the audio file exists and has a supported extension.", "kind": "function", "line": 163, "name": "validate_audio_file", "signature": "def validate_audio_file(file_path)"}, {"doc": "Print a formatted configuration summary.", "kind": "function", "line": 175, "name": "print_configuration_banner", "signature": "def print_configuration_banner(config, mode, audio_path)"}, {"doc": "Execute the training pipeline.", "kind": "function", "line": 215, "name": "run_training", "signature": "def run_training(args)"}, {"doc": "Execute the inference pipeline.", "kind": "function", "line": 226, "name": "run_inference", "signature": "def run_inference(args)"}, {"doc": "Main entry point.", "kind": "function", "line": 236, "name": "main", "signature": "def main()"}]}, {"id": "audio/metrics.py", "kind": "module", "label": "metrics.py", "language": "py", "sha256": "db887d65746cc11f", "symbol_count": 25, "symbols": [{"doc": "Tracks and computes all Hamiltonian mechanics metrics during\ntraining and inference.\n\nEach metric method is a pure computation with no side effects\nbeyond updating internal accumulators, following the\nInterface Segregation Principle by exposing granular metric methods.", "kind": "class", "line": 26, "name": "HamiltonianMetricsTracker", "signature": "class HamiltonianMetricsTracker"}, {"kind": "method", "line": 36, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Pre-allocate deque buffers for each tracked metric.", "kind": "method", "line": 43, "name": "_initialize_history_buffers", "signature": "def _initialize_history_buffers(self)"}, {"doc": "Compute the Hamiltonian H(q, p) = T(p) + V(q).\n\nT(p) = 0.5 * ||p||^2 (kinetic energy)\nV(q) = 0.5 * ||q||^2 (potential energy in harmonic approximation)\n\nArgs:\n    q: Generalized coordinates tensor (position in phase space).\n    p: Conjugate momenta tensor.\n\nReturns:\n    Scalar Hamiltonian energy value.", "kind": "method", "line": 74, "name": "compute_hamiltonian_energy", "signature": "def compute_hamiltonian_energy(self, q, p)"}, {"doc": "Compute the symplectic 2-form omega(dq, dp) = sum(dq_i ^ dp_i).\n\nMeasures preservation of the canonical symplectic structure\nunder Hamiltonian flow. Should remain invariant for symplectic\nintegrators.\n\nArgs:\n    q: Generalized coordinates.\n    p: Conjugate momenta.\n    dq: Variation in coordinates.\n    dp: Variation in momenta.\n\nReturns:\n    Scalar symplectic form magnitude.", "kind": "method", "line": 97, "name": "compute_symplectic_form", "signature": "def compute_symplectic_form(self, q, p, dq, dp)"}, {"doc": "Compute Liouville measure |det(J)| for the flow map Jacobian.\n\nBy Liouville's theorem, Hamiltonian flow preserves phase space\nvolume, so det(J) should equal 1 for exact symplectic evolution.\n\nArgs:\n    jacobian: The Jacobian matrix of the phase space transformation.\n\nReturns:\n    Absolute determinant of the Jacobian.", "kind": "method", "line": 125, "name": "compute_liouville_measure", "signature": "def compute_liouville_measure(self, jacobian)"}, {"doc": "Estimate phase space volume occupied by the state (q, p).\n\nUses the covariance ellipsoid approximation:\nV ~ sqrt(det(Cov([q, p])))\n\nArgs:\n    q: Generalized coordinates (flattened).\n    p: Conjugate momenta (flattened).\n\nReturns:\n    Estimated phase space volume.", "kind": "method", "line": 153, "name": "compute_phase_space_volume", "signature": "def compute_phase_space_volume(self, q, p)"}, {"doc": "Compute the action integral S = integral(L dt) along a trajectory.\n\nL = T - V = 0.5*||p||^2 - 0.5*||q||^2 (Lagrangian)\n\nArgs:\n    q_trajectory: Sequence of coordinate states [T, ...].\n    p_trajectory: Sequence of momentum states [T, ...].\n    dt: Time step between trajectory points.\n\nReturns:\n    Total action along the trajectory.", "kind": "method", "line": 181, "name": "compute_action_integral", "signature": "def compute_action_integral(self, q_trajectory, p_trajectory, dt)"}, {"doc": "Estimate the Poisson bracket {f, g} = sum(df/dq * dg/dp - df/dp * dg/dq).\n\nUses finite differences on the discretized phase space.\n\nArgs:\n    f_values: Observable f evaluated on phase space grid.\n    g_values: Observable g evaluated on phase space grid.\n    q: Coordinate grid.\n    p: Momentum grid.\n\nReturns:\n    Estimated Poisson bracket scalar.", "kind": "method", "line": 208, "name": "compute_poisson_bracket", "signature": "def compute_poisson_bracket(self, f_values, g_values, q, p)"}, {"doc": "Compute spectral entropy H = -sum(p_i * log(p_i)).\n\nMeasures the disorder/uniformity of the spectral distribution.\nMaximum entropy indicates uniform spectrum (white noise),\nminimum indicates pure tone (single frequency).\n\nArgs:\n    spectrum: Power spectrum tensor (non-negative).\n\nReturns:\n    Scalar spectral entropy.", "kind": "method", "line": 238, "name": "compute_spectral_entropy", "signature": "def compute_spectral_entropy(self, spectrum)"}, {"doc": "Compute Signal-to-Noise Ratio in dB.\n\nSNR = 10 * log10(||original||^2 / ||original - reconstructed||^2)\n\nArgs:\n    original: Ground truth signal.\n    reconstructed: Reconstructed signal.\n\nReturns:\n    SNR in decibels.", "kind": "method", "line": 259, "name": "compute_reconstruction_snr", "signature": "def compute_reconstruction_snr(self, original, reconstructed)"}, {"doc": "Compute spectral convergence metric.\n\nSC = ||S_orig - S_recon||_F / ||S_orig||_F\n\nLower values indicate better spectral fidelity.\n\nArgs:\n    original_spectrum: Original frequency domain representation.\n    reconstructed_spectrum: Reconstructed frequency domain representation.\n\nReturns:\n    Spectral convergence ratio.", "kind": "method", "line": 281, "name": "compute_spectral_convergence", "signature": "def compute_spectral_convergence(self, original_spectrum, reconstructed_spectrum)"}, {"doc": "Compute phase coherence between original and reconstructed signals.\n\nPC = |mean(exp(i * (phi_orig - phi_recon)))|\n\nValue of 1.0 indicates perfect phase alignment.\n\nArgs:\n    phase_original: Phase spectrum of original signal.\n    phase_reconstructed: Phase spectrum of reconstructed signal.\n\nReturns:\n    Phase coherence in [0, 1].", "kind": "method", "line": 305, "name": "compute_phase_coherence", "signature": "def compute_phase_coherence(self, phase_original, phase_reconstructed)"}, {"doc": "Compute relative energy drift from initial state.\n\ndrift = |E_current - E_initial| / (|E_initial| + epsilon)\n\nArgs:\n    energy_initial: Hamiltonian energy at t=0.\n    energy_current: Hamiltonian energy at current time.\n\nReturns:\n    Relative energy drift.", "kind": "method", "line": 328, "name": "compute_energy_drift", "signature": "def compute_energy_drift(self, energy_initial, energy_current)"}, {"doc": "Compute and record the total gradient norm across all parameters.", "kind": "method", "line": 348, "name": "record_gradient_norm", "signature": "def record_gradient_norm(self, model_parameters)"}, {"doc": "Compute and record the total parameter norm.", "kind": "method", "line": 359, "name": "record_parameter_norm", "signature": "def record_parameter_norm(self, model_parameters)"}, {"doc": "Record current learning rate.", "kind": "method", "line": 369, "name": "record_learning_rate", "signature": "def record_learning_rate(self, lr)"}, {"doc": "Record an individual loss component value.", "kind": "method", "line": 374, "name": "record_loss_component", "signature": "def record_loss_component(self, name, value)"}, {"doc": "Store a metric value in history and current snapshot.", "kind": "method", "line": 379, "name": "_record", "signature": "def _record(self, metric_name, value)"}, {"doc": "Return a snapshot of all current metric values.", "kind": "method", "line": 388, "name": "get_current_metrics", "signature": "def get_current_metrics(self)"}, {"doc": "Compute moving averages for all tracked metrics.", "kind": "method", "line": 392, "name": "get_moving_averages", "signature": "def get_moving_averages(self)"}, {"doc": "Format all current metrics into a human-readable string for progress bars.", "kind": "method", "line": 400, "name": "get_formatted_metrics_string", "signature": "def get_formatted_metrics_string(self)"}, {"doc": "Advance the global step counter.", "kind": "method", "line": 413, "name": "increment_step", "signature": "def increment_step(self)"}, {"kind": "method", "line": 418, "name": "step_count", "signature": "def step_count(self)"}, {"doc": "Determine if metrics should be logged at this step.", "kind": "method", "line": 421, "name": "should_log", "signature": "def should_log(self)"}]}, {"id": "audio/model.py", "kind": "module", "label": "model.py", "language": "py", "sha256": "ba02ca3c7063ae7a", "symbol_count": 11, "symbols": [{"doc": "Single Hamiltonian spectral evolution layer.\n\nPerforms frequency-domain evolution using learnable complex kernels.\nKernel shape: [hidden_dim, hidden_dim, kernel_base_height, kernel_base_width]\nmatching the original experiment2 architecture.", "kind": "class", "line": 27, "name": "SpectralEvolutionLayer", "signature": "class SpectralEvolutionLayer(Module)"}, {"doc": "Complete Hamiltonian Neural Network with parametric architecture.\n\nArchitecture (matching experiment2):\n    1. Input projection: Conv2d(1, hidden_dim, kernel, pad)\n    2. N spectral evolution layers with learnable complex kernels\n    3. Output projection: Conv2d(hidden_dim, 1, kernel, pad)", "kind": "class", "line": 162, "name": "HamiltonianNeuralNetwork", "signature": "class HamiltonianNeuralNetwork(Module)"}, {"kind": "method", "line": 36, "name": "__init__", "signature": "def __init__(self, hidden_dim, kernel_base_height, kernel_base_width, init_std)"}, {"doc": "Apply one step of Hamiltonian spectral evolution via RFFT2.\n\nArgs:\n    x: Input tensor [B, C, H, W] in spatial domain.\n\nReturns:\n    Evolved tensor [B, C, H, W] in spatial domain.", "kind": "method", "line": 51, "name": "forward", "signature": "def forward(self, x)"}, {"doc": "Full complex FFT evolution for amplitude and phase extraction.\n\nUses full FFT2 (not RFFT2) to preserve complete complex structure.\n\nArgs:\n    x: Input tensor [B, C, H, W].\n    target_height: Output spatial height.\n    target_width: Output spatial width.\n\nReturns:\n    Complex-valued evolved field in spatial domain.", "kind": "method", "line": 85, "name": "evolve_complex", "signature": "def evolve_complex(self, x, target_height, target_width)"}, {"doc": "Real FFT evolution for action map computation.\n\nArgs:\n    x: Input tensor [B, C, H, W].\n    target_height: Output spatial height.\n    target_width: Output spatial width.\n\nReturns:\n    Real-valued evolved field in spatial domain.", "kind": "method", "line": 124, "name": "evolve_real", "signature": "def evolve_real(self, x, target_height, target_width)"}, {"kind": "method", "line": 172, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Full forward pass: project -> evolve -> reconstruct.\n\nArgs:\n    x: Input tensor [B, 1, H, W].\n\nReturns:\n    Reconstructed tensor [B, 1, H, W].", "kind": "method", "line": 200, "name": "forward", "signature": "def forward(self, x)"}, {"doc": "Forward pass returning intermediate hidden states for analysis.\n\nArgs:\n    x: Input tensor [B, 1, H, W].\n\nReturns:\n    Tuple of (output, list of intermediate states).", "kind": "method", "line": 216, "name": "forward_with_intermediates", "signature": "def forward_with_intermediates(self, x)"}, {"doc": "Extract the three Hamiltonian field representations:\n1. Amplitude map (energy density / resonance)\n2. Phase map (topological structure / vortices)\n3. Action map (constructive interference = clear vision)\n\nMirrors the visual processing logic from the original code.\n\nArgs:\n    x: Input tensor [B, 1, H, W].\n\nReturns:\n    Tuple of (amplitude_map, phase_map, action_map) each [H, W].", "kind": "method", "line": 237, "name": "extract_hamiltonian_fields", "signature": "def extract_hamiltonian_fields(self, x)"}, {"doc": "Compute the Hamiltonian energy mask for spectral reconstruction.\n\nThis method implements the constructive interference principle:\nthe complex FFT evolution reveals amplitude (resonance) and\nphase (topology). Their constructive sum amplitude * cos(phase)\nidentifies WHERE in the time-frequency plane the model detects\ncoherent energy structure.\n\nThe real FFT evolution provides a complementary view (action),\nwhich highlights WHERE the model sees change/structure.\n\nBoth are combined and normalized to [0, 1] as a mask that\ncan be applied to the original STFT magnitude to produce\nthe reconstructed audio.\n\nThis is the audio equivalent of the \"clear vision\" (action map)\nin the visual domain.\n\nArgs:\n    x: STFT magnitude input [B, 1, freq_bins, time_frames],\n       normalized to [0, 1].\n\nReturns:\n    Energy mask [B, 1, freq_bins, time_frames] in [0, 1].", "kind": "method", "line": 270, "name": "compute_energy_mask", "signature": "def compute_energy_mask(self, x)"}]}, {"id": "audio/trainer.py", "kind": "module", "label": "trainer.py", "language": "py", "sha256": "a0dab0db89567b18", "symbol_count": 11, "symbols": [{"doc": "Builds a TensorDataset of spectrogram patches from an audio file.\n\nSegments the full mel spectrogram into overlapping patches\nof size (n_mels, matrix_size_width) for training.", "kind": "class", "line": 38, "name": "AudioSpectrogramDatasetBuilder", "signature": "class AudioSpectrogramDatasetBuilder"}, {"doc": "Complete training pipeline for the Hamiltonian Audio Network.\n\nManages:\n- Model initialization and checkpoint recovery\n- Optimizer and scheduler configuration\n- Training and validation loops\n- Full metric reporting at every step\n- Time-based checkpointing\n- Early stopping", "kind": "class", "line": 79, "name": "HamiltonianAudioTrainer", "signature": "class HamiltonianAudioTrainer"}, {"kind": "method", "line": 46, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Segment a mel spectrogram into training patches.\n\nArgs:\n    mel_spectrogram: Full spectrogram [1, 1, n_mels, time_frames].\n\nReturns:\n    TensorDataset of (input_patch, target_patch) pairs.", "kind": "method", "line": 49, "name": "build_dataset", "signature": "def build_dataset(self, mel_spectrogram)"}, {"kind": "method", "line": 92, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Load existing checkpoint if available.", "kind": "method", "line": 129, "name": "_attempt_checkpoint_recovery", "signature": "def _attempt_checkpoint_recovery(self)"}, {"doc": "Execute the full training pipeline on an audio file.\n\nArgs:\n    audio_file_path: Path to the input audio file.", "kind": "method", "line": 151, "name": "train", "signature": "def train(self, audio_file_path)"}, {"doc": "Execute one training epoch with full metric tracking.", "kind": "method", "line": 255, "name": "_train_one_epoch", "signature": "def _train_one_epoch(self, train_loader, epoch)"}, {"doc": "Run validation pass and return metrics.", "kind": "method", "line": 344, "name": "_validate", "signature": "def _validate(self, val_loader, epoch)"}, {"kind": "method", "line": 377, "name": "model", "signature": "def model(self)"}, {"kind": "method", "line": 381, "name": "audio_processor", "signature": "def audio_processor(self)"}]}, {"id": "audio/visualization.py", "kind": "module", "label": "visualization.py", "language": "py", "sha256": "a959c254455541de", "symbol_count": 8, "symbols": [{"doc": "Generates all scientific visualizations for Hamiltonian audio analysis.", "kind": "class", "line": 29, "name": "HamiltonianAudioVisualizer", "signature": "class HamiltonianAudioVisualizer"}, {"kind": "method", "line": 34, "name": "__init__", "signature": "def __init__(self, vis_config, audio_config)"}, {"doc": "Generate the complete suite of Hamiltonian analysis visualizations.\n\nArgs:\n    amplitude_map: Energy density field [H, W].\n    phase_map: Phase topology field [H, W].\n    action_map: Action density field [H, W].\n    original_spectrogram: Original mel spectrogram [1, 1, H, W].\n    reconstructed_spectrogram: Reconstructed mel spectrogram [1, 1, H, W].\n    original_waveform: Original audio waveform [1, T].\n    reconstructed_waveform: Reconstructed audio waveform [1, T].\n    output_prefix: Filename prefix for all outputs.", "kind": "method", "line": 43, "name": "render_complete_analysis", "signature": "def render_complete_analysis(self, amplitude_map, phase_map, action_map, original_spectrogram, reconstructed_spectrogram, original_waveform, reconstructed_waveform, output_prefix)"}, {"doc": "Render the three Hamiltonian field visualizations.", "kind": "method", "line": 91, "name": "_render_hamiltonian_fields", "signature": "def _render_hamiltonian_fields(self, amplitude_map, phase_map, action_map, output_prefix)"}, {"doc": "Render original vs reconstructed spectrogram comparison.", "kind": "method", "line": 140, "name": "_render_spectrogram_comparison", "signature": "def _render_spectrogram_comparison(self, original, reconstructed, output_prefix)"}, {"doc": "Render 2D phase portrait (amplitude vs phase histogram).", "kind": "method", "line": 185, "name": "_render_phase_portrait", "signature": "def _render_phase_portrait(self, amplitude_map, phase_map, output_prefix)"}, {"doc": "Render energy landscape as a 3D surface plot.", "kind": "method", "line": 218, "name": "_render_energy_landscape", "signature": "def _render_energy_landscape(self, amplitude_map, action_map, output_prefix)"}, {"doc": "Render original vs reconstructed waveform comparison.", "kind": "method", "line": 270, "name": "_render_waveform_comparison", "signature": "def _render_waveform_comparison(self, original_waveform, reconstructed_waveform, output_prefix)"}]}, {"id": "check_fase_berry.py", "kind": "module", "label": "check_fase_berry.py", "language": "py", "sha256": "00e04d89d26618fb", "symbol_count": 0, "symbols": []}, {"id": "diff_weights.py", "kind": "module", "label": "diff_weights.py", "language": "py", "sha256": "773ed0d0ac1df8ff", "symbol_count": 1, "symbols": [{"kind": "function", "line": 5, "name": "analize_checkpoint", "signature": "def analize_checkpoint(path)"}]}, {"id": "dirac.py", "kind": "module", "label": "dirac.py", "language": "py", "sha256": "34cde3b6d0b577dc", "symbol_count": 19, "symbols": [{"kind": "class", "line": 25, "name": "DiracConfig", "signature": "class DiracConfig"}, {"kind": "class", "line": 38, "name": "DiracDeltaAnalyzer", "signature": "class DiracDeltaAnalyzer"}, {"kind": "class", "line": 328, "name": "DiracVisualizer", "signature": "class DiracVisualizer"}, {"kind": "method", "line": 574, "name": "analyze_checkpoint", "signature": "def analyze_checkpoint(checkpoint_path, output_dir)"}, {"kind": "method", "line": 620, "name": "analyze_multiple_checkpoints", "signature": "def analyze_multiple_checkpoints(checkpoint_dir, n_latest, output_dir)"}, {"kind": "method", "line": 652, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 40, "name": "__init__", "signature": "def __init__(self, checkpoint_path, device)"}, {"kind": "method", "line": 64, "name": "extract_charge_distribution", "signature": "def extract_charge_distribution(self)"}, {"kind": "method", "line": 77, "name": "compute_dirac_delta_approximation", "signature": "def compute_dirac_delta_approximation(self, charge_density)"}, {"kind": "method", "line": 112, "name": "compute_electric_field", "signature": "def compute_electric_field(self, dirac_data, eval_points)"}, {"kind": "method", "line": 157, "name": "compute_electric_flux", "signature": "def compute_electric_flux(self, electric_field, surface_points)"}, {"kind": "method", "line": 192, "name": "compute_divergence", "signature": "def compute_divergence(self, electric_field)"}, {"kind": "method", "line": 200, "name": "verify_gauss_law", "signature": "def verify_gauss_law(self, dirac_data, flux_data)"}, {"kind": "method", "line": 223, "name": "analyze_all", "signature": "def analyze_all(self)"}, {"kind": "method", "line": 279, "name": "_print_report", "signature": "def _print_report(self, results)"}, {"kind": "method", "line": 331, "name": "plot_charge_distribution", "signature": "def plot_charge_distribution(charge_density, point_positions, point_charges, output_path)"}, {"kind": "method", "line": 379, "name": "plot_electric_field", "signature": "def plot_electric_field(electric_field, output_path)"}, {"kind": "method", "line": 441, "name": "plot_divergence", "signature": "def plot_divergence(divergence, output_path)"}, {"kind": "method", "line": 490, "name": "plot_combined_analysis", "signature": "def plot_combined_analysis(charge_density, point_positions, point_charges, electric_field, divergence, output_path)"}]}, {"id": "expand.py", "kind": "module", "label": "expand.py", "language": "py", "sha256": "fc9f20e83e76197a", "symbol_count": 5, "symbols": [{"kind": "function", "line": 18, "name": "load_config", "signature": "def load_config(toml_path)"}, {"doc": "Expand spectral kernels via zero-padding in frequency domain.", "kind": "function", "line": 23, "name": "expand_spectral_weights", "signature": "def expand_spectral_weights(kernel_real, kernel_imag, target_size, source_size)"}, {"doc": "Create a new model with expanded spectral weights.", "kind": "function", "line": 43, "name": "expand_model", "signature": "def expand_model(model, target_resolution, source_resolution)"}, {"doc": "Evaluate expanded model on synthetic data.", "kind": "function", "line": 74, "name": "evaluate_model", "signature": "def evaluate_model(model, resolution, device)"}, {"kind": "function", "line": 105, "name": "main", "signature": "def main()"}]}, {"id": "experiment.py", "kind": "module", "label": "experiment.py", "language": "py", "sha256": "25adec6b3c9d64e9", "symbol_count": 57, "symbols": [{"kind": "class", "line": 41, "name": "Config", "signature": "class Config"}, {"kind": "method", "line": 88, "name": "set_seed", "signature": "def set_seed(seed)"}, {"kind": "method", "line": 96, "name": "setup_logger", "signature": "def setup_logger(name, level)"}, {"kind": "class", "line": 109, "name": "IAnalysisStrategy", "signature": "class IAnalysisStrategy(ABC)"}, {"kind": "class", "line": 115, "name": "IMetricsCalculator", "signature": "class IMetricsCalculator(ABC)"}, {"doc": "True Hamiltonian operator H = -nabla^2 on torus.", "kind": "class", "line": 121, "name": "HamiltonianOperator", "signature": "class HamiltonianOperator"}, {"doc": "Fast dataset for Hamiltonian operator learning.", "kind": "class", "line": 146, "name": "FastDataset", "signature": "class FastDataset(Dataset)"}, {"doc": "Spectral layer with correct complex multiplication.", "kind": "class", "line": 203, "name": "SpectralLayer", "signature": "class SpectralLayer(Module)"}, {"doc": "Compact network for Hamiltonian operator learning.", "kind": "class", "line": 255, "name": "SimpleHamiltonianNet", "signature": "class SimpleHamiltonianNet(Module)"}, {"kind": "class", "line": 293, "name": "LocalComplexityAnalyzer", "signature": "class LocalComplexityAnalyzer"}, {"kind": "class", "line": 310, "name": "SuperpositionAnalyzer", "signature": "class SuperpositionAnalyzer"}, {"kind": "class", "line": 340, "name": "CrystallographyMetrics", "signature": "class CrystallographyMetrics"}, {"kind": "class", "line": 444, "name": "ThermodynamicMetrics", "signature": "class ThermodynamicMetrics"}, {"kind": "class", "line": 470, "name": "SpectroscopyMetrics", "signature": "class SpectroscopyMetrics"}, {"kind": "class", "line": 500, "name": "CheckpointManager", "signature": "class CheckpointManager"}, {"kind": "class", "line": 573, "name": "TrainingMonitor", "signature": "class TrainingMonitor"}, {"kind": "class", "line": 611, "name": "GlassStopper", "signature": "class GlassStopper"}, {"doc": "Train model with early stopping for glass detection.", "kind": "method", "line": 670, "name": "train_with_early_glass_stop", "signature": "def train_with_early_glass_stop(model, optimizer, seed, epochs)"}, {"doc": "Mine for crystal seeds by trying sequential seeds.", "kind": "method", "line": 803, "name": "seed_miner", "signature": "def seed_miner(total_attempts)"}, {"kind": "method", "line": 856, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 111, "name": "analyze", "signature": "def analyze(self, model)"}, {"kind": "method", "line": 117, "name": "compute", "signature": "def compute(self, model)"}, {"kind": "method", "line": 124, "name": "__init__", "signature": "def __init__(self, grid_size)"}, {"kind": "method", "line": 128, "name": "_precompute_spectral_operators", "signature": "def _precompute_spectral_operators(self)"}, {"kind": "method", "line": 135, "name": "apply", "signature": "def apply(self, field)"}, {"kind": "method", "line": 140, "name": "time_evolution", "signature": "def time_evolution(self, field, dt)"}, {"kind": "method", "line": 149, "name": "__init__", "signature": "def __init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)"}, {"kind": "method", "line": 193, "name": "__len__", "signature": "def __len__(self)"}, {"kind": "method", "line": 196, "name": "__getitem__", "signature": "def __getitem__(self, idx)"}, {"kind": "method", "line": 199, "name": "get_val_batch", "signature": "def get_val_batch(self)"}, {"kind": "method", "line": 206, "name": "__init__", "signature": "def __init__(self, channels, grid_size)"}, {"kind": "method", "line": 219, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 258, "name": "__init__", "signature": "def __init__(self, grid_size, hidden_dim, num_spectral_layers)"}, {"kind": "method", "line": 279, "name": "forward", "signature": "def forward(self, x)"}, {"doc": "Compute Local Complexity (LC) metric for weight matrix.", "kind": "method", "line": 295, "name": "compute_local_complexity", "signature": "def compute_local_complexity(weights, epsilon)"}, {"doc": "Compute Superposition (SP) metric for weight matrix.", "kind": "method", "line": 312, "name": "compute_superposition", "signature": "def compute_superposition(weights)"}, {"kind": "method", "line": 342, "name": "compute_kappa", "signature": "def compute_kappa(model, dataloader, num_batches)"}, {"kind": "method", "line": 378, "name": "compute_discretization_margin", "signature": "def compute_discretization_margin(coeffs)"}, {"kind": "method", "line": 387, "name": "compute_alpha_purity", "signature": "def compute_alpha_purity(coeffs)"}, {"kind": "method", "line": 394, "name": "compute_kappa_quantum", "signature": "def compute_kappa_quantum(coeffs, hbar)"}, {"kind": "method", "line": 411, "name": "compute_poynting_vector", "signature": "def compute_poynting_vector(coeffs)"}, {"kind": "method", "line": 426, "name": "compute_all_metrics", "signature": "def compute_all_metrics(model, dataloader)"}, {"kind": "method", "line": 446, "name": "compute_effective_temperature", "signature": "def compute_effective_temperature(gradient_buffer, learning_rate)"}, {"kind": "method", "line": 460, "name": "compute_specific_heat", "signature": "def compute_specific_heat(loss_history, temp_history, cv_threshold)"}, {"kind": "method", "line": 472, "name": "compute_weight_diffraction", "signature": "def compute_weight_diffraction(coeffs)"}, {"kind": "method", "line": 491, "name": "_compute_spectral_entropy", "signature": "def _compute_spectral_entropy(power_spectrum)"}, {"kind": "method", "line": 501, "name": "__init__", "signature": "def __init__(self, interval_minutes, max_checkpoints)"}, {"kind": "method", "line": 509, "name": "should_save_checkpoint", "signature": "def should_save_checkpoint(self)"}, {"kind": "method", "line": 514, "name": "save_checkpoint", "signature": "def save_checkpoint(self, model, optimizer, epoch, metrics)"}, {"kind": "method", "line": 574, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 594, "name": "update_metrics", "signature": "def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat, poynting_magnitude)"}, {"kind": "method", "line": 612, "name": "__init__", "signature": "def __init__(self, patience_epochs)"}, {"doc": "Check if the system is in glass state and should stop mining.", "kind": "method", "line": 616, "name": "should_stop", "signature": "def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)"}, {"kind": "class", "line": 879, "name": "BoltzmannAnalysisProgram", "signature": "class BoltzmannAnalysisProgram"}, {"kind": "method", "line": 880, "name": "__init__", "signature": "def __init__(self, checkpoint_path, results_dir)"}, {"kind": "method", "line": 886, "name": "load_and_analyze_checkpoint", "signature": "def load_and_analyze_checkpoint(self)"}, {"kind": "method", "line": 903, "name": "dataloader", "signature": "def dataloader()"}]}, {"id": "experiment2.py", "kind": "module", "label": "experiment2.py", "language": "py", "sha256": "772c52a21febab12", "symbol_count": 83, "symbols": [{"kind": "class", "line": 22, "name": "Config", "signature": "class Config"}, {"kind": "class", "line": 74, "name": "SeedManager", "signature": "class SeedManager"}, {"kind": "class", "line": 84, "name": "LoggerFactory", "signature": "class LoggerFactory"}, {"kind": "class", "line": 99, "name": "IAnalysisStrategy", "signature": "class IAnalysisStrategy(ABC)"}, {"kind": "class", "line": 105, "name": "IMetricsCalculator", "signature": "class IMetricsCalculator(ABC)"}, {"kind": "class", "line": 111, "name": "HamiltonianOperator", "signature": "class HamiltonianOperator"}, {"kind": "class", "line": 133, "name": "HamiltonianDataset", "signature": "class HamiltonianDataset(Dataset)"}, {"kind": "class", "line": 183, "name": "SpectralLayer", "signature": "class SpectralLayer(Module)"}, {"kind": "class", "line": 227, "name": "HamiltonianNeuralNetwork", "signature": "class HamiltonianNeuralNetwork(Module)"}, {"kind": "class", "line": 257, "name": "LocalComplexityAnalyzer", "signature": "class LocalComplexityAnalyzer"}, {"kind": "class", "line": 273, "name": "SuperpositionAnalyzer", "signature": "class SuperpositionAnalyzer"}, {"kind": "class", "line": 302, "name": "CrystallographyMetricsCalculator", "signature": "class CrystallographyMetricsCalculator(IMetricsCalculator)"}, {"kind": "class", "line": 737, "name": "ThermodynamicMetricsCalculator", "signature": "class ThermodynamicMetricsCalculator(IMetricsCalculator)"}, {"kind": "class", "line": 770, "name": "SpectroscopyMetricsCalculator", "signature": "class SpectroscopyMetricsCalculator(IMetricsCalculator)"}, {"kind": "class", "line": 804, "name": "CheckpointManager", "signature": "class CheckpointManager"}, {"kind": "class", "line": 874, "name": "TrainingMetricsMonitor", "signature": "class TrainingMetricsMonitor"}, {"kind": "class", "line": 912, "name": "GlassStateDetector", "signature": "class GlassStateDetector"}, {"kind": "class", "line": 973, "name": "TrainingEngine", "signature": "class TrainingEngine"}, {"kind": "class", "line": 1113, "name": "SeedMiningSystem", "signature": "class SeedMiningSystem"}, {"kind": "class", "line": 1164, "name": "SingleExperimentRunner", "signature": "class SingleExperimentRunner"}, {"kind": "class", "line": 1223, "name": "CheckpointAnalyzer", "signature": "class CheckpointAnalyzer"}, {"kind": "class", "line": 1275, "name": "Application", "signature": "class Application"}, {"kind": "method", "line": 1334, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 76, "name": "set_seed", "signature": "def set_seed(seed)"}, {"kind": "method", "line": 86, "name": "create_logger", "signature": "def create_logger(name, level)"}, {"kind": "method", "line": 101, "name": "analyze", "signature": "def analyze(self, model)"}, {"kind": "method", "line": 107, "name": "compute", "signature": "def compute(self, model)"}, {"kind": "method", "line": 112, "name": "__init__", "signature": "def __init__(self, grid_size)"}, {"kind": "method", "line": 116, "name": "_precompute_spectral_operators", "signature": "def _precompute_spectral_operators(self)"}, {"kind": "method", "line": 122, "name": "apply", "signature": "def apply(self, field)"}, {"kind": "method", "line": 127, "name": "time_evolution", "signature": "def time_evolution(self, field, dt)"}, {"kind": "method", "line": 134, "name": "__init__", "signature": "def __init__(self, num_samples, grid_size, time_steps, dt, train_ratio)"}, {"kind": "method", "line": 173, "name": "__len__", "signature": "def __len__(self)"}, {"kind": "method", "line": 176, "name": "__getitem__", "signature": "def __getitem__(self, idx)"}, {"kind": "method", "line": 179, "name": "get_validation_batch", "signature": "def get_validation_batch(self)"}, {"kind": "method", "line": 184, "name": "__init__", "signature": "def __init__(self, channels, grid_size)"}, {"kind": "method", "line": 195, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 228, "name": "__init__", "signature": "def __init__(self, grid_size, hidden_dim, num_spectral_layers)"}, {"kind": "method", "line": 243, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 259, "name": "compute_local_complexity", "signature": "def compute_local_complexity(weights, epsilon)"}, {"kind": "method", "line": 275, "name": "compute_superposition", "signature": "def compute_superposition(weights)"}, {"doc": "Implementación de interfaz IMetricsCalculator.\nDelega a compute_all_metrics con los argumentos correctos.", "kind": "method", "line": 303, "name": "compute", "signature": "def compute(self, model, val_x, val_y)"}, {"kind": "method", "line": 311, "name": "compute_gradient_covariance_kappa", "signature": "def compute_gradient_covariance_kappa(model, dataloader, num_batches)"}, {"doc": "Calcula el margen de discretización desde los parámetros del modelo.\nVersión estática que no requiere diccionario externo.", "kind": "method", "line": 348, "name": "compute_discretization_margin_from_state_dict", "signature": "def compute_discretization_margin_from_state_dict(model)"}, {"doc": "Calcula el margen de discretización desde un diccionario de coeficientes.", "kind": "method", "line": 361, "name": "compute_discretization_margin", "signature": "def compute_discretization_margin(coeffs)"}, {"doc": "Calcula el índice de pureza alpha directamente desde el modelo.", "kind": "method", "line": 373, "name": "compute_alpha_purity_from_model", "signature": "def compute_alpha_purity_from_model(model)"}, {"doc": "Calcula el índice de pureza alpha desde un diccionario de coeficientes.", "kind": "method", "line": 383, "name": "compute_alpha_purity", "signature": "def compute_alpha_purity(coeffs)"}, {"doc": "Número de condición de la matriz de covarianza de gradientes.", "kind": "method", "line": 393, "name": "compute_kappa", "signature": "def compute_kappa(model, val_x, val_y, num_batches)"}, {"doc": "Versión del cálculo cuántico de kappa que opera directamente sobre el modelo.", "kind": "method", "line": 464, "name": "compute_kappa_quantum", "signature": "def compute_kappa_quantum(model, hbar)"}, {"doc": "Versión del cálculo cuántico de kappa desde diccionario de coeficientes.", "kind": "method", "line": 492, "name": "compute_kappa_quantum_from_coeffs", "signature": "def compute_kappa_quantum_from_coeffs(coeffs, hbar)"}, {"doc": "Métricas cristalográficas con aislamiento completo de errores.", "kind": "method", "line": 511, "name": "_compute_crystallography_metrics", "signature": "def _compute_crystallography_metrics(self, model, val_x, val_y)"}, {"doc": "Verifica integridad de pesos: NaN, Inf, y estadísticas básicas.", "kind": "method", "line": 539, "name": "_check_weight_integrity", "signature": "def _check_weight_integrity(self, model)"}, {"doc": "Vector de Poynting: flujo de energía en el espacio de parámetros.\nAnálogo electromagnético para redes neuronales.", "kind": "method", "line": 603, "name": "compute_poynting_vector", "signature": "def compute_poynting_vector(model)"}, {"doc": "Calcula todas las métricas cristalográficas con manejo de errores.", "kind": "method", "line": 679, "name": "compute_all_metrics", "signature": "def compute_all_metrics(model, val_x, val_y)"}, {"kind": "method", "line": 738, "name": "compute", "signature": "def compute(self, model, gradient_buffer, learning_rate, loss_history, temp_history)"}, {"kind": "method", "line": 747, "name": "compute_effective_temperature", "signature": "def compute_effective_temperature(gradient_buffer, learning_rate)"}, {"kind": "method", "line": 760, "name": "compute_specific_heat", "signature": "def compute_specific_heat(loss_history, temp_history, cv_threshold)"}, {"kind": "method", "line": 771, "name": "compute", "signature": "def compute(self, model)"}, {"kind": "method", "line": 776, "name": "compute_weight_diffraction", "signature": "def compute_weight_diffraction(coeffs)"}, {"kind": "method", "line": 795, "name": "_compute_spectral_entropy", "signature": "def _compute_spectral_entropy(power_spectrum)"}, {"kind": "method", "line": 805, "name": "__init__", "signature": "def __init__(self, interval_minutes, max_checkpoints)"}, {"kind": "method", "line": 813, "name": "should_save_checkpoint", "signature": "def should_save_checkpoint(self)"}, {"kind": "method", "line": 818, "name": "save_checkpoint", "signature": "def save_checkpoint(self, model, optimizer, epoch, metrics)"}, {"kind": "method", "line": 875, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 895, "name": "update_metrics", "signature": "def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat, poynting_magnitude)"}, {"kind": "method", "line": 913, "name": "__init__", "signature": "def __init__(self, patience_epochs)"}, {"kind": "method", "line": 918, "name": "should_stop", "signature": "def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)"}, {"kind": "method", "line": 963, "name": "is_crystal_formed", "signature": "def is_crystal_formed(self, lc, sp, kappa, delta, temp, cv)"}, {"kind": "method", "line": 974, "name": "__init__", "signature": "def __init__(self, model, optimizer, device, logger)"}, {"kind": "method", "line": 997, "name": "train_epoch", "signature": "def train_epoch(self, dataloader, epoch)"}, {"kind": "method", "line": 1027, "name": "validate", "signature": "def validate(self, val_x, val_y)"}, {"kind": "method", "line": 1040, "name": "compute_weight_metrics", "signature": "def compute_weight_metrics(self)"}, {"kind": "method", "line": 1056, "name": "execute_training", "signature": "def execute_training(self, dataloader, val_x, val_y, epochs, seed, early_stopping)"}, {"kind": "method", "line": 1114, "name": "__init__", "signature": "def __init__(self, max_attempts)"}, {"kind": "method", "line": 1118, "name": "mine", "signature": "def mine(self)"}, {"kind": "method", "line": 1165, "name": "__init__", "signature": "def __init__(self, seed, epochs, grid_size, hidden_dim, num_spectral_layers, learning_rate)"}, {"kind": "method", "line": 1175, "name": "run", "signature": "def run(self)"}, {"kind": "method", "line": 1224, "name": "__init__", "signature": "def __init__(self, checkpoint_path, results_dir)"}, {"kind": "method", "line": 1230, "name": "analyze", "signature": "def analyze(self)"}, {"kind": "method", "line": 1276, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 1280, "name": "_create_argument_parser", "signature": "def _create_argument_parser(self)"}, {"kind": "method", "line": 1294, "name": "run", "signature": "def run(self)"}, {"kind": "method", "line": 694, "name": "safe_compute", "signature": "def safe_compute(func)"}]}, {"id": "export.py", "kind": "module", "label": "export.py", "language": "py", "sha256": "600cb2a1a9c688c8", "symbol_count": 0, "symbols": []}, {"id": "get_meditions.py", "kind": "module", "label": "get_meditions.py", "language": "py", "sha256": "ee0046b1a510cdd5", "symbol_count": 52, "symbols": [{"doc": "Configuración termodinámica para análisis de HPU Core", "kind": "class", "line": 28, "name": "ThermodynamicConfig", "signature": "class ThermodynamicConfig"}, {"kind": "method", "line": 51, "name": "setup_logger", "signature": "def setup_logger(name, level)"}, {"doc": "Potencial de Helmholtz: F = U - T*S + μ*N + α_term*C", "kind": "class", "line": 72, "name": "ThermodynamicPotential", "signature": "class ThermodynamicPotential"}, {"doc": "Métricas de cristalografía para redes neuronales Hamiltonianas.\nMide la \"pureza\" estructural de los pesos aprendidos.", "kind": "class", "line": 98, "name": "CrystallographyMetrics", "signature": "class CrystallographyMetrics"}, {"doc": "Análisis termodinámico del proceso de entrenamiento.\nTemperatura efectiva, calor específico, transiciones de fase.", "kind": "class", "line": 394, "name": "ThermodynamicMetrics", "signature": "class ThermodynamicMetrics"}, {"doc": "Análisis espectroscópico de pesos: difracción, descomposición, parámetros de red.", "kind": "class", "line": 634, "name": "SpectroscopyMetrics", "signature": "class SpectroscopyMetrics"}, {"kind": "class", "line": 740, "name": "CheckpointVerifier", "signature": "class CheckpointVerifier"}, {"doc": "Verifica los N checkpoints más recientes con análisis completo", "kind": "method", "line": 1439, "name": "verify_latest_checkpoints", "signature": "def verify_latest_checkpoints(checkpoint_dir, n)"}, {"kind": "method", "line": 1500, "name": "main", "signature": "def main()"}, {"doc": "F = U - T*S (a μ y N constantes)", "kind": "method", "line": 81, "name": "helmholtz_free_energy", "signature": "def helmholtz_free_energy(self)"}, {"doc": "G = F + μ*N + P*V (presión algorítmica)", "kind": "method", "line": 85, "name": "gibbs_free_energy", "signature": "def gibbs_free_energy(self)"}, {"doc": "Criterio de estabilidad: dG < 0", "kind": "method", "line": 90, "name": "is_stable", "signature": "def is_stable(self)"}, {"doc": "Contenedor para coeficientes espectrales de HPU Core", "kind": "class", "line": 105, "name": "SpectralCoefficients", "signature": "class SpectralCoefficients"}, {"doc": "Número de condición de la matriz de covarianza de gradientes.\nFIX: Limita tamaño de gradientes y usa método iterativo si es necesario.", "kind": "method", "line": 127, "name": "compute_kappa", "signature": "def compute_kappa(model, val_x, val_y, num_batches)"}, {"doc": "δ = max |w - round(w)| sobre todos los parámetros.\nMide qué tan cerca están los pesos de valores enteros.", "kind": "method", "line": 187, "name": "compute_discretization_margin", "signature": "def compute_discretization_margin(model)"}, {"doc": "α = -log(δ). Pureza cristalina.\nα > 7 indica estructura cristalina perfecta.", "kind": "method", "line": 203, "name": "compute_alpha_purity", "signature": "def compute_alpha_purity(model)"}, {"doc": "Fracción de parámetros \"activos\" (no cerca de cero).", "kind": "method", "line": 214, "name": "compute_local_complexity", "signature": "def compute_local_complexity(model)"}, {"doc": "κ cuántico: número de condición con regularización cuántica.\nFIX: Usa método iterativo de potencia en lugar de matriz densa.", "kind": "method", "line": 226, "name": "compute_kappa_quantum", "signature": "def compute_kappa_quantum(model, hbar)"}, {"doc": "Método de potencia para estimar κ sin construir matriz.\nEstima λ_max y λ_min de la matriz de covarianza regularizada.", "kind": "method", "line": 255, "name": "_compute_kappa_iterative", "signature": "def _compute_kappa_iterative(params, hbar, n, max_iters, tol)"}, {"doc": "Vector de Poynting: flujo de energía en el espacio de parámetros.\nAnálogo electromagnético para redes neuronales.", "kind": "method", "line": 312, "name": "compute_poynting_vector", "signature": "def compute_poynting_vector(model)"}, {"doc": "Calcula todas las métricas cristalográficas.", "kind": "method", "line": 366, "name": "compute_all_metrics", "signature": "def compute_all_metrics(model, val_x, val_y)"}, {"doc": "T_eff = (lr/2) * Var(∇L). Temperatura de fluctuaciones.", "kind": "method", "line": 401, "name": "compute_effective_temperature", "signature": "def compute_effective_temperature(gradient_buffer, learning_rate)"}, {"doc": "C_v = Var(U) / T^2. Detecta transiciones de fase (picos en C_v).", "kind": "method", "line": 418, "name": "compute_specific_heat", "signature": "def compute_specific_heat(loss_history, temp_history, cv_threshold)"}, {"doc": "Exponentes críticos cerca de transiciones de fase.", "kind": "method", "line": 436, "name": "compute_critical_exponents", "signature": "def compute_critical_exponents(temp_history, cv_history, alpha_history)"}, {"doc": "Ecuación de estado: T_c(α) = T_0 * exp(-c*α)\nRelación constitutiva cristal-vidrio.", "kind": "method", "line": 504, "name": "compute_equation_of_state", "signature": "def compute_equation_of_state(temp_eff, alpha, kappa)"}, {"doc": "Información mutua pesos-gradientes.", "kind": "method", "line": 539, "name": "compute_mutual_information", "signature": "def compute_mutual_information(weights, gradients)"}, {"doc": "ħ algorítmico efectivo.", "kind": "method", "line": 561, "name": "estimate_hbar_algorithmic", "signature": "def estimate_hbar_algorithmic(model_complexity, weight_dim, mutual_information)"}, {"doc": "Matriz de información de Fisher.", "kind": "method", "line": 572, "name": "compute_fisher_information_matrix", "signature": "def compute_fisher_information_matrix(model, samples)"}, {"doc": "Curvatura de Ricci escalar.", "kind": "method", "line": 593, "name": "compute_ricci_curvature", "signature": "def compute_ricci_curvature(fisher_matrix)"}, {"doc": "Eficiencia de Carnot del proceso de aprendizaje.", "kind": "method", "line": 608, "name": "calculate_carnot_efficiency", "signature": "def calculate_carnot_efficiency(delta_alpha, total_flops, initial_alpha)"}, {"doc": "Patrón de difracción de pesos (FFT).\nDetecta periodicidad cristalina (picos de Bragg).", "kind": "method", "line": 640, "name": "compute_weight_diffraction", "signature": "def compute_weight_diffraction(model)"}, {"doc": "Entropía espectral de Shannon.", "kind": "method", "line": 672, "name": "_compute_spectral_entropy", "signature": "def _compute_spectral_entropy(power_spectrum)"}, {"doc": "Extrae parámetros de red vía SVD.", "kind": "method", "line": 680, "name": "extract_lattice_parameters", "signature": "def extract_lattice_parameters(weight_tensor, rank)"}, {"doc": "Energía libre de Gibbs.", "kind": "method", "line": 732, "name": "compute_gibbs_free_energy", "signature": "def compute_gibbs_free_energy(loss, temp, entropy)"}, {"kind": "method", "line": 741, "name": "__init__", "signature": "def __init__(self, checkpoint_path, device)"}, {"doc": "Calcula TODAS las métricas desde cero y compara con las guardadas", "kind": "method", "line": 782, "name": "verify_all_metrics", "signature": "def verify_all_metrics(self)"}, {"doc": "Verifica que los pesos no tengan NaN/Inf.\nFIX: Evita calcular std en tensores con 1 elemento.", "kind": "method", "line": 849, "name": "_check_weight_integrity", "signature": "def _check_weight_integrity(self)"}, {"doc": "Calcula MSE y accuracy de validación desde cero", "kind": "method", "line": 915, "name": "_compute_validation_metrics", "signature": "def _compute_validation_metrics(self)"}, {"doc": "Calcula delta, alpha, purity, etc.", "kind": "method", "line": 942, "name": "_compute_discretization_metrics", "signature": "def _compute_discretization_metrics(self)"}, {"doc": "Calcula la penalización de cuantización", "kind": "method", "line": 986, "name": "_compute_quantization_metrics", "signature": "def _compute_quantization_metrics(self)"}, {"doc": "Reconstruye el loss total", "kind": "method", "line": 1005, "name": "_compute_loss_metrics", "signature": "def _compute_loss_metrics(self)"}, {"doc": "Métricas cristalográficas completas", "kind": "method", "line": 1031, "name": "_compute_crystallography_metrics", "signature": "def _compute_crystallography_metrics(self)"}, {"doc": "Métricas termodinámicas", "kind": "method", "line": 1038, "name": "_compute_thermodynamic_metrics", "signature": "def _compute_thermodynamic_metrics(self)"}, {"doc": "Aproximación de curvatura de Ricci para HPU Core", "kind": "method", "line": 1113, "name": "_approximate_ricci_curvature", "signature": "def _approximate_ricci_curvature(self)"}, {"doc": "Análisis espectroscópico", "kind": "method", "line": 1134, "name": "_compute_spectroscopy", "signature": "def _compute_spectroscopy(self)"}, {"doc": "Calcula potencial termodinámico completo", "kind": "method", "line": 1164, "name": "_compute_thermodynamic_potential", "signature": "def _compute_thermodynamic_potential(self, results)"}, {"doc": "Compara métricas calculadas vs almacenadas en el checkpoint", "kind": "method", "line": 1188, "name": "_compare_with_stored", "signature": "def _compare_with_stored(self, computed)"}, {"doc": "Verifica consistencia entre métricas relacionadas", "kind": "method", "line": 1226, "name": "_check_internal_consistency", "signature": "def _check_internal_consistency(self, results)"}, {"doc": "Calcula un score de salud del checkpoint (0-100)", "kind": "method", "line": 1267, "name": "_compute_health_score", "signature": "def _compute_health_score(self, results)"}, {"doc": "Asigna grado cristalográfico", "kind": "method", "line": 1336, "name": "_assign_crystallographic_grade", "signature": "def _assign_crystallographic_grade(self, delta, alpha)"}, {"doc": "Imprime reporte formateado con todas las métricas nuevas", "kind": "method", "line": 1349, "name": "_print_report", "signature": "def _print_report(self, results)"}, {"doc": "Extrae coeficientes del modelo HPU Core", "kind": "method", "line": 112, "name": "from_model", "signature": "def from_model(cls, model)"}]}, {"id": "hamiltonian_mbl.py", "kind": "module", "label": "hamiltonian_mbl.py", "language": "py", "sha256": "5b774bf1a608fa83", "symbol_count": 126, "symbols": [{"doc": "Configuration for Hamiltonian Neural Network architecture.\nAll architectural hyperparameters are centralized here.", "kind": "class", "line": 37, "name": "HamiltonianArchitectureConfig", "signature": "class HamiltonianArchitectureConfig"}, {"doc": "Comprehensive configuration for MBL analysis of Hamiltonian NN crystallization.\nAll analysis parameters are centralized following SOLID principles.", "kind": "class", "line": 67, "name": "MBLAnalysisConfig", "signature": "class MBLAnalysisConfig"}, {"doc": "Configuration for training process.", "kind": "class", "line": 141, "name": "TrainingConfig", "signature": "class TrainingConfig"}, {"doc": "Protocol for models compatible with MBL analysis.", "kind": "class", "line": 160, "name": "IModel", "signature": "class IModel(Protocol)"}, {"doc": "Protocol for level spacing ratio calculation.", "kind": "class", "line": 167, "name": "ILevelSpacingCalculator", "signature": "class ILevelSpacingCalculator(Protocol)"}, {"doc": "Protocol for participation ratio calculation.", "kind": "class", "line": 173, "name": "IParticipationRatioCalculator", "signature": "class IParticipationRatioCalculator(Protocol)"}, {"doc": "Protocol for synthetic Planck's constant calculation.", "kind": "class", "line": 179, "name": "ISyntheticPlanckCalculator", "signature": "class ISyntheticPlanckCalculator(Protocol)"}, {"doc": "Protocol for discretization dial analysis.", "kind": "class", "line": 185, "name": "IDiscretizationDialAnalyzer", "signature": "class IDiscretizationDialAnalyzer(Protocol)"}, {"doc": "Protocol for checkpoint management.", "kind": "class", "line": 191, "name": "ICheckpointManager", "signature": "class ICheckpointManager(Protocol)"}, {"doc": "Protocol for collecting all training metrics.", "kind": "class", "line": 199, "name": "ITrainingMetricsCollector", "signature": "class ITrainingMetricsCollector(Protocol)"}, {"doc": "Migra pesos de SimpleHamiltonianNet (Conv2d) a HamiltonianNeuralNetwork (Linear).", "kind": "class", "line": 209, "name": "ArchitectureMigrator", "signature": "class ArchitectureMigrator"}, {"doc": "Spectral layer implementing Hamiltonian dynamics in Fourier space.\nPreserves energy conservation through symplectic integration.", "kind": "class", "line": 346, "name": "SpectralHamiltonianLayer", "signature": "class SpectralHamiltonianLayer(Module)"}, {"doc": "Complete Hamiltonian Neural Network for learning dynamical systems.\nUses spectral layers to ensure energy conservation and symplectic structure.", "kind": "class", "line": 412, "name": "HamiltonianNeuralNetwork", "signature": "class HamiltonianNeuralNetwork(Module)"}, {"doc": "Generates physics-informed training data for Hamiltonian NN.\nCreates trajectories from known dynamical systems.", "kind": "class", "line": 555, "name": "HamiltonianDataset", "signature": "class HamiltonianDataset"}, {"doc": "Calculates the level spacing ratio r for MBL phase detection.\n\nThe ratio r_n = min(delta_n, delta_{n+1}) / max(delta_n, delta_{n+1})\nwhere delta_n = E_{n+1} - E_n (energy level spacing).", "kind": "class", "line": 608, "name": "LevelSpacingRatioCalculator", "signature": "class LevelSpacingRatioCalculator"}, {"doc": "Calculates Inverse Participation Ratio (IPR) for localization analysis.\nIPR = sum_i |c_i|^4 where c_i are coefficients in the chosen basis.", "kind": "class", "line": 719, "name": "ParticipationRatioCalculator", "signature": "class ParticipationRatioCalculator"}, {"doc": "Calculates effective synthetic Planck's constant (hbar_eff) from model properties.\nBased on the relation: hbar_eff ∝ 1 / sqrt(PR * Energy_Gap)", "kind": "class", "line": 801, "name": "SyntheticPlanckConstantCalculator", "signature": "class SyntheticPlanckConstantCalculator"}, {"doc": "Analyzes the discretization parameter delta as a phase transition control.", "kind": "class", "line": 844, "name": "DiscretizationDialAnalyzer", "signature": "class DiscretizationDialAnalyzer"}, {"doc": "Calculates the 'crystallinity' of the weight distribution.", "kind": "class", "line": 947, "name": "PurityIndexCalculator", "signature": "class PurityIndexCalculator"}, {"doc": "Calculates effective temperature from loss history.", "kind": "class", "line": 1004, "name": "EffectiveTemperatureCalculator", "signature": "class EffectiveTemperatureCalculator"}, {"doc": "Calculates Krylov complexity as a measure of operator growth and scrambling.\nBased on the spread of operators in Krylov space.", "kind": "class", "line": 1048, "name": "KrylovComplexityCalculator", "signature": "class KrylovComplexityCalculator"}, {"doc": "Calculates crystallinity index through spectral analysis of weight matrices.\nAnalogous to X-ray diffraction for physical crystals.", "kind": "class", "line": 1089, "name": "CrystallinityIndexCalculator", "signature": "class CrystallinityIndexCalculator"}, {"doc": "Measures algorithmic resilience through controlled perturbations.\nTests stability across different subspaces and noise levels.", "kind": "class", "line": 1144, "name": "ResilienceSpectrometer", "signature": "class ResilienceSpectrometer"}, {"doc": "Classifies the crystallization phase based on alpha and temperature.", "kind": "class", "line": 1243, "name": "PhaseClassifier", "signature": "class PhaseClassifier"}, {"doc": "Handles migration between different checkpoint formats.", "kind": "class", "line": 1270, "name": "CheckpointMigrator", "signature": "class CheckpointMigrator"}, {"doc": "Manages checkpoint saving with 5-minute intervals and latest file maintenance.", "kind": "class", "line": 1310, "name": "MBLCheckpointManager", "signature": "class MBLCheckpointManager"}, {"doc": "Collects all MBL metrics for comprehensive training monitoring.\nIncludes all metrics from the crystallography paper.", "kind": "class", "line": 1386, "name": "HamiltonianMBLMetricsCollector", "signature": "class HamiltonianMBLMetricsCollector"}, {"doc": "Training system for Hamiltonian Neural Networks with integrated MBL monitoring.", "kind": "class", "line": 1540, "name": "HamiltonianTrainer", "signature": "class HamiltonianTrainer"}, {"doc": "Comprehensive analyzer for Hamiltonian NN checkpoints with migration support.", "kind": "class", "line": 1710, "name": "HamiltonianCheckpointAnalyzer", "signature": "class HamiltonianCheckpointAnalyzer"}, {"doc": "Main pipeline for processing checkpoints and generating reports.", "kind": "class", "line": 1875, "name": "HamiltonianMBLPipeline", "signature": "class HamiltonianMBLPipeline"}, {"kind": "method", "line": 2031, "name": "main", "signature": "def main()"}, {"doc": "Calculate input dimension from grid size.", "kind": "method", "line": 55, "name": "get_input_dim", "signature": "def get_input_dim(self)"}, {"doc": "Estimate total parameter count.", "kind": "method", "line": 59, "name": "get_total_parameters", "signature": "def get_total_parameters(self)"}, {"doc": "Calculate reduced dimension for analysis.", "kind": "method", "line": 135, "name": "get_reduced_dimension", "signature": "def get_reduced_dimension(self)"}, {"kind": "method", "line": 162, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 163, "name": "forward", "signature": "def forward(self)"}, {"kind": "method", "line": 169, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 175, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 181, "name": "calculate", "signature": "def calculate(self, participation_ratio, energy_gap)"}, {"kind": "method", "line": 187, "name": "analyze_robustness", "signature": "def analyze_robustness(self, model, noise_levels)"}, {"kind": "method", "line": 193, "name": "save_checkpoint", "signature": "def save_checkpoint(self, model, epoch, metrics, loss_history, path)"}, {"kind": "method", "line": 195, "name": "load_checkpoint", "signature": "def load_checkpoint(self, path)"}, {"kind": "method", "line": 201, "name": "collect", "signature": "def collect(self, model, loss, epoch, loss_history)"}, {"kind": "method", "line": 214, "name": "__init__", "signature": "def __init__(self, source_config, target_config)"}, {"doc": "Migra estado de SimpleHamiltonianNet a HamiltonianNeuralNetwork.\n\nSimpleHamiltonianNet tiene:\n- input_proj: Conv2d(1, hidden_dim, 1) -> weight [hidden_dim, 1, 1, 1]\n- spectral_layers.{i}.kernel_real: [hidden_dim, hidden_dim, grid//2+1, grid]\n- output_proj: Conv2d(hidden_dim, 1, 1) -> weight [1, hidden_dim, 1, 1]\n\nHamiltonianNeuralNetwork necesita:\n- q_projection, p_projection: Linear(input_dim, hidden_dim)\n- spectral_layers.{i}.spectral_weights: [hidden_dim, spectral_modes]\n- q_output, p_output: Linear(hidden_dim, input_dim)", "kind": "method", "line": 218, "name": "migrate_state_dict", "signature": "def migrate_state_dict(self, source_state)"}, {"doc": "Crea parámetro por defecto.", "kind": "method", "line": 320, "name": "_create_default_parameter", "signature": "def _create_default_parameter(self, key)"}, {"kind": "method", "line": 352, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Initialize with physics-informed priors.", "kind": "method", "line": 366, "name": "_initialize_spectral_parameters", "signature": "def _initialize_spectral_parameters(self)"}, {"doc": "Symplectic Euler integration of Hamilton's equations.\ndq/dt = dH/dp, dp/dt = -dH/dq", "kind": "method", "line": 373, "name": "forward", "signature": "def forward(self, q, p, dt)"}, {"doc": "Compute Hamiltonian H = T + V in spectral space.", "kind": "method", "line": 401, "name": "get_hamiltonian", "signature": "def get_hamiltonian(self, q, p)"}, {"kind": "method", "line": 418, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Orthogonal initialization for Hamiltonian structure preservation.", "kind": "method", "line": 438, "name": "_initialize_weights", "signature": "def _initialize_weights(self)"}, {"doc": "Forward pass through Hamiltonian dynamics.", "kind": "method", "line": 443, "name": "forward", "signature": "def forward(self, q, p, dt)"}, {"doc": "Generate trajectory through time evolution.", "kind": "method", "line": 459, "name": "time_evolution", "signature": "def time_evolution(self, q_initial, p_initial, num_steps, dt)"}, {"doc": "Compute total Hamiltonian.", "kind": "method", "line": 474, "name": "get_hamiltonian", "signature": "def get_hamiltonian(self, q, p)"}, {"kind": "method", "line": 486, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"doc": "Returns all parameters flattened for Hamiltonian construction.", "kind": "method", "line": 496, "name": "get_flat_parameters", "signature": "def get_flat_parameters(self)"}, {"doc": "MÉTODO CORREGIDO - No usa 65GB de RAM.", "kind": "method", "line": 503, "name": "construct_hessian_approximation", "signature": "def construct_hessian_approximation(self, max_dim, method)"}, {"kind": "method", "line": 561, "name": "__init__", "signature": "def __init__(self, grid_size, num_samples, device)"}, {"doc": "Generate harmonic oscillator initial conditions.", "kind": "method", "line": 567, "name": "generate_harmonic_oscillator", "signature": "def generate_harmonic_oscillator(self, omega)"}, {"doc": "Generate double-well potential trajectories.", "kind": "method", "line": 591, "name": "generate_double_well", "signature": "def generate_double_well(self, barrier_height)"}, {"kind": "method", "line": 616, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calculate level spacing statistics from model weights.", "kind": "method", "line": 619, "name": "calculate", "signature": "def calculate(self, model)"}, {"doc": "Alternative Hessian construction for generic models.", "kind": "method", "line": 651, "name": "_construct_hessian_from_weights", "signature": "def _construct_hessian_from_weights(self, model)"}, {"doc": "Compute sorted eigenvalues of the Hamiltonian.", "kind": "method", "line": 666, "name": "_compute_eigenvalues", "signature": "def _compute_eigenvalues(self, hessian)"}, {"doc": "Calculate adjacent gap ratios r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}).", "kind": "method", "line": 671, "name": "_calculate_spacing_ratios", "signature": "def _calculate_spacing_ratios(self, spacings)"}, {"doc": "Classify the quantum phase based on level spacing ratio.", "kind": "method", "line": 684, "name": "_classify_phase", "signature": "def _classify_phase(self, mean_ratio)"}, {"doc": "Estimate Brody parameter for intermediate statistics.\n0 = Poisson (integrable), 1 = Wigner-Dyson (chaotic)", "kind": "method", "line": 699, "name": "_estimate_brody_parameter", "signature": "def _estimate_brody_parameter(self, ratios)"}, {"kind": "method", "line": 725, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calculate participation ratios for all weight layers.", "kind": "method", "line": 728, "name": "calculate", "signature": "def calculate(self, model)"}, {"doc": "Calculate standard Inverse Participation Ratio.", "kind": "method", "line": 771, "name": "_calculate_ipr", "signature": "def _calculate_ipr(self, coefficients)"}, {"doc": "Calculate q-th order Rényi IPR.", "kind": "method", "line": 782, "name": "_calculate_renyi_ipr", "signature": "def _calculate_renyi_ipr(self, coefficients, q)"}, {"doc": "Calculate fractal dimension D_q from IPR.", "kind": "method", "line": 793, "name": "_calculate_fractal_dimension", "signature": "def _calculate_fractal_dimension(self, ipr, n)"}, {"kind": "method", "line": 807, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calculate synthetic Planck's constant.", "kind": "method", "line": 810, "name": "calculate", "signature": "def calculate(self, participation_ratio, energy_gap)"}, {"doc": "Comprehensive calculation from model and previous analyses.", "kind": "method", "line": 820, "name": "calculate_from_model", "signature": "def calculate_from_model(self, model, level_spacing_results, pr_results)"}, {"kind": "method", "line": 849, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calculate the base discretization level from weight rounding error.", "kind": "method", "line": 853, "name": "calculate_base_discretization", "signature": "def calculate_base_discretization(self, model)"}, {"doc": "Test robustness by applying noise and measuring gap collapse.", "kind": "method", "line": 877, "name": "analyze_robustness", "signature": "def analyze_robustness(self, model, noise_levels)"}, {"doc": "Apply noise to model and measure resulting metrics.", "kind": "method", "line": 921, "name": "_perturb_and_measure", "signature": "def _perturb_and_measure(self, model, noise_level)"}, {"doc": "Convert discretization error to purity alpha.", "kind": "method", "line": 940, "name": "_delta_to_alpha", "signature": "def _delta_to_alpha(self, delta)"}, {"kind": "method", "line": 950, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 953, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 982, "name": "_compute_layer_purity", "signature": "def _compute_layer_purity(self, weights)"}, {"kind": "method", "line": 988, "name": "_delta_to_alpha", "signature": "def _delta_to_alpha(self, delta)"}, {"kind": "method", "line": 993, "name": "_assess_purity_quality", "signature": "def _assess_purity_quality(self, alpha, variance)"}, {"kind": "method", "line": 1007, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 1010, "name": "calculate", "signature": "def calculate(self, loss_history)"}, {"kind": "method", "line": 1054, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calculate Krylov complexity from model dynamics.", "kind": "method", "line": 1057, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 1095, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calculate crystallinity index from weight spectra.", "kind": "method", "line": 1098, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 1150, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Comprehensive resilience measurement.", "kind": "method", "line": 1153, "name": "measure", "signature": "def measure(self, model)"}, {"doc": "Measure baseline performance metrics.", "kind": "method", "line": 1176, "name": "_measure_base_performance", "signature": "def _measure_base_performance(self, model)"}, {"doc": "Test resilience to specific perturbation.", "kind": "method", "line": 1195, "name": "_test_perturbation", "signature": "def _test_perturbation(self, model, dimension, noise_level)"}, {"doc": "Aggregate resilience scores by perturbation dimension.", "kind": "method", "line": 1220, "name": "_aggregate_by_dimension", "signature": "def _aggregate_by_dimension(self, results)"}, {"doc": "Aggregate resilience scores by noise level.", "kind": "method", "line": 1231, "name": "_aggregate_by_noise", "signature": "def _aggregate_by_noise(self, results)"}, {"kind": "method", "line": 1246, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 1249, "name": "classify", "signature": "def classify(self, alpha, temperature)"}, {"kind": "method", "line": 1273, "name": "__init__", "signature": "def __init__(self, arch_config)"}, {"kind": "method", "line": 1277, "name": "migrate", "signature": "def migrate(self, raw_data, device)"}, {"doc": "Detecta el formato y aplica migración si es necesario.", "kind": "method", "line": 1290, "name": "_migrate_if_needed", "signature": "def _migrate_if_needed(self, state_dict, device)"}, {"kind": "method", "line": 1315, "name": "__init__", "signature": "def __init__(self, config, arch_config)"}, {"doc": "Check if 5 minutes have elapsed since last checkpoint.", "kind": "method", "line": 1322, "name": "should_save_checkpoint", "signature": "def should_save_checkpoint(self)"}, {"doc": "Save checkpoint with all MBL metrics.", "kind": "method", "line": 1328, "name": "save_checkpoint", "signature": "def save_checkpoint(self, model, epoch, metrics, loss_history, checkpoint_dir)"}, {"doc": "Load checkpoint with automatic device placement and migration.", "kind": "method", "line": 1361, "name": "load_checkpoint", "signature": "def load_checkpoint(self, path)"}, {"kind": "method", "line": 1392, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Collect core metrics for the current training state.", "kind": "method", "line": 1405, "name": "collect", "signature": "def collect(self, model, loss, epoch, loss_history, step)"}, {"doc": "Collect comprehensive metrics including expensive calculations.", "kind": "method", "line": 1493, "name": "collect_comprehensive", "signature": "def collect_comprehensive(self, model, loss, epoch, loss_history, step)"}, {"doc": "Classify combined quantum phase.", "kind": "method", "line": 1520, "name": "_classify_quantum_phase", "signature": "def _classify_quantum_phase(self, level_spacing, hbar_results)"}, {"kind": "method", "line": 1545, "name": "__init__", "signature": "def __init__(self, model, arch_config, mbl_config, train_config)"}, {"doc": "Single training step with Hamiltonian loss.", "kind": "method", "line": 1571, "name": "train_step", "signature": "def train_step(self, q_batch, p_batch, q_target, p_target)"}, {"doc": "Train for one epoch with MBL monitoring.", "kind": "method", "line": 1603, "name": "train_epoch", "signature": "def train_epoch(self, dataset, epoch)"}, {"doc": "Log metrics to console in scientific format.", "kind": "method", "line": 1658, "name": "_log_metrics", "signature": "def _log_metrics(self, metrics)"}, {"doc": "Full training loop.", "kind": "method", "line": 1672, "name": "train", "signature": "def train(self, dataset, num_epochs)"}, {"kind": "method", "line": 1713, "name": "__init__", "signature": "def __init__(self, checkpoint_path, arch_config, mbl_config)"}, {"doc": "Load and migrate checkpoint.", "kind": "method", "line": 1723, "name": "_load_checkpoint", "signature": "def _load_checkpoint(self)"}, {"doc": "Perform complete MBL analysis.", "kind": "method", "line": 1763, "name": "analyze", "signature": "def analyze(self)"}, {"doc": "Generate executive summary.", "kind": "method", "line": 1787, "name": "_generate_summary", "signature": "def _generate_summary(self, metrics)"}, {"doc": "Print formatted analysis report.", "kind": "method", "line": 1807, "name": "_print_report", "signature": "def _print_report(self, results)"}, {"kind": "method", "line": 1878, "name": "__init__", "signature": "def __init__(self, arch_config, mbl_config)"}, {"doc": "Process single checkpoint and save results.", "kind": "method", "line": 1882, "name": "process_checkpoint", "signature": "def process_checkpoint(self, checkpoint_path, output_dir)"}, {"doc": "Process multiple checkpoints from directory.", "kind": "method", "line": 1901, "name": "process_directory", "signature": "def process_directory(self, checkpoint_dir, n_latest, output_dir)"}, {"doc": "Generate aggregate summary report.", "kind": "method", "line": 1941, "name": "generate_summary", "signature": "def generate_summary(self, all_results, output_dir)"}, {"doc": "Generate human-readable text report.", "kind": "method", "line": 1982, "name": "_generate_text_report", "signature": "def _generate_text_report(self, summary, output_dir)"}]}, {"id": "hpu_view.py", "kind": "module", "label": "hpu_view.py", "language": "py", "sha256": "451282a9f13845ab", "symbol_count": 0, "symbols": []}, {"id": "install.sh", "kind": "module", "label": "install.sh", "language": "sh", "sha256": "c907d80fd6734993", "symbol_count": 0, "symbols": []}, {"id": "mining_seeds.py", "kind": "module", "label": "mining_seeds.py", "language": "py", "sha256": "077aff054f1983a5", "symbol_count": 57, "symbols": [{"kind": "class", "line": 41, "name": "Config", "signature": "class Config"}, {"kind": "method", "line": 88, "name": "set_seed", "signature": "def set_seed(seed)"}, {"kind": "method", "line": 96, "name": "setup_logger", "signature": "def setup_logger(name, level)"}, {"kind": "class", "line": 109, "name": "IAnalysisStrategy", "signature": "class IAnalysisStrategy(ABC)"}, {"kind": "class", "line": 115, "name": "IMetricsCalculator", "signature": "class IMetricsCalculator(ABC)"}, {"doc": "True Hamiltonian operator H = -nabla^2 on torus.", "kind": "class", "line": 121, "name": "HamiltonianOperator", "signature": "class HamiltonianOperator"}, {"doc": "Fast dataset for Hamiltonian operator learning.", "kind": "class", "line": 146, "name": "FastDataset", "signature": "class FastDataset(Dataset)"}, {"doc": "Spectral layer with correct complex multiplication.", "kind": "class", "line": 203, "name": "SpectralLayer", "signature": "class SpectralLayer(Module)"}, {"doc": "Compact network for Hamiltonian operator learning.", "kind": "class", "line": 255, "name": "SimpleHamiltonianNet", "signature": "class SimpleHamiltonianNet(Module)"}, {"kind": "class", "line": 293, "name": "LocalComplexityAnalyzer", "signature": "class LocalComplexityAnalyzer"}, {"kind": "class", "line": 310, "name": "SuperpositionAnalyzer", "signature": "class SuperpositionAnalyzer"}, {"kind": "class", "line": 340, "name": "CrystallographyMetrics", "signature": "class CrystallographyMetrics"}, {"kind": "class", "line": 444, "name": "ThermodynamicMetrics", "signature": "class ThermodynamicMetrics"}, {"kind": "class", "line": 470, "name": "SpectroscopyMetrics", "signature": "class SpectroscopyMetrics"}, {"kind": "class", "line": 500, "name": "CheckpointManager", "signature": "class CheckpointManager"}, {"kind": "class", "line": 573, "name": "TrainingMonitor", "signature": "class TrainingMonitor"}, {"kind": "class", "line": 611, "name": "GlassStopper", "signature": "class GlassStopper"}, {"doc": "Train model with early stopping for glass detection.", "kind": "method", "line": 670, "name": "train_with_early_glass_stop", "signature": "def train_with_early_glass_stop(model, optimizer, seed, epochs)"}, {"doc": "Mine for crystal seeds by trying sequential seeds.", "kind": "method", "line": 803, "name": "seed_miner", "signature": "def seed_miner(total_attempts)"}, {"kind": "method", "line": 856, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 111, "name": "analyze", "signature": "def analyze(self, model)"}, {"kind": "method", "line": 117, "name": "compute", "signature": "def compute(self, model)"}, {"kind": "method", "line": 124, "name": "__init__", "signature": "def __init__(self, grid_size)"}, {"kind": "method", "line": 128, "name": "_precompute_spectral_operators", "signature": "def _precompute_spectral_operators(self)"}, {"kind": "method", "line": 135, "name": "apply", "signature": "def apply(self, field)"}, {"kind": "method", "line": 140, "name": "time_evolution", "signature": "def time_evolution(self, field, dt)"}, {"kind": "method", "line": 149, "name": "__init__", "signature": "def __init__(self, num_samples, grid_size, time_steps, dt, seed, train_ratio)"}, {"kind": "method", "line": 193, "name": "__len__", "signature": "def __len__(self)"}, {"kind": "method", "line": 196, "name": "__getitem__", "signature": "def __getitem__(self, idx)"}, {"kind": "method", "line": 199, "name": "get_val_batch", "signature": "def get_val_batch(self)"}, {"kind": "method", "line": 206, "name": "__init__", "signature": "def __init__(self, channels, grid_size)"}, {"kind": "method", "line": 219, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 258, "name": "__init__", "signature": "def __init__(self, grid_size, hidden_dim, num_spectral_layers)"}, {"kind": "method", "line": 279, "name": "forward", "signature": "def forward(self, x)"}, {"doc": "Compute Local Complexity (LC) metric for weight matrix.", "kind": "method", "line": 295, "name": "compute_local_complexity", "signature": "def compute_local_complexity(weights, epsilon)"}, {"doc": "Compute Superposition (SP) metric for weight matrix.", "kind": "method", "line": 312, "name": "compute_superposition", "signature": "def compute_superposition(weights)"}, {"kind": "method", "line": 342, "name": "compute_kappa", "signature": "def compute_kappa(model, dataloader, num_batches)"}, {"kind": "method", "line": 378, "name": "compute_discretization_margin", "signature": "def compute_discretization_margin(coeffs)"}, {"kind": "method", "line": 387, "name": "compute_alpha_purity", "signature": "def compute_alpha_purity(coeffs)"}, {"kind": "method", "line": 394, "name": "compute_kappa_quantum", "signature": "def compute_kappa_quantum(coeffs, hbar)"}, {"kind": "method", "line": 411, "name": "compute_poynting_vector", "signature": "def compute_poynting_vector(coeffs)"}, {"kind": "method", "line": 426, "name": "compute_all_metrics", "signature": "def compute_all_metrics(model, dataloader)"}, {"kind": "method", "line": 446, "name": "compute_effective_temperature", "signature": "def compute_effective_temperature(gradient_buffer, learning_rate)"}, {"kind": "method", "line": 460, "name": "compute_specific_heat", "signature": "def compute_specific_heat(loss_history, temp_history, cv_threshold)"}, {"kind": "method", "line": 472, "name": "compute_weight_diffraction", "signature": "def compute_weight_diffraction(coeffs)"}, {"kind": "method", "line": 491, "name": "_compute_spectral_entropy", "signature": "def _compute_spectral_entropy(power_spectrum)"}, {"kind": "method", "line": 501, "name": "__init__", "signature": "def __init__(self, interval_minutes, max_checkpoints)"}, {"kind": "method", "line": 509, "name": "should_save_checkpoint", "signature": "def should_save_checkpoint(self)"}, {"kind": "method", "line": 514, "name": "save_checkpoint", "signature": "def save_checkpoint(self, model, optimizer, epoch, metrics)"}, {"kind": "method", "line": 574, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 594, "name": "update_metrics", "signature": "def update_metrics(self, epoch, loss, val_loss, val_acc, lc, sp, alpha, kappa, delta, temperature, specific_heat, poynting_magnitude)"}, {"kind": "method", "line": 612, "name": "__init__", "signature": "def __init__(self, patience_epochs)"}, {"doc": "Check if the system is in glass state and should stop mining.", "kind": "method", "line": 616, "name": "should_stop", "signature": "def should_stop(self, epoch, lc, sp, kappa, delta, temp, cv)"}, {"kind": "class", "line": 879, "name": "BoltzmannAnalysisProgram", "signature": "class BoltzmannAnalysisProgram"}, {"kind": "method", "line": 880, "name": "__init__", "signature": "def __init__(self, checkpoint_path, results_dir)"}, {"kind": "method", "line": 886, "name": "load_and_analyze_checkpoint", "signature": "def load_and_analyze_checkpoint(self)"}, {"kind": "method", "line": 903, "name": "dataloader", "signature": "def dataloader()"}]}, {"id": "plank.py", "kind": "module", "label": "plank.py", "language": "py", "sha256": "96b61e37e92a16b9", "symbol_count": 5, "symbols": [{"doc": "Calcula ħ efectiva desde checkpoint HPU usando física realista.", "kind": "class", "line": 26, "name": "HBarCalculator", "signature": "class HBarCalculator"}, {"kind": "method", "line": 213, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 29, "name": "__init__", "signature": "def __init__(self, checkpoint_path, device)"}, {"doc": "Ejecuta todos los cálculos de ħ.", "kind": "method", "line": 54, "name": "calculate_all", "signature": "def calculate_all(self)"}, {"doc": "Imprime reporte formateado.", "kind": "method", "line": 170, "name": "print_report", "signature": "def print_report(self, results)"}]}, {"id": "polos.py", "kind": "module", "label": "polos.py", "language": "py", "sha256": "5977862a9aff0d04", "symbol_count": 42, "symbols": [{"kind": "class", "line": 27, "name": "ControlConfig", "signature": "class ControlConfig"}, {"kind": "class", "line": 48, "name": "TransferFunctionExtractor", "signature": "class TransferFunctionExtractor"}, {"kind": "class", "line": 142, "name": "PoleZeroAnalyzer", "signature": "class PoleZeroAnalyzer"}, {"kind": "class", "line": 299, "name": "FrequencyResponseAnalyzer", "signature": "class FrequencyResponseAnalyzer"}, {"kind": "class", "line": 415, "name": "TimeResponseAnalyzer", "signature": "class TimeResponseAnalyzer"}, {"kind": "class", "line": 516, "name": "ControllerDesigner", "signature": "class ControllerDesigner"}, {"kind": "class", "line": 599, "name": "ControlSystemAnalyzer", "signature": "class ControlSystemAnalyzer"}, {"kind": "class", "line": 798, "name": "ControlVisualizer", "signature": "class ControlVisualizer"}, {"kind": "method", "line": 1152, "name": "analyze_checkpoint", "signature": "def analyze_checkpoint(checkpoint_path, output_dir)"}, {"kind": "method", "line": 1208, "name": "analyze_multiple_checkpoints", "signature": "def analyze_multiple_checkpoints(checkpoint_dir, n_latest, output_dir)"}, {"kind": "method", "line": 1240, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 50, "name": "__init__", "signature": "def __init__(self, model, device)"}, {"kind": "method", "line": 55, "name": "extract_state_space_representation", "signature": "def extract_state_space_representation(self)"}, {"kind": "method", "line": 105, "name": "compute_transfer_function", "signature": "def compute_transfer_function(self, A, B, C, D)"}, {"kind": "method", "line": 144, "name": "__init__", "signature": "def __init__(self, numerator, denominator)"}, {"kind": "method", "line": 153, "name": "_compute_poles_zeros", "signature": "def _compute_poles_zeros(self)"}, {"kind": "method", "line": 166, "name": "analyze_stability", "signature": "def analyze_stability(self)"}, {"kind": "method", "line": 207, "name": "classify_poles", "signature": "def classify_poles(self)"}, {"kind": "method", "line": 238, "name": "compute_damping_frequency", "signature": "def compute_damping_frequency(self)"}, {"kind": "method", "line": 278, "name": "compute_time_constants", "signature": "def compute_time_constants(self)"}, {"kind": "method", "line": 301, "name": "__init__", "signature": "def __init__(self, numerator, denominator)"}, {"kind": "method", "line": 309, "name": "compute_bode_plot_data", "signature": "def compute_bode_plot_data(self)"}, {"kind": "method", "line": 328, "name": "compute_gain_phase_margins", "signature": "def compute_gain_phase_margins(self)"}, {"kind": "method", "line": 357, "name": "compute_nyquist_data", "signature": "def compute_nyquist_data(self)"}, {"kind": "method", "line": 379, "name": "evaluate_nyquist_stability", "signature": "def evaluate_nyquist_stability(self, nyquist_data)"}, {"kind": "method", "line": 417, "name": "__init__", "signature": "def __init__(self, numerator, denominator)"}, {"kind": "method", "line": 425, "name": "compute_step_response", "signature": "def compute_step_response(self)"}, {"kind": "method", "line": 441, "name": "compute_impulse_response", "signature": "def compute_impulse_response(self)"}, {"kind": "method", "line": 457, "name": "analyze_step_response_characteristics", "signature": "def analyze_step_response_characteristics(self, step_data)"}, {"kind": "method", "line": 518, "name": "__init__", "signature": "def __init__(self, poles, zeros)"}, {"kind": "method", "line": 522, "name": "design_pid_controller", "signature": "def design_pid_controller(self, desired_damping, desired_settling_time)"}, {"kind": "method", "line": 542, "name": "design_lead_compensator", "signature": "def design_lead_compensator(self, desired_phase_margin)"}, {"kind": "method", "line": 572, "name": "compute_root_locus", "signature": "def compute_root_locus(self, num, den)"}, {"kind": "method", "line": 601, "name": "__init__", "signature": "def __init__(self, checkpoint_path, device)"}, {"kind": "method", "line": 625, "name": "analyze_complete_system", "signature": "def analyze_complete_system(self)"}, {"kind": "method", "line": 715, "name": "_print_report", "signature": "def _print_report(self, results)"}, {"kind": "method", "line": 801, "name": "plot_pole_zero_map", "signature": "def plot_pole_zero_map(poles, zeros, output_path)"}, {"kind": "method", "line": 865, "name": "plot_bode_diagram", "signature": "def plot_bode_diagram(bode_data, margins, output_path)"}, {"kind": "method", "line": 940, "name": "plot_nyquist_diagram", "signature": "def plot_nyquist_diagram(nyquist_data, output_path)"}, {"kind": "method", "line": 990, "name": "plot_time_responses", "signature": "def plot_time_responses(step_data, impulse_data, output_path)"}, {"kind": "method", "line": 1036, "name": "plot_root_locus", "signature": "def plot_root_locus(root_locus_data, output_path)"}, {"kind": "method", "line": 1096, "name": "plot_combined_analysis", "signature": "def plot_combined_analysis(poles, zeros, bode_data, step_data, output_path)"}]}, {"id": "precision.py", "kind": "module", "label": "precision.py", "language": "py", "sha256": "70f25d49d34b4a56", "symbol_count": 18, "symbols": [{"kind": "class", "line": 26, "name": "MassiveLambdaConfig", "signature": "class MassiveLambdaConfig"}, {"kind": "class", "line": 36, "name": "CrystallizationLossMassive", "signature": "class CrystallizationLossMassive(Module)"}, {"kind": "class", "line": 72, "name": "ContinuationEngine", "signature": "class ContinuationEngine"}, {"kind": "method", "line": 506, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 37, "name": "__init__", "signature": "def __init__(self, lambda_quant)"}, {"kind": "method", "line": 42, "name": "quantization_penalty", "signature": "def quantization_penalty(self, model)"}, {"kind": "method", "line": 54, "name": "forward", "signature": "def forward(self, predictions, targets, model)"}, {"kind": "method", "line": 73, "name": "__init__", "signature": "def __init__(self, checkpoint_path, device)"}, {"kind": "method", "line": 150, "name": "_setup_logger", "signature": "def _setup_logger(self)"}, {"kind": "method", "line": 162, "name": "_find_latest_checkpoint", "signature": "def _find_latest_checkpoint(self)"}, {"kind": "method", "line": 191, "name": "_compute_initial_metrics", "signature": "def _compute_initial_metrics(self, model)"}, {"kind": "method", "line": 208, "name": "compute_discretization_metrics", "signature": "def compute_discretization_metrics(self)"}, {"kind": "method", "line": 241, "name": "validate", "signature": "def validate(self)"}, {"kind": "method", "line": 250, "name": "train_epoch", "signature": "def train_epoch(self, epoch)"}, {"kind": "method", "line": 288, "name": "refine", "signature": "def refine(self)"}, {"doc": "Guarda/sobrescribe latest.pth - rápido, para danger zone", "kind": "method", "line": 430, "name": "_save_latest_checkpoint", "signature": "def _save_latest_checkpoint(self, epoch, metrics, val_acc)"}, {"kind": "method", "line": 456, "name": "_save_crystal_checkpoint", "signature": "def _save_crystal_checkpoint(self, epoch, metrics, val_acc, final, force_save, emergency)"}, {"kind": "method", "line": 490, "name": "_compile_results", "signature": "def _compile_results(self, success, final_epoch)"}]}, {"id": "refinamiento.py", "kind": "module", "label": "refinamiento.py", "language": "py", "sha256": "f534abc800cbe8d6", "symbol_count": 23, "symbols": [{"doc": "Configuración agresiva para forzar discretización", "kind": "class", "line": 33, "name": "CrystallizationConfig", "signature": "class CrystallizationConfig"}, {"doc": "Pérdida combinada: MSE + penalización de cuantización\nFuerza los pesos a caer en {-1, 0, 1}", "kind": "class", "line": 57, "name": "CrystallizationLoss", "signature": "class CrystallizationLoss(Module)"}, {"doc": "Implementa poda progresiva de pesos pequeños", "kind": "class", "line": 96, "name": "StructuralPruner", "signature": "class StructuralPruner"}, {"doc": "Motor de refinamiento que carga un checkpoint y fuerza discretización", "kind": "class", "line": 144, "name": "CrystallizationEngine", "signature": "class CrystallizationEngine"}, {"doc": "Análisis detallado de la discretización de un checkpoint", "kind": "method", "line": 498, "name": "analyze_discretization", "signature": "def analyze_discretization(checkpoint_path)"}, {"kind": "method", "line": 575, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 62, "name": "__init__", "signature": "def __init__(self, lambda_quant)"}, {"doc": "Penalización L2 de la distancia al entero más cercano", "kind": "method", "line": 67, "name": "quantization_penalty", "signature": "def quantization_penalty(self, model)"}, {"kind": "method", "line": 81, "name": "forward", "signature": "def forward(self, predictions, targets, model)"}, {"kind": "method", "line": 98, "name": "__init__", "signature": "def __init__(self, thresholds)"}, {"doc": "Determina si es momento de podar (cada 500 épocas)", "kind": "method", "line": 103, "name": "should_prune", "signature": "def should_prune(self, epoch)"}, {"doc": "Poda pesos con |w| < threshold\nRetorna número de parámetros podados", "kind": "method", "line": 107, "name": "prune", "signature": "def prune(self, model, force_threshold)"}, {"doc": "Calcula porcentaje de pesos exactamente en cero", "kind": "method", "line": 131, "name": "get_sparsity", "signature": "def get_sparsity(self, model)"}, {"kind": "method", "line": 148, "name": "__init__", "signature": "def __init__(self, checkpoint_path, device)"}, {"kind": "method", "line": 186, "name": "_setup_logger", "signature": "def _setup_logger(self)"}, {"doc": "Carga el checkpoint y retorna modelo, época y métricas", "kind": "method", "line": 198, "name": "_load_checkpoint", "signature": "def _load_checkpoint(self)"}, {"doc": "Calcula métricas iniciales si no vienen en el checkpoint", "kind": "method", "line": 234, "name": "_compute_initial_metrics", "signature": "def _compute_initial_metrics(self, model)"}, {"doc": "Calcula métricas de cristalinidad actuales", "kind": "method", "line": 253, "name": "compute_discretization_metrics", "signature": "def compute_discretization_metrics(self)"}, {"doc": "Valida el modelo manteniendo accuracy", "kind": "method", "line": 291, "name": "validate", "signature": "def validate(self)"}, {"doc": "Entrena una época con pérdida de cuantización", "kind": "method", "line": 302, "name": "train_epoch", "signature": "def train_epoch(self, epoch)"}, {"doc": "Ejecuta el refinamiento hasta alcanzar δ < TARGET_DELTA o MAX_EPOCHS", "kind": "method", "line": 344, "name": "refine", "signature": "def refine(self)"}, {"doc": "Guarda checkpoint cristalino", "kind": "method", "line": 459, "name": "_save_crystal_checkpoint", "signature": "def _save_crystal_checkpoint(self, epoch, metrics, val_acc, final)"}, {"doc": "Compila resultados finales", "kind": "method", "line": 482, "name": "_compile_results", "signature": "def _compile_results(self, success, final_epoch)"}]}, {"id": "simple_hpu_view.py", "kind": "module", "label": "simple_hpu_view.py", "language": "py", "sha256": "2c470311a2f28a94", "symbol_count": 0, "symbols": []}, {"id": "test_grokkit.py", "kind": "module", "label": "test_grokkit.py", "language": "py", "sha256": "d45d48c6e156df13", "symbol_count": 11, "symbols": [{"doc": "Validates grokking phenomenon in Hamiltonian operator learning.\n\nImplements Theorem 1.1 requirements:\n1. Spectral convergence to true H operator\n2. Operator kernel representation in weights\n3. Phase transition from memorization to generalization\n\nThis class implements a battery of tests to confirm that the trained\nmodel has successfully transitioned from the memorization phase to\nthe generalization phase, exhibiting the characteristic properties\nof spectral convergence as predicted by Theorem 1.1.", "kind": "class", "line": 41, "name": "GrokkingValidator", "signature": "class GrokkingValidator"}, {"doc": "Executes a quick validation test with minimal output.\n\nThis function provides a streamlined testing interface for\nrapid verification of model performance.", "kind": "method", "line": 507, "name": "run_quick_test", "signature": "def run_quick_test()"}, {"kind": "method", "line": 56, "name": "__init__", "signature": "def __init__(self, weights_dir)"}, {"doc": "Loads the trained model from checkpoint.\n\nReturns:\n    Tuple of (model, checkpoint)\n    \nRaises:\n    FileNotFoundError: If no checkpoint exists in weights directory.", "kind": "method", "line": 64, "name": "load_model", "signature": "def load_model(self)"}, {"doc": "Generates test dataset using the true Hamiltonian operator.\n\nCreates random initial fields and evolves them under H to produce\nground truth targets for validation.\n\nArgs:\n    num_samples: Number of test samples\n    \nReturns:\n    Tuple of (inputs, targets) tensors", "kind": "method", "line": 119, "name": "generate_test_dataset", "signature": "def generate_test_dataset(self, num_samples)"}, {"doc": "Computes Local Complexity (LC) metric for the model.\n\nLC measures the effective dimensionality of the model's\nlearned representations. High LC indicates diverse, independent\nfeature utilization - a key indicator of operator learning.\n\nArgs:\n    model: Neural network model\n    \nReturns:\n    LC value in [0, 1] range", "kind": "method", "line": 158, "name": "compute_local_complexity", "signature": "def compute_local_complexity(self, model)"}, {"doc": "Computes Superposition (SP) metric for the model.\n\nSP measures the correlation between weight vectors.\nLow SP indicates orthogonal, non-redundant representations.\n\nArgs:\n    model: Neural network model\n    \nReturns:\n    SP value in [0, 1] range", "kind": "method", "line": 181, "name": "compute_superposition", "signature": "def compute_superposition(self, model)"}, {"doc": "Computes operator approximation error.\n\nMeasures how well the learned model approximates the true\nHamiltonian operator on held-out test data.\n\nArgs:\n    model: Trained model\n    inputs: Test input fields\n    targets: True evolved fields under H\n    \nReturns:\n    Mean squared error between prediction and target", "kind": "method", "line": 203, "name": "compute_operator_error", "signature": "def compute_operator_error(self, model, inputs, targets)"}, {"doc": "Estimates the spectral gap in weight singular values.\n\nThe spectral gap provides insight into the model's capacity\nutilization and the degree of weight superposition.\n\nArgs:\n    model: Neural network model\n    \nReturns:\n    Ratio of largest to smallest non-zero singular value", "kind": "method", "line": 228, "name": "compute_spectral_gap", "signature": "def compute_spectral_gap(self, model)"}, {"doc": "Executes the complete validation suite.\n\nRuns all tests and aggregates results into a comprehensive\nreport documenting the grokking phenomenon characteristics\nas predicted by Theorem 1.1.\n\nReturns:\n    Dictionary containing all test results and metrics", "kind": "method", "line": 259, "name": "run_validation", "signature": "def run_validation(self)"}, {"doc": "Generates a formal validation report.\n\nReturns:\n    Markdown formatted validation report", "kind": "method", "line": 382, "name": "generate_report", "signature": "def generate_report(self)"}]}, {"id": "verify.py", "kind": "module", "label": "verify.py", "language": "py", "sha256": "2a8b97c525664a06", "symbol_count": 14, "symbols": [{"kind": "class", "line": 14, "name": "CheckpointVerifier", "signature": "class CheckpointVerifier"}, {"doc": "Verifica los N checkpoints más recientes", "kind": "method", "line": 444, "name": "verify_latest_checkpoints", "signature": "def verify_latest_checkpoints(checkpoint_dir, n)"}, {"kind": "method", "line": 486, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 15, "name": "__init__", "signature": "def __init__(self, checkpoint_path, device)"}, {"doc": "Calcula TODAS las métricas desde cero y compara con las guardadas", "kind": "method", "line": 50, "name": "verify_all_metrics", "signature": "def verify_all_metrics(self)"}, {"doc": "Verifica que los pesos no tengan NaN/Inf", "kind": "method", "line": 101, "name": "_check_weight_integrity", "signature": "def _check_weight_integrity(self)"}, {"doc": "Calcula MSE y accuracy de validación desde cero", "kind": "method", "line": 146, "name": "_compute_validation_metrics", "signature": "def _compute_validation_metrics(self)"}, {"doc": "Calcula delta, alpha, purity, etc.", "kind": "method", "line": 173, "name": "_compute_discretization_metrics", "signature": "def _compute_discretization_metrics(self)"}, {"doc": "Calcula la penalización de cuantización", "kind": "method", "line": 225, "name": "_compute_quantization_metrics", "signature": "def _compute_quantization_metrics(self)"}, {"doc": "Reconstruye el loss total", "kind": "method", "line": 244, "name": "_compute_loss_metrics", "signature": "def _compute_loss_metrics(self)"}, {"doc": "Compara métricas calculadas vs almacenadas en el checkpoint", "kind": "method", "line": 269, "name": "_compare_with_stored", "signature": "def _compare_with_stored(self, computed)"}, {"doc": "Verifica consistencia entre métricas relacionadas", "kind": "method", "line": 302, "name": "_check_internal_consistency", "signature": "def _check_internal_consistency(self, results)"}, {"doc": "Calcula un score de salud del checkpoint (0-100)", "kind": "method", "line": 331, "name": "_compute_health_score", "signature": "def _compute_health_score(self, results)"}, {"doc": "Imprime reporte formateado", "kind": "method", "line": 374, "name": "_print_report", "signature": "def _print_report(self, results)"}]}], "type": "CodePropertyGraph", "version": "1.0"}
```

---

## Architecture Reference

### PY (31 files)

#### `app.py`
**Path:** `app.py`
**File Doc:** *_*_ coding: utf8 _*_*

**Classes:**
- `SimpleConfig` (line 29) `class SimpleConfig`
- `HamiltonianOperator` (line 88) `class HamiltonianOperator` - *True Hamiltonian operator H = -nabla^2 on torus.*
- `FastDataset` (line 113) `class FastDataset(Dataset)` - *Fast dataset for Hamiltonian operator learning.*
- `SpectralLayer` (line 170) `class SpectralLayer(Module)` - *Spectral layer with correct complex multiplication.*
- `SimpleHamiltonianNet` (line 222) `class SimpleHamiltonianNet(Module)` - *Compact network for Hamiltonian operator learning.*

**Methods:**
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

**Methods:**
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

**Methods:**
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

**Methods:**
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

**Methods:**
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
- `SpectralLayer` (line 183) `class SpectralLayer(Module)`
- `HamiltonianNeuralNetwork` (line 227) `class HamiltonianNeuralNetwork(Module)`
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

**Methods:**
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

**Methods:**
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

**Methods:**
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

**Methods:**
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
- `SpectralEvolutionLayer` (line 27) `class SpectralEvolutionLayer(Module)` - *Single Hamiltonian spectral evolution layer.

Performs frequency-domain evolution using learnable complex kernels.
Kernel shape: [hidden_dim, hidden_dim, kernel_base_height, kernel_base_width]
matching the original experiment2 architecture.*
- `HamiltonianNeuralNetwork` (line 162) `class HamiltonianNeuralNetwork(Module)` - *Complete Hamiltonian Neural Network with parametric architecture.

Architecture (matching experiment2):
    1. Input projection: Conv2d(1, hidden_dim, kernel, pad)
    2. N spectral evolution layers with learnable complex kernels
    3. Output projection: Conv2d(hidden_dim, 1, kernel, pad)*

**Methods:**
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

**Methods:**
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

**Methods:**
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

**Methods:**
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
- `SpectralLayer` (line 203) `class SpectralLayer(Module)` - *Spectral layer with correct complex multiplication.*
- `SimpleHamiltonianNet` (line 255) `class SimpleHamiltonianNet(Module)` - *Compact network for Hamiltonian operator learning.*
- `LocalComplexityAnalyzer` (line 293) `class LocalComplexityAnalyzer`
- `SuperpositionAnalyzer` (line 310) `class SuperpositionAnalyzer`
- `CrystallographyMetrics` (line 340) `class CrystallographyMetrics`
- `ThermodynamicMetrics` (line 444) `class ThermodynamicMetrics`
- `SpectroscopyMetrics` (line 470) `class SpectroscopyMetrics`
- `CheckpointManager` (line 500) `class CheckpointManager`
- `TrainingMonitor` (line 573) `class TrainingMonitor`
- `GlassStopper` (line 611) `class GlassStopper`
- `BoltzmannAnalysisProgram` (line 879) `class BoltzmannAnalysisProgram`

**Methods:**
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
- `SpectralLayer` (line 183) `class SpectralLayer(Module)`
- `HamiltonianNeuralNetwork` (line 227) `class HamiltonianNeuralNetwork(Module)`
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

**Methods:**
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

**Methods:**
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
- `SpectralHamiltonianLayer` (line 346) `class SpectralHamiltonianLayer(Module)` - *Spectral layer implementing Hamiltonian dynamics in Fourier space.
Preserves energy conservation through symplectic integration.*
- `HamiltonianNeuralNetwork` (line 412) `class HamiltonianNeuralNetwork(Module)` - *Complete Hamiltonian Neural Network for learning dynamical systems.
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

**Methods:**
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
- `SpectralLayer` (line 203) `class SpectralLayer(Module)` - *Spectral layer with correct complex multiplication.*
- `SimpleHamiltonianNet` (line 255) `class SimpleHamiltonianNet(Module)` - *Compact network for Hamiltonian operator learning.*
- `LocalComplexityAnalyzer` (line 293) `class LocalComplexityAnalyzer`
- `SuperpositionAnalyzer` (line 310) `class SuperpositionAnalyzer`
- `CrystallographyMetrics` (line 340) `class CrystallographyMetrics`
- `ThermodynamicMetrics` (line 444) `class ThermodynamicMetrics`
- `SpectroscopyMetrics` (line 470) `class SpectroscopyMetrics`
- `CheckpointManager` (line 500) `class CheckpointManager`
- `TrainingMonitor` (line 573) `class TrainingMonitor`
- `GlassStopper` (line 611) `class GlassStopper`
- `BoltzmannAnalysisProgram` (line 879) `class BoltzmannAnalysisProgram`

**Methods:**
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

**Methods:**
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

**Methods:**
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
- `CrystallizationLossMassive` (line 36) `class CrystallizationLossMassive(Module)`
- `ContinuationEngine` (line 72) `class ContinuationEngine`

**Methods:**
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
- `CrystallizationLoss` (line 57) `class CrystallizationLoss(Module)` - *Pérdida combinada: MSE + penalización de cuantización
Fuerza los pesos a caer en {-1, 0, 1}*
- `StructuralPruner` (line 96) `class StructuralPruner` - *Implementa poda progresiva de pesos pequeños*
- `CrystallizationEngine` (line 144) `class CrystallizationEngine` - *Motor de refinamiento que carga un checkpoint y fuerza discretización*

**Methods:**
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

**Methods:**
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

**Methods:**
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
