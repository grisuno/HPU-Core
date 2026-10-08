# Gotchas

## God Nodes (high connectivity)

These files have the most connections. Changes here have high blast radius.

- `experiment2.py` (score: 26.30, imported by 9 files)
- `audio/config.py` (score: 19.30, imported by 9 files)
- `audio/trainer.py` (score: 15.10, imported by 1 files)
- `audio/inference.py` (score: 14.70, imported by 1 files)
- `hamiltonian_mbl.py` (score: 12.60)
- `audio/experiment2.py` (score: 10.30, imported by 1 files)
- `audio/main.py` (score: 8.70, imported by 1 files)
- `audio/metrics.py` (score: 8.50, imported by 2 files)
- `audio/audio_io.py` (score: 7.20, imported by 2 files)
- `get_meditions.py` (score: 7.20)

## Blast Radius (change impact)

Editing these files can break the listed number of dependents. Run their tests after any change.

- `audio/config.py` -- 9 direct, 10 total dependents
- `experiment2.py` -- 9 direct, 9 total dependents
- `audio/audio_io.py` -- 2 direct, 4 total dependents
- `audio/checkpoint_manager.py` -- 2 direct, 4 total dependents
- `audio/metrics.py` -- 2 direct, 4 total dependents
- `audio/model.py` -- 2 direct, 4 total dependents
- `audio/losses.py` -- 1 direct, 3 total dependents
- `audio/visualization.py` -- 1 direct, 3 total dependents
- `audio/inference.py` -- 1 direct, 2 total dependents
- `audio/trainer.py` -- 1 direct, 2 total dependents

## Hotspots (complexity + centrality)

- `experiment2.py` -- complexity: 0.7, centrality: 1.0, combined: 0.9
- `hamiltonian_mbl.py` -- complexity: 1.0, centrality: 0.8, combined: 0.9
- `experiment.py` -- complexity: 0.5, centrality: 0.9, combined: 0.7
- `mining_seeds.py` -- complexity: 0.5, centrality: 0.9, combined: 0.7
- `audio/experiment2.py` -- complexity: 0.7, centrality: 0.7, combined: 0.7
- `get_meditions.py` -- complexity: 0.4, centrality: 0.8, combined: 0.6
- `polos.py` -- complexity: 0.3, centrality: 0.8, combined: 0.6
- `audio/audios.py` -- complexity: 0.4, centrality: 0.7, combined: 0.6
- `audio/trainer.py` -- complexity: 0.1, centrality: 0.8, combined: 0.5
- `dirac.py` -- complexity: 0.2, centrality: 0.7, combined: 0.5
