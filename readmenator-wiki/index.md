# Second Brain

*Last synthesized: 2026-10-07 | 32 files | 4 concept pages | offline, zero tokens*

> Raw sources -> readmenator wiki -> links (Karpathy LLM Wiki Pattern, deterministic).
> Start here, then open one community page. Prefer grep over full reads.

## Vault Overview

The codebase centres on `experiment2.py`, `config.py`, `trainer.py`. Architecturally it is 6 layers, dominant utility (26 files) across 4 import-based communities. Recorded risk surface: 0 security findings and 0 dependency cycles.

Surprising tissue lives between audio: metrics, root, audio: experiment2: 0 extracted cross-community imports and 6 inferred bridges. Follow `connections.json` sorted by strength before refactoring.

Open work clusters around documentation (59% file coverage), 0 security findings, 0 taint paths, and 5 suggested exploration questions in `queries.md`.

## Stats

| Metric | Value |
|--------|-------|
| Files | 32 |
| Symbols | 776 |
| Resolved imports | 33 |
| Languages | py, sh |
| Communities | 4 |
| Doc coverage | 59% (19/32 files) |
| Security findings | 0 |
| Estimated read cost | ~14778 tokens (chars/4, offline so $0) |

## Reading Order

1. Skim Stats and God Nodes below for blast radius.
2. Open the largest community page first, then follow Connections.
3. Use `queries.md` for the next question; log the answer there.

```
grep -rn '<keyword>' index.md community_*.md
readmenator query "<question>" --target readmenator_HPU-Core_zghb0t6k
```

## Concept Wiki

- [audio: metrics (11 files, cohesion 1.00)](./community_0_audio_metrics.md)
- [root (10 files, cohesion 1.00)](./community_1_root.md)
- [audio: experiment2 (2 files, cohesion 1.00)](./community_2_audio_experiment2.md)
- [orphans (9 files, cohesion 0.00)](./community_3_orphans.md)

## God Nodes

| File | Score |
|------|-------|
| `experiment2.py` | 26.3 |
| `audio/config.py` | 19.3 |
| `audio/trainer.py` | 15.1 |
| `audio/inference.py` | 14.7 |
| `hamiltonian_mbl.py` | 12.6 |

## Strongest Connections

- 1 -> 2: duplicates (strength 0.53, INFERRED)
- 0 -> 1: shares_context (strength 0.5, INFERRED)
- 0 -> 2: shares_context (strength 0.5, INFERRED)
- 0 -> 3: shares_context (strength 0.5, INFERRED)
- 1 -> 3: shares_context (strength 0.5, INFERRED)
- 2 -> 3: shares_context (strength 0.5, INFERRED)

## Navigation Tips

- Obsidian Graph View works: every community page links back here.
- `connections.json` is machine-readable for GraphRAG pipelines.
- `REPORT.md` states what was extracted vs inferred and current limits.
- Regenerate offline: `readmenator . --rebuild` (no network, no tokens).
