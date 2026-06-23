# e017 — Traceable Simulator Core

**Date:** 2026-06-22
**Status:** completed

## Purpose
Export full time-resolved trace data from the canonical branch-accessibility
simulator across all ten protocol phases, enabling reviewer-inspectable figures.

## How to reproduce
```bash
python experiments/exp017_traceable_simulator_core.py
```

## Outputs
- `traces/branch_traces.csv` — per-branch state at every step
- `traces/trace_support.csv` — recall support at every step
- `traces/linking_trace.csv` — linking score at every step
- `summary/signature_summary.csv` — SIG-A to SIG-E values
- `summary/run_metadata.json` — full run parameters, hashes, git commit
- `figures/Fig_e017_01_branch_state_traces.png`
- `figures/Fig_e017_02_structural_accessibility_traces.png`
- `figures/Fig_e017_03_recall_and_linking_traces.png`
- `figures/Fig_e017_04_signature_barplot.png`

## Claim scope
This experiment supports only:
> "The simulator emits reproducible branch-level and trace-level dynamics
> that can be inspected as time-resolved traces."

It does NOT support:
- biological validation
- robustness across parameters
- comparator baseline claims
- DANDI evidence claims
