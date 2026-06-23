# e017 — Claim Ledger

| Claim | Status | Evidence | Limitation | Next action |
|-------|--------|----------|------------|-------------|
| Simulator emits reproducible time-resolved traces | Internal validated result | branch_traces.csv, run_metadata.json, figures 1–3 | Canonical params only; no noise; no parameter sweep | e018 comparator traces |
| SIG-A: overlap branch gains more M_b than non-overlap | Internal validated result | effect_summary.md | Single canonical run | e018 across baselines |
| SIG-B: linking increases post-consolidation | Internal validated result | effect_summary.md | Single canonical run | e018 |
| SIG-C: context separation | Internal validated result | separate context-sim; score only | Not in main trace CSV | e018 |
| SIG-D: linking more fragile than recall under damage | Internal validated result | effect_summary.md | Damage modelled as decay-rate increase only | e018 |
| SIG-E: targeted rescue selectivity | Internal validated result | effect_summary.md | Targeted rescue vs standard only; no generic rescue | e018 |
| Joint signature profile supported | Internal validated result | signature_summary.csv | Canonical params only | e018 comparators |
| Simulator separates slow writing from fast gating | Pending | Requires e018 comparator baseline | Not tested in e017 | e018 |
| Biological validation | Not supported | e017 scope is instrumentation only | — | — |
