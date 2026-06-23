# e018 — Claim Ledger

| Claim | Status | Evidence | Limitation | Next action |
|-------|--------|----------|------------|-------------|
| Simulator can compare full model against baselines using same trace machinery | Internal validated result | All 5 comparators ran; traces exported | Canonical params only | E019 parameter robustness |
| Full model passes joint SIG-A–SIG-E profile | Internal validated result | comparator_pass_fail_matrix.csv | Canonical params; directional+protected threshold | E019 |
| At least one simpler comparator fails joint profile | Internal validated result | comparator_pass_fail_matrix.csv | Tested comparators only | E018R if needed |
| No simpler comparator passes all five signatures | Internal validated result | comparator_pass_fail_matrix.csv | Canonical params; not robustness tested | E019 |
| Model discrimination claim is internally testable | Internal validated result | joint_pass_summary.json | Not externally validated | — |
| Model robust across parameters | Pending | Requires E019 | — | E019 |
| Biological validation | Not supported | E018 scope: instrumentation + discrimination | — | — |
| DANDI evidence validates model | Not supported | E018 has no DANDI analysis | — | — |
