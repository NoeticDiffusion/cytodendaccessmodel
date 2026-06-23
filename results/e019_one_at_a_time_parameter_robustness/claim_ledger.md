# e019 — Claim Ledger

| Claim | Status | Evidence | Limitation | Next action |
|-------|--------|----------|------------|-------------|
| Joint signature profile is not a single-run artifact | Validated | Survives parameter variation in 9/9 sweeps | One-at-a-time only | E020 pairwise |
| Profile survives OAT variation across defined ranges | Validated | sweeps CSVs | No pairwise or noise-robustness claim | E020 |
| Signatures have identifiable failure boundaries | Validated | failure_boundary_summary.csv | OAT only | E020 |
| SIG-C is a fast-context / allocation signature | Validated (if context_gain OAT confirms) | context_gain sweep Fig e019-04 | Allocation architecture driven | Paper prose |
| SIG-D non-diagnostic alone (E018 finding replicated) | Validated | Wide pass range across params | — | Paper prose |
| Model robust to all parameter combinations | Pending | Requires E020 | — | E020 |
| Model scales beyond 4-branch motif | Not supported | — | — | Future |
| Biological validation | Not supported | — | — | Future |
