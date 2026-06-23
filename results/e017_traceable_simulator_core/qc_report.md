# e017 — QC Report

## Variable availability
All required trace variables present: x_b, fast_access, slow_access,
effective_access, eligibility, translation_readiness, structural_accessibility,
recall_support, readout_value, linking_score.

## Missing variables
- `input_drive` is NaN during consolidation phases (no cue applied). Expected.
- `context_value` is not currently exported (not in BranchState). Noted.

## Determinism
Fixed seed: 42. Structural noise = 0.0. Run is deterministic.
Re-run and compare hashes from `run_metadata.json` to verify.

## Phase coverage
All 10 phases present in branch_traces.csv.

## SIG-C note
SIG-C uses a separate context-probe simulator (not the main canonical sim).
Traces for SIG-C are therefore not in branch_traces.csv. Score is reported
in signature_summary.csv. This is expected and documented.

## Signature summary

| Signature | Value | Direction pass |
|-----------|-------|---------------|
| SIG-A overlap advantage     | 0.1281 | PASS |
| SIG-B linking gain          | 0.2397 | PASS |
| SIG-C context separation    | 0.1545 | PASS |
| SIG-D dissociation (pp)     | 20.98 | PASS |
| SIG-E rescue advantage (pp) | 33.29 | PASS |

Note: "direction pass" = signature is in the predicted positive direction.
Magnitude and robustness are assessed in later experiments (e018, e019).
