# e018 — Comparator Trace Matrix

**Date:** 2026-06-22
**Comparators:** full_model, fast_context_only, replay_no_structure, random_slow_drift, fixed_allocation_only

## Purpose
Model-discrimination experiment: does the full model uniquely reproduce the
joint SIG-A–SIG-E signature profile among the tested comparators?

## Run
```bash
python experiments/exp018_comparator_trace_matrix.py
```

## Key result
See `summary/joint_pass_summary.json` and
`figures/Fig_e018_01_comparator_signature_matrix.png`.

## Claim scope
Only if full model passes and no simpler comparator passes the joint profile:
> "Under canonical parameters, the joint SIG-A to SIG-E profile is specific
> to the full replay-dependent slow structural writing model."
