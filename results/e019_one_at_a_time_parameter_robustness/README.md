# e019 — One-at-a-Time Parameter Robustness

**Date:** 2026-06-22
**Parameters swept:** structural_lr, replay_gain, eligibility_decay, structural_decay, structural_noise, context_gain, timing_gap, overlap_strength, readout_threshold

## Purpose
Test whether the full model's joint SIG-A–E profile is robust to one-at-a-time
parameter variation, or is narrow around the canonical set.

## Run
```bash
python experiments/exp019_one_at_a_time_parameter_robustness.py
```

## Key outputs
- `summary/robustness_summary_by_parameter.csv` — per-parameter joint-pass statistics
- `summary/failure_boundary_summary.csv` — first failure value per (param, sig)
- `figures/Fig_e019_01_joint_pass_by_parameter.png` — main overview figure

## Claim scope (if successful)
> "The full model's canonical joint signature profile is not a single-run artifact;
> it survives one-at-a-time variation across defined parameter ranges."
