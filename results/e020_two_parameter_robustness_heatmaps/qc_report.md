# E020 — QC Report

## Signature module
All computations use ``src/cytodend_accessmodel/signatures.py`` (E019R locked).

## SIG-E unit
Normalized recovery difference (NOT percentage points). Can exceed 1.0.

## SIG-C cache
Cached per unique (context_gain, structural_lr, eligibility_decay, replay_gain, structural_decay, seed).

## Noise grids
structural_noise × replay_gain uses seeds: [0, 1, 2, 3, 4, 42, 101, 202, 303, 404].

## Thresholds
{'SIG_A': 0.02, 'SIG_B': 0.05, 'SIG_C': 0.05, 'SIG_D': 5.0, 'SIG_E_normalized': 0.1, 'SIG_E_raw': 0.02}
