# e019 — QC Report

## Determinism
All deterministic sweeps use default seed 42.
structural_noise sweep uses seeds: [0, 1, 2, 3, 4, 42, 101, 202, 303, 404].

## SIG-C
Computed in a dedicated context-probe simulation per run. Identical protocol to E018.

## SIG-E
Uses E018 generic rescue protocol (plain consolidation, no pre-cueing).

## Thresholds
Predeclared and identical to E018: {'SIG_A': 0.02, 'SIG_B': 0.05, 'SIG_C': 0.05, 'SIG_D': 5.0, 'SIG_E': 10.0}.
