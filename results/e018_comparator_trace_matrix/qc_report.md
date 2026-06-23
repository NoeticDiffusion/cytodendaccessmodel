# e018 — QC Report

## Determinism
All comparators except random_slow_drift use fixed seed 42 and structural_noise=0.
random_slow_drift uses fixed seed 42; drift is Gaussian but seeded.

## SIG-C
Computed in separate context-probe sim. Traces exported to context_probe_traces.csv per comparator.
See summary/context_probe_limitations.md.

## SIG-E
Now compares targeted_overlap_rescue vs generic_all_branch_rescue.
See summary/rescue_protocol.md.

## Variable coverage
All required trace columns exported. context_value not in main branch_traces (known limitation).

## Threshold predeclaration
Thresholds set before analysis: {'SIG_A': 0.02, 'SIG_B': 0.05, 'SIG_C': 0.05, 'SIG_D': 5.0, 'SIG_E': 10.0}.
