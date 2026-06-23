# E019R — QC Report

## Shared module
`src/cytodend_accessmodel/signatures.py` is the single source of truth.

## Rescue conditions audited
5 conditions: ['no_rescue', 'targeted_overlap_rescue', 'generic_plain_consolidation', 'generic_all_branch_precue', 'nonoverlap_branch_rescue']

## Threshold provenance
DEFAULT_THRESHOLDS from signatures.py: {'SIG_A': 0.02, 'SIG_B': 0.05, 'SIG_C': 0.05, 'SIG_D': 5.0, 'SIG_E_normalized': 0.1, 'SIG_E_raw': 0.02}

## Protocol
E019 canonical (no inter-phase probe cues).
