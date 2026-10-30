# E024 claim ledger

## Primary model-independent readout

The outcome is the post-minus-pre change in target-trace readout for a one-hot cue to a shared branch relative to a matched source-private branch. Values are simulator readout scores, not empirical probabilities.

Full-model baseline-matched macro mean D: `0.0104874` (95% bootstrap CI `0.0104406` to `0.0105282`).
## Comparator decisions

### Baseline Matched

- `fast_context_only`: practical_equivalence; mean full-minus-comparator `0.0104874`; paired seeds 50.
- `replay_no_structure`: practical_equivalence; mean full-minus-comparator `0.0123601`; paired seeds 50.
- `random_slow_drift`: practical_equivalence; mean full-minus-comparator `0.0123221`; paired seeds 50.
- `fixed_allocation_only`: practical_equivalence; mean full-minus-comparator `0.0104874`; paired seeds 50.
- `hebbian_weight_only`: inconclusive; mean full-minus-comparator `-0.0439481`; paired seeds 50.
- `soma_global_gain_only`: practical_equivalence; mean full-minus-comparator `0.0104874`; paired seeds 50.
- `shuffled_replay`: practical_equivalence; mean full-minus-comparator `0.00391929`; paired seeds 50.
- `eligibility_only`: practical_equivalence; mean full-minus-comparator `0.0100894`; paired seeds 50.
- `resource_only`: inconclusive; mean full-minus-comparator `-0.0421069`; paired seeds 50.

### Published Parameters

- `fast_context_only`: practical_equivalence; mean full-minus-comparator `0.0104874`; paired seeds 50.
- `replay_no_structure`: practical_equivalence; mean full-minus-comparator `0.0123601`; paired seeds 50.
- `random_slow_drift`: practical_equivalence; mean full-minus-comparator `0.0123221`; paired seeds 50.
- `fixed_allocation_only`: practical_equivalence; mean full-minus-comparator `0.0104874`; paired seeds 50.
- `hebbian_weight_only`: inconclusive; mean full-minus-comparator `-0.0439481`; paired seeds 50.
- `soma_global_gain_only`: practical_equivalence; mean full-minus-comparator `0.0104874`; paired seeds 50.
- `shuffled_replay`: practical_equivalence; mean full-minus-comparator `0.00391929`; paired seeds 50.
- `eligibility_only`: practical_equivalence; mean full-minus-comparator `0.0100517`; paired seeds 50.
- `resource_only`: inconclusive; mean full-minus-comparator `-0.0421069`; paired seeds 50.

## Interpretation limits

- `N/A` motifs or adapters are not failures; see `probe_manifest.csv` and `primary_by_seed.csv`.
- Internal `M_b`, `E_b`, and `P_b` summaries are mechanism diagnostics and do not establish biological identity or validation.
- A comparator classified as practically equivalent prevents a claim that the full model is required for this readout.
- Results apply only to the motifs, parameters, adapter equations, and common task recorded in `metadata/run_config.json`.
