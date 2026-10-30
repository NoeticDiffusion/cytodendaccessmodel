# E025 inference sensitivity note (2026-10-30)

This is a retrospective analysis note. It was written after E025 holdout results
were available and is not part of the preregistration.

## Sign-flip assumption

The preregistered runner applies random sign multipliers to topology-level paired
contrasts. This is a sign-flip test, not an exact randomization test arising from
the design: model labels/signs were not randomized. Its null calibration requires
the paired contrast distribution to be sign-exchangeable, equivalently symmetric
around the tested boundary under the relevant null. The simulator design alone
does not establish this assumption.

## Complementary centered cluster-bootstrap sensitivity

The companion script
experiments/exp025_cluster_bootstrap_sensitivity.py reuses the saved E025 holdout
draw-level macro scores and tests each null boundary by resampling empirical paired
topology-draw residuals after centering them at that boundary. It uses 200,000
resamples, seed 250926, the same ±0.02 margin and 54-hypothesis Holm family.
The selected candidates and adapters are held fixed, and the 100 topology-draw
macros are treated as independent draws from the declared topology generator.
This is an approximate bootstrap sensitivity analysis conditional on those
selections and that draw-generating distribution; it is not an exact randomization
procedure and was not preregistered.

The output is
results/e025_common_probe_tuning/sensitivity/centered_cluster_bootstrap_holm.csv;
run metadata and hashes are in the neighboring JSON. The analysis reproduced
the immediate registered decisions for the testable families, including Hebbian
and resource-only comparator superiority and practical equivalence to shuffled
replay. Monte Carlo p-values have finite resolution determined by the 200,000
resamples; the output records the estimated values and corrected decisions.
