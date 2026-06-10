#import "template.typ": essay-template

#show: doc => essay-template(
  short_title: [S2 Appendix],
  doc,
)

= S2 Appendix. Executable simulator architecture, parameters, and comparator baselines

== Scope

The results reported in Table 1 and in the Executable Instantiation section of the main text were produced by a branch-resolved Python simulator (`cytodend_keylock`) whose source code and experiment scripts are maintained in the public `cytodendaccessmodel` repository and archived in the manuscript-matched Zenodo snapshot (DOI `10.5281/zenodo.17686203`). This appendix documents the canonical parameter set, the network architecture, the encoding and consolidation protocol, and the four comparator baselines used for claim sharpening. Reviewers can reproduce all numerical results by running the scripts listed in `S3 Appendix`, specifically `experiments/exp001_minimal_branch_linking.py` through `experiments/exp015_comparator_baselines.py`.

== Network Architecture

The canonical simulation uses a minimal four-branch network:

- *Branches:* `b0`, `b1`, `b2`, `b3`
- *Overlap structure:* `b1` is the overlap branch shared by both traces; `b0` and `b2` are single-trace branches; `b3` is an unrelated background branch
- *Spines per branch:* 3 (for spine-level calcium proxy tracking in exp001; 0 in the comparator baseline experiments where branch-level dynamics are the focus)
- *Initial structural state:* $M_b = 0.5$ for all branches at $t = 0$

*Trace allocations:*

- $mu_1$: `b0` = 0.90, `b1` = 0.85, `b2` = 0.05, `b3` = 0.05
- $mu_2$: `b0` = 0.05, `b1` = 0.85, `b2` = 0.90, `b3` = 0.05

*Cue inputs used during encoding and recall probes:*

- $mu_1$ cue: `b0` = 1.0, `b1` = 0.8, `b2` = 0.0, `b3` = 0.0
- $mu_2$ cue: `b0` = 0.0, `b1` = 0.8, `b2` = 1.0, `b3` = 0.0
- Ambiguous cue (context experiments): all branches = 0.5

== Canonical Parameter Set

#table(
  columns: (auto, auto, auto, auto),
  inset: 6pt,
  stroke: 0.5pt + black,
  align: (left, left, right, left),
  table.header(
    [*Parameter*], [*Symbol / role*], [*Value*], [*Biological interpretation*],
  ),
  [`structural_lr`], [$eta$ — learning rate], [0.18], [scales how strongly eligibility + capture drive $dot(M)_b$],
  [`structural_gain`], [gain on $A_b^s = sigma(g dot.c M_b)$], [6.0], [controls slope of slow-access sigmoid],
  [`structural_decay`], [$lambda_M$ — turnover], [0.005], [ongoing structural destabilization per step],
  [`structural_max`], [$M_"max"$], [1.0], [finite local accessibility capacity],
  [`structural_noise`], [$T_"eff"$ scale], [0.0], [set to zero for deterministic reference runs; nonzero in robustness sweeps],
  [`eligibility_decay`], [decay of $E_b$], [0.12], [tag lifetime; matches STC tag window of ~minutes–hours phenomenologically],
  [`translation_decay`], [decay of $P_b$], [0.05], [capture resource turnover per step],
  [`replay_gain`], [weight of replay on $P_b$], [0.80], [scales how strongly replay recruits $P_b$],
  [`sleep_gain`], [weight of sleep drive on $P_b$], [0.0], [set to zero in main runs (replay-only); nonzero in sleep-window extension],
  [`fast_gain`], [gain on fast access $A_b^f$], [2.0 (exp001); implicit via context], [amplifies cue drive into fast sigmoid],
  [`context_gain`], [weight of context bias term], [1.0], [multiplicative context gating in fast access],
  [`readout_gain`], [sigmoid slope at recall readout], [5.0], [sharpness of expressed recall threshold],
  [`readout_threshold`], [$theta_mu$ at readout], [0.3 (exp001) / 0.5 (exp015)], [threshold for expressed retrieval],
  [`context_mismatch_penalty`], [penalty factor on wrong-context recall], [0.25], [reduces recall support by 25% when context mismatches],
)

#par(first-line-indent: 0pt)[
  #emph[Table 5. Canonical parameter set for the branch-resolved simulator. All values are dimensionless phenomenological parameters. The structural_lr, eligibility_decay, and replay_gain parameters jointly determine the slow write dynamics and are the most diagnosis-relevant for the paper's core claims.]
]

== Encoding and Consolidation Protocol

All main experiments follow a three-phase protocol:

1. *Encoding:* Two cue passes for $mu_1$, then two cue passes for $mu_2$ (four `apply_cue` calls total). Each cue pass updates fast access, eligibility traces, and spine calcium proxies.

2. *Pre-consolidation probe:* Recall support is computed immediately after encoding to provide a baseline.

3. *Consolidation:* Nine passes of `run_consolidation` with a `ConsolidationWindow` specifying `replay_trace_ids = [mu_1, mu_2]` and `modulatory_drive = 1.0`. Each pass executes two sub-steps: first updating $P_b$ from replay overlap and sleep drive; then updating $M_b$ via the bounded tag-dependent rule; then decaying $E_b$.

The linking metric is computed after consolidation as:

$ L_(mu_1 mu_2) = sum_b a_(mu_1 b) a_(mu_2 b) M_b $

*Focal damage protocol (SIG-D and SIG-E):* An elevated per-branch decay rate (`decay_rate = 0.030` versus the canonical 0.005) is applied to the overlap branch `b1`, followed by nine null consolidation passes (`modulatory_drive = 0.0`, no replay), to simulate selective structural destabilization.

== Comparator Baselines

Four simpler models were run against the full model on the five signature families (SIG-A through SIG-E). Each baseline corresponds to a specific mechanistic ablation or substitution.

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  stroke: 0.5pt + black,
  align: (left, left, left),
  table.header(
    [*Baseline label*], [*Mechanistic change*], [*Key parameter difference*],
  ),
  [`full_model`], [standard simulator], [`structural_lr = 0.18`],
  [`fast_context_only`], [structural state fixed throughout; no slow writing or decay], [`structural_lr = 0.0`, `structural_decay = 0.0`],
  [`replay_no_structure`], [replay updates $P_b$ and $E_b$ but $M_b$ cannot be written], [`structural_lr = 0.0` (identical to exp014 gate ablation)],
  [`random_slow_drift`], [slow term of matched scale drives $M_b$ randomly (Gaussian noise, $sigma = 0.025$ per pass) instead of tag-dependent writing], [full params; consolidation replaced by unstructured drift],
  [`fixed_allocation_only`], [hand-designed branch overlap preserved as initial $M_b$; all dynamic updating removed], [`structural_lr = 0.0`, `replay_gain = 0.0`, `eligibility_decay = 0.0`, `structural_decay = 0.0`],
)

#par(first-line-indent: 0pt)[
  #emph[Table 6. Comparator baseline definitions. Random seed fixed to 42 for the random-drift baseline. The joint criterion is that a baseline must pass all five signature thresholds simultaneously; no simpler baseline meets this criterion.]
]

The signature thresholds used to define a directional pass are:

- *SIG-A* (overlap-branch structural advantage): $Delta M_(b_1) - Delta M_(b_0) >$ half the full-model value
- *SIG-B* (linking gain): linking metric increase $> 5\%$
- *SIG-C* (context separation): correct-context minus wrong-context support $>$ half the full-model value
- *SIG-D* (linking vs. recall dissociation): linking drop exceeds recall drop by $> 5$ percentage points
- *SIG-E* (targeted rescue advantage): overlap-targeted rescue exceeds standard rescue by $> 10$ percentage points

== Initial Conditions and Stochastic Reproducibility

All deterministic experiments use `structural_noise = 0.0` and therefore produce identical results on repeated runs without a random seed. The random-drift comparator baseline (`random_slow_drift`) uses `random.seed(42)` at the start of `exp015_comparator_baselines.py`. Robustness sweeps across noise levels and initial structural states are reported in `experiments/exp004_robustness.py` and `experiments/exp_seed_validation.py`; these confirm that the directional pass rates on the five protected claims reach 100% across the tested parameter variation range.

== Parameter variation and sensitivity analysis

To show that the simulator is not a knife-edge construction, we also ran a reviewer-facing one-at-a-time sweep over six parameter families in `experiments/exp004_robustness.py`. This package uses `10` stochastic seeds per parameter value and keeps a small background structural noise term (`structural_noise = 0.005`) active so that the robustness check samples genuinely different trajectories rather than repeated deterministic copies. The tracked metrics span the same executable logic used in the article: overlap advantage, linking growth, context separation, context-gap widening after consolidation, timing sensitivity, and replay dependence.

#table(
  columns: (auto, auto, auto, auto),
  inset: 6pt,
  stroke: 0.5pt + black,
  align: (left, left, center, left),
  table.header(
    [*Sweep family*], [*Tested values*], [*Minimum pass rate*], [*Interpretation*],
  ),
  [`structural_lr`], [`0.08`, `0.12`, `0.20`, `0.28`, `0.35`], [`100%`], [all six metrics preserve direction across a ~4x write-rate range],
  [`replay_gain`], [`0.5`, `0.8`, `1.2`, `1.6`, `2.0`], [`100%`], [replay dependence and linking signatures remain intact across a broad replay-strength sweep],
  [`eligibility_decay`], [`0.05`, `0.08`, `0.10`, `0.15`, `0.20`], [`100%`], [protected claims survive shorter and longer effective tag windows],
  [`structural_noise`], [`0.0`, `0.005`, `0.010`, `0.015`, `0.020`], [`100%`], [directional claims survive stochastic structural perturbation],
  [`gap`], [`0`, `4`, `8`, `12`, `16`], [`60%` at `0`; `100%` for all `> 0`], [the only weakened entry is the degenerate `gap = 0` case, where the timing comparison is definitionally absent],
  [`ctx_bias_scale`], [`0.3`, `0.5`, `0.7`, `0.9`, `1.1`], [`100%`], [context-related metrics remain positive even under weak contextual bias],
)

#par(first-line-indent: 0pt)[
  #emph[Table S2-1. Reviewer-facing sensitivity summary for the executable model. Pass rate denotes the fraction of stochastic seeds for which the metric retained its predicted direction under the indicated sweep.]
]

These sweeps are complementary to, rather than replacements for, the deterministic reference runs in the main text. Their role is to show that the article's directional claims do not depend on a single hand-tuned parameter choice. The strongest caveat is also the most interpretable one: the timing metric weakens only when the comparison gap is set to zero, which removes the timing manipulation itself. Additional high-sensitivity seed validation under `structural_noise = 0.02` is reported in `experiments/exp_seed_validation.py`, where the protected claim panel again reaches full directional pass rates under the designated seed tiers.

#bibliography("references_cytoskeletal_dendritic_accesibility_model.bib")
