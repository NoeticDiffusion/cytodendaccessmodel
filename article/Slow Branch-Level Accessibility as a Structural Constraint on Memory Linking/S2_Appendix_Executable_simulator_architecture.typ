#import "template.typ": essay-template, clean-table

#show: doc => essay-template(
  short_title: [S2 Appendix],
  doc,
)

// Tables in this appendix are numbered S2.1, S2.2, …
#set figure(numbering: (..num) => "S2." + str(num.at(-1)))

= S2 Appendix. Executable simulator architecture, canonical parameters, and reproducibility routing

== Scope

The main text's simulator-first evidence ladder is produced by the branch-resolved Python simulator package `cytodend_accessmodel` maintained in the `cytodendaccessmodel` repository. E025 supplies the primary equal-budget tuned common-probe comparison; E017--E022R provide the historical simulator traces, mechanism signatures, robustness, and motif context, and E023 assembles their figures. E024 established the common probe and is retained as protocol-development history. This appendix documents the canonical architecture and parameterization, historical comparator definitions, and the E025 adapters, tuning design, and holdout outcome. Reviewers should use `reviewer_slow_branch_level_accessibility.ipynb` and `CLAIMS_TO_EXPERIMENTS.md` for exact script-to-claim routing. S4 Appendix collects software identifiers, computational reporting conventions, and the consolidated reproducibility-entry table. Older experiments `exp001--exp016` remain available as legacy lineage and optional cross-checks rather than as the main evidence ladder.

== Current article routing

The simulator-facing article results are organized as follows:

- `E025`: equal-budget tuned common-probe comparison across ten simulator families; the full model has no unique advantage on the primary readout
- `E024`: historical common-probe analysis and protocol-development step; not pooled with E025 holdout results
- `E017`: traceable canonical simulator export and the locked SIG-A through SIG-E reference profile
- `E018`: baseline comparator matrix for fast-context-only, replay-without-structure, random-drift, and fixed-allocation baselines
- `E019`: one-at-a-time robustness across nine parameter families
- `E020`: paired-parameter heatmaps defining write, replay, timing, and overlap boundaries
- `E021` and `E021R`: scaling, motif generalization, and specificity boundaries
- `E022` and `E022R`: hard comparators and shuffled-replay audit

The older `exp001--exp016` family is still useful for onboarding and historical lineage, but it should not be read as the main evidence ladder for the current manuscript. E018/E022 comparator classifications describe their original mechanism-specific score definitions; E025 supplies the current common-readout result after equal-budget tuning.

This appendix should therefore be read together with S4 Appendix: S2 documents the
canonical simulator setup and experiment-family logic, whereas S4 documents the broader
computational materials, software resources, and reporting conventions.

== Network Architecture

The canonical simulation uses a minimal four-branch network:

- *Branches:* `b0`, `b1`, `b2`, `b3`
- *Overlap structure:* `b1` is the overlap branch shared by both traces; `b0` and `b2` are single-trace branches; `b3` is an unrelated background branch
- *Spines per branch:* 3 in the canonical simulator; manuscript claims are evaluated at the branch level even when spine proxies are exported in legacy demos
- *Initial structural state:* $M_b = 0.5$ for all branches at $t = 0$

*Trace allocations:*

- $mu_1$: `b0` = 0.90, `b1` = 0.85, `b2` = 0.05, `b3` = 0.05
- $mu_2$: `b0` = 0.05, `b1` = 0.85, `b2` = 0.90, `b3` = 0.05

*Cue inputs used during encoding and recall probes:*

- $mu_1$ cue: `b0` = 1.0, `b1` = 0.8, `b2` = 0.0, `b3` = 0.0
- $mu_2$ cue: `b0` = 0.0, `b1` = 0.8, `b2` = 1.0, `b3` = 0.0
- Ambiguous cue (context experiments): all branches = 0.5

== Discrete updates and scale

The primary E017--E025 results use discrete calls to `apply_cue` and
`run_consolidation`; their step count is not calibrated to physical time. All values
below are dimensionless simulator quantities. See the main text's variable table for
the complete symbol, code field, biological interpretation, and observability mapping.

On each cue call, effective access gates the raw cue to produce branch activation:

$ A_b^f = sigma(g_f I_b + g_c c_b + g_"sp" q_b - g_i h_b), quad
  A_b^s = sigma(g_M M_b), quad
  A_b = op("clip")_(0,1)(A_b^f A_b^s), quad
  x_b = A_b I_b, quad
  E_b^"cue" = op("clip")_(0,1)((1-d_E)E_b + abs(x_b)), quad
  P_b^"cue" = op("clip")_(0,1)((1-d_P)P_b). $

Here $q_b$ is mean local spine access (or 1 when no spines are present), $c_b$ is
context bias, and $h_b$ is inhibitory tone. A cue call does not integrate activity or
deliver replay. It does update $E_b$ and decay $P_b$, so independent probes need copies
of the same saved state.

During consolidation, replay IDs determine branch overlap $r_b$: selected traces are
weighted by replay priority and branch allocation, with optional adjacency-based
spillover. A nonempty `replay_trace_ids` list selects those traces; an empty list invokes
the default selection of all context-compatible traces. $P_b$ is updated from $r_b$ and
sleep drive, clipped to $[0,1]$, and optionally normalized if a positive global
`translation_budget` is exceeded. Then:

$ P_b^+ = op("clip")_(0,1)((1-d_P)P_b + g_R r_b + g_"sleep" s r_b), quad
  M_b^+ = op("clip")_(0,M_"max")(M_b + eta E_b P_b^+ max(0,W)(1-M_b/M_"max")
    - lambda_b M_b + epsilon_b), quad
  E_b^+ = op("clip")_(0,1)((1-d_E)E_b). $

In the $M_b$ update, $P_b^+$ is the readiness value after optional budget
normalization, and $M_"max"$ is positive.
$W$ is the consolidation window's `modulatory_drive`; $s$ is `sleep_drive`; and
$epsilon_b$ is a per-branch Gaussian draw with standard deviation `noise_scale` on each
pass. Setting `modulatory_drive=0` removes positive structural writing but leaves
structural decay, optional noise, and replay-driven $P_b$ updates active. In particular,
an empty replay-ID list is the default selection, not a no-replay condition. The E022
damage condition increases the target branch's decay rate during these passes; it is a
simulator intervention and does not specify a biological lesion.

Recall support is the allocation-weighted sum of access-gated activation, followed by a
context-mismatch penalty only when both context labels are present and differ. Structural
linking is $L_(mu nu)=sum_b a_(mu b)a_(nu b)M_b$; because it contains $M_b$, it is an
internal structural diagnostic and not a model-independent behavioral outcome.

== Canonical Parameter Set

#figure(
  kind: table,
  supplement: [Table],
  caption: [Canonical parameter set for the branch-resolved simulator. All values are dimensionless phenomenological parameters. `structural_lr`, `eligibility_decay`, and `replay_gain` jointly determine the slow write dynamics and are the most diagnosis-relevant for the paper's core claims.],
  table(
    columns: (auto, auto, auto, auto),
    inset: 6pt,
    stroke: 0.5pt + black,
    align: (left, left, right, left),
    table.header(
      [*Parameter*], [*Symbol / role*], [*Value*], [*Biological interpretation*],
    ),
    [`structural_lr`], [$eta$ — learning coefficient], [0.18], [scales the eligibility × readiness write per consolidation pass; no physical time calibration],
    [`structural_gain`], [gain on $A_b^s = sigma(g_M M_b)$], [6.0], [controls the slow-access sigmoid slope],
    [`structural_decay`], [$lambda_b$ — turnover], [0.005], [structural decay per consolidation pass; no physical time calibration],
    [`structural_max`], [$M_"max"$], [1.0], [finite local accessibility capacity],
    [`structural_noise`], [SD of $epsilon_b$], [0.0], [Gaussian noise SD per consolidation pass; zero in deterministic reference runs],
    [`eligibility_decay`], [decay of $E_b$], [0.12], [per cue/consolidation call; a tag-like proxy with no calibrated biological lifetime @RedondoMorris2011STC @Gerstner2018EligibilityTraces @Rogerson2014SynapticTaggingAllocation],
    [`translation_decay`], [decay of $P_b$], [0.05], [per cue/consolidation call; a resource-turnover proxy @Govindarajan2011DendriticBranch @Das2023LocalTranslationMemory @Rangaraju2019MitoCompartments],
    [`replay_gain`], [weight of replay on $P_b$], [0.80], [scales replay-driven readiness per consolidation pass @Wang2024SleepDependentEngramReactivation @PeyracheSeibt2020Spindles],
    [`sleep_gain`], [weight of sleep drive on $P_b$], [0.0], [set to zero in the primary runs; nonzero in a sleep-window extension @Seibt2017DendriticSpindles @TononiCirelli2020SleepPlasticity],
    [`fast_gain`], [gain on fast access $A_b^f$], [2.0], [amplifies cue drive into fast sigmoid],
    [`context_gain`], [weight of context bias term], [1.0], [multiplicative context gating in fast access],
    [`readout_gain`], [sigmoid slope at recall readout], [5.0], [sharpness of expressed recall threshold],
    [`readout_threshold`], [$theta_mu$ at readout], [0.3], [threshold for expressed retrieval in the canonical E017/E018 reference stack],
    [`translation_budget`], [optional global cap on total $P_b$], [0.0], [zero disables normalization; positive values scale all $P_b$ when the total exceeds the cap],
    [`spillover_rate`], [neighbor contribution to $r_b$], [0.0], [zero disables adjacency-based replay spillover],
  ),
) <tab-s2-params>

== Encoding and Consolidation Protocol

All main experiments follow a three-phase protocol:

1. *Encoding:* Two cue passes for $mu_1$, then two cue passes for $mu_2$ (four `apply_cue` calls total). Each cue pass updates fast access, eligibility traces, and spine calcium proxies.

2. *Pre-consolidation probe:* Recall support is computed immediately after encoding to provide a baseline.

3. *Consolidation:* Nine passes of `run_consolidation` with a `ConsolidationWindow` specifying `replay_trace_ids = [mu_1, mu_2]` and `modulatory_drive = 1.0`. Each pass executes two sub-steps: first updating $P_b$ from replay overlap and sleep drive; then updating $M_b$ via the bounded tag-dependent rule; then decaying $E_b$.

The linking metric is computed after consolidation as:

$ L_(mu_1 mu_2) = sum_b a_(mu_1 b) a_(mu_2 b) M_b $

*Focal damage protocol (SIG-D and SIG-E):* An elevated per-branch decay rate (`decay_rate = 0.030` versus the canonical 0.005) is applied to the overlap branch `b1`, followed by nine zero-modulation consolidation passes (`modulatory_drive = 0.0`). In the current simulator, an empty replay-ID list selects the default traces, so replay can still update $P_b$ during these passes; they are not a no-replay condition. This is a simulator-level decay intervention, not a direct biological lesion.

== Baseline comparator definitions (E018)

Four simpler models were run against the full model on the five signature families (SIG-A through SIG-E). Each baseline corresponds to a specific mechanistic ablation or substitution.

#figure(
  kind: table,
  supplement: [Table],
  caption: [Historical comparator definitions for E018. Random seed fixed to 42 for the random-drift baseline. The original joint criterion required each baseline to pass all five mechanism signatures simultaneously; it is not a common-readout model-selection criterion.],
  table(
    columns: (auto, auto, auto),
    inset: 6pt,
    stroke: 0.5pt + black,
    align: (left, left, left),
    table.header(
      [*Baseline label*], [*Mechanistic change*], [*Key parameter difference*],
    ),
    [`full_model`], [standard simulator], [`structural_lr = 0.18`],
    [`fast_context_only`], [structural state fixed throughout; no slow writing or decay], [`structural_lr = 0.0`, `structural_decay = 0.0`],
    [`replay_no_structure`], [replay updates $P_b$ and $E_b$ but $M_b$ cannot be written], [`structural_lr = 0.0` (same mechanistic ablation later mirrored by legacy `exp014`)],
    [`random_slow_drift`], [slow term of matched scale drives $M_b$ randomly (Gaussian noise, $sigma = 0.025$ per pass) instead of tag-dependent writing], [full params; consolidation replaced by unstructured drift],
    [`fixed_allocation_only`], [hand-designed branch overlap preserved as initial $M_b$; all dynamic updating removed], [`structural_lr = 0.0`, `replay_gain = 0.0`, `eligibility_decay = 0.0`, `structural_decay = 0.0`],
  ),
) <tab-s2-comparators>

The locked signature thresholds used in the locked reference stack are:

- *SIG-A* (overlap-branch structural advantage): $> 0.02$
- *SIG-B* (linking gain): $> 0.05$
- *SIG-C* (context separation): $> 0.05$
- *SIG-D* (linking vs. recall dissociation): $> 5$ percentage points
- *SIG-E* (targeted rescue selectivity): normalized recovery difference $> 0.10$ under the locked E019R/E022 protocol

_Note:_ earlier probe-warmed protocol variants (E018 warm-probe) reported a percentage-point rescue advantage; the current manuscript uses the normalized recovery convention throughout.

== Reproducibility, robustness, and later extensions

The canonical reference runs in `E017` and `E018` use `structural_noise = 0.0` and are therefore deterministic once the protocol is fixed. The random-drift baseline in `E018` uses a fixed seed so that the comparator remains inspectable rather than sampling a new drift realization on every run.

For the current article, robustness and extension analyses should be read through the newer experiment stack rather than through the older `exp004`-style reviewer sweeps:

- `E019` performs one-at-a-time robustness across nine parameter families and shows that the joint signature profile is not a single-run artifact.
- `E020` adds six paired-parameter heatmaps, including the key `structural_lr × replay_gain`, `eligibility_decay × timing_gap`, and `overlap_strength × replay_gain` boundaries.
- `E021` and `E021R` extend the canonical motif to larger branch counts and additional overlap topologies, revealing expected weak-overlap failure and hub-overlinking boundary conditions.
- `E022` and `E022R` add hard comparators and the shuffled-replay audit. Their conclusions concern the original mixed proxy definitions and implemented structural-specificity metric.
- `E024` applied the shared one-hot cue/probe schedule and sigmoid output equation to ten simulator families under four primary motifs, with 50 paired evaluation seed clusters. Six of eight alternatives meeting baseline calibration were practically equivalent to the full model; Hebbian and resource-only alternatives had larger primary point estimates but were classified inconclusive under the frozen E024 decision rule. E024 is retained as protocol history and is not pooled with E025.

== Historical E024 common-probe analysis

The protocol in `project/neural_computation/model_comparison_protocol.md` was frozen
before E024 evaluation. The primary outcome is the post-minus-pre change in target-trace
readout for a one-hot cue to a shared branch relative to a matched source-private cue.
Every probe starts from an independent copy of the saved pre- or post-consolidation
state. The shared output equation is `s_j = sum_b W_jb h_b x_b` followed by one fixed
sigmoid function. The comparator state adapters are model-specific and are fully
described in `results/e024_common_probe_comparison/summary/model_registry.json`; the
comparison therefore concerns these implementations.

The full-model mean was `D=0.010487` (95% seed-bootstrap CI `0.010441–0.010528`). Six of
eight baseline-matched alternatives were practically equivalent under the registered
Holm-corrected tests. Hebbian weight-only and resource-only had means `0.054435` and
`0.052594`, respectively; the registered rule did not test comparator superiority and
classified both as inconclusive. Eligibility-only missed the baseline-match criterion
and is exploratory. Published-parameter decisions were the same.

Five motifs were built per seed, but hub-overlap had no unlinked pairs and remained a
boundary case. Rows without an eligible shared/private probe pair also remained `N/A`.
The raw primary probes, manifest, calibration grid, model registry, per-seed outcomes,
paired comparisons, and figure are saved in `results/e024_common_probe_comparison/`.
Recovery-arm aggregation is reproduced by `experiments/exp024_summarize_secondary.py`;
the figure-only English rendering is reproduced by `experiments/exp024_render_english_figure.py`.

== E025 equal-budget tuning and locked holdout

The preregistered E025 plan, parameter registry, topology manifest, and probe manifest
are stored under `project/neural_computation/`. The plan and generated manifests were
frozen before outcomes were inspected; their recorded hashes are preserved alongside
the files. The runner is `experiments/exp025_tuned_common_probe.py`, and the publication
figure can be regenerated with `experiments/exp025_render_common_probe_figure.py`.

All ten families received 32 candidate configurations under the same evaluation budget.
The five initial parameter points included the E024 nominal configuration and gain grid;
the remaining points were generated from the frozen low-discrepancy design over each
family's active parameters. Adapters and non-searchable mechanism choices were fixed by
the E025 registry. The 40 development draws ranked candidates, 40 independent validation
draws selected one configuration per qualifying family, and the 100 locked holdout draws
were opened once for the selected candidate and the nominal setting. Holdout results were
not used to retune parameters or alter adapters, filters, topology weights, or hypotheses.

Every holdout draw contains the same four primary topology strata. Cue-pair scores are
averaged within directed pair, then within stratum; the four stratum means receive equal
weight for each draw. The 100 draw-level macros are the independent units for bootstrap intervals and the
paired sign-flip tests. The separate `hub_overlap` condition is a sensitivity
stratum, not part of the four-stratum mean. E025's probe and readout match the E024
definition; independent probe copies prevent probe-induced changes in eligibility or
readiness from carrying into another probe.

Candidate qualification requires (1) pre-consolidation private-cue readout within ±0.02
of the E024 nominal full-model reference on every primary stratum, (2) no more than 0.02
loss in correct private-cue recall after consolidation, and (3) nonnegative correct-minus-
largest-incorrect private-cue margin. Nine families produced a selected tuned candidate.
Eligibility-only produced no qualifying candidate during development and validation.
Its nominal holdout scores are descriptive only; inferential outcomes against that
family are unavailable.

#figure(
  kind: "table",
  supplement: "Table",
  caption: [E025 primary holdout means and Holm-corrected decision for each selected tuned family. The decision column refers to the paired full-minus-comparator difference on immediate $D$.],
  clean-table(
    columns: (1.7fr, 1.35fr, 0.8fr, 1.8fr),
    header: ([*Family*], [*Selected candidate*], [*Mean D*], [*Decision vs full*]),
    [Full model], [`full_model_c16`], [0.02270], [Reference],
    [Fast context only], [`fast_context_only_c02`], [0.00000], [Full superiority],
    [Replay, no structure], [`replay_no_structure_c24`], [-0.00094], [Full superiority],
    [Random slow drift], [`random_slow_drift_c24`], [-0.00025], [Full superiority],
    [Fixed allocation], [`fixed_allocation_only_c02`], [0.00000], [Full superiority],
    [Hebbian weights], [`hebbian_weight_only_c05`], [0.09330], [Comparator superiority],
    [Global gain], [`soma_global_gain_only_c01`], [0.00000], [Full superiority],
    [Shuffled replay], [`shuffled_replay_c12`], [0.00703], [Practical equivalence],
    [Eligibility only], [No qualifying candidate], [N/A], [Unavailable],
    [Resource only], [`resource_only_c16`], [0.09435], [Comparator superiority],
  ),
) <tab-e025-holdout>

The full model's mean was $D=0.022703$ (95% bootstrap CI 0.022538--0.022864).
Hebbian weight-only produced $D=0.093304$ (0.092157--0.094452), and resource-only
produced $D=0.094349$ (0.093087--0.095636). The paired full-minus-Hebbian contrast was
$-0.070601$ (95% CI $-0.071621$ to $-0.069593$); full-minus-resource was $-0.071645$
($-0.072817$ to $-0.070484$). Both support comparator superiority after the joint
Holm correction. Full-model superiority was supported against five alternatives;
shuffled replay met the registered equivalence test. Eligibility-only was not inferentially
tested because it had no qualifying tuned candidate.

After nine passive-decay steps, the full model's secondary $D_"late"$ was 0.017951
(95% CI 0.017801--0.018097), compared with 0.092944 (0.091807--0.094081) for Hebbian
weight-only and 0.071630 (0.070997--0.072270) for resource-only. The two comparators
again exceeded the full model after correction. This secondary endpoint does not replace
the immediate primary outcome.

For each endpoint and comparator, the analysis separately tests full-model superiority
($Delta > 0.02$), comparator superiority ($Delta < -0.02$), and practical equivalence
within ±0.02. The reported 95% percentile confidence intervals use 10,000 shared-index
bootstrap resamples; they are pointwise and are not adjusted across the 54 registered
hypotheses. The hypothesis tests use 100,000 paired sign-flip resamples with a joint
Holm correction over 54 claims (9 comparators × 2 endpoints × 3 outcomes). Sign-flip
calibration assumes the paired topology-draw differences are sign-exchangeable
(symmetric around each tested null boundary); comparator signs were not randomized by
the simulator design, so these are not exact randomization tests. The margin applies to
paired contrasts, never to absolute $D$ values. The hub sensitivity scores are preserved in
`results/e025_common_probe_tuning/summary/hub_sensitivity.csv`; they are reported
separately because hub-overlap is not part of the primary macro-effect.

A post-hoc centered cluster-bootstrap sensitivity analysis was added after the holdout
results were available; it was not preregistered and does not replace the registered
sign-flip analysis. It resamples empirical paired topology-draw residuals centered
at each null boundary (200,000 resamples; seed 250926), then applies Holm correction
across the same 54 hypotheses. This approximate bootstrap test holds the selected candidates and adapters fixed
and assumes independent topology draws from the declared generator. It reproduced the immediate decisions for
the testable families, including comparator superiority for Hebbian and resource-only
and practical equivalence for shuffled replay. Full p-values are in
results/e025_common_probe_tuning/sensitivity/centered_cluster_bootstrap_holm.csv;
the analysis script and assumptions are documented in
project/neural_computation/e025_inference_sensitivity_20261030.md.

Complete candidate scores, holdout draw scores, nominal scores, comparisons, hypotheses,
run summary, and output hashes are stored in `results/e025_common_probe_tuning/`. The
main analysis uses simulated topologies and technical simulator draws; these counts are
not animal, subject, or experimental-replicate sample sizes.

The older `exp013_paper_summary.py` and `exp_seed_validation.py` scripts remain useful as optional historical cross-checks for reviewers who want lineage or parameter-history context, but they are not the main reproducibility route for the current article.

#bibliography("references_slow_branch_level_accessibility.bib")
