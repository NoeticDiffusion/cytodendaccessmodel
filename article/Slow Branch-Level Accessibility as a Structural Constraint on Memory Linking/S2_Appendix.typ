#import "template.typ": essay-template

#show: doc => essay-template(
  short_title: [S2 Appendix],
  doc,
)

// Tables in this appendix are numbered S2.1, S2.2, …
#set figure(numbering: (..num) => "S2." + str(num.at(-1)))

= S2 Appendix. Executable simulator architecture, canonical parameters, and reproducibility routing

== Scope

The main text's simulator-first evidence ladder is produced by the branch-resolved Python simulator package `cytodend_accessmodel` maintained in the public `cytodendaccessmodel` repository and archived in the manuscript-matched Zenodo snapshot. For the current manuscript, the primary reproducibility path is `E017--E022R`, with `E023` assembling the manuscript figures. This appendix documents the canonical architecture and parameterization used by that stack, the baseline comparator definitions, and how later robustness, scaling, specificity, and hard-comparator analyses extend the core simulator. Reviewers should use `reviewer_slow_branch_level_accessibility.ipynb` and `CLAIMS_TO_EXPERIMENTS.md` for exact script-to-claim routing. S4 Appendix collects software identifiers, computational reporting conventions, and the consolidated reproducibility-entry table. Older experiments `exp001--exp016` remain available as legacy lineage and optional cross-checks rather than as the main evidence ladder.

== Current article routing

The simulator-facing article results are organized as follows:

- `E017`: traceable canonical simulator export and the locked SIG-A through SIG-E reference profile
- `E018`: baseline comparator matrix for fast-context-only, replay-without-structure, random-drift, and fixed-allocation baselines
- `E019`: one-at-a-time robustness across nine parameter families
- `E020`: paired-parameter heatmaps defining write, replay, timing, and overlap boundaries
- `E021` and `E021R`: scaling, motif generalization, and specificity boundaries
- `E022` and `E022R`: hard comparators and shuffled-replay audit

The older `exp001--exp016` family is still useful for onboarding and historical lineage, but it should not be read as the main evidence ladder for the current manuscript.

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
    [`structural_lr`], [$eta$ — learning rate], [0.18], [scales how strongly eligibility + capture drive $dot(M)_b$],
    [`structural_gain`], [gain on $A_b^s = sigma(g dot.c M_b)$], [6.0], [controls slope of slow-access sigmoid],
    [`structural_decay`], [$lambda_M$ — turnover], [0.005], [ongoing structural destabilization per step],
    [`structural_max`], [$M_"max"$], [1.0], [finite local accessibility capacity],
    [`structural_noise`], [$T_"eff"$ scale], [0.0], [set to zero for deterministic reference runs; nonzero in robustness sweeps],
    [`eligibility_decay`], [decay of $E_b$], [0.12], [tag lifetime; matches STC tag window of ~minutes–hours phenomenologically @RedondoMorris2011STC @Gerstner2018EligibilityTraces @Rogerson2014SynapticTaggingAllocation],
    [`translation_decay`], [decay of $P_b$], [0.05], [capture resource turnover per step @Govindarajan2011DendriticBranch @Das2023LocalTranslationMemory @Rangaraju2019MitoCompartments],
    [`replay_gain`], [weight of replay on $P_b$], [0.80], [scales how strongly replay recruits $P_b$ @Wang2024SleepDependentEngramReactivation @PeyracheSeibt2020Spindles],
    [`sleep_gain`], [weight of sleep drive on $P_b$], [0.0], [set to zero in main runs (replay-only); nonzero in sleep-window extension @Seibt2017DendriticSpindles @TononiCirelli2020SleepPlasticity],
    [`fast_gain`], [gain on fast access $A_b^f$], [2.0], [amplifies cue drive into fast sigmoid],
    [`context_gain`], [weight of context bias term], [1.0], [multiplicative context gating in fast access],
    [`readout_gain`], [sigmoid slope at recall readout], [5.0], [sharpness of expressed recall threshold],
    [`readout_threshold`], [$theta_mu$ at readout], [0.3], [threshold for expressed retrieval in the canonical E017/E018 reference stack],
  ),
) <tab-s2-params>

== Encoding and Consolidation Protocol

All main experiments follow a three-phase protocol:

1. *Encoding:* Two cue passes for $mu_1$, then two cue passes for $mu_2$ (four `apply_cue` calls total). Each cue pass updates fast access, eligibility traces, and spine calcium proxies.

2. *Pre-consolidation probe:* Recall support is computed immediately after encoding to provide a baseline.

3. *Consolidation:* Nine passes of `run_consolidation` with a `ConsolidationWindow` specifying `replay_trace_ids = [mu_1, mu_2]` and `modulatory_drive = 1.0`. Each pass executes two sub-steps: first updating $P_b$ from replay overlap and sleep drive; then updating $M_b$ via the bounded tag-dependent rule; then decaying $E_b$.

The linking metric is computed after consolidation as:

$ L_(mu_1 mu_2) = sum_b a_(mu_1 b) a_(mu_2 b) M_b $

*Focal damage protocol (SIG-D and SIG-E):* An elevated per-branch decay rate (`decay_rate = 0.030` versus the canonical 0.005) is applied to the overlap branch `b1`, followed by nine null consolidation passes (`modulatory_drive = 0.0`, no replay), to simulate selective structural destabilization.

== Baseline comparator definitions (E018)

Four simpler models were run against the full model on the five signature families (SIG-A through SIG-E). Each baseline corresponds to a specific mechanistic ablation or substitution.

#figure(
  kind: table,
  supplement: [Table],
  caption: [Comparator baseline definitions. Random seed fixed to 42 for the random-drift baseline. The joint criterion is that a baseline must pass all five signature thresholds simultaneously; no simpler baseline meets this criterion.],
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
- `E022` and `E022R` add hard comparators and the shuffled-replay audit, showing that partial behavioral mimicry does not preserve scalable structural specificity.

The older `exp013_paper_summary.py` and `exp_seed_validation.py` scripts remain useful as optional historical cross-checks for reviewers who want lineage or parameter-history context, but they are not the main reproducibility route for the current article.

#bibliography("references_slow_branch_level_accessibility.bib")
