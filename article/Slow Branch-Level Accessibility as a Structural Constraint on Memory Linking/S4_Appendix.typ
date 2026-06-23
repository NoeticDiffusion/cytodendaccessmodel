#import "template.typ": essay-template

#show: doc => essay-template(
  short_title: [S4 Appendix],
  doc,
)

// Tables in this appendix are numbered S4.1, S4.2, …
#set figure(numbering: (..num) => "S4." + str(num.at(-1)))

= S4 Appendix. Computational materials, software resources, and reporting conventions

== Scope

This appendix consolidates the computational materials-and-methods details that are most
useful for reproducibility review and for detector-style reporting checks such as
SciScore. The main article retains the short study-design statements and availability
language needed for the simulator-first narrative. S2 Appendix retains simulator
architecture and canonical protocol framing. S3 Appendix retains the exploratory
open-data bridge pipelines. This appendix collects software identifiers, computational
study-design conventions, stochastic-replication policy, and reproducibility entry
points in one place.

== Code and dataset identifiers

#figure(
  kind: table,
  supplement: [Table],
  caption: [Primary code, data, and reviewer-routing identifiers for the simulator-first manuscript.],
  table(
    columns: (1.2fr, 0.8fr, 1.4fr, 2fr),
    inset: 6pt,
    stroke: 0.5pt + black,
    align: (left, left, left, left),
    table.header(
      [*Resource*], [*Type*], [*Identifier*], [*Role in this study*],
    ),
    [`cytodendaccessmodel` GitHub repository], [code], [`https://github.com/NoeticDiffusion/cytodendaccessmodel`], [public source repository for simulator, analysis scripts, figures, manuscript files, and reviewer routing],
    [`cytodendaccessmodel` Zenodo snapshot], [archived code], [`10.5281/zenodo.19498499`], [manuscript-matched archived release used for manuscript reproducibility],
    [`DANDI 000718`], [dataset], [`https://dandiarchive.org/dandiset/000718`], [supplementary offline enrichment and replay-linked reuse bridge in S3],
    [`DANDI 000336`], [dataset], [`https://dandiarchive.org/dandiset/000336`], [supplementary cross-plane coupling bridge in S3],
    [`DANDI 001710`], [dataset], [`https://dandiarchive.org/dandiset/001710`], [supplementary perturbation-sensitive stabilization bridge in S3],
    [`reviewer_slow_branch_level_accessibility.ipynb`], [reviewer guide], [root-level notebook], [top-level reviewer routing for article evidence tiers, legacy lineage, and supplementary bridges],
    [`CLAIMS_TO_EXPERIMENTS.md`], [claim map], [root-level markdown file], [maps article claims and figures to experiments, result directories, and reviewer routing],
  ),
) <tab-s4-identifiers>

== Software environment and identifiers

Unless otherwise noted, the local manuscript build environment used:

#figure(
  kind: table,
  supplement: [Table],
  caption: [Software environment and identifiers used for the local manuscript build. RRIDs are included where a validated software identifier was available during article preparation.],
  table(
    columns: (1.0fr, 0.9fr, 1.0fr, 1.9fr),
    inset: 6pt,
    stroke: 0.5pt + black,
    align: (left, left, left, left),
    table.header(
      [*Software*], [*Version*], [*Identifier*], [*Role*],
    ),
    [Python], [`3.12.0`], [`RRID:SCR_008394`], [main implementation and analysis language; project requirement is `>=3.10`],
    [NumPy], [`1.26.4`], [`RRID:SCR_008633`], [array and numerical operations in simulator and analysis pipelines],
    [PyYAML], [`6.0.2`], [no RRID listed here], [dataset-selection and configuration parsing],
    [PyTorch], [`2.5.1+cu121`], [no RRID listed here], [declared project dependency for computational workflows],
    [DANDI Python package / CLI], [`0.74.3`], [`RRID:SCR_017571`], [archive listing, download, and metadata workflow for supplementary open-data analyses],
    [PyNWB], [`3.1.3`], [`RRID:SCR_017452`], [reading NWB files for the supplementary DANDI pipelines],
    [matplotlib], [`3.9.2`], [no RRID listed here], [article and analysis figure generation],
    [pytest], [`9.0.3`], [no RRID listed here], [test-suite execution],
    [Typst], [`0.14.1`], [no RRID listed here], [manuscript and appendix compilation],
  ),
) <tab-s4-software>

== Computational study design

The primary study is a simulator-first model-discrimination analysis rather than a
subject-sampling experiment. Its main evidence tier is the `E017--E022R` stack:

- `E017`: canonical traceable simulator run and locked SIG-A through SIG-E reference
- `E018`: baseline comparator matrix
- `E019`: one-at-a-time robustness
- `E020`: paired-parameter robustness heatmaps
- `E021` and `E021R`: branch-count scaling, motif generalization, and specificity
  boundaries
- `E022` and `E022R`: hard comparators and shuffled-replay audit

Legacy experiments `exp001--exp016`, `exp013_paper_summary.py`, and
`exp_seed_validation.py` are retained for lineage and reviewer cross-checks but are not
the main evidence ladder for the current manuscript.

== Canonical simulator protocol summary

The canonical simulator used by `E017` and `E018` contains four branches and two traces,
with one shared overlap branch and private branches for each trace. The canonical
parameterization matches the locked reference stack:

- `structural_lr = 0.18`
- `replay_gain = 0.80`
- `eligibility_decay = 0.12`
- `structural_decay = 0.005`
- `structural_gain = 6.0`
- `readout_threshold = 0.3`

The canonical protocol consists of:

1. two cue passes for the first trace,
2. two cue passes for the second trace,
3. pre-consolidation probes,
4. nine replay-dependent consolidation passes,
5. overlap-branch damage plus nine null passes,
6. targeted rescue with three rescue rounds and post-rescue probes.

Branch-level trace export includes fast activity, fast access, slow access, effective
access, eligibility, consolidation support, structural accessibility, replay/input drive,
and linking contributions.

== Comparator implementation routing

Two comparator families were used in the primary evidence stack.

#table(
  columns: (1.2fr, 1.0fr, 1.8fr),
  inset: 6pt,
  stroke: 0.5pt + black,
  align: (left, left, left),
  table.header(
    [*Comparator*], [*Family*], [*Implementation note*],
  ),
  [`fast_context_only`], [baseline], [fast gating retained; `M_b` frozen],
  [`replay_no_structure`], [baseline], [replay updates transient variables but cannot write `M_b`],
  [`random_slow_drift`], [baseline], [matched-magnitude slow drift replaces replay-specific writing],
  [`fixed_allocation_only`], [baseline], [static overlap geometry without dynamic update],
  [`hebbian_weight_only`], [hard], [weight-only alternative without overlap-specific slow writing],
  [`soma_global_gain_only`], [hard], [uniform branch-independent gain producing non-specific recruitment],
  [`shuffled_replay`], [hard], [replay preserved but branch identity shuffled],
  [`eligibility_only`], [hard], [transient eligibility without retained structure],
  [`resource_only`], [hard], [replay-linked resources without retained `M_b`],
)

#par(first-line-indent: 0pt)[
  #emph[Table S4-3. Comparator routing for the simulator-first manuscript. Baseline comparators are formalized in E018; hard comparators are added in E022.]
]

== Robustness, scaling, and specificity design

The robustness and extension experiments are organized as follows:

- `E019` varies nine parameter families one at a time. Only the structural-noise sweep
  is stochastic; it uses `10` predeclared seeds (`0`, `1`, `2`, `3`, `4`, `42`, `101`,
  `202`, `303`, `404`). Non-noisy sweeps use the deterministic default seed.
- `E020` evaluates six paired-parameter heatmaps. The `structural_noise × replay_gain`
  grid uses the same `10` predeclared seeds per noisy cell; the remaining grids are
  deterministic.
- `E021` evaluates canonical, weak-overlap, strong-overlap, chain-overlap, hub-overlap,
  and sparse-random motifs across branch counts from `4` to `32`. Sparse-random motifs
  use seeds `42` and `123`.
- `E021R` converts those motif outcomes into reviewer-facing specificity classes such as
  expected weak-overlap failure and hub overlinking boundary.
- `E022R` evaluates shuffled replay with `20` shuffled seeds per
  `(motif_type, n_branches)` condition, with the full-model reference held at a fixed
  base seed.

== Replication, randomness, and exclusions

The canonical reference experiments are deterministic once the protocol is fixed. The
stochastic experiments are technical robustness replicates of the implemented simulator,
not biological replicates. Seed lists are predeclared in code rather than tuned after
inspection of the outcomes.

No outcome-driven exclusions are used in the primary simulator evidence ladder. Invalid
or missing outputs are intended to surface as failures that require inspection rather
than being silently imputed into the summary tables. Open-data QC rules, partial-day
coverage handling, and dataset-specific exclusions are reported separately in S3
Appendix because they belong to the exploratory bridge tier rather than to the primary
simulator evidence ladder.

== Reporting conventions and not-applicable items

The simulator-first article does not report new human-participant recruitment, live
animal experimentation by the author, cell-line work, antibody use, surgery, or specimen
collection. No investigator blinding was performed for the simulator experiments because
conditions are defined and labeled in automated scripts. Subject randomization is not
applicable to the primary simulator experiments. A formal power analysis was not used to
choose the deterministic canonical reference runs; stochastic seed counts were chosen as
robustness and specificity audits.

Experimental subjects were not randomized into groups because the primary evidence ladder
is a computational simulator study and does not assign biological subjects to
intervention groups. We did not check for sample sizes using a power analysis because the
primary simulator evidence ladder is based on deterministic reference runs plus
predeclared robustness audits rather than on biological subject sampling. No simulator
runs in the primary evidence ladder were excluded on the basis of their outcomes.

For the supplementary DANDI analyses, the relevant public archive identifiers, null
constructions, and QC rules are documented in S3 Appendix and in the original archive
records.

== Reproducibility entry points

Reviewers who want the shortest route through the current manuscript should begin with:

- `reviewer_slow_branch_level_accessibility.ipynb`
- `CLAIMS_TO_EXPERIMENTS.md`
- `results/e017_traceable_simulator_core/`
- `results/e018_comparator_trace_matrix/`
- `results/e019_one_at_a_time_parameter_robustness/`
- `results/e020_two_parameter_robustness_heatmaps/`
- `results/e021_scaling_and_motif_generalization/`
- `results/e021r_generalized_specificity_gate/`
- `results/e022_hard_comparators/`
- `results/e022r_shuffled_replay_scaling_audit/`

This routing keeps the current article's main simulator evidence, older lineage
experiments, and supplementary open-data bridges explicitly separated.

#bibliography("references_slow_branch_level_accessibility.bib")
