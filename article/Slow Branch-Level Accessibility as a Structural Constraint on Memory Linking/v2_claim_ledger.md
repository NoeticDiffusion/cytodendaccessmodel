# Claim Ledger — Slow Branch-Level Accessibility

Article: *Testing Slow Branch-Level Accessibility in a Simulator of Memory Linking*

**Status key:** `SIMULATOR-SUPPORTED` means reproduced inside the specified simulator; `BOUNDED` means supported only for tested parameter or motif ranges; `ARCHITECTURAL` means it follows from task or model structure; `HISTORICAL` means it describes E017–E022R but does not rank models on the tuned E025 outcome; `NOT SUPPORTED` means outside the evidence.

## Primary common-probe claims (E025)

The primary outcome is the post-minus-pre change in target-trace sigmoid readout for a
one-hot shared-branch cue relative to a matched source-private cue. E025 uses equal-budget
tuning and a locked holdout of 100 generated topology draws. The four primary strata are
weighted equally within each draw. Scores are simulator values, not empirical
probabilities; the inferential unit is a topology draw, not an animal or subject. The
95% percentile bootstrap intervals are pointwise and are not multiplicity-adjusted;
Holm correction applies to the registered hypothesis tests.

The registered Holm decisions use paired sign-flip tests. Their calibration assumes
sign-exchangeability of paired draw-level differences at each null boundary; model signs
were not randomized. A post-hoc centered topology-cluster bootstrap sensitivity analysis,
conditional on selected candidates and the declared topology generator, reproduced the
registered decisions. It is approximate and was not preregistered; see S2 Appendix and
project/neural_computation/e025_inference_sensitivity_20261030.md.

| ID | Claim | Status | Evidence | Interpretive limit |
|---|---|---|---|---|
| E25-1 | The tuned full model mean was `D=0.022703` (95% bootstrap CI `0.022538–0.022864`). | SIMULATOR-SUPPORTED | `results/e025_common_probe_tuning/summary/holdout_comparisons.csv`; 100 topology draws | A common-probe simulator score, not behavior or an empirical probability. |
| E25-2 | The full model exceeded five testable families, was practically equivalent to shuffled replay, and was exceeded by Hebbian weight-only and resource-only models after the preregistered Holm correction. | SIMULATOR-SUPPORTED | `holdout_comparisons.csv`, immediate `D` rows | The result applies to the registered adapters, search space, task, and readout. |
| E25-3 | Hebbian weight-only (`D=0.093304`) and resource-only (`D=0.094349`) exceeded the full model; full-minus-comparator estimates were `-0.070601` and `-0.071645`. | SIMULATOR-SUPPORTED | Paired 95% CIs: `[-0.071621, -0.069593]` and `[-0.072817, -0.070484]`; both comparator-superiority decisions survived Holm correction | Does not show that every implementation of these mechanism families would outperform the full model. |
| E25-4 | Nine families yielded a qualifying tuned candidate; eligibility-only yielded none under the predeclared baseline and functional filters. | SIMULATOR-SUPPORTED | `summary/candidate_selection.csv` | Eligibility-only has no confirmatory comparison; its nominal holdout score is descriptive. |
| E25-5 | After nine passive-decay washout steps, Hebbian and resource-only still exceeded the full model on the secondary `D_late` endpoint. | SIMULATOR-SUPPORTED | `holdout_comparisons.csv`, `D_late` rows | Secondary persistence endpoint; it does not replace immediate `D`. |
| E25-6 | Hub-overlap has a separately defined `D` and was analyzed as sensitivity, outside the four-stratum primary macro-effect. | REPRODUCIBILITY CHECK | `summary/hub_sensitivity.csv` | Not a primary confirmatory stratum. |

No unique full-model advantage is supported on the registered E025 outcome. This is not
evidence that $M_b$ has no possible function; it is a limitation on what the tested
simulator implementations demonstrate.

## Historical common-probe claims (E024)

The primary outcome is the post-minus-pre change in target-trace readout for a one-hot cue to a shared branch, relative to a matched source-private cue. The same probe schedule, sigmoid output equation, scale, motif list, and paired seed clusters were used. The state-to-support adapter was defined per model family in the frozen model registry. Readout values are simulator scores, not empirical probabilities.

| ID | Claim | Status | Evidence | Interpretive limit |
|---|---|---|---|---|
| E24-1 | Full-model macro mean was `D=0.010487` (95% seed-bootstrap CI `0.010441–0.010528`). | SIMULATOR-SUPPORTED | `results/e024_common_probe_comparison/summary/primary_model_summary.csv`; 50 paired seed clusters | Estimate is below the prespecified 0.02 comparison margin; task and adapters are simulator-defined. |
| E24-2 | Six of eight alternatives that met baseline calibration were practically equivalent to the full model under Holm-corrected TOST. | SIMULATOR-SUPPORTED | `paired_comparator_summary.csv`: fast-context-only, replay-no-structure, random slow drift, fixed allocation, global gain, shuffled replay | Does not imply equivalence outside the registered readout and implementations. |
| E24-3 | Hebbian weight-only (`D=0.054435`) and resource-only (`D=0.052594`) had larger point estimates than the full model (`D_full=0.010487`). | DESCRIPTIVE; PREDECLARED DECISION INCONCLUSIVE | Paired differences: `-0.043948` (95% CI `-0.044381` to `-0.043405`) and `-0.042107` (95% CI `-0.042511` to `-0.041578`) | The frozen rule tested full-model superiority and equivalence, not formal superiority of the alternatives. |
| E24-4 | Eligibility-only was practically equivalent on the primary readout, but failed the predeclared baseline-match criterion. | EXPLORATORY | `metadata/run_config.json`; `calibration_grid.csv`; paired summary | Do not treat it as a well-matched comparison. |
| E24-5 | The published-parameter sensitivity analysis gave the same comparator decisions. | SIMULATOR-SUPPORTED | `paired_comparator_summary.csv` | The same task and adapter limits apply. |
| E24-6 | Undefined primary cells were retained as `N/A`: 500 hub-boundary cells and 30 cells with no eligible probe pair per analysis. | REPRODUCIBILITY CHECK | `probe_manifest.csv`; `primary_by_seed.csv` | `N/A` is not a zero or model failure. |

No comparator showed predeclared evidence that the full model exceeded it by more than `0.02`. E024 therefore does not support a claim that slow writing is necessary or uniquely improves this primary output. The intervals for Hebbian and resource-only were below `-0.02` for `D_full − D_comparator`, but the registered decision rule did not test comparator superiority.

### Registered secondary outcomes

The recovery challenge is an adapter-defined simulator intervention. The seed-clustered mean targeted-minus-control change under the common branch-gain reduction was `0.00135` for the full model (95% descriptive bootstrap CI `0.00134–0.00136`), `0.27233` for Hebbian weight-only (`0.27196–0.27272`), and `0.11065` for eligibility-only (`0.11049–0.11086`). These secondary estimates are descriptive; the protocol did not define a corrected cross-model inference rule for them. See `results/e024_common_probe_comparison/summary/recovery_model_summary.csv` and `experiments/exp024_summarize_secondary.py`.

Context separation and branch-mask outcomes are also retained in `secondary_by_seed.csv`. Context separation was not unique to the full model (mean `0.715`; Hebbian `0.763`; eligibility-only `0.765`). The shared-minus-private mask effect was similar for the full model (`0.629`) and fast-context-only (`0.617`).

## Canonical mechanism signatures (E017–E018)

| ID | Claim | Status | Evidence | Interpretive limit |
|---|---|---|---|---|
| C01 | The canonical full model produces positive overlap-branch slow writing (SIG-A = `0.1281`). | SIMULATOR-SUPPORTED | E017 trace export | Internal mechanism result; not a cross-model behavioral advantage. |
| C02 | The `M_b`-based linking diagnostic increases after consolidation (SIG-B = `0.2397`). | SIMULATOR-SUPPORTED | E017 trace export | Directly contains `M_b`; it is not model-independent. |
| C03 | Context separation is present (SIG-C = `0.1545`). | ARCHITECTURAL | E017 | Also arises from fast-access geometry and allocation. |
| C04 | Linking is more damage-sensitive than single-trace recall (SIG-D = `20.98` percentage points). | BOUNDED | E017 | Depends on the tested perturbation geometry. |
| C05 | The canonical recovery protocol shows targeted recovery (gSIG-E = `1.57`, normalized). | SIMULATOR-SUPPORTED | E017; E019R/E022 convention | Model-specific intervention and readout; not evidence of biological rescue. |

## Historical comparator and specificity claims (E018–E022R)

| ID | Original claim | Current status | Evidence and limit |
|---|---|---|---|
| C06–C10 | Baseline ablations fail parts of the joint SIG-A–SIG-E profile. | HISTORICAL | E018 signatures include `M_b`-dependent quantities; failure does not show failure on the E025 common readout. |
| C11 | Hard alternatives do not reproduce the full structural-rescue profile. | HISTORICAL | E022 mixes `M_b` diagnostics and comparator-specific B1–B7 proxies; it is not a common model-selection test. |
| C12 | The internal signature profile occupies a bounded write–replay regime. | BOUNDED | E019 OAT sweeps and E020 heatmaps; selected parameters only. |
| C13 | Internal signatures generalize across tested branch counts (4–32) and non-hub motif classes. | BOUNDED | E021/E021R; this is not realistic dendritic-tree scaling or E025 behavioral superiority. |
| C14 | Weak overlap can fail and hub overlap can over-link. | SIMULATOR-SUPPORTED | E021/E021R; topology boundary conditions for this implementation. |
| C15 | Shuffled replay's apparent small-motif structural match declines with branch count. | BOUNDED | E022R, 20 seeds; applies to the tested replay-identity disruption and structural metric. |

## Explicit non-claims

| ID | Non-claim | Status | Reason |
|---|---|---|---|
| NC1 | The model establishes a unique molecular or cytoskeletal memory code. | NOT SUPPORTED | `M_b` is phenomenological; no molecule is identified. |
| NC2 | `M_b` is directly biologically validated. | NOT SUPPORTED | E025 and E017–E022R are simulator experiments; S3 is exploratory. |
| NC3 | Slow branch-level writing is necessary for memory linking or uniquely improves the E025 primary output. | NOT SUPPORTED | E025 found no unique full-model advantage; Hebbian and resource-only outperformed the full model on the registered outcome. |
| NC4 | The E025 result generalizes to all implementations of weight, eligibility, resource, replay, or gain mechanisms. | NOT SUPPORTED | One frozen adapter per family was tested. |
| NC5 | DANDI bridge analyses are primary evidence for `M_b`. | NOT SUPPORTED | They are exploratory observable bridges and do not directly measure `M_b`. |

---

*Updated after E025, 2026-09-26. Earlier E017–E022R outputs and the E024 protocol and results remain unchanged.*
