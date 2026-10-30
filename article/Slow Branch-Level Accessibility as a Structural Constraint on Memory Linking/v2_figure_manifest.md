# Figure Manifest — Slow Branch-Level Accessibility

Article: *Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking*

## Figure 1 — Branch-resolved model concept

**File:** `figures2/Fig_e023_01_model_concept.png`  
**Source:** E023 concept assembly; original full-resolution source is `figures2/fig1_slow-branch-level-accessability-as-a-structural-constraint-on-memory-linking.png`.  
**Key message:** A phenomenological branch-level accessibility state can bias later branch recruitment in the simulator.  
**Claims supported:** Model definition and motivation only; no unique mechanism or biological validation.  
**Caption:** *Model overview. Fast branch access and a slow, replay- and eligibility-dependent state are separated in a four-branch linking motif. The slow state is a simulator variable, not an identified molecule.*

## Figure 2 — Tuned common-probe comparison (E025)

**File:** `figures2/Fig_e025_01_tuned_common_probe.png`  
**Source:** E025 holdout draw scores and paired comparisons; rendered by `experiments/exp025_render_common_probe_figure.py` with the registered shared bootstrap indices.  
**Key message:** After equal-budget tuning, the full model exceeds five comparator families, is practically equivalent to shuffled replay, and is exceeded by Hebbian weight-only and resource-only models. Eligibility-only has no qualifying candidate.  
**Claims supported:** E25-1 through E25-5 in `v2_claim_ledger.md`.  
**Caption:** *E025 tuned common-probe comparison. Panel A shows immediate D means and 95% topology-draw bootstrap intervals for each qualifying tuned family. Panel B shows paired full-minus-comparator contrasts, with the ±0.02 practical-equivalence region applied to those contrasts. Holm-corrected decisions show full-model superiority to five families, equivalence to shuffled replay, and comparator superiority for Hebbian weight-only and resource-only. Eligibility-only is unavailable because no candidate met the preregistered criteria. Readouts are simulator scores, not empirical probabilities.*

## Figure 3 — Canonical simulator traces (E017)

**File:** `figures2/Fig_e023_02_canonical_traces.png`  
**Source:** E017 trace export, assembled by E023.  
**Key message:** Branch-level variables and linking diagnostics can be inspected across encoding, consolidation, perturbation, and recovery.  
**Claims supported:** C01, C02, C04, C05 as simulator-internal outcomes.  
**Caption:** *Canonical simulator traces. The full implementation produces overlap-branch slow-state change, an increase in the structural linking diagnostic, damage sensitivity, and targeted recovery under the specified canonical protocol.*

## Figure 4 — Historical comparator matrix (E018, E022)

**File:** `figures2/Fig_e023_03_comparator_matrix.png`  
**Source:** E018 and E022 matrices, assembled by E023.  
**Key message:** Historical mechanism signatures and B1/B3/B5 proxy labels use mixed, sometimes model-specific definitions.  
**Claims supported:** Historical behavior only; not a common model-selection claim.  
**Caption:** *Historical E018/E022 comparator summary. Colors reflect the original thresholds and protocol-specific scores. They do not rank model families on the tuned E025 readout; `N/A` denotes an undefined measure.*

## Figure 5 — Robustness landscape (E019, E020)

**File:** `figures2/Fig_e023_04_robustness_landscape.png`  
**Source:** E019 one-at-a-time sweeps and E020 parameter-pair heatmaps.  
**Key message:** The full model's internal signature profile occupies a bounded parameter regime.  
**Claims supported:** C12, bounded to tested ranges and model-internal signatures.  
**Caption:** *Internal signature robustness. Parameter sweeps identify write, replay, timing, and overlap boundaries; they do not establish cross-model superiority on E025's common readout.*

## Figure 6 — Motif and scaling boundaries (E021, E021R)

**File:** `figures2/Fig_e023_05_motif_scaling.png`  
**Source:** E021 and E021R motif and specificity analyses.  
**Key message:** Internal signatures generalize across tested branch counts and motifs with weak-overlap and hub-overlap limits.  
**Claims supported:** C13, C14, within tested topologies.  
**Caption:** *Motif scaling and specificity. The implemented structural diagnostics vary with branch count and topology, including weak-overlap failure and hub-overlap over-linking.*

## Figure 7 — Shuffled-replay audit (E022R)

**File:** `figures2/Fig_e023_06_shuffled_replay_audit.png`  
**Source:** E022R shuffled-replay scaling audit, 4–32 branches and 20 seeds per condition.  
**Key message:** The small-motif structural match under shuffled replay declines with branch count.  
**Claims supported:** C15, limited to the implemented replay-identity disruption and structural measure.  
**Caption:** *Shuffled-replay scaling audit. Structural specificity in this implementation declines as allocation space expands; this is not evidence that all alternative models fail the E025 common readout.*

---

*Updated 2026-09-26 after E025. E017–E023 source figures remain unchanged.*
