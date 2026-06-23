# v2 Claim Ledger — Slow Branch-Level Accessibility

Article: *Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking*

Status legend: **SUPPORTED** = reproducible from E017–E022R; **BOUNDED** = supported within defined parameter/motif regime; **ARCHITECTURAL** = arises from model structure, not uniquely diagnostic of slow writing; **NOT SUPPORTED** = explicitly outside the paper's claim boundary.

---

## Canonical signature claims (E017–E018)

| ID | Claim | Status | Evidence | Interpretive limit |
|----|-------|--------|----------|--------------------|
| C01 | Overlap-branch slow structural writing is positive after replay consolidation (SIG-A > 0.02) | SUPPORTED | E017 trace export; SIG-A = 0.1281 | Simulator result, not biological proof |
| C02 | Linking gain increases after consolidation (SIG-B > 0.05) | SUPPORTED | E017 trace export; SIG-B = 0.2397 | Simulator result |
| C03 | Context separation is present (SIG-C > 0.05) | ARCHITECTURAL | E017; SIG-C = 0.1545 | Arises from fast-access geometry; not uniquely diagnostic of slow writing |
| C04 | Linking is more damage-sensitive than single-trace recall (SIG-D > 5 pp) | BOUNDED | E017; SIG-D = 20.98 pp | Damage sensitivity is not unique to slow writing alone |
| C05 | Targeted rescue outperforms generic rescue (gSIG-E > 0.10 normalized) | SUPPORTED | E017; gSIG-E = 1.57 (E019R/E022 protocol) | Normalized recovery difference; old E018 warm-probe value was 33.29 pp |

## Comparator discrimination claims (E018, E022)

| ID | Claim | Status | Evidence | Interpretive limit |
|----|-------|--------|----------|--------------------|
| C06 | No tested baseline comparator reproduces the full joint SIG-A–SIG-E profile | SUPPORTED | E018 comparator matrix | Alternatives outside the tested set remain possible |
| C07 | fast-context-only fails SIG-A, SIG-B, SIG-E; passes only SIG-C and SIG-D | SUPPORTED | E018 | Confirms SIG-C is architectural |
| C08 | replay-without-structure fails durable linking (SIG-B < 0) | SUPPORTED | E018 | Transient replay is insufficient |
| C09 | random-slow-drift produces non-specific or negative structural signal | SUPPORTED | E018 | Scale-matched null controls specificity |
| C10 | fixed-allocation-only preserves context geometry but fails replay-dependent gain | SUPPORTED | E018 | Static overlap is insufficient |
| C11 | No hard comparator (Hebbian, global-gain, eligibility-only, resource-only, shuffled-replay) reproduces the full structural-rescue profile | SUPPORTED | E022 | Hard comparators evaluated on both structural and output-level metrics |

## Robustness and motif generalization claims (E019, E020, E021, E021R)

| ID | Claim | Status | Evidence | Interpretive limit |
|----|-------|--------|----------|--------------------|
| C12 | The joint profile occupies a bounded write–replay regime, not a single tuned point | BOUNDED | E019 (OAT sweeps); E020 (heatmaps) | Robustness is broad but not universal; selected parameter pairs only |
| C13 | The mechanism generalizes across tested branch counts (4–32) and motif classes in non-hub topologies | BOUNDED | E021, E021R | Motif generalization, not realistic dendritic-tree scaling |
| C14 | Weak-overlap motifs fail as expected; hub-overlap motifs produce over-linking | SUPPORTED | E021, E021R | Boundary conditions, not model defects |

## Shuffled-replay specificity audit (E022R)

| ID | Claim | Status | Evidence | Interpretive limit |
|----|-------|--------|----------|--------------------|
| C15-audit | Shuffled replay produces a small apparent overlap advantage at n=4 that decays rapidly with branch count | SUPPORTED | E022R; 20 seeds | Effect is sampling artefact; concerns only the implemented replay-identity disruption |

## Explicit non-claims (claim boundary)

| ID | Non-claim | Status | Reason |
|----|-----------|--------|--------|
| C-NC1 | The model establishes a unique molecular or cytoskeletal memory code | NOT SUPPORTED | M_b is a phenomenological variable; the simulator does not require any specific molecule |
| C-NC2 | Direct biological validation of M_b is provided | NOT SUPPORTED BY DESIGN | Open-data analyses are supplementary exploratory bridges; they do not directly measure M_b |
| C-NC3 | DANDI results constitute primary evidence for the slow-branch claim | NOT SUPPORTED | DANDI analyses are in S3 Appendix as supplementary bridge material only |

---

*Last updated: E028 reference pass, article polish E026–E028.*
