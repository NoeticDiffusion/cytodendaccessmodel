# v2 Figure Manifest — Slow Branch-Level Accessibility

Article: *Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking*

All figures below are assembled by `experiments/exp023_article_figure_assembly.py` from results in `results/e017_*` to `results/e022r_*`.

---

## Fig_e023_01_model_concept.png

**File**: `figures2/Fig_e023_01_model_concept.png`  
**Source**: E017 canonical trace export  
**Key message**: Conceptual schematic of the slow branch-level accessibility variable M_b and how it gates linking gain after replay consolidation.  
**Claims supported**: C01, C02  
**Caption stub**: *Model overview. Branch-level accessibility variable M_b accumulates during slow write cycles and gates linking gain. The four-branch canonical motif used throughout this paper is shown with overlap branches (grey) and non-overlap branches (white).*

---

## Fig_e023_02_canonical_traces.png

**File**: `figures2/Fig_e023_02_canonical_traces.png`  
**Source**: E017  
**Key message**: Canonical signature traces (SIG-A to SIG-E) across the write–replay–consolidation cycle, showing positive linking gain, context separation, damage sensitivity, and targeted rescue differential.  
**Claims supported**: C01, C02, C03, C04, C05  
**Caption stub**: *Canonical simulator signatures. Time-course traces of SIG-A (overlap writing), SIG-B (linking gain), SIG-C (context separation), SIG-D (damage sensitivity), and gSIG-E (targeted rescue differential) for the four-branch canonical motif.*

---

## Fig_e023_03_comparator_matrix.png

**File**: `figures2/Fig_e023_03_comparator_matrix.png`  
**Source**: E018, E022  
**Key message**: Comparator discrimination heatmap showing which comparators fail which signatures. No tested comparator reproduces the full joint SIG-A–SIG-E profile.  
**Claims supported**: C06, C07, C08, C09, C10, C11  
**Caption stub**: *Comparator discrimination matrix. Each cell indicates whether a comparator variant passes (green) or fails (red) a given signature criterion. The full joint profile is unique to the slow-branch model.*

---

## Fig_e023_04_robustness_landscape.png

**File**: `figures2/Fig_e023_04_robustness_landscape.png`  
**Source**: E019 (one-at-a-time sweeps), E020 (two-parameter heatmaps)  
**Key message**: The joint signature profile occupies a broad but bounded write–replay regime; the result is not sensitive to single-parameter tuning.  
**Claims supported**: C12  
**Caption stub**: *Robustness landscape. One-at-a-time parameter sweeps (left) and two-parameter heatmaps (right) showing that the joint signature profile is maintained across a broad write–replay regime, not a narrow tuned point.*

---

## Fig_e023_05_motif_scaling.png

**File**: `figures2/Fig_e023_05_motif_scaling.png`  
**Source**: E021, E021R  
**Key message**: Linking gain and specificity generalize across 4–32 branches; hub-overlap motifs produce over-linking as expected; weak-overlap motifs fail at expected boundary.  
**Claims supported**: C13, C14  
**Caption stub**: *Motif scaling and specificity gate. SIG-B (linking gain) and the specificity gate metric across branch counts (4, 8, 16, 32) and motif types (canonical, hub, weak-overlap). Hub-overlap produces over-linking; weak-overlap fails as expected.*

---

## Fig_e023_06_shuffled_replay_audit.png

**File**: `figures2/Fig_e023_06_shuffled_replay_audit.png`  
**Source**: E022R (shuffled replay scaling audit, n=4–32, 20 seeds)  
**Key message**: The small apparent overlap advantage under shuffled replay is a sampling artefact that decays with branch count, confirming that observed gain depends on replay identity structure.  
**Claims supported**: C15-audit  
**Caption stub**: *Shuffled replay audit. Linking gain under identity-disrupted replay across branch counts (4–32) and 20 seeds, showing that apparent advantage decays with branch count and does not replicate the structured-replay signal.*

---

*Last updated: E023 assembly pass.*
