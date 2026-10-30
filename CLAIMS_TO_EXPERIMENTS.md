# Claims to Experiments

This is the reviewer-facing map for the **current** manuscript:
*Testing Slow Branch-Level Accessibility in a Simulator of Memory Linking*.

It is organized around three explicit tiers:

1. **Primary model comparison**: E025 equal-budget tuned common-probe outcomes; `E017` to `E022R` provide historical simulator mechanism, robustness, and motif results; `E023` assembles the earlier figures; E024 is protocol-development history.
2. **Foundational legacy**: earlier experiment families (`exp001` to `exp016`, `exp013`, `exp_seed_validation.py`) retained for lineage, onboarding, and historical context.
3. **Supplementary framing**: DANDI and open-data bridge material, older conceptual assets, and the companion conceptual article.

The older root map has been preserved in `CLAIMS_TO_EXPERIMENTS_legacy.md`.

## Reviewer Start Here

Use these files together:

- `reviewer_slow_branch_level_accessibility.ipynb`: primary reviewer notebook for the current article.
- `README.md`: top-level orientation and reviewer path.
- `article/Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking/Slow Branch-Level Accessibility as a Structural Constraint.typ`: current manuscript source.
- `article/Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking/v2_claim_ledger.md`: article-level claim ledger.
- `article/Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking/v2_figure_manifest.md`: figure-by-figure article map.

## Claim Boundary

The current manuscript supports a narrow simulator-first claim:

> A branch-resolved simulator makes a slow, replay- and eligibility-dependent
> accessibility hypothesis executable, but E025 does not show a unique output-level
> advantage for the full model after equal-budget tuning and holdout evaluation.

The tuned full model mean was `D=0.022703` (95% topology-draw bootstrap CI
`0.022538–0.022864`). It exceeded five comparator families, was practically equivalent
to shuffled replay, and was exceeded by Hebbian weight-only (`D=0.093304`) and
resource-only (`D=0.094349`) after the preregistered Holm correction. Eligibility-only
had no qualifying tuned candidate, so its inference is unavailable. These results apply
to the tested adapters, parameter ranges, task, and readout. They do **not** directly
measure `M_b`, establish a unique cytoskeletal memory code, or turn DANDI analyses into
main evidence.

## Primary Evidence Tier

These are the main experiments for the current article and should be reviewed first.

| Current article claim or section | Primary experiments | Main results directory or artifact | Article figure or table | Notebook section |
| --- | --- | --- | --- | --- |
| `E25-1` through `E25-6` tuned common-probe outcome and comparator decisions | `E025` | `results/e025_common_probe_tuning/`; plan and manifests in `project/neural_computation/` | Fig. 2; S2 Table | `E025 primary result and limitations` |
| Historical E024 common-probe outcome | `E024` | `results/e024_common_probe_comparison/`; original protocol in `project/neural_computation/model_comparison_protocol.md` | S2 Appendix | `Historical E024 protocol-development result` |
| Conceptual framing and theory-to-executable bridge | `E023`, concept pack | `article/.../figures2/`, `article/.../figures3_concepts/` | `Fig 1`, evidence and comparator concept figures | `Repository map and evidence tiers`; `Current claim ladder for the slow-branch article` |
| `C01-C05` canonical internal mechanism signatures | `E017`, `E018` | `results/e017_traceable_simulator_core/`, `results/e018_comparator_trace_matrix/` | `Fig 3`, `Table 2`; historical context in Fig. 4 | `Canonical mechanism signatures` |
| Historical mechanism-specific comparator signatures | `E018`, `E022` | `results/e018_comparator_trace_matrix/`, `results/e022_hard_comparators/` | `Fig 4`, `Table 3` | `Historical comparator diagnostics` |
| `C15` shuffled-replay specificity under the implemented structural metric | `E022R` | `results/e022r_shuffled_replay_scaling_audit/` | `Fig 7` | `Current claim ladder for the slow-branch article` |
| `C12` bounded internal robustness regime | `E019`, `E020` | `results/e019_one_at_a_time_parameter_robustness/`, `results/e020_two_parameter_robustness_heatmaps/` | `Fig 5`, `Table 4` | `Where the model generalizes and where it fails` |
| `C13-C14` internal scaling, motifs, weak-overlap failure, hub overlinking | `E021`, `E021R` | `results/e021_scaling_and_motif_generalization/`, `results/e021r_generalized_specificity_gate/` | `Fig 6`, `Table 5` | `Where the model generalizes and where it fails` |
| Explicit non-claims and article limits | E025 plus discussion and ledger | `v2_claim_ledger.md`; `results/e025_common_probe_tuning/summary/claim_status.json` | Fig. 2; Discussion; S3 framing | `Scope boundary`; `E025 primary result and limitations` |

### Primary Evidence Ladder in Order

1. `experiments/exp025_tuned_common_probe.py` (primary tuned holdout comparison)
2. `experiments/exp025_render_common_probe_figure.py` (figure from saved holdout outputs)
3. `experiments/exp017_traceable_simulator_core.py`
4. `experiments/exp018_comparator_trace_matrix.py`
5. `experiments/exp019_one_at_a_time_parameter_robustness.py`
6. `experiments/exp020_two_parameter_robustness_heatmaps.py`
7. `experiments/exp021_scaling_and_motif_generalization.py`
8. `experiments/exp021r_generalized_specificity_gate.py`
9. `experiments/exp022_hard_comparators.py`
10. `experiments/exp022r_shuffled_replay_scaling_audit.py`
11. `experiments/exp023_article_figure_assembly.py` (historical E017–E022R figures)
12. `experiments/exp024_common_probe_comparison.py` (historical protocol-development run)

## Foundational Legacy Tier

These files are still important, but they are no longer the main evidence ladder for the current article.

| Legacy family | Role now | Representative files | Why retained |
| --- | --- | --- | --- |
| `exp001-exp003` | Early minimal demos | `exp001_minimal_branch_linking.py`, `exp002_context_sensitive_recall.py`, `exp003_timing_replay_linking.py` | Best retained as optional lineage demos in the reviewer notebook, not as current article evidence |
| `exp004`, seed validation | Early robustness and reproducibility checks | `exp004_robustness.py`, `exp_seed_validation.py` | `exp004` is superseded by `E019-E020`; `exp_seed_validation.py` remains a useful optional reproducibility cross-check |
| `exp005`, `exp009`, `exp014` | Perturbation, rescue, ablation lineage | `exp005_pathology.py`, `exp009_rescue_linking.py`, `exp014_structural_gate_ablation.py` | Important precursors to the current damage and rescue framing |
| `exp010`, `exp016` | Earlier motif and topology explorations | `exp010_multitrace_overlap.py`, `exp016_task_family.py` | Hand-built precursors to `E021-E021R`, retained as lineage rather than primary evidence |
| `exp011`, `exp012` | Exploratory extensions beyond the current paper scope | `exp011_branch_topology.py`, `exp012_retrieval_readout.py` | Potentially interesting spillover and readout checks, but outside the current manuscript claim surface |
| `exp013` | Legacy canonical summary | `exp013_paper_summary.py` and `data/reviewer/013_canonical_values.json` | Useful optional reviewer cross-check, but numerically distinct from the E017-E022R stack |
| Older reviewer walkthrough | Legacy reviewer path | `reviewer_branch_resolved_walkthrough.ipynb` | Keep available as a companion, not the default entry point |

For this layer, use `CLAIMS_TO_EXPERIMENTS_legacy.md` if you want the older article-oriented mapping in full.

## Supplementary Framing Tier

These materials are retained explicitly rather than implicitly, but they should not be mistaken for the main evidence ladder.

| Supplementary family | Current status | Main files |
| --- | --- | --- |
| Older conceptual assets | Framing and intuition only | `article/.../figures/` and `A Cytoskeletal-Dendritic Accessibility Model.typ` |
| New concept pack | Reviewer-facing framing for current paper | `article/.../figures3_concepts/` |
| DANDI `000718`, `000336`, `001710` | Exploratory observable bridges | `data/dandi/triage/`, S3 appendix materials |
| DANDI `000871` and related scripts | Legacy supplementary workflow | `configs/dandi/dataset_000871.yaml`, `experiments/dandi_000871_*.py` |

## Article Figure and Experiment Routing

| Article asset | Main source |
| --- | --- |
| `Fig 1` | `E023` assembly plus current concept figure assets |
| `Fig 2` | `E025` |
| `Fig 3` | `E017` |
| `Fig 4` | `E018` and `E022` (historical proxies) |
| `Fig 5` | `E019` and `E020` |
| `Fig 6` | `E021` and `E021R` |
| `Fig 7` | `E022R` |
| `Table 2` | `E017` |
| `Table 3` | `E018` and `E022` |
| `Table 4` | `E019` and `E020` |
| `Table 5` | `E021` and `E021R` |

## Recommended Reviewer Flow

1. Open `reviewer_slow_branch_level_accessibility.ipynb`.
2. Read the current manuscript and `v2_claim_ledger.md`.
3. Inspect E025's locked holdout outputs first, then E017–E022R for historical mechanism and boundary results.
4. Use E024 only for protocol-development history; its estimates are not pooled with E025.
5. Use `CLAIMS_TO_EXPERIMENTS_legacy.md` only if you want the earlier lineage.
6. Treat DANDI and S3 as supplementary unless you are explicitly reviewing the open-data bridge.

## Data and Runtime Note

The **primary simulator-first evidence tier** requires no DANDI download.

The full DANDI-backed reproduction for `000718`, `000336`, and `001710` is much larger
and should be treated as optional reviewer work for the supplementary bridge layer.
