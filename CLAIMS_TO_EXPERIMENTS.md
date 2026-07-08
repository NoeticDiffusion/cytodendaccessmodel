# Claims to Experiments

This is the reviewer-facing map for the **current** manuscript:
*Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking*.

It is organized around three explicit tiers:

1. **Primary evidence**: the simulator-first article stack (`E017` to `E022R`, plus `E023` figure assembly).
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

This repository supports a narrow simulator-first claim:

> replay-dependent slow branch-level writing better reproduces the tested joint
> signature profile for memory linking than the tested comparators do.

This repository does **not** directly measure `M_b`, establish a unique cytoskeletal
memory code, or turn the DANDI analyses into main evidence for the current paper.

## Primary Evidence Tier

These are the main experiments for the current article and should be reviewed first.

| Current article claim or section | Primary experiments | Main results directory or artifact | Article figure or table | Notebook section |
| --- | --- | --- | --- | --- |
| Conceptual framing and theory-to-executable bridge | `E023`, concept pack | `article/.../figures2/`, `article/.../figures3_concepts/` | `Fig 1`, evidence and comparator concept figures | `Repository map and evidence tiers`; `Current claim ladder for the slow-branch article` |
| `C01` overlap-branch structural strengthening (`SIG-A`) | `E017` | `results/e017_traceable_simulator_core/` | `Fig 2`, `Table 2` | `Current claim ladder for the slow-branch article` |
| `C02` linking gain after consolidation (`SIG-B`) | `E017` | `results/e017_traceable_simulator_core/` | `Fig 2`, `Table 2` | `Current claim ladder for the slow-branch article` |
| `C03` context separation is present but not discriminating (`SIG-C`) | `E018` with reference to canonical behavior | `results/e018_comparator_trace_matrix/` | `Fig 3`, `Table 2`, `Table 3` | `Current claim ladder for the slow-branch article` |
| `C04-C05` damage sensitivity and targeted rescue (`SIG-D`, `SIG-E`) | `E017` | `results/e017_traceable_simulator_core/` | `Fig 2`, overlap-rescue concept figure, `Table 2` | `Current claim ladder for the slow-branch article` |
| `C06-C09` comparator discrimination across baseline and hard alternatives | `E018`, `E022` | `results/e018_comparator_trace_matrix/`, `results/e022_hard_comparators/` | `Fig 3`, `Table 3` | `Current claim ladder for the slow-branch article` |
| `C10` replay identity requirement and shuffled-replay scaling failure | `E022R` | `results/e022r_shuffled_replay_scaling_audit/` | `Fig 6` | `Current claim ladder for the slow-branch article` |
| `C11` bounded robustness regime | `E019`, `E020` | `results/e019_one_at_a_time_parameter_robustness/`, `results/e020_two_parameter_robustness_heatmaps/` | `Fig 4`, `Table 4` | `Current main evidence path` |
| `C12-C14` scaling, motifs, weak-overlap failure, hub overlinking boundary | `E021`, `E021R` | `results/e021_scaling_and_motif_generalization/`, `results/e021r_generalized_specificity_gate/` | `Fig 5`, `Table 5` | `Current main evidence path` |
| `C15-C16` explicit non-claims and article limits | article discussion plus ledger | `v2_claim_ledger.md` | Discussion; S3 framing | `Scope boundary`; `Supplementary and DANDI boundary` |

### Primary Evidence Ladder in Order

1. `exp017_traceable_simulator_core.py`
2. `exp018_comparator_trace_matrix.py`
3. `exp019_one_at_a_time_parameter_robustness.py`
4. `exp020_two_parameter_robustness_heatmaps.py`
5. `exp021_scaling_and_motif_generalization.py`
6. `exp021r_generalized_specificity_gate.py`
7. `exp022_hard_comparators.py`
8. `exp022r_shuffled_replay_scaling_audit.py`
9. `exp023_article_figure_assembly.py`

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
| `Fig 2` | `E017` |
| `Fig 3` | `E018` and `E022` |
| `Fig 4` | `E019` and `E020` |
| `Fig 5` | `E021` and `E021R` |
| `Fig 6` | `E022R` |
| `Table 2` | `E017` |
| `Table 3` | `E018` and `E022` |
| `Table 4` | `E019` and `E020` |
| `Table 5` | `E021` and `E021R` |

## Recommended Reviewer Flow

1. Open `reviewer_slow_branch_level_accessibility.ipynb`.
2. Read the current manuscript and `v2_claim_ledger.md`.
3. Inspect `results/e017_*` to `results/e022r_*` in order.
4. Use `CLAIMS_TO_EXPERIMENTS_legacy.md` only if you want the earlier lineage or the older manuscript posture.
5. Treat DANDI and S3 as supplementary unless you are explicitly reviewing the open-data bridge.

## Data and Runtime Note

The **primary simulator-first evidence tier** requires no DANDI download.

The full DANDI-backed reproduction for `000718`, `000336`, and `001710` is much larger
and should be treated as optional reviewer work for the supplementary bridge layer.
