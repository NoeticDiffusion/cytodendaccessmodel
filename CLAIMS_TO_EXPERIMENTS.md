# Claims to Experiments

This file is a reviewer-facing map from the manuscript claims to the scripts,
data requirements, outputs, and figures that reproduce or constrain them. Use
`RUN.md` for exact setup and command order; use this file to understand why each
experiment exists.

## Claim Boundaries

The repository can reproduce the executable simulator results and the reported
open-data analyses. It cannot, by itself, directly observe the proposed slow
cytoskeletal accessibility field. The biological hypotheses remain falsifiable
targets for future branch-resolved perturbation experiments, while the code here
reproduces the downstream signatures reported in the article.

## Data Download Size

The no-data simulator claims require no DANDI download. The full open-data
reproduction for DANDI `000718`, `000336`, and `001710` is much larger: reviewers
should plan for approximately **180-200 GB** of free disk space before starting
the Level 2 downloads in `RUN.md`. If disk space is limited, audit one dataset
claim at a time rather than downloading all open-data assets at once.

## Manuscript Claim Map

| Manuscript area | Claim or signature | Primary commands | Required data | Expected reviewer-facing output | Figure/table link | Reviewer verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Testable Predictions, H1 | Cytoskeletal or structural perturbation should alter branch-level access, not only global excitability. | External wet-lab test proposed; executable stress tests: `python experiments/exp005_pathology.py`, `python experiments/exp009_rescue_linking.py`, `python experiments/exp014_structural_gate_ablation.py` | None for simulator scripts | Terminal verdicts for vulnerability, rescue, and structural-gate ablation | Main text H1; Table 1 vulnerability/rescue rows | Reproduces an executable proxy, not direct biological validation |
| Testable Predictions, H2 | Temporally adjacent memories should link more strongly when they share branch allocation and compatible support states. | `python experiments/exp001_minimal_branch_linking.py`, `python experiments/exp003_timing_replay_linking.py`, `python experiments/exp010_multitrace_overlap.py`, `python experiments/exp011_branch_topology.py` | None | Terminal summaries; `data/reviewer/013_canonical_values.json` after `exp013` | Table 1; S2 Appendix | Reproducible simulator claim |
| Testable Predictions, H3 | Contextual retrieval depends on accessibility of the relevant branch subset, not cue quality alone. | `python experiments/exp002_context_sensitive_recall.py`, `python experiments/exp012_retrieval_readout.py`, `python experiments/exp015_comparator_baselines.py` | None | Terminal context/retrieval verdicts and comparator panel | Table 1; S2 Appendix | Reproducible simulator claim with bounded interpretation because fast contextual gating contributes |
| Executable Instantiation | Replay-dependent consolidation writes overlap-branch structural accessibility and increases linking. | `python experiments/exp001_minimal_branch_linking.py`, `python experiments/exp006_asymmetric_consolidation.py`, `python experiments/exp013_paper_summary.py` | None | Terminal metrics; `data/reviewer/013_canonical_values.json` | Table 1; S2 Appendix | Reproducible no-data result |
| Executable Instantiation | Robustness sweeps preserve protected directional claims. | `python experiments/exp004_robustness.py`, `python experiments/exp_seed_validation.py` | None | Terminal pass-rate summaries and canonical RNG-state check | Table 1 robustness row; S2 Appendix | Reproducible no-data result |
| Executable Instantiation | Simpler baselines do not reproduce the joint signature profile. | `python experiments/exp014_structural_gate_ablation.py`, `python experiments/exp015_comparator_baselines.py`, `python experiments/exp016_task_family.py` | None | Terminal ablation/comparator verdicts | Table 1; S2 Appendix | Reproducible no-data result |
| Open-Data Evaluation, DANDI `000718` | NeutralExposure-defined core units show modest excess enrichment during high-synchrony offline events above a strong burst baseline. | `python experiments/dandi_000718_01_inventory.py` through `python experiments/dandi_000718_14_h1_pri_enrichment.py` as listed in `RUN.md` | DANDI `000718` selected subjects | `data/dandi/triage/000718/h1_pri_enrichment.json`, `.md`, `.log`; supporting `h1_robustness.*`, `h1_specificity.*`, `coreactivation_baseline.*` | Figures 6 and 7; S3 Appendix Table 7 | Reproducible open-data bridge; not sequence-level replay proof |
| Open-Data Evaluation, DANDI `000336` | Cross-plane coupling is structured and above null across analyzed bundle pairs; strict bilateral access-constraint match is cleanest in the supplementary cross-area pair. | `python experiments/dandi_000336_01_inventory.py` through `python experiments/dandi_000336_06_full_bundle.py` | DANDI `000336` selected subjects | `data/dandi/triage/000336/full_bundle_coupling.json`, `.md`, `.log`; supporting cross-plane outputs | Table 2; Figures 8 and 9; S3 Appendix Table 8 | Reproducible open-data bridge; speaks to structured access constraints, not direct slow-field measurement |
| Open-Data Evaluation, DANDI `001710` | SparseKO has lower subject-level cross-day stability than Cre under the implemented subject-level null, with weaker separation from Ctrl and channel sensitivity. | `python experiments/dandi_001710_01_inventory.py` through `python experiments/dandi_001710_08_robustness_and_nulls.py` | DANDI `001710` full selected bundle | `data/dandi/triage/001710/replication_bundle/`; `data/dandi/triage/001710/robustness/group_null_tests.json`, `.md`, `claim_boundary.md`, `day_lag_similarity.*` | Table 3; S3 Appendix Tables 9-12 | Reproducible open-data bridge; bounded by channel sensitivity and indirect arm-label audit |
| Simulator x DANDI bridge | Simulator parameter sensitivity and bootstrap summaries connect model-scale predictions to DANDI-facing signatures. | `python experiments/dandi_simulator_07_bootstrap_ci.py`, `python experiments/dandi_simulator_09_sensitivity.py` | None for simulator bridge | `data/dandi/triage/model/simulator_sensitivity.json`, `.md`, `.log` and bootstrap outputs | Supporting robustness context | Reviewer audit / sensitivity layer |

## Legacy or Supplementary Pipelines

`configs/dandi/dataset_000871.yaml` and `experiments/dandi_000871_*.py` are
kept as legacy or supplementary cross-plane analyses. They are not required for
the current manuscript claims, which use DANDI `000336`, `000718`, and `001710`.
Run them only if you want to inspect the older cross-plane workflow or compare
the `000336` migration against prior outputs.

## Recommended Reviewer Flow

1. Read the abstract, Table 1, Open-Data Evaluation, S2 Appendix, and S3 Appendix.
2. Follow `RUN.md` Level 0 and Level 1 to reproduce all no-data simulator claims.
3. Download only the DANDI datasets needed for the open-data claim you want to audit.
4. Run the corresponding Level 2 commands in `RUN.md`.
5. Regenerate open-data figures with `python -m dandi_analysis.visualisation.cli`.
6. Open `notebooks/reviewer_reproduction_walkthrough.py` after the scripts finish
   to inspect a compact, derived-output summary.

## Latest Local Dry-Run Status

Local dry-run on 2026-04-28:

- Install step: passed with `python -m pip install -e ".[dev,viz]"`.
- Test suite: passed, `127 passed`, `2 warnings`.
- Level 1 no-data simulator suite: passed after install, including `exp001`
  through `exp016`, `exp_seed_validation.py`, and `gen_figures_executable.py`.
- Reviewer walkthrough: passed and found
  `data/reviewer/013_canonical_values.json`.
- DANDI data-backed pass: skipped because no local NWB files were available
  under `data/dandi/raw/` during this dry-run.
