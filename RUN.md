# Reproduction workflow for the revised simulator manuscript

Python `>=3.10` is required. E025 is the primary equal-budget tuned common-probe
comparison; E024 is retained as earlier protocol-development history. E025 environment,
freeze, and output hashes are recorded under `results/e025_common_probe_tuning/metadata/`.

## Reviewer route

1. Read the byte-exact archived E025 preregistration at project/neural_computation/e025_preregistration_plan_v1_0_frozen_20260925.md, the annotated current plan, the E025 parameter registry, and the topology and probe manifests.
2. Inspect `results/e025_common_probe_tuning/summary/holdout_comparisons.csv`, `candidate_selection.csv`, `scores/holdout_draw_scores.csv`, and `output_artifact_manifest.json`.
3. Read the manuscript and `article/Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking/v2_claim_ledger.md`.
4. Inspect E017–E022R as historical mechanism, robustness, and motif evidence. E023 assembled those earlier figures; E024 is the earlier common-probe protocol-development run.

## Reproduce E025

The E025 preregistration, parameter registry, topology and probe manifests, runner, and
output hash manifest are saved before any rerun. To reproduce the tuning and holdout run
from a clean checkout, install the project dependencies and run:

```powershell
python -m pip install -e ".[viz]"
python experiments/exp025_tuned_common_probe.py
```

The runner refuses to overwrite an existing output directory. It writes candidate,
draw-level, nominal, and holdout comparison files under
`results/e025_common_probe_tuning/`.

To regenerate only the publication-facing figure from saved E025 holdout results:

```powershell
python experiments/exp025_render_common_probe_figure.py
```

To reproduce the explicitly post-hoc centered cluster-bootstrap sensitivity analysis (not part of the preregistration):

```powershell
python experiments/exp025_cluster_bootstrap_sensitivity.py
```

It writes its full 54-hypothesis table and input/output hashes under results/e025_common_probe_tuning/sensitivity/.

## Reproduce historical E024

Install the package and visualization extra:

```powershell
python -m pip install -e ".[viz]"
```

In a checkout without an existing `results/e024_common_probe_comparison/` run, execute:

```powershell
python experiments/exp024_common_probe_comparison.py
```

The run freezes calibration, model adapters, probe pairs, environment details, and the
protocol hash before evaluation. It then writes raw primary probes, per-seed summaries,
paired comparator tests, secondary outputs, mechanism diagnostics, and the original
figure under `results/e024_common_probe_comparison/`. The script refuses to overwrite an
existing run by default.

After the result files exist, the publication-language figure and registered secondary
recovery aggregation can be regenerated without rerunning the experiment:

```powershell
python experiments/exp024_render_english_figure.py
python experiments/exp024_summarize_secondary.py
```

These post-processing steps read saved summary CSVs. Their inputs, scripts, outputs, and
hashes are documented in `results/e024_common_probe_comparison/metadata/figure_postprocessing.md`.

## Rebuild historical figures and manuscript

The E017–E022R experiment outputs are historical inputs to the current article. If those
outputs are absent, run their scripts in sequence and then run
`python experiments/exp023_article_figure_assembly.py`. E024 is kept for historical
lineage and is not combined with the E025 holdout results.

To rebuild the main manuscript PDF from the repository root:

```powershell
typst compile "article/Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking/Slow Branch-Level Accessibility as a Structural Constraint.typ" "article/Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking/Slow Branch-Level Accessibility as a Structural Constraint.pdf"
```

The Typst source includes bibliography and figure paths relative to the manuscript
directory. `reviewer_slow_branch_level_accessibility.ipynb` and `CLAIMS_TO_EXPERIMENTS.md`
provide the reviewer-facing claim and artifact map.
Before submission, include the E025 source and outputs in a new public,
manuscript-matched versioned archive; the earlier Zenodo snapshot does not contain these
E025 additions.
