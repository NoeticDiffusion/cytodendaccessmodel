# RUN - Reviewer Reproducibility Guide

This file describes how to set up the data and run every experiment in the
repository from a clean checkout.  All commands are run from the **repository
root** unless stated otherwise.

Use `CLAIMS_TO_EXPERIMENTS.md` as the claim map while running this guide. Use
`OUTPUTS.md` when you need to know where a script writes reviewer-facing
artifacts.

Reviewer levels:

- **Level 0 - sanity checks:** install, import/test checks, seed validation, and
  DANDI inventories when data are present.
- **Level 1 - no-data simulator reproduction:** all executable model claims,
  ablations, paper summary, and executable figures.
- **Level 2 - manuscript open-data reproduction:** DANDI `000718`, `000336`,
  and `001710` in the order used by the manuscript claims.
- **Level 3 - full audit:** robustness/null checks, simulator-DANDI bridge
  scripts, figure regeneration, and integrity checks.

Python `>=3.10` is required. The simulator and DANDI analyses are CPU-sufficient;
CUDA is not required. The DANDI `001710` bundle is the main disk/time cost
(`~139` NWB files; plan for several GB and longer wall time than the simulator
suite).

---

## Level 0 - Install and Sanity Checks

```bash
python --version
python -m pip install --upgrade pip
python -m pip install -e ".[dev,viz]"
```

Core dependencies (`numpy`, `pyyaml`, `torch`, `dandi`, `pynwb`) are declared
in `pyproject.toml` and installed automatically.

Run the lightweight local checks:

```bash
pytest
python experiments/exp_seed_validation.py
```

---

## Level 2 Setup - Download DANDI Data

The manuscript open-data claims require three DANDI datasets: `000718`,
`000336`, and `001710`. The expected layout after download is:

```
data/dandi/raw/
  000336/
    sub-644972/   (2 NWB files)
    sub-656228/   (4 NWB files)
  000718/
    sub-Ca-EEG2-1/  (5 NWB files)
    sub-Ca-EEG3-4/  (7 NWB files)
  001710/
    sub-Cre-1/ … sub-Cre-7/          (6 sessions each)
    sub-Ctrl-1/ … sub-Ctrl-9/        (6 sessions each)
    sub-SparseKO-1/ … sub-SparseKO-7/ (6 sessions each, SparseKO-4 has 5)
```

### Using the `dandi` CLI (recommended)

Install the CLI if needed: `pip install dandi`

```bash
# Dandiset 000336 — two subjects, 6 NWB files total
dandi download --output-dir data/dandi/raw "DANDI:000336/sub-644972"
dandi download --output-dir data/dandi/raw "DANDI:000336/sub-656228"

# Dandiset 000718 — two subjects, 12 NWB files total
dandi download --output-dir data/dandi/raw "DANDI:000718/sub-Ca-EEG2-1"
dandi download --output-dir data/dandi/raw "DANDI:000718/sub-Ca-EEG3-4"

# Dandiset 001710 — full dandiset (~139 NWB files, ~several GB)
dandi download --output-dir data/dandi/raw "DANDI:001710"
```

> **Note:** `dandi download --output-dir data/dandi/raw DANDI:<id>` places
> files at `data/dandi/raw/<id>/sub-<subject>/`.  This is the path the
> analysis modules expect.

### File inventory per dataset

**000336** (`data/dandi/raw/000336/`)

| Subject | File |
|---------|------|
| sub-644972 | `sub-644972_ses-1237338784-acq-1237809217_ophys.nwb` |
| sub-644972 | `sub-644972_ses-1237338784-acq-1237809219_ophys.nwb` |
| sub-656228 | `sub-656228_ses-1245548523-acq-1245937727_ophys.nwb` |
| sub-656228 | `sub-656228_ses-1245548523-acq-1245937736_ophys.nwb` |
| sub-656228 | `sub-656228_ses-1247233186-acq-1247385128_ophys.nwb` |
| sub-656228 | `sub-656228_ses-1247233186-acq-1247385130_ophys.nwb` |

**000718** (`data/dandi/raw/000718/`)

| Subject | Session file |
|---------|--------------|
| sub-Ca-EEG2-1 | `sub-Ca-EEG2-1_ses-FC_image+ophys.nwb` |
| sub-Ca-EEG2-1 | `sub-Ca-EEG2-1_ses-NeutralExposure_image+ophys.nwb` |
| sub-Ca-EEG2-1 | `sub-Ca-EEG2-1_ses-OfflineDay2Session1_ophys.nwb` |
| sub-Ca-EEG2-1 | `sub-Ca-EEG2-1_ses-Recall1_image+ophys.nwb` |
| sub-Ca-EEG2-1 | `sub-Ca-EEG2-1_ses-Week.nwb` |
| sub-Ca-EEG3-4 | `sub-Ca-EEG3-4_ses-FC_image+ophys.nwb` |
| sub-Ca-EEG3-4 | `sub-Ca-EEG3-4_ses-NeutralExposure_image+ophys.nwb` |
| sub-Ca-EEG3-4 | `sub-Ca-EEG3-4_ses-OfflineDay1Session1_ophys.nwb` |
| sub-Ca-EEG3-4 | `sub-Ca-EEG3-4_ses-OfflineDay2Session1_ophys.nwb` |
| sub-Ca-EEG3-4 | `sub-Ca-EEG3-4_ses-Recall1_image+ophys.nwb` |
| sub-Ca-EEG3-4 | `sub-Ca-EEG3-4_ses-Recall2_image+ophys.nwb` |
| sub-Ca-EEG3-4 | `sub-Ca-EEG3-4_ses-Recall3_image+ophys.nwb` |

**001710** (`data/dandi/raw/001710/`)

Three groups × subjects × 6 sessions (day0–day5), pattern
`sub-<ID>_ses-ymaze-day<N>-scan0-novel-arm[-]1_behavior+ophys.nwb`.

| Group | Subjects | Sessions per subject |
|-------|----------|----------------------|
| Cre | sub-Cre-1 … sub-Cre-7 (7) | day0–day5 (6 each) |
| Ctrl | sub-Ctrl-1 … sub-Ctrl-9 (9) | day0–day5 (6 each, sub-Ctrl-2 has an extra scan1 on day3) |
| SparseKO | sub-SparseKO-1 … sub-SparseKO-7 (7) | day0–day5 (sub-SparseKO-4 missing day2) |

---

### Legacy `000871` note

The repository also contains `configs/dandi/dataset_000871.yaml` and
`experiments/dandi_000871_*.py`. These are legacy/supplementary cross-plane
scripts and are not required for the current manuscript claims. The current
article uses `000336`, `000718`, and `001710`.

---

## Level 0 - Verify the Download

Quick sanity checks — these scripts read metadata only and require no heavy
computation:

```bash
python experiments/dandi_000336_01_inventory.py
python experiments/dandi_000718_01_inventory.py
python experiments/dandi_001710_01_inventory.py
```

---

## Level 1 - Run the Simulator Experiments

These experiments use only the in-repo scaffold (`src/cytodend_accessmodel`)
and need no data download. Most scripts print reviewer-facing terminal tables.
`exp013_paper_summary.py` also writes `data/reviewer/013_canonical_values.json`.
`gen_figures_executable.py` writes SVG figures under `.article/Executable
Structural Accessibility - A Biologically Constrained Cytoskeletal-Dendritic
Model of Memory Linking/figures/`.

```bash
# Core model properties
python experiments/exp001_minimal_branch_linking.py
python experiments/exp002_context_sensitive_recall.py
python experiments/exp003_timing_replay_linking.py
python experiments/exp004_robustness.py
python experiments/exp005_pathology.py
python experiments/exp006_asymmetric_consolidation.py
python experiments/exp007_branch_heterogeneity.py
python experiments/exp008_local_competition.py
python experiments/exp009_rescue_linking.py
python experiments/exp010_multitrace_overlap.py
python experiments/exp011_branch_topology.py
python experiments/exp012_retrieval_readout.py

# Ablations and comparisons
python experiments/exp014_structural_gate_ablation.py
python experiments/exp015_comparator_baselines.py
python experiments/exp016_task_family.py

# Summary and figures
python experiments/exp013_paper_summary.py
python experiments/gen_figures_executable.py
```

Seed validation (reproduces canonical RNG state):

```bash
python experiments/exp_seed_validation.py
```

---

## Level 2 - Run the DANDI Open-Data Experiments

Run scripts in numbered order within each dataset series. Dataset `000336`
and `000718` must be downloaded first; `001710` is required for the `001710`
series. The `000718` numbering intentionally has no `07` or `09` script in the
current retained pipeline.

### Dandiset 000336

```bash
python experiments/dandi_000336_01_inventory.py
python experiments/dandi_000336_02_header_probe.py
python experiments/dandi_000336_03_crossplane_coupling.py
python experiments/dandi_000336_04_sub656228_replication.py
python experiments/dandi_000336_05_ses1245548523.py
python experiments/dandi_000336_06_full_bundle.py
```

### Dandiset 000718

```bash
python experiments/dandi_000718_01_inventory.py
python experiments/dandi_000718_02_header_probe.py
python experiments/dandi_000718_03_offline_epoch_candidates.py
python experiments/dandi_000718_04_activity_matrix_smoke_test.py
python experiments/dandi_000718_05_pairwise_coreactivation_baseline.py
python experiments/dandi_000718_06_ensemble_reactivation.py
python experiments/dandi_000718_08_h1_neutral_offline.py
python experiments/dandi_000718_10_h1_event_registration.py
python experiments/dandi_000718_11_h1_robustness.py
python experiments/dandi_000718_12_h1_specificity.py
python experiments/dandi_000718_13_h1_pri.py
python experiments/dandi_000718_14_h1_pri_enrichment.py
```

### Dandiset 001710

```bash
python experiments/dandi_001710_01_inventory.py
python experiments/dandi_001710_02_header_probe.py
python experiments/dandi_001710_03_trial_reconstruction.py
python experiments/dandi_001710_04_activity_matrix_smoke_test.py
python experiments/dandi_001710_05_within_day_place_tuning.py
python experiments/dandi_001710_06_cross_day_remapping_baseline.py
python experiments/dandi_001710_07_replication_bundle_full_pass.py
python experiments/dandi_001710_08_robustness_and_nulls.py
```

### Simulator × DANDI bridge

```bash
python experiments/dandi_simulator_07_bootstrap_ci.py
python experiments/dandi_simulator_09_sensitivity.py
```

---

## Level 3 - Regenerate Manuscript-Facing Open-Data Figures

After the DANDI triage JSON outputs exist, regenerate the manuscript-facing
open-data PNGs:

```bash
python -m dandi_analysis.visualisation.cli
```

By default this renders the current manuscript figures for `000718` and `000336`
under `.article/A Cytoskeletal-Dendritic Accessibility Model of Associative Memory/figures/`.
To also render legacy `000871` cross-plane figures, pass:

```bash
python -m dandi_analysis.visualisation.cli --include-legacy-000871
```

---

## Level 3 - Run the Test Suite

```bash
pytest
```

---

## Level 3 - Integrity Check

```bash
python experiments/integrity_check_article2.py
```

This check is a secondary integrity gate for older article/table harmonization.
Treat failures as a prompt to inspect the reported deltas rather than as a
replacement for the claim map in `CLAIMS_TO_EXPERIMENTS.md`.

---

## Troubleshooting

- If DANDI downloads resume into the wrong folder, verify that files are under
  `data/dandi/raw/<dandiset>/sub-...` before running experiments.
- If a script reports missing NWB files, rerun the matching inventory script and
  compare against the file inventory above.
- On Windows, run commands from the repository root and keep paths quoted if
  calling tools manually from a directory with spaces.
- If figure regeneration fails, confirm that the prerequisite triage JSON files
  exist under `data/dandi/triage/...`.
- If installing `torch` is slow, a CPU-only install is sufficient for the
  reviewer reproduction path.
