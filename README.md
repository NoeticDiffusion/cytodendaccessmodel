# cytodendaccessmodel

Companion code repository for the article
*Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking*.

The repository contains the executable simulator, result artifacts, article source,
and supplementary open-data tooling required to inspect and reproduce the
simulator-first manuscript claims.

---

## Repository layout

```
src/cytodend_accessmodel/   # associative-memory simulator and contracts
src/dandi_analysis/         # dataset-specific analysis helpers (000336, 000718, 001710)
src/dandi_io/               # generic DANDI listing, downloading, and probing CLI
experiments/                # runnable experiment scripts
configs/                    # YAML configuration files for DANDI datasets
tests/                      # pytest test suite
article/                    # manuscript source, figures, bibliography, and PDFs
data/                       # local data root (not tracked by git — see RUN.md)
```

---

## Installation

Python `>=3.10` is required. The primary reproduction path is CPU-sufficient;
CUDA is not required.

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,viz]"
```

Core dependencies declared in `pyproject.toml`:

- `numpy`
- `pyyaml`
- `torch`
- `dandi`
- `pynwb`

Optional extras: `dev` adds `pytest`, `viz` adds `matplotlib`.
Use `docs` for Sphinx/ReadTheDocs builds and `article` for PDF/document-processing helpers.

---

## Quick start

Run the test suite:

```bash
pytest
```

Run the primary canonical simulator experiment (no data required):

```bash
python experiments/exp017_traceable_simulator_core.py
```

### Reproduction entry points

The primary entry point is the root-level notebook:

**`reviewer_slow_branch_level_accessibility.ipynb`** — top-level routing for the
simulator evidence tiers, legacy lineage, and supplementary bridges.

The repository is organized into three tiers:

- **Primary evidence:** `E017–E022R`, assembled figures in `figures2/`, and the
  Typst article source under `article/`.
- **Legacy lineage:** `exp001–exp016`, earlier claim maps, and validation scripts
  retained for lineage and optional cross-checks.
- **Supplementary framing:** `figures3_concepts/`, DANDI bridge material, and S3
  appendix analyses.

Supporting reference files:

- `CLAIMS_TO_EXPERIMENTS.md` — claim map for the current article.
- `RUN.md` — level-by-level reproduction workflow (Levels 0–3).
- `OUTPUTS.md` — lists where scripts write output artifacts.

Current article source:

- `article/Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking/Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking.typ`
- `article/Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking/v2_claim_ledger.md`
- `article/Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking/v2_figure_manifest.md`

---

## DANDI open-data workflow

The repository includes a CLI for listing, downloading, and probing DANDI
assets from YAML configs:

```bash
python -m dandi_io.cli list     --config configs/dandi/dataset_000718.yaml
python -m dandi_io.cli download --config configs/dandi/dataset_000718.yaml
python -m dandi_io.cli probe    --config configs/dandi/dataset_000718.yaml
```

Equivalent configs exist for datasets `000336` and `001710` under `configs/dandi/`.
Dataset `000871` scripts are retained as legacy supplementary material and are not
required for the current manuscript claims.

The DANDI analyses belong to the **supplementary bridge layer**, not the primary
evidence ladder. Readers who only need to verify the main article claims can stop
after the no-data simulator stack (`E017–E022R`).

DANDI data and derived artifacts are written under:

```
data/dandi/raw/
data/dandi/cache/
data/dandi/triage/
```

For exact download commands, expected file inventory, full experiment run order,
resource notes, and troubleshooting see **`RUN.md`**.

---

## Notes

- All experiment scripts assume they are run from the repository root.
- The `data/` directory is excluded from version control via `.gitignore`.
- The manuscript source and supporting information are included under `article/`;
  rebuild with Typst only if you need to regenerate PDFs from the `.typ` sources.
- Older article framing, notebooks, and claim maps are retained as legacy materials.
