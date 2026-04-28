# cytodendaccessmodel

Companion code repository for the article
*A Cytoskeletal-Dendritic Accessibility Model of Associative Memory*.

The repository brings together the executable memory scaffold, DANDI open-data
tooling, and experiment entry points so that the theoretical article, the
computational model, and the empirical analyses can be inspected and reproduced
together.

---

## Repository layout

```
src/cytodend_accessmodel/   # associative-memory simulator and contracts
src/dandi_analysis/         # dataset-specific analysis helpers (000336, 000718, 001710; 000871 legacy)
src/dandi_io/               # generic DANDI listing, downloading, and probing CLI
experiments/                # runnable experiment scripts
configs/                    # YAML configuration files (benchmark and DANDI datasets)
tests/                      # pytest test suite
article/                    # manuscript source, figures, bibliography, and PDFs
data/                       # local data root (not tracked by git — see RUN.md)
```

---

## Installation

Python `>=3.10` is required. The reviewer reproduction path is CPU-sufficient;
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

> The installable project name in `pyproject.toml` is `asomemm` (legacy
> identifier kept for backward compatibility with earlier experiment logs).

---

## Quick start

Run the test suite:

```bash
pytest
```

Run the first simulator experiment (no data required):

```bash
python experiments/exp001_minimal_branch_linking.py
```

Reviewer orientation:

- `CLAIMS_TO_EXPERIMENTS.md` maps article claims to commands, outputs, and
  manuscript figures/tables.
- `RUN.md` gives the level-by-level reproduction workflow.
- `OUTPUTS.md` lists where scripts write reviewer-facing artifacts.

---

## DANDI open-data workflow

The repository includes a CLI for listing, downloading, and probing DANDI
assets from YAML configs:

```bash
python -m dandi_io.cli list     --config configs/dandi/dataset_000718.yaml
python -m dandi_io.cli download --config configs/dandi/dataset_000718.yaml
python -m dandi_io.cli probe    --config configs/dandi/dataset_000718.yaml
```

Equivalent current-manuscript configs exist for datasets `000336` and `001710`
under `configs/dandi/`. Dataset `000871` configs/scripts are retained as
legacy or supplementary cross-plane analyses and are not required for the
current manuscript claims.

DANDI data and derived artefacts are written under:

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
- The manuscript source and supporting information are included under
  `article/`; rebuild with Typst only if you need to regenerate PDFs from the
  `.typ` sources.
