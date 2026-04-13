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
src/dandi_analysis/         # dataset-specific analysis helpers (000336, 000718, 000871, 001710)
src/dandi_io/               # generic DANDI listing, downloading, and probing CLI
experiments/                # runnable experiment scripts
configs/                    # YAML configuration files (benchmark and DANDI datasets)
tests/                      # pytest test suite
article/                    # manuscript PDF
data/                       # local data root (not tracked by git — see RUN.md)
```

---

## Installation

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

---

## DANDI open-data workflow

The repository includes a CLI for listing, downloading, and probing DANDI
assets from YAML configs:

```bash
python -m dandi_io.cli list     --config configs/dandi/dataset_000718.yaml
python -m dandi_io.cli download --config configs/dandi/dataset_000718.yaml
python -m dandi_io.cli probe    --config configs/dandi/dataset_000718.yaml
```

Equivalent configs exist for datasets `000336`, `000871`, and `001710`
under `configs/dandi/`.

DANDI data and derived artefacts are written under:

```
data/dandi/raw/
data/dandi/cache/
data/dandi/triage/
```

For exact download commands, expected file inventory, and the full
experiment run order see **`RUN.md`**.

---

## Notes

- All experiment scripts assume they are run from the repository root.
- The `data/` directory is excluded from version control via `.gitignore`.
- The manuscript is included as a PDF under `article/`.
