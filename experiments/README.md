# Experiments

Runnable experiment entry points live here. Run them from the repository root
with `python experiments/<script>.py`.

Rules:

- import reusable simulator logic from `src/cytodend_accessmodel/`
- import reusable open-data analysis logic from `src/dandi_analysis/`
- import generic DANDI listing/downloading/probing logic from `src/dandi_io/`
- do not re-implement core module logic here
- keep reviewer-facing scripts deterministic where possible and document their
  outputs in `OUTPUTS.md`

The current source packages used by these experiments are the three packages
listed above. See `CLAIMS_TO_EXPERIMENTS.md` for the article-claim mapping and
`RUN.md` for the canonical run order.
