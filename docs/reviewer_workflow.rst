Reviewer Workflow
=================

The reviewer path is organized around three repository files:

``RUN.md``
   Command-level reproduction guide, organized into Level 0 through Level 3.

``CLAIMS_TO_EXPERIMENTS.md``
   Claim-to-command map linking manuscript sections to scripts, outputs, and
   claim boundaries.

``OUTPUTS.md``
   Output manifest describing where experiment scripts write JSON, Markdown,
   logs, and figures.

Recommended order
-----------------

1. Install the package in editable mode with ``.[dev,viz]``.
2. Run the Level 0 checks in ``RUN.md``.
3. Run Level 1 to reproduce no-data simulator claims.
4. Download the DANDI datasets needed for the open-data claim under review.
5. Run the matching Level 2 scripts.
6. Render open-data figures with ``python -m dandi_analysis.visualisation.cli``.
7. Inspect ``notebooks/reviewer_reproduction_walkthrough.py`` after derived
   outputs exist.

Scope boundary
--------------

The code reproduces the executable model and downstream open-data signatures.
It does not directly observe the proposed slow cytoskeletal accessibility field;
that remains a biological hypothesis requiring branch-resolved perturbation
experiments.
