Reviewer Workflow
=================

The primary reviewer entry point is the root-level Jupyter notebook:

``reviewer_branch_resolved_walkthrough.ipynb``
   Self-contained walkthrough combining a direct simulator primer, guided
   execution of the minimal and full Level 1 paths, an artifact-status audit
   of all expected outputs, and the optional DANDI open-data workflow.  Run all
   cells from the repository root.  The notebook ships with pre-executed outputs
   so the key results are visible without running anything.

Supporting reference files
--------------------------

``RUN.md``
   Command-level reproduction guide, organized into Level 0 through Level 3.

``CLAIMS_TO_EXPERIMENTS.md``
   Claim-to-command map linking manuscript sections to scripts, outputs, and
   claim boundaries.

``OUTPUTS.md``
   Output manifest describing where experiment scripts write JSON, Markdown,
   logs, and figures.

``notebooks/reviewer_reproduction_walkthrough.py``
   Earlier Python-script equivalent of the notebook, kept for reference.

Recommended order
-----------------

1. Install the package in editable mode with ``.[dev,viz]``.
2. Open ``reviewer_branch_resolved_walkthrough.ipynb`` and run all cells.
3. For a CLI-only path, follow the Level 0 and Level 1 steps in ``RUN.md``.
4. Download the DANDI datasets needed for the open-data claim under review.
5. Run the matching Level 2 scripts.
6. Render open-data figures with ``python -m dandi_analysis.visualisation.cli``.

Scope boundary
--------------

The code reproduces the executable model and downstream open-data signatures.
It does not directly observe the proposed slow cytoskeletal accessibility field;
that remains a biological hypothesis requiring branch-resolved perturbation
experiments.
