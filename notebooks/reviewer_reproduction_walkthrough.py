# %% [markdown]
# # Reviewer Reproduction Walkthrough
#
# This percent-format notebook is a lightweight companion to `RUN.md`.
# Run the experiment commands first, then use this file to inspect which
# reviewer-facing artifacts exist and how they map back to the manuscript.
#
# It intentionally reads only small derived artifacts such as JSON and Markdown
# files under `data/reviewer/` and `data/dandi/triage/`. It does not load NWB
# files or rerun the heavy analyses.

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def status(path: Path) -> str:
    return "present" if path.exists() else "missing"


def print_artifact_status(title: str, paths: list[Path]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for path in paths:
        print(f"{status(path):>7}  {rel(path)}")


def summarize_json_shape(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        keys = ", ".join(list(value)[:6])
        suffix = "..." if len(value) > 6 else ""
        return f"dict[{len(value)}]: {keys}{suffix}"
    return type(value).__name__


# %% [markdown]
# ## Claim-to-Artifact Map
#
# The claims below mirror `CLAIMS_TO_EXPERIMENTS.md`.

# %%
artifacts = {
    "Simulator canonical summary": [
        REPO_ROOT / "data" / "reviewer" / "013_canonical_values.json",
    ],
    "DANDI 000718 offline enrichment": [
        REPO_ROOT / "data" / "dandi" / "triage" / "000718" / "h1_pri_enrichment.json",
        REPO_ROOT / "data" / "dandi" / "triage" / "000718" / "h1_pri_enrichment.md",
        REPO_ROOT / "data" / "dandi" / "triage" / "000718" / "h1_robustness.json",
    ],
    "DANDI 000336 full-bundle coupling": [
        REPO_ROOT / "data" / "dandi" / "triage" / "000336" / "full_bundle_coupling.json",
        REPO_ROOT / "data" / "dandi" / "triage" / "000336" / "full_bundle_coupling.md",
    ],
    "DANDI 001710 robustness and nulls": [
        REPO_ROOT / "data" / "dandi" / "triage" / "001710" / "robustness" / "group_null_tests.json",
        REPO_ROOT / "data" / "dandi" / "triage" / "001710" / "robustness" / "claim_boundary.md",
        REPO_ROOT / "data" / "dandi" / "triage" / "001710" / "robustness" / "day_lag_similarity.md",
    ],
}

for title, paths in artifacts.items():
    print_artifact_status(title, paths)


# %% [markdown]
# ## Quick JSON Inspection
#
# These summaries are intentionally shallow. Open the matching `.md` or `.json`
# files for the full reviewer audit.

# %%
json_paths = [
    REPO_ROOT / "data" / "reviewer" / "013_canonical_values.json",
    REPO_ROOT / "data" / "dandi" / "triage" / "000718" / "h1_pri_enrichment.json",
    REPO_ROOT / "data" / "dandi" / "triage" / "000718" / "h1_robustness.json",
    REPO_ROOT / "data" / "dandi" / "triage" / "000336" / "full_bundle_coupling.json",
    REPO_ROOT / "data" / "dandi" / "triage" / "001710" / "robustness" / "group_null_tests.json",
]

for path in json_paths:
    print(f"{rel(path)}: {summarize_json_shape(load_json(path))}")


# %% [markdown]
# ## Reviewer Claim Boundaries
#
# - `000718`: positive but modest excess enrichment above a strong
#   population-burst baseline; not direct sequence-level replay proof.
# - `000336`: structured above-null cross-plane coupling across the analyzed
#   bundle; the cleanest strict access-constraint match is the supplementary
#   cross-area pair.
# - `001710`: SparseKO is lower than Cre under the implemented subject-level
#   null; separation from Ctrl is weaker and channel sensitivity remains a
#   boundary condition.
# - Simulator: executable signatures are reproducible from local scripts, but
#   they are mechanistic stress tests rather than direct molecular validation.

# %%
print("Next steps:")
print("1. Compare missing artifacts above with the relevant RUN.md level.")
print("2. Read CLAIMS_TO_EXPERIMENTS.md for article-section mapping.")
print("3. Read OUTPUTS.md for exact output filenames.")
