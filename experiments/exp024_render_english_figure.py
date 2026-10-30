"""Render the publication-language E024 primary-effect figure from frozen summaries.

This is figure-only post-processing. It does not alter E024 observations or summaries.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "e024_common_probe_comparison" / "summary" / "primary_model_summary.csv"
OUTPUT = ROOT / "results" / "e024_common_probe_comparison" / "figures" / "Fig_e024_01_primary_effects_en.png"
MODEL_ORDER = (
    "full_model",
    "fast_context_only",
    "replay_no_structure",
    "random_slow_drift",
    "fixed_allocation_only",
    "hebbian_weight_only",
    "soma_global_gain_only",
    "shuffled_replay",
    "eligibility_only",
    "resource_only",
)
MODEL_LABELS = (
    "Full model",
    "Fast context only",
    "Replay, no structure",
    "Random slow drift",
    "Fixed allocation",
    "Hebbian weights",
    "Global gain",
    "Shuffled replay",
    "Eligibility only",
    "Resource only",
)
ANALYSES = (
    ("baseline_matched", "Baseline-matched input gain"),
    ("published_parameters", "Published parameters"),
)


def main() -> None:
    with SUMMARY.open(newline="", encoding="utf-8") as handle:
        data = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, (analysis, title) in zip(axes, ANALYSES):
        rows = {
            row["model"]: row
            for row in data
            if row["analysis"] == analysis and row["motif_type"] == "macro_mean"
        }
        x = np.arange(len(MODEL_ORDER))
        means = np.array([float(rows[name]["mean_D"]) for name in MODEL_ORDER])
        low = np.array([float(rows[name]["ci95_low"]) for name in MODEL_ORDER])
        high = np.array([float(rows[name]["ci95_high"]) for name in MODEL_ORDER])
        yerr = np.vstack((means - low, high - means))
        colors = ["#222222" if name == "full_model" else "#225ea8" for name in MODEL_ORDER]
        for idx, color in enumerate(colors):
            ax.errorbar(
                x[idx], means[idx],
                yerr=np.array([[yerr[0, idx]], [yerr[1, idx]]]),
                fmt="o", capsize=3, color=color, ecolor="#7aa6d8" if idx else "#555555",
            )
        ax.axhline(0.0, color="#555555", linewidth=0.8)
        ax.axhline(0.02, color="#b2182b", linestyle="--", linewidth=0.9)
        ax.set_xticks(x, MODEL_LABELS, rotation=55, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Primary D (readout scale 0–1)")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("E024 — common probe: shared branch minus source-private control, post − pre")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
