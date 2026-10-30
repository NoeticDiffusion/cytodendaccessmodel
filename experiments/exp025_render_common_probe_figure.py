"""Render the publication-facing E025 holdout comparison figure.

The absolute-D intervals use the same 10,000 topology-draw bootstrap index
matrix as the frozen E025 analysis. The contrast panel is read directly from
the E025 Holm-adjusted holdout summary.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "e025_common_probe_tuning"
OUT = (
    ROOT
    / "article"
    / "Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking"
    / "figures2"
    / "Fig_e025_01_tuned_common_probe.png"
)
BOOTSTRAPS = 10_000
SEED = 250925

LABELS = {
    "full_model": "Full model",
    "fast_context_only": "Fast context only",
    "replay_no_structure": "Replay, no structure",
    "random_slow_drift": "Random slow drift",
    "fixed_allocation_only": "Fixed allocation",
    "hebbian_weight_only": "Hebbian weights",
    "soma_global_gain_only": "Global gain",
    "shuffled_replay": "Shuffled replay",
    "eligibility_only": "Eligibility only",
    "resource_only": "Resource only",
}

MODEL_ORDER = [
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
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def render() -> None:
    draw_rows = [
        row
        for row in read_csv(RESULTS / "scores" / "holdout_draw_scores.csv")
        if row["partition"] == "holdout" and row["role"] in {"tuned", "tuned_and_nominal"}
    ]
    model_draws: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    for model in MODEL_ORDER:
        rows = sorted(
            (row for row in draw_rows if row["model"] == model),
            key=lambda row: row["draw_id"],
        )
        if not rows:
            continue
        draw_ids = [row["draw_id"] for row in rows]
        if len(draw_ids) != 100 or len(set(draw_ids)) != 100:
            raise ValueError(f"Expected 100 unique holdout topology draws for {model}.")
        immediate = np.asarray([float(row["D_macro"]) for row in rows], dtype=float)
        late = np.asarray([float(row["D_late_macro"]) for row in rows], dtype=float)
        model_draws[model] = (draw_ids, immediate, late)

    if "full_model" not in model_draws:
        raise ValueError("E025 tuned full-model holdout rows are missing.")

    # The frozen E025 analysis uses a single common bootstrap index matrix for
    # every family and every paired contrast.
    rng = np.random.default_rng(SEED)
    bootstrap_indices = rng.integers(0, 100, size=(BOOTSTRAPS, 100))
    summaries: dict[str, tuple[float, float, float]] = {}
    for model, (_, values, _) in model_draws.items():
        bootstrap_means = np.mean(values[bootstrap_indices], axis=1)
        summaries[model] = (
            float(np.mean(values)),
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        )

    comparisons = {
        row["comparator"]: row
        for row in read_csv(RESULTS / "summary" / "holdout_comparisons.csv")
        if row["endpoint"] == "D"
    }
    if len(comparisons) != 9:
        raise ValueError("Expected nine preregistered E025 immediate-D comparisons.")

    status_colors = {
        "full_superiority": "#2468a2",
        "practical_equivalence": "#69727d",
        "comparator_superiority": "#bb4c40",
        "inconclusive": "#b07a22",
        "unavailable_no_qualifying_tuned_candidate": "#aab0b5",
    }
    model_colors = {
        "full_model": "#174f7a",
        "hebbian_weight_only": "#b8493e",
        "resource_only": "#b8493e",
    }

    fig, (ax_d, ax_delta) = plt.subplots(
        2,
        1,
        figsize=(7.2, 8.7),
        gridspec_kw={"height_ratios": [1.15, 1]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    visible_models = [model for model in MODEL_ORDER if model != "eligibility_only"]
    y_positions = np.arange(len(MODEL_ORDER), dtype=float)
    for y, model in zip(y_positions, MODEL_ORDER):
        if model not in summaries:
            ax_d.text(0.002, y, "No qualifying tuned candidate", va="center", color="#72777c", fontsize=8)
            continue
        mean, low, high = summaries[model]
        color = model_colors.get(model, "#527b91")
        ax_d.errorbar(
            mean,
            y,
            xerr=np.asarray([[mean - low], [high - mean]]),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=4.5,
            capsize=2.5,
            linewidth=1.15,
        )

    ax_d.set_yticks(y_positions, [LABELS[model] for model in MODEL_ORDER])
    ax_d.invert_yaxis()
    ax_d.set_xlim(-0.002, 0.112)
    ax_d.set_xlabel("Immediate shared-branch effect, D")
    ax_d.set_title("A  Tuned-model means on the locked holdout", loc="left", fontweight="bold")
    ax_d.grid(axis="x", color="#d8dde1", linewidth=0.65)
    ax_d.set_axisbelow(True)
    ax_d.spines[["top", "right", "left"]].set_visible(False)
    ax_d.tick_params(axis="y", length=0, pad=7)

    comparators = [model for model in MODEL_ORDER if model != "full_model"]
    y_delta = np.arange(len(comparators), dtype=float)
    ax_delta.axvspan(-0.02, 0.02, color="#e7eaed", zorder=0)
    ax_delta.axvline(0, color="#414950", linewidth=0.85, zorder=1)
    ax_delta.axvline(-0.02, color="#777f86", linestyle="--", linewidth=0.8, zorder=1)
    ax_delta.axvline(0.02, color="#777f86", linestyle="--", linewidth=0.8, zorder=1)
    for y, model in zip(y_delta, comparators):
        row = comparisons[model]
        if row["status"] != "testable":
            ax_delta.text(-0.115, y, "Not testable", va="center", color="#72777c", fontsize=8)
            continue
        mean = float(row["mean_delta_full_minus_comparator"])
        low = float(row["delta_ci95_low"])
        high = float(row["delta_ci95_high"])
        color = status_colors[row["decision"]]
        ax_delta.errorbar(
            mean,
            y,
            xerr=np.asarray([[mean - low], [high - mean]]),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=4.5,
            capsize=2.5,
            linewidth=1.15,
        )

    ax_delta.set_yticks(y_delta, [LABELS[model] for model in comparators])
    ax_delta.invert_yaxis()
    ax_delta.set_xlim(-0.12, 0.04)
    ax_delta.set_xlabel("Paired difference, D(full) - D(comparator)")
    ax_delta.set_title("B  Paired holdout contrasts", loc="left", fontweight="bold")
    ax_delta.grid(axis="x", color="#d8dde1", linewidth=0.65)
    ax_delta.set_axisbelow(True)
    ax_delta.spines[["top", "right", "left"]].set_visible(False)
    ax_delta.tick_params(axis="y", length=0, pad=7)
    fig.suptitle(
        "E025: equal-budget tuning does not reveal a unique full-model advantage",
        x=0.5,
        y=1.015,
        fontsize=12,
        fontweight="bold",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    render()
