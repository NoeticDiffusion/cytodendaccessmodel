"""Summarize the prespecified E024 targeted-versus-control recovery challenge.

This post-processing script preserves seed clusters, averages eligible pairs within
motif and motifs within seed, and does not change the frozen E024 evaluation data.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
E024_ROOT = ROOT / "results" / "e024_common_probe_comparison"
INPUT = E024_ROOT / "summary" / "secondary_by_seed.csv"
SEED_OUTPUT = E024_ROOT / "summary" / "recovery_seed_summary.csv"
MODEL_OUTPUT = E024_ROOT / "summary" / "recovery_model_summary.csv"
BOOTSTRAP_SEED = 240925
BOOTSTRAP_REPLICATES = 10_000


def main() -> None:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    pair_values: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row["analysis"] != "baseline_matched" or not row["outcome"].startswith("recovery_"):
            continue
        if row["status"] != "defined" or row["value"] in ("", "nan", "NaN"):
            continue
        arm = row["arm"]
        pair_values[(row["model"], row["motif_type"], row["seed"], arm)]["value"].append(
            float(row["value"])
        )

    motif_seed_values: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for (model, motif, seed, arm), values in pair_values.items():
        motif_seed_values[(model, motif, seed)][arm] = statistics.mean(values["value"])

    seed_values: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for (model, motif, seed), arms in motif_seed_values.items():
        if "targeted" not in arms or "control" not in arms:
            continue
        seed_values[(model, seed)].append({
            "targeted": arms["targeted"],
            "control": arms["control"],
            "targeted_minus_control": arms["targeted"] - arms["control"],
        })

    per_seed_rows: list[dict[str, object]] = []
    per_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for (model, seed), motif_rows in sorted(seed_values.items()):
        result = {
            "analysis": "baseline_matched",
            "model": model,
            "seed": int(seed),
            "n_defined_motifs": len(motif_rows),
            "targeted_mean": statistics.mean(row["targeted"] for row in motif_rows),
            "control_mean": statistics.mean(row["control"] for row in motif_rows),
            "targeted_minus_control": statistics.mean(
                row["targeted_minus_control"] for row in motif_rows
            ),
        }
        per_seed_rows.append(result)
        per_model[model].append(result)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    model_rows: list[dict[str, object]] = []
    for model, values in sorted(per_model.items()):
        model_result: dict[str, object] = {
            "analysis": "baseline_matched",
            "model": model,
            "n_seed_clusters": len(values),
            "n_seed_motif_cells": sum(int(row["n_defined_motifs"]) for row in values),
        }
        for field in ("targeted_mean", "control_mean", "targeted_minus_control"):
            vector = np.array([float(row[field]) for row in values])
            samples = rng.choice(vector, size=(BOOTSTRAP_REPLICATES, len(vector)), replace=True).mean(axis=1)
            model_result[f"{field}_mean"] = float(vector.mean())
            model_result[f"{field}_ci95_low"] = float(np.quantile(samples, 0.025))
            model_result[f"{field}_ci95_high"] = float(np.quantile(samples, 0.975))
        model_rows.append(model_result)

    SEED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for path, output_rows in ((SEED_OUTPUT, per_seed_rows), (MODEL_OUTPUT, model_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    print(f"Wrote {SEED_OUTPUT}")
    print(f"Wrote {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
