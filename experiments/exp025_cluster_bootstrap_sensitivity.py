#!/usr/bin/env python
"""Post-hoc centered cluster-bootstrap sensitivity for E025 paired contrasts.

Approximate bootstrap sensitivity, not an exact randomization test and not part
of the preregistered E025 analysis. Treats the 100 locked topology-draw macros
as independent draws from the declared topology generator.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/e025_common_probe_tuning/scores/holdout_draw_scores.csv"
COMPARISONS = ROOT / "results/e025_common_probe_tuning/summary/holdout_comparisons.csv"
DEFAULT_OUTPUT = ROOT / "results/e025_common_probe_tuning/sensitivity/centered_cluster_bootstrap_holm.csv"
DEFAULT_METADATA = ROOT / "results/e025_common_probe_tuning/sensitivity/centered_cluster_bootstrap_metadata.json"
ENDPOINTS = {"D": "D_macro", "D_late": "D_late_macro"}
HYPOTHESES = ("full_superiority", "comparator_superiority", "practical_equivalence")
MARGIN = 0.02
SEED = 250926
REPLICATES = 200_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def holm(pvalues: list[float]) -> list[float]:
    size = len(pvalues)
    order = sorted(range(size), key=lambda i: (float("inf") if not np.isfinite(pvalues[i]) else pvalues[i], i))
    adjusted = [float("nan")] * size
    running = 0.0
    for rank, index in enumerate(order):
        pvalue = pvalues[index]
        if not np.isfinite(pvalue):
            continue
        running = max(running, min(1.0, (size - rank) * pvalue))
        adjusted[index] = running
    return adjusted


def centered_bootstrap_pvalues(
    delta: np.ndarray, index: np.ndarray, margin: float = MARGIN
) -> dict[str, float]:
    delta = np.asarray(delta, dtype=float)
    index = np.asarray(index)
    if delta.ndim != 1 or index.ndim != 2 or index.shape[1] != delta.size:
        raise ValueError("Expected one paired contrast per bootstrap index column.")
    observed = float(delta.mean())
    residual_means = (delta - observed)[index].mean(axis=1)

    def p_upper(boundary: float) -> float:
        cutoff = observed - boundary
        return (int(np.count_nonzero(residual_means >= cutoff - 1e-15)) + 1) / (len(index) + 1)

    def p_lower(boundary: float) -> float:
        cutoff = observed - boundary
        return (int(np.count_nonzero(residual_means <= cutoff + 1e-15)) + 1) / (len(index) + 1)

    return {
        "full_superiority": p_upper(margin),
        "comparator_superiority": p_lower(-margin),
        "practical_equivalence": max(p_upper(-margin), p_lower(margin)),
    }


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--comparisons", type=Path, default=COMPARISONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")

    with args.input.open(newline="", encoding="utf-8") as handle:
        draws = list(csv.DictReader(handle))
    with args.comparisons.open(newline="", encoding="utf-8") as handle:
        comparison_rows = list(csv.DictReader(handle))
    comparators = list(dict.fromkeys(row["comparator"] for row in comparison_rows))
    if len(comparators) != 9:
        raise RuntimeError("Expected nine registered comparators.")
    registered_status = {
        (row["endpoint"], row["comparator"]): row["status"]
        for row in comparison_rows
    }

    tuned_roles = {"tuned", "tuned_and_nominal"}
    full = {
        endpoint: {
            row["draw_id"]: float(row[field])
            for row in draws
            if row["model"] == "full_model" and row["role"] in tuned_roles
        }
        for endpoint, field in ENDPOINTS.items()
    }
    rng = np.random.default_rng(SEED)
    index = rng.integers(0, 100, size=(args.replicates, 100), dtype=np.uint8)
    records = []
    for endpoint, field in ENDPOINTS.items():
        for comparator in comparators:
            other = {
                row["draw_id"]: float(row[field])
                for row in draws
                if row["model"] == comparator and row["role"] in tuned_roles
            }
            keys = sorted(set(full[endpoint]) & set(other))
            status = registered_status.get((endpoint, comparator), "unavailable_missing_registered_status")
            if status != "testable":
                for hypothesis in HYPOTHESES:
                    records.append({
                        "endpoint": endpoint, "comparator": comparator, "hypothesis": hypothesis,
                        "n_topology_draws": len(keys), "mean_delta_full_minus_comparator": float("nan"),
                        "p_raw": float("nan"), "status": status,
                    })
                continue
            if len(keys) != 100:
                raise RuntimeError(
                    f"Registered testable pair {endpoint}/{comparator} has {len(keys)} holdout draws, expected 100."
                )
            delta = np.asarray([full[endpoint][key] - other[key] for key in keys], dtype=float)
            if not np.isfinite(delta).all():
                raise RuntimeError(f"Registered testable pair {endpoint}/{comparator} contains non-finite scores.")
            observed = float(delta.mean())
            pvalues = centered_bootstrap_pvalues(delta, index, MARGIN)
            for hypothesis in HYPOTHESES:
                records.append({
                    "endpoint": endpoint, "comparator": comparator, "hypothesis": hypothesis,
                    "n_topology_draws": 100, "mean_delta_full_minus_comparator": observed,
                    "p_raw": pvalues[hypothesis], "status": "testable",
                })

    adjusted = holm([float(row["p_raw"]) for row in records])
    for row, pvalue in zip(records, adjusted):
        row["p_holm_54"] = pvalue
        row["decision"] = (
            "supported" if np.isfinite(pvalue) and pvalue < 0.05
            else "not_supported" if np.isfinite(pvalue)
            else "inconclusive"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "endpoint", "comparator", "hypothesis", "n_topology_draws",
            "mean_delta_full_minus_comparator", "p_raw", "p_holm_54", "status", "decision",
        ])
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "analysis": "post-hoc centered topology-cluster bootstrap sensitivity",
        "preregistered": False,
        "inference": "Approximate nonparametric bootstrap of the 100 paired topology-draw macros, centered at each null boundary; not an exact randomization test.",
        "scope": "Conditional on the selected candidates/adapters, frozen topology generator, and independent topology-draw sampling.",
        "replicates": args.replicates, "seed": SEED, "margin": MARGIN, "holm_family_size": 54,
        "script": display_path(Path(__file__)), "script_sha256": sha256(Path(__file__)),
        "input": display_path(args.input), "input_sha256": sha256(args.input),
        "comparison_input": display_path(args.comparisons),
        "comparison_sha256": sha256(args.comparisons), "output": display_path(args.output),
        "output_sha256": sha256(args.output), "metadata": display_path(args.metadata),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {len(records)} hypothesis rows to {args.output}")
    print(f"Wrote analysis metadata to {args.metadata}")
    for row in records:
        if row["endpoint"] == "D" and row["comparator"] in {"hebbian_weight_only", "resource_only", "shuffled_replay"}:
            print(row["comparator"], row["hypothesis"], f"p={float(row['p_raw']):.8g}",
                  f"Holm={float(row['p_holm_54']):.8g}", row["decision"])


if __name__ == "__main__":
    main()
