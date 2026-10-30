import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from experiments.exp025_cluster_bootstrap_sensitivity import (
    centered_bootstrap_pvalues,
    holm,
)


def test_centered_bootstrap_and_holm_use_expected_boundaries():
    delta = np.asarray([0.03, 0.04, 0.05, 0.06])
    index = np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]])
    pvalues = centered_bootstrap_pvalues(delta, index)

    assert pvalues["full_superiority"] == 1 / 3
    assert pvalues["comparator_superiority"] == 1.0
    assert pvalues["practical_equivalence"] == 1.0
    adjusted = holm([0.01, 0.04, float("nan")])
    assert adjusted[:2] == [0.03, 0.08]
    assert np.isnan(adjusted[2])


def test_cli_honors_registered_unavailable_status_and_external_paths(tmp_path):
    endpoints = {"D": "D_macro", "D_late": "D_late_macro"}
    comparators = [f"model_{i}" for i in range(9)]
    draw_path = tmp_path / "draws.csv"
    comparison_path = tmp_path / "comparisons.csv"
    output_path = tmp_path / "external" / "result.csv"
    metadata_path = tmp_path / "external" / "metadata.json"

    with draw_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["partition", "role", "draw_id", "model", *endpoints.values()]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for draw_index in range(100):
            draw_id = f"holdout_{draw_index}"
            writer.writerow({
                "partition": "holdout", "role": "tuned", "draw_id": draw_id,
                "model": "full_model", "D_macro": 0.10, "D_late_macro": 0.08,
            })
            for comparator in comparators:
                writer.writerow({
                    "partition": "holdout", "role": "tuned", "draw_id": draw_id,
                    "model": comparator, "D_macro": 0.03, "D_late_macro": 0.02,
                })

    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["endpoint", "comparator", "status"])
        writer.writeheader()
        for endpoint in endpoints:
            for comparator_index, comparator in enumerate(comparators):
                writer.writerow({
                    "endpoint": endpoint,
                    "comparator": comparator,
                    "status": (
                        "unavailable_no_qualifying_tuned_candidate"
                        if comparator_index == 0 else "testable"
                    ),
                })

    script = Path(__file__).resolve().parents[1] / "experiments/exp025_cluster_bootstrap_sensitivity.py"
    subprocess.run([
        sys.executable, str(script),
        "--input", str(draw_path),
        "--comparisons", str(comparison_path),
        "--output", str(output_path),
        "--metadata", str(metadata_path),
        "--replicates", "20",
    ], check=True, capture_output=True, text=True)

    with output_path.open(newline="", encoding="utf-8") as handle:
        results = list(csv.DictReader(handle))
    unavailable = [row for row in results if row["comparator"] == comparators[0]]
    assert len(unavailable) == 6
    assert all(row["status"] == "unavailable_no_qualifying_tuned_candidate" for row in unavailable)
    assert all(row["decision"] == "inconclusive" for row in unavailable)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["input"] == str(draw_path.resolve())
    assert metadata["output"] == str(output_path.resolve())
    assert metadata["preregistered"] is False
