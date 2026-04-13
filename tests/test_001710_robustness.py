"""Tests for group-aware 001710 robustness helpers."""
from __future__ import annotations

import numpy as np

from dandi_analysis.dataset_001710.remapping import SimilarityMatrix
from dandi_analysis.dataset_001710.robustness import (
    aggregate_group_day_lag,
    group_null_tests,
)


def _sim_matrix(off_diag: float) -> SimilarityMatrix:
    mat = np.array(
        [
            [1.0, off_diag, off_diag],
            [off_diag, 1.0, off_diag],
            [off_diag, off_diag, 1.0],
        ],
        dtype=float,
    )
    return SimilarityMatrix(labels=["day0", "day1", "day2"], matrix=mat)


def test_group_null_tests_compare_subject_level_groups():
    subject_sim_matrices = {
        "SparseKO-1": _sim_matrix(0.10),
        "SparseKO-2": _sim_matrix(0.15),
        "Cre-1": _sim_matrix(0.45),
        "Cre-2": _sim_matrix(0.50),
        "Ctrl-1": _sim_matrix(0.30),
        "Ctrl-2": _sim_matrix(0.35),
    }
    subject_groups = {
        "SparseKO-1": "SparseKO",
        "SparseKO-2": "SparseKO",
        "Cre-1": "Cre",
        "Cre-2": "Cre",
        "Ctrl-1": "Ctrl",
        "Ctrl-2": "Ctrl",
    }

    results = group_null_tests(
        subject_sim_matrices,
        subject_groups,
        target_group="SparseKO",
        n_perms=200,
        seed=0,
    )

    by_name = {row["comparison"]: row for row in results}
    assert set(by_name) == {"SparseKO vs Cre", "SparseKO vs Ctrl"}
    assert by_name["SparseKO vs Cre"]["target_n_subjects"] == 2
    assert by_name["SparseKO vs Ctrl"]["other_n_subjects"] == 2
    assert by_name["SparseKO vs Cre"]["observed_diff"] < 0
    assert by_name["SparseKO vs Ctrl"]["observed_diff"] < 0


def test_aggregate_group_day_lag_averages_subject_means():
    subject_lag_data = {
        "SparseKO-1": {
            1: {"mean": 0.10},
            2: {"mean": 0.20},
            3: {"mean": float("nan")},
            4: {"mean": float("nan")},
            5: {"mean": float("nan")},
        },
        "SparseKO-2": {
            1: {"mean": 0.30},
            2: {"mean": float("nan")},
            3: {"mean": 0.05},
            4: {"mean": float("nan")},
            5: {"mean": float("nan")},
        },
        "Cre-1": {
            1: {"mean": 0.50},
            2: {"mean": 0.40},
            3: {"mean": 0.30},
            4: {"mean": float("nan")},
            5: {"mean": float("nan")},
        },
    }
    subject_groups = {
        "SparseKO-1": "SparseKO",
        "SparseKO-2": "SparseKO",
        "Cre-1": "Cre",
    }

    aggregated = aggregate_group_day_lag(subject_lag_data, subject_groups)

    assert aggregated["SparseKO"][1]["mean"] == 0.2
    assert aggregated["SparseKO"][1]["n_subjects"] == 2
    assert aggregated["SparseKO"][2]["mean"] == 0.2
    assert aggregated["SparseKO"][2]["n_subjects"] == 1
    assert aggregated["Cre"][1]["mean"] == 0.5
