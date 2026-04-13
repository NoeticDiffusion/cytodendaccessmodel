"""Tests for dandi_analysis.dataset_000718.observables — no NWB files required."""
from __future__ import annotations

import numpy as np
import pytest

from dandi_analysis.contracts import ActivityMatrix, OfflineWindow


def _make_matrix(
    data: np.ndarray,
    session_id: str = "s1",
    unit_ids: tuple[str, ...] | None = None,
) -> ActivityMatrix:
    T, N = data.shape
    timestamps = np.linspace(0.0, T * 0.1, T)
    if unit_ids is None:
        unit_ids = tuple(str(i) for i in range(N))
    return ActivityMatrix(
        session_id=session_id,
        data=data,
        unit_ids=unit_ids,
        timestamps=timestamps,
        sampling_rate=10.0,
    )


def _dummy_window(session_id: str = "s1") -> OfflineWindow:
    return OfflineWindow(
        session_id=session_id,
        label="test_window",
        start_sec=0.0,
        stop_sec=5.0,
        epoch_type="rest",
    )


# ---------------------------------------------------------------------------
# pairwise_coactivity_matrix
# ---------------------------------------------------------------------------

def test_pairwise_matrix_shape():
    from dandi_analysis.dataset_000718.observables import pairwise_coactivity_matrix
    data = np.random.default_rng(0).standard_normal((100, 5))
    mat = _make_matrix(data)
    corr = pairwise_coactivity_matrix(mat)
    assert corr.shape == (5, 5)


def test_pairwise_matrix_is_symmetric():
    from dandi_analysis.dataset_000718.observables import pairwise_coactivity_matrix
    data = np.random.default_rng(1).standard_normal((80, 4))
    corr = pairwise_coactivity_matrix(_make_matrix(data))
    non_nan = ~np.isnan(corr)
    assert np.allclose(corr[non_nan], corr.T[non_nan])


def test_pairwise_matrix_diagonal_nan():
    from dandi_analysis.dataset_000718.observables import pairwise_coactivity_matrix
    data = np.random.default_rng(2).standard_normal((50, 3))
    corr = pairwise_coactivity_matrix(_make_matrix(data))
    assert all(np.isnan(corr[i, i]) for i in range(3))


def test_pairwise_matrix_known_perfect_correlation():
    from dandi_analysis.dataset_000718.observables import pairwise_coactivity_matrix
    t = np.linspace(0, 10, 200)
    col = np.sin(t)
    data = np.column_stack([col, col])  # two identical traces
    corr = pairwise_coactivity_matrix(_make_matrix(data))
    assert corr[0, 1] == pytest.approx(1.0, abs=1e-10)


def test_pairwise_matrix_known_negative_correlation():
    from dandi_analysis.dataset_000718.observables import pairwise_coactivity_matrix
    t = np.linspace(0, 10, 200)
    col = np.sin(t)
    data = np.column_stack([col, -col])
    corr = pairwise_coactivity_matrix(_make_matrix(data))
    assert corr[0, 1] == pytest.approx(-1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# offline_coreactivation_score
# ---------------------------------------------------------------------------

def test_coreactivation_returns_correct_type():
    from dandi_analysis.dataset_000718.observables import offline_coreactivation_score
    from dandi_analysis.contracts import PairwiseCoreactivationResult
    data = np.random.default_rng(0).standard_normal((50, 2))
    mat = _make_matrix(data, unit_ids=("u0", "u1"))
    result = offline_coreactivation_score(mat, _dummy_window(), "u0", "u1")
    assert isinstance(result, PairwiseCoreactivationResult)


def test_coreactivation_perfect_correlation():
    from dandi_analysis.dataset_000718.observables import offline_coreactivation_score
    t = np.linspace(0, 5, 100)
    col = np.sin(t)
    data = np.column_stack([col, col])
    mat = _make_matrix(data, unit_ids=("a", "b"))
    result = offline_coreactivation_score(mat, _dummy_window(), "a", "b")
    assert result.co_reactivation_score == pytest.approx(1.0, abs=1e-10)


def test_coreactivation_missing_unit_id_returns_nan():
    from dandi_analysis.dataset_000718.observables import offline_coreactivation_score
    data = np.random.default_rng(0).standard_normal((30, 2))
    mat = _make_matrix(data, unit_ids=("x", "y"))
    result = offline_coreactivation_score(mat, _dummy_window(), "x", "MISSING")
    assert np.isnan(result.co_reactivation_score)


def test_coreactivation_with_null_matrices():
    from dandi_analysis.dataset_000718.observables import offline_coreactivation_score
    from dandi_analysis.dataset_000718.nulls import circular_time_shift
    rng = np.random.default_rng(42)
    data = rng.standard_normal((100, 2))
    mat = _make_matrix(data, unit_ids=("u0", "u1"))
    nulls = [circular_time_shift(mat, shift=s) for s in range(5, 50, 5)]
    result = offline_coreactivation_score(mat, _dummy_window(), "u0", "u1", null_matrices=nulls)
    assert not np.isnan(result.null_mean)
    assert not np.isnan(result.null_std)
