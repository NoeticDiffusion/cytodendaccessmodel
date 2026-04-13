"""Tests for dandi_analysis.dataset_000718.nulls — no NWB files required."""
from __future__ import annotations

import numpy as np
import pytest

from dandi_analysis.contracts import ActivityMatrix


def _make_matrix(data: np.ndarray, session_id: str = "s1") -> ActivityMatrix:
    T, N = data.shape
    timestamps = np.linspace(0.0, T * 0.1, T)
    unit_ids = tuple(str(i) for i in range(N))
    return ActivityMatrix(
        session_id=session_id,
        data=data,
        unit_ids=unit_ids,
        timestamps=timestamps,
        sampling_rate=10.0,
    )


# ---------------------------------------------------------------------------
# circular_time_shift
# ---------------------------------------------------------------------------

def test_circular_shift_preserves_shape():
    from dandi_analysis.dataset_000718.nulls import circular_time_shift
    data = np.random.default_rng(0).standard_normal((50, 4))
    mat = _make_matrix(data)
    shifted = circular_time_shift(mat, shift=10)
    assert shifted.data.shape == data.shape


def test_circular_shift_changes_values():
    from dandi_analysis.dataset_000718.nulls import circular_time_shift
    data = np.arange(20, dtype=float).reshape(20, 1)
    mat = _make_matrix(data)
    shifted = circular_time_shift(mat, shift=5)
    assert not np.array_equal(shifted.data, data)


def test_circular_shift_zero_is_identity():
    from dandi_analysis.dataset_000718.nulls import circular_time_shift
    data = np.random.default_rng(1).standard_normal((30, 3))
    mat = _make_matrix(data)
    shifted = circular_time_shift(mat, shift=0)
    assert np.array_equal(shifted.data, data)


def test_circular_shift_does_not_return_original_object():
    from dandi_analysis.dataset_000718.nulls import circular_time_shift
    data = np.ones((10, 2))
    mat = _make_matrix(data)
    shifted = circular_time_shift(mat, shift=3)
    assert shifted is not mat


def test_circular_shift_preserves_unit_ids():
    from dandi_analysis.dataset_000718.nulls import circular_time_shift
    data = np.ones((10, 3))
    mat = _make_matrix(data)
    shifted = circular_time_shift(mat, shift=2)
    assert shifted.unit_ids == mat.unit_ids


# ---------------------------------------------------------------------------
# unit_label_permutation
# ---------------------------------------------------------------------------

def test_unit_permutation_preserves_shape():
    from dandi_analysis.dataset_000718.nulls import unit_label_permutation
    rng = np.random.default_rng(42)
    data = rng.standard_normal((40, 5))
    mat = _make_matrix(data)
    perm = unit_label_permutation(mat, rng)
    assert perm.data.shape == data.shape


def test_unit_permutation_preserves_values_as_set():
    from dandi_analysis.dataset_000718.nulls import unit_label_permutation
    rng = np.random.default_rng(42)
    data = np.arange(20, dtype=float).reshape(4, 5)
    mat = _make_matrix(data)
    perm = unit_label_permutation(mat, rng)
    # The column-wise sorted data should match (same columns in different order)
    orig_sorted = np.sort(data, axis=1)
    perm_sorted = np.sort(np.array(perm.data), axis=1)
    assert np.allclose(orig_sorted, perm_sorted)


def test_unit_permutation_deterministic_with_fixed_seed():
    from dandi_analysis.dataset_000718.nulls import unit_label_permutation
    data = np.random.default_rng(0).standard_normal((20, 6))
    mat = _make_matrix(data)
    perm1 = unit_label_permutation(mat, np.random.default_rng(7))
    perm2 = unit_label_permutation(mat, np.random.default_rng(7))
    assert np.array_equal(perm1.data, perm2.data)


# ---------------------------------------------------------------------------
# matched_count_shuffle
# ---------------------------------------------------------------------------

def test_matched_count_preserves_shape():
    from dandi_analysis.dataset_000718.nulls import matched_count_shuffle
    rng = np.random.default_rng(0)
    data = rng.standard_normal((50, 4))
    mat = _make_matrix(data)
    shuffled = matched_count_shuffle(mat, rng)
    assert shuffled.data.shape == data.shape


def test_matched_count_preserves_column_sum():
    from dandi_analysis.dataset_000718.nulls import matched_count_shuffle
    rng = np.random.default_rng(99)
    data = np.abs(rng.standard_normal((30, 4)))
    mat = _make_matrix(data)
    shuffled = matched_count_shuffle(mat, np.random.default_rng(99))
    assert np.allclose(
        np.sort(data, axis=0),
        np.sort(np.array(shuffled.data), axis=0),
    )


def test_matched_count_deterministic_with_fixed_seed():
    from dandi_analysis.dataset_000718.nulls import matched_count_shuffle
    data = np.random.default_rng(5).standard_normal((15, 3))
    mat = _make_matrix(data)
    s1 = matched_count_shuffle(mat, np.random.default_rng(11))
    s2 = matched_count_shuffle(mat, np.random.default_rng(11))
    assert np.array_equal(s1.data, s2.data)
