"""Tests for dandi_analysis.dataset_001710.placecode — synthetic data only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dandi_analysis.dataset_001710.behavior import BehaviorTable
from dandi_analysis.dataset_001710.ophys import OphysMatrix
from dandi_analysis.dataset_001710.placecode import (
    compute_tuning_curves,
    split_half_reliability,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ophys(n: int = 200, n_rois: int = 10, seed: int = 0) -> OphysMatrix:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, n_rois)).clip(0)
    return OphysMatrix(
        session_path=Path("synthetic.nwb"),
        signal="dff",
        data=data,
        timestamps=np.arange(n) / 15.0,
        roi_ids=tuple(range(n_rois)),
        sampling_rate=15.0,
        n_frames=n,
        n_rois=n_rois,
    )


def _make_behavior(n: int = 200, pos_min: float = 0.0, pos_max: float = 200.0) -> BehaviorTable:
    pos = np.linspace(pos_min, pos_max, n)
    return BehaviorTable(
        session_path=Path("synthetic.nwb"),
        source="2p",
        n_frames=n,
        timestamps=np.arange(n) / 15.0,
        channels={"position": pos},
    )


# ---------------------------------------------------------------------------
# compute_tuning_curves
# ---------------------------------------------------------------------------

def test_tuning_curves_shape():
    ophys = _make_ophys(n=200, n_rois=10)
    beh = _make_behavior(n=200)
    tc = compute_tuning_curves(ophys, beh, n_bins=20)
    assert tc.tuning_curves.shape == (10, 20)
    assert tc.n_rois == 10
    assert tc.n_bins == 20


def test_tuning_curves_occupancy_sums_to_n():
    n = 200
    ophys = _make_ophys(n=n, n_rois=5)
    beh = _make_behavior(n=n)
    tc = compute_tuning_curves(ophys, beh, n_bins=10)
    assert int(tc.occupancy.sum()) == n


def test_tuning_curves_missing_position_raises():
    ophys = _make_ophys(n=100)
    beh = BehaviorTable(
        session_path=Path("x.nwb"),
        source="2p",
        n_frames=100,
        timestamps=np.arange(100) / 15.0,
        channels={},  # no position
    )
    with pytest.raises(KeyError):
        compute_tuning_curves(ophys, beh)


def test_tuning_curves_min_occupancy_produces_nan():
    ophys = _make_ophys(n=20, n_rois=3)
    # All frames at position=0: np.digitize places them into the last bin edge
    # (edge case), so exactly one bin should be non-NaN.
    pos = np.zeros(20)
    beh = BehaviorTable(
        session_path=Path("x.nwb"),
        source="2p",
        n_frames=20,
        timestamps=np.arange(20) / 15.0,
        channels={"position": pos},
    )
    tc = compute_tuning_curves(ophys, beh, n_bins=10, min_occupancy_frames=2)
    # At most one bin should have non-NaN tuning (the bin that received all frames)
    non_nan_bins = np.sum(~np.isnan(tc.tuning_curves[0]))
    assert non_nan_bins <= 1


def test_tuning_curves_bin_edges_length():
    ophys = _make_ophys(n=100)
    beh = _make_behavior(n=100)
    tc = compute_tuning_curves(ophys, beh, n_bins=30)
    assert len(tc.bin_edges) == 31
    assert len(tc.bin_centers) == 30


# ---------------------------------------------------------------------------
# split_half_reliability
# ---------------------------------------------------------------------------

def test_split_half_reliability_shape():
    n_rois = 8
    ophys = _make_ophys(n=200, n_rois=n_rois, seed=42)
    beh = _make_behavior(n=200)
    rel = split_half_reliability(
        np.random.default_rng(0).standard_normal((n_rois, 20)),
        ophys,
        beh,
        n_splits=3,
    )
    assert rel.shape == (n_rois,)


def test_split_half_reliability_range():
    n_rois = 5
    ophys = _make_ophys(n=300, n_rois=n_rois, seed=7)
    beh = _make_behavior(n=300)
    tc = np.random.default_rng(7).standard_normal((n_rois, 30))
    rel = split_half_reliability(tc, ophys, beh, n_splits=5)
    finite = rel[np.isfinite(rel)]
    assert np.all(finite >= -1.0) and np.all(finite <= 1.0)


def test_split_half_reliability_deterministic():
    n_rois = 4
    ophys = _make_ophys(n=200, n_rois=n_rois, seed=1)
    beh = _make_behavior(n=200)
    tc = np.random.default_rng(1).standard_normal((n_rois, 20))
    rel1 = split_half_reliability(tc, ophys, beh, n_splits=3, seed=99)
    rel2 = split_half_reliability(tc, ophys, beh, n_splits=3, seed=99)
    np.testing.assert_array_equal(rel1, rel2)
