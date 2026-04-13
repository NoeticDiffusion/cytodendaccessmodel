"""Tests for dandi_analysis.dataset_001710.trials — synthetic data only."""
from __future__ import annotations

import numpy as np
import pytest

from dandi_analysis.dataset_001710.behavior import BehaviorTable
from dandi_analysis.dataset_001710.trials import (
    TrialRow,
    TrialTable,
    _detect_rising_edges,
    _majority_arm,
    _pair_ends,
    _build_from_behavior_signals,
)


# ---------------------------------------------------------------------------
# _detect_rising_edges
# ---------------------------------------------------------------------------

def test_detect_rising_edges_basic():
    arr = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    edges = _detect_rising_edges(arr)
    assert list(edges) == [2, 5]


def test_detect_rising_edges_none_input():
    edges = _detect_rising_edges(None)
    assert len(edges) == 0


def test_detect_rising_edges_all_zeros():
    arr = np.zeros(10)
    edges = _detect_rising_edges(arr)
    assert len(edges) == 0


def test_detect_rising_edges_all_ones():
    arr = np.ones(10)
    edges = _detect_rising_edges(arr)
    assert len(edges) == 1  # one rising edge at the very start


# ---------------------------------------------------------------------------
# _pair_ends
# ---------------------------------------------------------------------------

def test_pair_ends_normal():
    starts = np.array([0, 5, 10])
    ends = np.array([3, 8, 14])
    paired = _pair_ends(starts, ends, n_total=20)
    assert list(paired) == [3, 8, 14]


def test_pair_ends_missing_end_for_last_trial():
    starts = np.array([0, 5])
    ends = np.array([3])  # no end after start=5
    paired = _pair_ends(starts, ends, n_total=20)
    assert paired[0] == 3
    assert paired[1] == 19  # clamped to n_total - 1


def test_pair_ends_empty_ends():
    starts = np.array([0, 5])
    ends = np.array([], dtype=int)
    paired = _pair_ends(starts, ends, n_total=30)
    assert len(paired) == 2
    assert all(p == 29 for p in paired)


# ---------------------------------------------------------------------------
# _majority_arm
# ---------------------------------------------------------------------------

def test_majority_arm_left():
    arr = np.array([0.0, 0.0, 0.0, 0.1])
    assert _majority_arm(arr, 0, 4) == "left"


def test_majority_arm_right():
    arr = np.array([0.9, 1.0, 1.0, 1.0])
    assert _majority_arm(arr, 0, 4) == "right"


def test_majority_arm_none():
    assert _majority_arm(None, 0, 10) == "unknown"


def test_majority_arm_empty_window():
    arr = np.array([0.0, 1.0, 0.0])
    assert _majority_arm(arr, 2, 2) == "unknown"


# ---------------------------------------------------------------------------
# _build_from_behavior_signals (full synthetic BehaviorTable)
# ---------------------------------------------------------------------------

def _make_behavior(
    n: int = 200,
    *,
    n_trials: int = 5,
    arm_pattern: str = "alternating",
) -> BehaviorTable:
    """Build a synthetic BehaviorTable with n_trials embedded."""
    from pathlib import Path

    ts = np.arange(n) / 15.0  # 15 Hz

    trial_start = np.zeros(n)
    trial_end = np.zeros(n)
    trial_number = np.zeros(n)
    arm = np.zeros(n)
    block = np.zeros(n)
    reward = np.zeros(n)
    position = np.linspace(0, 200, n)

    trial_len = n // (n_trials + 1)
    for i in range(n_trials):
        s = i * trial_len + 5
        e = s + trial_len - 5
        if s >= n or e >= n:
            break
        trial_start[s] = 1.0
        trial_end[e] = 1.0
        trial_number[s:e] = float(i + 1)
        arm[s:e] = 0.0 if (i % 2 == 0 and arm_pattern == "alternating") else 1.0
        block[s:e] = float(i // 2 + 1)
        reward[e - 2 : e] = 1.0

    return BehaviorTable(
        session_path=Path("synthetic.nwb"),
        source="2p",
        n_frames=n,
        timestamps=ts,
        channels={
            "trial_start": trial_start,
            "trial_end": trial_end,
            "trial_number": trial_number,
            "arm": arm,
            "block": block,
            "reward": reward,
            "position": position,
        },
    )


def test_build_from_signals_returns_correct_count():
    beh = _make_behavior(n=300, n_trials=6)
    rows = _build_from_behavior_signals(beh, {}, day=0, min_duration_sec=0.0)
    assert len(rows) == 6


def test_build_from_signals_valid_flags():
    beh = _make_behavior(n=300, n_trials=4)
    rows = _build_from_behavior_signals(beh, {}, day=0, min_duration_sec=0.5)
    for row in rows:
        assert row.valid or row.duration_sec < 0.5


def test_build_from_signals_arm_propagation():
    beh = _make_behavior(n=300, n_trials=4, arm_pattern="alternating")
    rows = _build_from_behavior_signals(beh, {}, day=0, min_duration_sec=0.0)
    # Even-indexed trials have arm=0 → "left"
    # Arm alternates, first trial should be "left"
    assert rows[0].arm_label == "left"


def test_build_from_signals_empty_input():
    from pathlib import Path
    beh_empty = BehaviorTable(
        session_path=Path("empty.nwb"),
        source="2p",
        n_frames=0,
        timestamps=np.array([]),
        channels={},
    )
    rows = _build_from_behavior_signals(beh_empty, {}, day=0, min_duration_sec=0.0)
    assert rows == []


def test_trial_table_by_arm():
    beh = _make_behavior(n=400, n_trials=6, arm_pattern="alternating")
    rows = _build_from_behavior_signals(beh, {}, day=1, min_duration_sec=0.0)
    table = TrialTable(
        session_path=beh.session_path, day=1, trials=rows
    )
    left = table.by_arm("left")
    right = table.by_arm("right")
    assert len(left) + len(right) >= 4  # at least some arm-labelled trials


def test_trial_table_valid_trials_subset():
    beh = _make_behavior(n=200, n_trials=3)
    rows = _build_from_behavior_signals(beh, {}, day=0, min_duration_sec=999.0)
    table = TrialTable(session_path=beh.session_path, day=0, trials=rows)
    assert all(not t.valid for t in table.valid_trials() if t.duration_sec < 999.0)
    assert len(table.valid_trials()) == 0
