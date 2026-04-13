"""Tests for dandi_analysis.dataset_001710.nulls — no NWB files required."""
from __future__ import annotations

import numpy as np
import pytest

from dandi_analysis.dataset_001710.nulls import (
    arm_label_shuffle,
    circular_time_shift,
    generate_null_distribution,
    position_bin_shuffle,
    trial_label_shuffle,
)


# ---------------------------------------------------------------------------
# circular_time_shift
# ---------------------------------------------------------------------------

def test_circular_time_shift_shape_preserved():
    activity = np.random.default_rng(0).standard_normal((100, 10))
    shifted = circular_time_shift(activity, seed=0)
    assert shifted.shape == activity.shape


def test_circular_time_shift_is_rotation():
    # np.roll(a, shift, axis=0) puts the last `shift` rows at the front.
    # shifted[:shift]  == activity[-shift:]
    # shifted[shift:]  == activity[:-shift]
    activity = np.arange(20).reshape(10, 2).astype(float)
    shifted = circular_time_shift(activity, shift=3)
    np.testing.assert_array_equal(shifted[:3], activity[-3:])
    np.testing.assert_array_equal(shifted[3:], activity[:-3])


def test_circular_time_shift_actually_changes_data():
    activity = np.random.default_rng(1).standard_normal((50, 5))
    shifted = circular_time_shift(activity, seed=1)
    assert not np.allclose(activity, shifted)


def test_circular_time_shift_deterministic():
    activity = np.random.default_rng(2).standard_normal((60, 8))
    s1 = circular_time_shift(activity, seed=42)
    s2 = circular_time_shift(activity, seed=42)
    np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# position_bin_shuffle
# ---------------------------------------------------------------------------

def test_position_bin_shuffle_shape_preserved():
    n = 100
    activity = np.random.default_rng(0).standard_normal((n, 5))
    pos = np.linspace(0, 200, n)
    sh_act, sh_pos = position_bin_shuffle(activity, pos, seed=0)
    assert sh_act.shape == activity.shape
    assert sh_pos.shape == pos.shape


def test_position_bin_shuffle_marginals_preserved():
    """Shuffled arrays should have same set of values as originals."""
    n = 100
    activity = np.random.default_rng(3).standard_normal((n, 3))
    pos = np.linspace(0, 200, n)
    sh_act, sh_pos = position_bin_shuffle(activity, pos, seed=3)
    np.testing.assert_array_equal(np.sort(sh_act.ravel()), np.sort(activity.ravel()))
    np.testing.assert_array_equal(np.sort(sh_pos), np.sort(pos))


def test_position_bin_shuffle_deterministic():
    n = 80
    activity = np.random.default_rng(5).standard_normal((n, 4))
    pos = np.linspace(0, 100, n)
    r1 = position_bin_shuffle(activity, pos, seed=7)
    r2 = position_bin_shuffle(activity, pos, seed=7)
    np.testing.assert_array_equal(r1[0], r2[0])
    np.testing.assert_array_equal(r1[1], r2[1])


# ---------------------------------------------------------------------------
# trial_label_shuffle / arm_label_shuffle
# ---------------------------------------------------------------------------

def test_trial_label_shuffle_length():
    labels = ["left", "right", "left", "right", "left"]
    shuffled = trial_label_shuffle(labels, seed=0)
    assert len(shuffled) == len(labels)


def test_trial_label_shuffle_same_elements():
    labels = ["A", "B", "A", "C", "B", "A"]
    shuffled = trial_label_shuffle(labels, seed=0)
    assert sorted(shuffled) == sorted(labels)


def test_trial_label_shuffle_deterministic():
    labels = ["left", "right"] * 5
    s1 = trial_label_shuffle(labels, seed=99)
    s2 = trial_label_shuffle(labels, seed=99)
    np.testing.assert_array_equal(s1, s2)


def test_trial_label_shuffle_actually_shuffles():
    labels = list(range(20))
    shuffled = trial_label_shuffle(labels, seed=1)
    # With 20 elements, it is extremely unlikely a permutation is identity
    assert not np.array_equal(np.array(labels), shuffled)


def test_arm_label_shuffle_same_as_trial_shuffle():
    """arm_label_shuffle should behave identically to trial_label_shuffle."""
    labels = ["left", "right", "left", "right"]
    a = arm_label_shuffle(labels, seed=42)
    b = trial_label_shuffle(labels, seed=42)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# generate_null_distribution
# ---------------------------------------------------------------------------

def test_generate_null_distribution_length():
    def dummy(seed: int) -> float:
        return float(seed) * 0.1

    nulls = generate_null_distribution(dummy, n_nulls=10, base_seed=0)
    assert nulls.shape == (10,)


def test_generate_null_distribution_values():
    def dummy(seed: int) -> float:
        return float(seed)

    nulls = generate_null_distribution(dummy, n_nulls=5, base_seed=0)
    np.testing.assert_array_equal(nulls, [0.0, 1.0, 2.0, 3.0, 4.0])


def test_generate_null_distribution_base_seed_offset():
    def dummy(seed: int) -> float:
        return float(seed)

    nulls = generate_null_distribution(dummy, n_nulls=3, base_seed=10)
    np.testing.assert_array_equal(nulls, [10.0, 11.0, 12.0])
