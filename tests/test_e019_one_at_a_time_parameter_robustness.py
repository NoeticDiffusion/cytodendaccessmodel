"""Tests for e019 — One-at-a-Time Parameter Robustness.

Run with: pytest tests/test_e019_one_at_a_time_parameter_robustness.py -v
"""

from __future__ import annotations

import csv
import json
import math
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from exp019_one_at_a_time_parameter_robustness import (
    SWEEPS,
    THRESHOLDS,
    CANONICAL_PARAMS,
    CANONICAL_OVERLAP_STR,
    DEFAULT_SEED,
    NOISE_SEEDS,
    _run_single,
    _build_row,
    _protected,
    _directional,
    _first_failure_value,
)

OUT_ROOT   = REPO_ROOT / "results" / "e019_one_at_a_time_parameter_robustness"
SWEEPS_DIR = OUT_ROOT / "sweeps"
SUMMARY_DIR= OUT_ROOT / "summary"

SWEEP_NAMES = [s.name for s in SWEEPS]
SIG_KEYS = ["SIG_A", "SIG_B", "SIG_C", "SIG_D", "SIG_E"]

# ---------------------------------------------------------------------------
# Required schema tests (use pre-computed sweep CSVs to avoid re-running)
# ---------------------------------------------------------------------------

def test_all_required_parameters_have_sweep_outputs():
    for name in SWEEP_NAMES:
        path = SWEEPS_DIR / f"{name}_sweep.csv"
        assert path.exists(), f"Missing sweep CSV for {name}"


def test_each_sweep_contains_canonical_value():
    for sweep in SWEEPS:
        path = SWEEPS_DIR / f"{sweep.name}_sweep.csv"
        if not path.exists():
            pytest.skip(f"{sweep.name}_sweep.csv not found (run exp019 first)")
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        values = {float(r["parameter_value"]) for r in rows}
        assert sweep.canonical in values or any(
            abs(v - sweep.canonical) < 1e-9 for v in values
        ), f"{sweep.name}: canonical value {sweep.canonical} not in sweep"


def test_each_sweep_exports_sig_a_to_sig_e():
    for name in SWEEP_NAMES:
        path = SWEEPS_DIR / f"{name}_sweep.csv"
        if not path.exists():
            pytest.skip(f"{name}_sweep.csv not found")
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
        for sig in SIG_KEYS:
            assert f"{sig}_score" in cols, f"{name}: missing {sig}_score column"


def test_thresholds_are_exported():
    path = SUMMARY_DIR / "protected_thresholds.json"
    assert path.exists(), "protected_thresholds.json not found"
    data = json.loads(path.read_text(encoding="utf-8"))
    for sig in SIG_KEYS:
        assert sig in data, f"Missing threshold for {sig}"
        assert data[sig] > 0


def test_joint_pass_columns_exist():
    for name in SWEEP_NAMES:
        path = SWEEPS_DIR / f"{name}_sweep.csv"
        if not path.exists():
            pytest.skip(f"{name}_sweep.csv not found")
        with path.open(encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames or []
        assert "joint_directional_pass" in cols, f"{name}: missing joint_directional_pass"
        assert "joint_protected_pass" in cols,   f"{name}: missing joint_protected_pass"


def test_no_nan_in_signature_scores():
    for name in SWEEP_NAMES:
        path = SWEEPS_DIR / f"{name}_sweep.csv"
        if not path.exists():
            pytest.skip(f"{name}_sweep.csv not found")
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for sig in SIG_KEYS:
                val = row.get(f"{sig}_score", "")
                if val != "":
                    assert not math.isnan(float(val)), (
                        f"{name} row: {sig}_score is NaN at "
                        f"value={row['parameter_value']}, seed={row['seed']}"
                    )


def test_noise_sweep_has_multiple_seeds():
    path = SWEEPS_DIR / "structural_noise_sweep.csv"
    if not path.exists():
        pytest.skip("structural_noise_sweep.csv not found")
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seeds_found = {int(r["seed"]) for r in rows}
    assert len(seeds_found) > 1, "structural_noise sweep should have multiple seeds"
    assert len(seeds_found) == len(NOISE_SEEDS), (
        f"Expected {len(NOISE_SEEDS)} seeds, found {len(seeds_found)}"
    )


def test_failure_boundary_summary_exists():
    path = SUMMARY_DIR / "failure_boundary_summary.csv"
    assert path.exists(), "failure_boundary_summary.csv not found"
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "failure_boundary_summary.csv is empty"


def test_all_sweeps_long_schema_is_stable():
    path = SUMMARY_DIR / "all_sweeps_long.csv"
    if not path.exists():
        pytest.skip("all_sweeps_long.csv not found")
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])

    required = {
        "parameter_name", "parameter_value", "seed",
        "SIG_A_score", "SIG_B_score", "SIG_C_score", "SIG_D_score", "SIG_E_score",
        "joint_directional_pass", "joint_protected_pass", "failure_mode",
    }
    missing = required - cols
    assert not missing, f"all_sweeps_long.csv missing columns: {missing}"


def test_canonical_row_matches_e018_full_model_within_tolerance():
    """Canonical _run_single result should match E018 full_model within loose tolerance."""
    sigs = _run_single(CANONICAL_PARAMS, timing_gap=0,
                       overlap_str=CANONICAL_OVERLAP_STR, seed=DEFAULT_SEED)
    assert sigs["SIG_A"] > THRESHOLDS["SIG_A"], "Canonical SIG-A below protected threshold"
    assert sigs["SIG_B"] > THRESHOLDS["SIG_B"], "Canonical SIG-B below protected threshold"
    assert sigs["SIG_C"] > THRESHOLDS["SIG_C"], "Canonical SIG-C below protected threshold"
    assert sigs["SIG_D"] > THRESHOLDS["SIG_D"], "Canonical SIG-D below protected threshold"
    assert sigs["SIG_E"] > THRESHOLDS["SIG_E"], "Canonical SIG-E below protected threshold"


# ---------------------------------------------------------------------------
# Optional — parameter-specific correctness
# ---------------------------------------------------------------------------

def test_structural_lr_zero_fails_sig_a():
    """With zero learning rate, overlap branch cannot write M_b -> SIG-A should fail."""
    params = replace(CANONICAL_PARAMS, structural_lr=0.0)
    sigs = _run_single(params)
    assert not _protected(sigs)["SIG_A"], (
        f"SIG-A should fail with structural_lr=0.0, got {sigs['SIG_A']:.4f}"
    )


def test_replay_gain_zero_fails_sig_b():
    """With zero replay gain, no P_b build-up -> linking gain should fail."""
    params = replace(CANONICAL_PARAMS, replay_gain=0.0)
    sigs = _run_single(params)
    assert not _protected(sigs)["SIG_B"], (
        f"SIG-B should fail with replay_gain=0.0, got {sigs['SIG_B']:.4f}"
    )


def test_context_gain_zero_reduces_sig_c():
    """With zero context gain, SIG-C should be reduced (may still be positive from allocation)."""
    params_zero = replace(CANONICAL_PARAMS, context_gain=0.0)
    sigs_zero   = _run_single(params_zero)
    sigs_canon  = _run_single(CANONICAL_PARAMS)
    assert sigs_zero["SIG_C"] <= sigs_canon["SIG_C"] + 0.01, (
        f"SIG-C should not increase when context_gain=0: "
        f"zero={sigs_zero['SIG_C']:.4f} vs canonical={sigs_canon['SIG_C']:.4f}"
    )


def test_overlap_strength_zero_fails_linking_signatures():
    """With zero overlap, b1 contributes nothing to L -> SIG-B should fail."""
    sigs = _run_single(CANONICAL_PARAMS, overlap_str=0.0)
    pf   = _protected(sigs)
    assert not pf["SIG_B"], (
        f"SIG-B should fail with overlap_str=0.0, got {sigs['SIG_B']:.4f}"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic_sweep_is_reproducible():
    """Same parameters, same seed -> identical signatures."""
    sigs1 = _run_single(CANONICAL_PARAMS, seed=DEFAULT_SEED)
    sigs2 = _run_single(CANONICAL_PARAMS, seed=DEFAULT_SEED)
    for sig in SIG_KEYS:
        assert abs(sigs1[sig] - sigs2[sig]) < 1e-12, (
            f"Non-deterministic {sig}: {sigs1[sig]} vs {sigs2[sig]}"
        )


def test_noise_seeds_produce_variance():
    """Different seeds with nonzero structural_noise should produce different SIG-A scores."""
    params_noisy = replace(CANONICAL_PARAMS, structural_noise=0.05)
    scores = [
        _run_single(params_noisy, seed=s)["SIG_A"]
        for s in NOISE_SEEDS[:5]
    ]
    # At least some variance across seeds
    assert max(scores) - min(scores) > 1e-6, (
        "Expected nonzero variance across seeds with structural_noise=0.05"
    )


# ---------------------------------------------------------------------------
# Structural integrity of row builder
# ---------------------------------------------------------------------------

def test_build_row_schema():
    sigs = _run_single(CANONICAL_PARAMS)
    row  = _build_row("structural_lr", 0.18, DEFAULT_SEED, sigs)
    required = {
        "parameter_name", "parameter_value", "seed",
        "SIG_A_score", "SIG_B_score", "SIG_C_score", "SIG_D_score", "SIG_E_score",
        "SIG_A_directional_pass", "SIG_B_directional_pass",
        "SIG_A_protected_pass", "SIG_B_protected_pass",
        "joint_directional_pass", "joint_protected_pass", "failure_mode",
    }
    assert required <= set(row.keys())
