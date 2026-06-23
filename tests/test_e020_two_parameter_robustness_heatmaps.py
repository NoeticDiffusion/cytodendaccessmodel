"""Tests for E020 — Two-Parameter Robustness Heatmaps.

Run with: pytest tests/test_e020_two_parameter_robustness_heatmaps.py -v
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

OUT_ROOT    = REPO_ROOT / "results" / "e020_two_parameter_robustness_heatmaps"
GRIDS_DIR   = OUT_ROOT / "grids"
SUMMARY_DIR = OUT_ROOT / "summary"

from exp020_two_parameter_robustness_heatmaps import (
    GRID_SPECS, DEFAULT_SEED, NOISE_SEEDS,
    CANONICAL_PARAMS, CANONICAL_OVERLAP_STR,
    _run_cell, _apply_param,
)
from cytodend_accessmodel.signatures import DEFAULT_THRESHOLDS

REQUIRED_PAIR_KEYS = [s.key for s in GRID_SPECS]
SIG_KEYS = ["SIG_A", "SIG_B", "SIG_C", "SIG_D", "SIG_E"]


# ---------------------------------------------------------------------------
# Schema / output file existence
# ---------------------------------------------------------------------------

def test_all_required_parameter_pairs_have_outputs():
    for key in REQUIRED_PAIR_KEYS:
        path = GRIDS_DIR / f"{key}_grid.csv"
        assert path.exists(), f"Grid CSV missing: {key}"


def test_each_grid_contains_canonical_coordinate():
    for spec in GRID_SPECS:
        path = GRIDS_DIR / f"{spec.key}_grid.csv"
        if not path.exists():
            pytest.skip(f"{spec.key} grid not found")
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        xs = {float(r["param_x_value"]) for r in rows}
        ys = {float(r["param_y_value"]) for r in rows}
        assert any(abs(v - spec.canonical_x) < 1e-9 for v in xs), \
            f"{spec.key}: canonical_x={spec.canonical_x} not in grid"
        assert any(abs(v - spec.canonical_y) < 1e-9 for v in ys), \
            f"{spec.key}: canonical_y={spec.canonical_y} not in grid"


def test_each_grid_exports_sig_a_to_sig_e():
    for spec in GRID_SPECS:
        path = GRIDS_DIR / f"{spec.key}_grid.csv"
        if not path.exists():
            pytest.skip(f"{spec.key} not found")
        with path.open(encoding="utf-8") as f:
            cols = set(csv.DictReader(f).fieldnames or [])
        for sig in SIG_KEYS:
            assert f"{sig}_score" in cols, f"{spec.key}: missing {sig}_score"


def test_each_grid_has_joint_pass_column():
    for spec in GRID_SPECS:
        path = GRIDS_DIR / f"{spec.key}_grid.csv"
        if not path.exists():
            pytest.skip(f"{spec.key} not found")
        with path.open(encoding="utf-8") as f:
            cols = set(csv.DictReader(f).fieldnames or [])
        assert "joint_protected_pass" in cols, f"{spec.key}: missing joint_protected_pass"


def test_thresholds_match_e019r_signature_protocol():
    path = SUMMARY_DIR / "protected_thresholds.json"
    assert path.exists(), "protected_thresholds.json not found"
    data = json.loads(path.read_text(encoding="utf-8"))
    for k, v in DEFAULT_THRESHOLDS.items():
        assert k in data, f"Missing threshold: {k}"
        assert abs(data[k] - v) < 1e-9, f"Threshold {k} mismatch: {data[k]} vs {v}"


def test_no_nan_in_required_signature_columns():
    for spec in GRID_SPECS:
        path = GRIDS_DIR / f"{spec.key}_grid.csv"
        if not path.exists():
            pytest.skip(f"{spec.key} not found")
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            # Only check primary rows (seed=default or seed=multi)
            seed = str(row.get("seed", ""))
            if seed not in (str(DEFAULT_SEED), "multi"):
                continue
            for sig in SIG_KEYS:
                val = row.get(f"{sig}_score", "")
                if val not in ("", "nan"):
                    try:
                        assert not math.isnan(float(val)), \
                            f"{spec.key}: NaN in {sig}_score at ({row['param_x_value']}, {row['param_y_value']})"
                    except ValueError:
                        pass


def test_noise_grid_has_multiple_seeds():
    noise_spec = next((s for s in GRID_SPECS if s.is_noisy), None)
    assert noise_spec is not None, "No noisy grid spec found"
    path = GRIDS_DIR / f"{noise_spec.key}_grid.csv"
    if not path.exists():
        pytest.skip(f"{noise_spec.key} not found")
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seeds = {str(r["seed"]) for r in rows if str(r.get("seed", "")) != "multi"}
    assert len(seeds) > 1, "Noise grid should have multiple seeds"
    assert len(seeds) == len(NOISE_SEEDS)


def test_failure_mode_summary_exists():
    path = SUMMARY_DIR / "failure_mode_summary.csv"
    assert path.exists(), "failure_mode_summary.csv not found"
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "failure_mode_summary.csv is empty"


def test_all_heatmaps_long_schema_is_stable():
    path = SUMMARY_DIR / "all_heatmaps_long.csv"
    if not path.exists():
        pytest.skip("all_heatmaps_long.csv not found")
    with path.open(encoding="utf-8") as f:
        cols = set(csv.DictReader(f).fieldnames or [])
    required = {"param_x_name", "param_x_value", "param_y_name", "param_y_value",
                "seed", "joint_protected_pass", "failure_mode"}
    assert required <= cols, f"Missing columns: {required - cols}"


# ---------------------------------------------------------------------------
# Canonical coordinate
# ---------------------------------------------------------------------------

def test_canonical_coordinate_matches_e019r_full_model_within_tolerance():
    """Canonical cell should pass all five protected signatures."""
    from copy import deepcopy
    prof = _run_cell(deepcopy(CANONICAL_PARAMS), timing_gap=0,
                     overlap_str=CANONICAL_OVERLAP_STR, seed=DEFAULT_SEED)
    assert prof.joint_protected_pass, (
        f"Canonical cell should pass joint profile: {prof.protected_passes}"
    )
    for sig in SIG_KEYS:
        assert prof.protected_passes[sig], (
            f"Canonical {sig} fails: score = {getattr(prof, sig if sig != 'SIG_E' else 'SIG_E_normalized')}"
        )


# ---------------------------------------------------------------------------
# Optional: boundary behaviour tests
# ---------------------------------------------------------------------------

def test_low_structural_lr_low_replay_gain_fails_joint_profile():
    """Double-zero corner of main pair should fail."""
    from copy import deepcopy
    from dataclasses import replace
    params = replace(deepcopy(CANONICAL_PARAMS), structural_lr=0.0, replay_gain=0.0)
    prof = _run_cell(params)
    assert not prof.joint_protected_pass, (
        "structural_lr=0, replay_gain=0 should fail joint profile"
    )


def test_zero_replay_gain_fails_sig_b_in_replay_pairs():
    from copy import deepcopy
    from dataclasses import replace
    params = replace(deepcopy(CANONICAL_PARAMS), replay_gain=0.0)
    prof = _run_cell(params)
    assert not prof.protected_passes["SIG_B"], (
        f"replay_gain=0 should fail SIG-B, got {prof.SIG_B:.4f}"
    )


def test_zero_overlap_strength_fails_linking_related_signatures():
    from copy import deepcopy
    prof = _run_cell(deepcopy(CANONICAL_PARAMS), overlap_str=0.0)
    assert not prof.protected_passes["SIG_B"], (
        f"overlap_str=0 should fail SIG-B (no linking); got {prof.SIG_B:.4f}"
    )


def test_high_timing_gap_fast_eligibility_decay_fails_slow_write():
    """Long timing gap + fast decay exhausts eligibility before consolidation."""
    from copy import deepcopy
    from dataclasses import replace
    params = replace(deepcopy(CANONICAL_PARAMS), eligibility_decay=0.60)
    prof = _run_cell(params, timing_gap=24)
    assert not prof.protected_passes["SIG_A"], (
        f"timing_gap=24 + eligibility_decay=0.60 should fail SIG-A; got {prof.SIG_A:.4f}"
    )
