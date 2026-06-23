"""Tests for e018 — Comparator Trace Matrix.

Validates that the comparator matrix is correctly constructed, that the full
model passes the joint profile, and that simpler baselines fail as expected.

Run with: pytest tests/test_e018_comparator_trace_matrix.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from exp018_comparator_trace_matrix import (
    COMPARATOR_NAMES,
    COMPARATORS,
    THRESHOLDS,
    RANDOM_SEED,
    BRANCH_IDS,
    OVERLAP_BRANCH,
    run_comparator,
    _passes,
    _joint_pass,
)

# ---------------------------------------------------------------------------
# Module-level fixture: run all comparators once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_results():
    """Run all comparators and return result dicts."""
    results = {}
    for spec in COMPARATORS:
        branch_rows, support_rows, linking_rows, ctx_rows, sigs = run_comparator(spec)
        pf = _passes(sigs)
        results[spec.name] = {
            "branch_rows":  branch_rows,
            "support_rows": support_rows,
            "linking_rows": linking_rows,
            "ctx_rows":     ctx_rows,
            "sigs":         sigs,
            "pf":           pf,
        }
    return results


# ---------------------------------------------------------------------------
# Structural / required tests
# ---------------------------------------------------------------------------

def test_all_required_comparators_run(all_results):
    assert set(all_results.keys()) == set(COMPARATOR_NAMES)


def test_each_comparator_exports_trace_files(all_results):
    for name, data in all_results.items():
        assert data["branch_rows"],  f"{name}: branch_rows empty"
        assert data["support_rows"], f"{name}: support_rows empty"
        assert data["linking_rows"], f"{name}: linking_rows empty"


def test_comparator_signature_matrix_has_sig_a_to_sig_e(all_results):
    sig_keys = ["SIG_A", "SIG_B", "SIG_C", "SIG_D", "SIG_E"]
    for name, data in all_results.items():
        pf = data["pf"]
        missing = set(sig_keys) - set(pf.keys())
        assert not missing, f"{name}: missing sig keys {missing}"


def test_comparator_definitions_are_exported():
    """Comparator list must be non-empty and contain full_model."""
    assert COMPARATOR_NAMES, "COMPARATOR_NAMES must not be empty"
    assert "full_model" in COMPARATOR_NAMES


def test_signature_thresholds_are_exported():
    """Thresholds must all be predeclared (positive floats)."""
    for sig_key in ["SIG_A", "SIG_B", "SIG_C", "SIG_D", "SIG_E"]:
        assert sig_key in THRESHOLDS, f"Missing threshold for {sig_key}"
        assert THRESHOLDS[sig_key] > 0, f"Threshold for {sig_key} must be > 0"


def test_rescue_protocol_is_documented():
    path = REPO_ROOT / "results" / "e018_comparator_trace_matrix" / "summary" / "rescue_protocol.md"
    assert path.exists(), "rescue_protocol.md not found (run exp018 first)"
    content = path.read_text(encoding="utf-8")
    assert "targeted overlap rescue" in content.lower()
    assert "generic" in content.lower()


# ---------------------------------------------------------------------------
# Correctness / signature tests
# ---------------------------------------------------------------------------

def test_full_model_passes_joint_profile(all_results):
    pf = all_results["full_model"]["pf"]
    assert _joint_pass(pf), (
        f"full_model failed joint profile: {pf}"
    )


def test_at_least_one_comparator_fails_joint_profile(all_results):
    any_fail = any(
        not _joint_pass(data["pf"])
        for name, data in all_results.items()
        if name != "full_model"
    )
    assert any_fail, "All simpler comparators passed — unexpected result"


def test_no_simpler_comparator_passes_all_five_signatures(all_results):
    for name, data in all_results.items():
        if name == "full_model":
            continue
        assert not _joint_pass(data["pf"]), (
            f"{name} unexpectedly passed the full joint profile: {data['pf']}"
        )


def test_no_comparator_has_missing_signature_without_documented_reason(all_results):
    sig_keys = ["SIG_A", "SIG_B", "SIG_C", "SIG_D", "SIG_E"]
    for name, data in all_results.items():
        sigs = data["sigs"]
        raw_keys = [
            "SIG_A_overlap_advantage", "SIG_B_linking_gain",
            "SIG_C_context_separation", "SIG_D_linking_recall_dissociation",
            "SIG_E_targeted_rescue_advantage",
        ]
        for rk in raw_keys:
            assert rk in sigs, f"{name}: missing signature key {rk}"
            import math
            assert not math.isnan(float(sigs[rk])), f"{name}: NaN in {rk}"


# ---------------------------------------------------------------------------
# Comparator-specific correctness
# ---------------------------------------------------------------------------

def test_fast_context_only_has_no_structural_writing(all_results):
    """M_b should not increase from init to post-consolidation."""
    rows = all_results["fast_context_only"]["branch_rows"]
    init_mb    = {r["branch_id"]: r["structural_accessibility"]
                  for r in rows if r["phase"] == "init"}
    post_steps = [r for r in rows if r["phase"] == "consolidation_replay"]
    if not post_steps:
        pytest.skip("No consolidation_replay rows found")
    last_step  = max(r["step"] for r in post_steps)
    post_mb    = {r["branch_id"]: r["structural_accessibility"]
                  for r in post_steps if r["step"] == last_step}
    for bid in BRANCH_IDS:
        delta = post_mb[bid] - init_mb[bid]
        assert delta <= 1e-9, (
            f"fast_context_only: M_b increased on {bid} by {delta:.6f} — "
            "structural_lr should be 0 and structural_decay should be 0"
        )


def test_replay_no_structure_has_no_persistent_mb_write(all_results):
    """With structural_lr=0, M_b should not increase post-consolidation."""
    sigs = all_results["replay_no_structure"]["sigs"]
    assert sigs["SIG_A_overlap_advantage"] <= 0.0, (
        f"replay_no_structure SIG-A = {sigs['SIG_A_overlap_advantage']:.4f} — "
        "expected <= 0 (no structural write)"
    )


def test_fixed_allocation_only_has_no_post_consolidation_linking_gain(all_results):
    """fixed_allocation_only should not gain linking post-consolidation."""
    sigs = all_results["fixed_allocation_only"]["sigs"]
    assert abs(sigs["SIG_B_linking_gain"]) < 1e-9, (
        f"fixed_allocation_only SIG-B = {sigs['SIG_B_linking_gain']:.6f} — "
        "expected ~0 (no dynamic updating)"
    )


def test_random_slow_drift_uses_fixed_seed(all_results):
    """random_slow_drift should be deterministic under fixed seed."""
    from exp018_comparator_trace_matrix import COMPARATORS, run_comparator
    spec = next(s for s in COMPARATORS if s.name == "random_slow_drift")
    _, _, _, _, sigs1 = run_comparator(spec)
    _, _, _, _, sigs2 = run_comparator(spec)
    assert abs(sigs1["SIG_B_linking_gain"] - sigs2["SIG_B_linking_gain"]) < 1e-10, (
        "random_slow_drift not deterministic"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_repeated_run_is_deterministic_for_deterministic_comparators():
    """Deterministic comparators (no random drift) produce identical signatures on re-run."""
    from exp018_comparator_trace_matrix import COMPARATORS, run_comparator
    for spec in COMPARATORS:
        if spec.random_drift:
            continue  # random_slow_drift tested separately
        _, _, _, _, sigs1 = run_comparator(spec)
        _, _, _, _, sigs2 = run_comparator(spec)
        for key in ["SIG_A_overlap_advantage", "SIG_B_linking_gain",
                    "SIG_C_context_separation", "SIG_D_linking_recall_dissociation",
                    "SIG_E_targeted_rescue_advantage"]:
            assert abs(sigs1[key] - sigs2[key]) < 1e-12, (
                f"{spec.name} non-deterministic on '{key}': {sigs1[key]} vs {sigs2[key]}"
            )


# ---------------------------------------------------------------------------
# Full model signature magnitudes (sanity bounds)
# ---------------------------------------------------------------------------

def test_full_model_sig_a_exceeds_protected_threshold(all_results):
    sigs = all_results["full_model"]["sigs"]
    assert sigs["SIG_A_overlap_advantage"] > THRESHOLDS["SIG_A"]


def test_full_model_sig_b_exceeds_protected_threshold(all_results):
    sigs = all_results["full_model"]["sigs"]
    assert sigs["SIG_B_linking_gain"] > THRESHOLDS["SIG_B"]


def test_full_model_sig_d_exceeds_protected_threshold(all_results):
    sigs = all_results["full_model"]["sigs"]
    assert sigs["SIG_D_linking_recall_dissociation"] > THRESHOLDS["SIG_D"]


def test_full_model_sig_e_exceeds_protected_threshold(all_results):
    sigs = all_results["full_model"]["sigs"]
    assert sigs["SIG_E_targeted_rescue_advantage"] > THRESHOLDS["SIG_E"]
