"""Tests for e017 — Traceable Simulator Core.

Validates that the canonical trace-export run produces correct structure,
preserves biological invariants, and is deterministic.

Run with: pytest tests/test_e017_traceable_simulator_core.py -v
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

# Make src and experiments importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from exp017_traceable_simulator_core import (
    BRANCH_IDS,
    OVERLAP_BRANCH,
    run_canonical,
    _linking,
    _build_sim,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def canonical_run():
    """Run the canonical protocol once for the whole module."""
    branch_rows, support_rows, linking_rows, sigs = run_canonical()
    return branch_rows, support_rows, linking_rows, sigs


# ---------------------------------------------------------------------------
# Column / schema tests
# ---------------------------------------------------------------------------

REQUIRED_BRANCH_COLS = {
    "step", "phase", "branch_id", "is_overlap",
    "x_b", "fast_access", "slow_access", "effective_access",
    "eligibility", "translation_readiness", "structural_accessibility",
    "input_drive",
}

REQUIRED_SUPPORT_COLS = {
    "step", "phase", "trace_id",
    "recall_support", "readout_value", "context_label",
}

REQUIRED_LINKING_COLS = {
    "step", "phase", "trace_pair",
    "linking_score", "overlap_branch_contribution", "nonoverlap_contribution",
}


def test_trace_export_has_required_branch_columns(canonical_run):
    branch_rows, _, _, _ = canonical_run
    assert branch_rows, "branch_rows must not be empty"
    cols = set(branch_rows[0].keys())
    missing = REQUIRED_BRANCH_COLS - cols
    assert not missing, f"Missing branch trace columns: {missing}"


def test_trace_export_has_required_support_columns(canonical_run):
    _, support_rows, _, _ = canonical_run
    assert support_rows, "support_rows must not be empty"
    cols = set(support_rows[0].keys())
    missing = REQUIRED_SUPPORT_COLS - cols
    assert not missing, f"Missing support trace columns: {missing}"


def test_trace_export_has_required_linking_columns(canonical_run):
    _, _, linking_rows, _ = canonical_run
    assert linking_rows, "linking_rows must not be empty"
    cols = set(linking_rows[0].keys())
    missing = REQUIRED_LINKING_COLS - cols
    assert not missing, f"Missing linking trace columns: {missing}"


# ---------------------------------------------------------------------------
# Phase coverage tests
# ---------------------------------------------------------------------------

REQUIRED_PHASES = {
    "init",
    "encode_mu_1",
    "encode_mu_2",
    "pre_consolidation_probe",
    "consolidation_replay",
    "post_consolidation_probe",
    "overlap_damage",
    "post_damage_probe",
    "targeted_rescue",
    "post_rescue_probe",
}


def test_trace_export_has_all_expected_phases(canonical_run):
    branch_rows, _, _, _ = canonical_run
    phases_present = {r["phase"] for r in branch_rows}
    missing = REQUIRED_PHASES - phases_present
    assert not missing, f"Missing phases in branch_traces: {missing}"


def test_phase_order_is_monotonic(canonical_run):
    """Steps must be non-decreasing within each phase (phases progress forward)."""
    branch_rows, _, _, _ = canonical_run
    phase_first_step: dict[str, int] = {}
    phase_last_step:  dict[str, int] = {}
    for r in branch_rows:
        ph = r["phase"]
        s  = r["step"]
        if ph not in phase_first_step:
            phase_first_step[ph] = s
        phase_last_step[ph] = s

    phase_order = [
        "init", "encode_mu_1", "encode_mu_2",
        "pre_consolidation_probe", "consolidation_replay",
        "post_consolidation_probe", "overlap_damage",
        "post_damage_probe", "targeted_rescue", "post_rescue_probe",
    ]
    present = [p for p in phase_order if p in phase_first_step]
    for i in range(len(present) - 1):
        a, b = present[i], present[i + 1]
        assert phase_last_step[a] <= phase_first_step[b], (
            f"Phase {a} ends at step {phase_last_step[a]} but "
            f"{b} starts at step {phase_first_step[b]}"
        )


# ---------------------------------------------------------------------------
# Branch identity tests
# ---------------------------------------------------------------------------

def test_overlap_branch_identity_is_preserved(canonical_run):
    """Every row for b1 must have is_overlap=True; all others False."""
    branch_rows, _, _, _ = canonical_run
    for r in branch_rows:
        expected = r["branch_id"] == OVERLAP_BRANCH
        assert r["is_overlap"] == expected, (
            f"branch {r['branch_id']}: is_overlap={r['is_overlap']} "
            f"but expected {expected}"
        )


def test_branch_ids_are_unique_and_complete(canonical_run):
    """Each step should contain exactly one row per branch_id."""
    branch_rows, _, _, _ = canonical_run
    from collections import defaultdict
    per_step: dict[int, set] = defaultdict(set)
    for r in branch_rows:
        per_step[r["step"]].add(r["branch_id"])
    for step, ids in per_step.items():
        assert ids == set(BRANCH_IDS), (
            f"Step {step}: expected branch_ids {set(BRANCH_IDS)}, got {ids}"
        )


def test_trace_ids_are_unique_and_complete(canonical_run):
    """Each step in support_rows should contain both mu1 and mu2."""
    _, support_rows, _, _ = canonical_run
    from collections import defaultdict
    per_step: dict[int, set] = defaultdict(set)
    for r in support_rows:
        per_step[r["step"]].add(r["trace_id"])
    for step, ids in per_step.items():
        assert {"mu1", "mu2"}.issubset(ids), (
            f"Step {step}: mu1/mu2 not both present in support_rows, got {ids}"
        )


# ---------------------------------------------------------------------------
# Numerical consistency tests
# ---------------------------------------------------------------------------

def test_linking_score_recomputes_from_branch_traces(canonical_run):
    """Linking score in linking_rows must match manual recomputation from branch rows."""
    from exp017_traceable_simulator_core import MU1_ALLOC, MU2_ALLOC
    branch_rows, _, linking_rows, _ = canonical_run

    # Build per-step M_b lookup
    mb_by_step: dict[int, dict[str, float]] = {}
    for r in branch_rows:
        s = r["step"]
        if s not in mb_by_step:
            mb_by_step[s] = {}
        mb_by_step[s][r["branch_id"]] = r["structural_accessibility"]

    for lr in linking_rows:
        step = lr["step"]
        expected_lk = sum(
            MU1_ALLOC.branch_weights.get(b, 0.0)
            * MU2_ALLOC.branch_weights.get(b, 0.0)
            * mb_by_step[step][b]
            for b in BRANCH_IDS
        )
        assert abs(lr["linking_score"] - expected_lk) < 1e-9, (
            f"Step {step}: linking_score={lr['linking_score']:.8f} "
            f"but recomputed={expected_lk:.8f}"
        )


def test_no_nan_in_required_trace_columns(canonical_run):
    """No NaN allowed in the key numerical columns."""
    import math
    branch_rows, support_rows, linking_rows, _ = canonical_run
    nan_cols = ["x_b", "fast_access", "slow_access", "effective_access",
                "eligibility", "translation_readiness", "structural_accessibility"]
    for r in branch_rows:
        for col in nan_cols:
            assert not math.isnan(float(r[col])), (
                f"NaN in branch_traces col '{col}' at step {r['step']}"
            )
    for r in support_rows:
        for col in ["recall_support", "readout_value"]:
            assert not math.isnan(float(r[col])), (
                f"NaN in support col '{col}' at step {r['step']}"
            )


# ---------------------------------------------------------------------------
# Signature tests
# ---------------------------------------------------------------------------

def test_signature_summary_matches_trace_files(canonical_run):
    """Recompute SIG-A and SIG-B from raw trace data and compare to sigs dict."""
    from exp017_traceable_simulator_core import MU1_ALLOC, MU2_ALLOC, OVERLAP_BRANCH
    branch_rows, _, linking_rows, sigs = canonical_run

    # Find init and post_consolidation steps
    init_mb:  dict[str, float] = {}
    post_mb:  dict[str, float] = {}

    # Use last step of init for pre, last step of consolidation_replay for post
    init_steps = [r for r in branch_rows if r["phase"] == "init"]
    cons_steps = [r for r in branch_rows if r["phase"] == "consolidation_replay"]

    assert init_steps, "No init rows found"
    assert cons_steps, "No consolidation_replay rows found"

    last_init_step = max(r["step"] for r in init_steps)
    last_cons_step = max(r["step"] for r in cons_steps)

    for r in branch_rows:
        if r["step"] == last_init_step:
            init_mb[r["branch_id"]] = r["structural_accessibility"]
        if r["step"] == last_cons_step:
            post_mb[r["branch_id"]] = r["structural_accessibility"]

    # SIG-A recompute
    nonoverlap = [b for b in BRANCH_IDS if b != OVERLAP_BRANCH]
    delta_ovlp = post_mb[OVERLAP_BRANCH] - init_mb[OVERLAP_BRANCH]
    delta_nonovlp_mean = sum(post_mb[b] - init_mb[b] for b in nonoverlap) / len(nonoverlap)
    sig_a_recomputed = delta_ovlp - delta_nonovlp_mean

    assert abs(sigs["SIG_A_overlap_advantage"] - sig_a_recomputed) < 1e-6, (
        f"SIG-A mismatch: reported={sigs['SIG_A_overlap_advantage']:.6f}, "
        f"recomputed={sig_a_recomputed:.6f}"
    )


def test_sig_a_is_positive(canonical_run):
    """Overlap branch must gain more M_b than non-overlap branches."""
    _, _, _, sigs = canonical_run
    assert sigs["SIG_A_overlap_advantage"] > 0, (
        f"SIG-A = {sigs['SIG_A_overlap_advantage']:.4f} — expected positive"
    )


def test_sig_b_is_positive(canonical_run):
    """Linking score must increase after consolidation."""
    _, _, _, sigs = canonical_run
    assert sigs["SIG_B_linking_gain"] > 0, (
        f"SIG-B = {sigs['SIG_B_linking_gain']:.4f} — expected positive"
    )


def test_sig_c_is_positive(canonical_run):
    """Context separation score must be positive."""
    _, _, _, sigs = canonical_run
    assert sigs["SIG_C_context_separation"] > 0, (
        f"SIG-C = {sigs['SIG_C_context_separation']:.4f} — expected positive"
    )


def test_sig_d_is_positive(canonical_run):
    """Linking must drop more than recall under focal overlap damage."""
    _, _, _, sigs = canonical_run
    assert sigs["SIG_D_linking_recall_dissociation"] > 0, (
        f"SIG-D = {sigs['SIG_D_linking_recall_dissociation']:.2f}pp — expected positive"
    )


def test_sig_e_is_positive(canonical_run):
    """Targeted rescue must recover more linking than standard rescue."""
    _, _, _, sigs = canonical_run
    assert sigs["SIG_E_targeted_rescue_advantage"] > 0, (
        f"SIG-E = {sigs['SIG_E_targeted_rescue_advantage']:.2f}pp — expected positive"
    )


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------

def test_deterministic_canonical_run_reproduces_same_signatures():
    """Two independent calls to run_canonical must produce identical signatures."""
    _, _, _, sigs1 = run_canonical()
    _, _, _, sigs2 = run_canonical()
    for key in sigs1:
        v1, v2 = sigs1[key], sigs2[key]
        assert abs(v1 - v2) < 1e-12, (
            f"Non-determinism detected in '{key}': {v1} vs {v2}"
        )
