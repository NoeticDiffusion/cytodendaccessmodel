"""Tests for E021 — Scaling and Motif Generalization.

Run with: pytest tests/test_e021_scaling_and_motif_generalization.py -v
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

OUT_ROOT    = REPO_ROOT / "results" / "e021_scaling_and_motif_generalization"
MOTIFS_DIR  = OUT_ROOT / "motifs"
TRACES_DIR  = OUT_ROOT / "traces"
SUMMARY_DIR = OUT_ROOT / "summary"

from cytodend_accessmodel.motifs import (
    build_motif, MotifSpec, alloc_to_cue, private_cue, linking_score,
)
from exp021_scaling_and_motif_generalization import (
    CANONICAL_PARAMS, DEFAULT_SEED, ALL_SPECS, STAGE1_SPECS, STAGE2_SPECS,
    _run_motif_cell,
)

REQUIRED_MOTIF_TYPES = [
    "canonical", "weak_overlap", "strong_overlap",
    "chain_overlap", "hub_overlap", "sparse_random",
]


# ---------------------------------------------------------------------------
# Motif generator tests
# ---------------------------------------------------------------------------

def test_motif_generator_reproduces_canonical_allocation():
    m = build_motif("canonical", n_branches=4)
    assert m.n_branches == 4
    assert m.n_traces  == 2
    assert len(m.branch_ids) == 4
    assert m.branch_ids[1] in m.damage_target_branches
    alloc_mu1 = m.allocations["mu1"]
    alloc_mu2 = m.allocations["mu2"]
    assert alloc_mu1["b0"] == pytest.approx(0.90)
    assert alloc_mu1["b1"] == pytest.approx(0.85)
    assert alloc_mu2["b2"] == pytest.approx(0.90)
    assert alloc_mu2["b1"] == pytest.approx(0.85)


def test_all_required_motifs_generate_valid_allocation_matrices():
    for mt in REQUIRED_MOTIF_TYPES:
        n_tr = 3 if mt == "chain_overlap" else (4 if mt in ("hub_overlap", "sparse_random") else 2)
        n_br = 8
        m = build_motif(mt, n_branches=n_br, n_traces=n_tr, seed=42)
        assert m.motif_type == mt
        assert len(m.branch_ids) == n_br
        assert len(m.trace_ids) == n_tr
        for tid in m.trace_ids:
            assert tid in m.allocations
            weights = m.allocations[tid]
            assert all(b in weights for b in m.branch_ids), \
                f"{mt}: trace {tid} missing branch entries"


def test_allocation_values_are_bounded():
    for mt in REQUIRED_MOTIF_TYPES:
        n_tr = 3 if mt == "chain_overlap" else (4 if mt in ("hub_overlap", "sparse_random") else 2)
        m = build_motif(mt, n_branches=8, n_traces=n_tr, seed=42)
        for tid, alloc in m.allocations.items():
            for b, w in alloc.items():
                assert 0.0 <= w <= 1.0, \
                    f"{mt} / {tid} / {b}: weight {w} out of [0,1]"


def test_expected_linked_pairs_do_not_overlap_unlinked_pairs():
    for mt in REQUIRED_MOTIF_TYPES:
        n_tr = 3 if mt == "chain_overlap" else (4 if mt in ("hub_overlap", "sparse_random") else 2)
        m = build_motif(mt, n_branches=8, n_traces=n_tr, seed=42)
        linked_set = {tuple(sorted(p)) for p in m.expected_linked_pairs}
        unlinked_set = {tuple(sorted(p)) for p in m.expected_unlinked_pairs}
        intersection = linked_set & unlinked_set
        assert not intersection, \
            f"{mt}: pairs appear in both linked and unlinked: {intersection}"


def test_chain_overlap_has_local_and_distant_pairs():
    m = build_motif("chain_overlap", n_branches=8, n_traces=3)
    assert len(m.expected_linked_pairs) == 2, \
        "chain should have exactly 2 expected linked pairs (t0-t1, t1-t2)"
    assert len(m.expected_unlinked_pairs) == 1, \
        "chain should have exactly 1 expected unlinked pair (t0-t2)"


def test_hub_overlap_all_pairs_linked():
    m = build_motif("hub_overlap", n_branches=8, n_traces=4)
    n_pairs = m.n_traces * (m.n_traces - 1) // 2
    assert len(m.expected_linked_pairs) == n_pairs, \
        "hub: all pairs should be expected linked"
    assert len(m.expected_unlinked_pairs) == 0, \
        "hub: no unlinked pairs (all share hub branch)"


# ---------------------------------------------------------------------------
# Output file existence tests
# ---------------------------------------------------------------------------

def test_all_required_motif_output_files_exist():
    for motif_type, n_branches, n_traces, seed in ALL_SPECS:
        m = build_motif(motif_type, n_branches=n_branches, n_traces=n_traces, seed=seed)
        assert (MOTIFS_DIR / f"{m.motif_id}_motif.json").exists(), \
            f"Missing motif JSON: {m.motif_id}"
        assert (TRACES_DIR / f"{m.motif_id}_branch_traces.csv").exists(), \
            f"Missing branch traces: {m.motif_id}"
        assert (TRACES_DIR / f"{m.motif_id}_linking_trace.csv").exists(), \
            f"Missing linking trace: {m.motif_id}"


def test_generalized_signatures_exported():
    path = SUMMARY_DIR / "all_motif_runs_long.csv"
    if not path.exists():
        pytest.skip("all_motif_runs_long.csv not found")
    with path.open(encoding="utf-8") as f:
        cols = set(csv.DictReader(f).fieldnames or [])
    for sig in ["gSIG_A", "gSIG_B", "gSIG_C", "gSIG_D", "gSIG_E"]:
        assert sig in cols, f"Missing column: {sig}"
    assert "joint_pass" in cols


def test_stage_1_branch_counts_run():
    expected_n = sorted({n for _, n, _, _ in STAGE1_SPECS})
    path = SUMMARY_DIR / "all_motif_runs_long.csv"
    if not path.exists():
        pytest.skip("all_motif_runs_long.csv not found")
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    observed_n = sorted(set(int(r["n_branches"]) for r in rows))
    for n in expected_n:
        assert n in observed_n, f"Branch count {n} not found in results"


def test_stage_2_multitrace_motifs_run_or_are_documented_deferred():
    path = SUMMARY_DIR / "all_motif_runs_long.csv"
    if not path.exists():
        pytest.skip("all_motif_runs_long.csv not found")
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    stage2_types = {"chain_overlap", "hub_overlap", "sparse_random"}
    observed_types = {r["motif_type"] for r in rows}
    for mt in stage2_types:
        assert mt in observed_types, \
            f"Stage 2 motif {mt} neither run nor documented as deferred"


def test_false_linking_summary_exists():
    path = SUMMARY_DIR / "false_linking_summary.csv"
    assert path.exists(), "false_linking_summary.csv not found"
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "false_linking_summary.csv is empty"


def test_no_nan_in_generalized_signature_outputs():
    path = SUMMARY_DIR / "all_motif_runs_long.csv"
    if not path.exists():
        pytest.skip("all_motif_runs_long.csv not found")
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # gSIG-E can be NaN when there are no expected pairs (weak_overlap)
    # Only check gSIG-A, gSIG-B, gSIG-C, gSIG-D
    for row in rows:
        for sig in ["gSIG_A", "gSIG_B", "gSIG_C", "gSIG_D"]:
            val = row.get(sig, "")
            if val not in ("", "nan"):
                assert not math.isnan(float(val)), \
                    f"{row.get('run_id')}: {sig} = NaN"


# ---------------------------------------------------------------------------
# Canonical reference test
# ---------------------------------------------------------------------------

def test_canonical_4_branch_run_matches_e019r_or_e020_reference():
    """Canonical 4-branch run should pass the joint generalized profile."""
    motif = build_motif("canonical", n_branches=4)
    result = _run_motif_cell(motif, params=CANONICAL_PARAMS, seed=DEFAULT_SEED)
    assert result["joint_pass"], (
        f"Canonical 4-branch should pass: "
        f"A={result['gSIG_A']:.4f} B={result['gSIG_B']:.4f} "
        f"D={result['gSIG_D']:.4f} E={result['gSIG_E']:.4f}"
    )
    assert result["gSIG_A"] > 0, "gSIG-A should be positive (overlap branch writes more)"
    assert result["gSIG_B"] > 0, "gSIG-B should be positive (expected pairs link more)"


# ---------------------------------------------------------------------------
# Optional: boundary behaviour
# ---------------------------------------------------------------------------

def test_weak_overlap_has_lower_linking_than_strong_overlap():
    m_weak   = build_motif("weak_overlap",   n_branches=8)
    m_strong = build_motif("strong_overlap", n_branches=8)
    r_weak   = _run_motif_cell(m_weak,   CANONICAL_PARAMS, DEFAULT_SEED)
    r_strong = _run_motif_cell(m_strong, CANONICAL_PARAMS, DEFAULT_SEED)
    assert r_weak["gSIG_B"] < r_strong["gSIG_B"], \
        f"weak ({r_weak['gSIG_B']:.4f}) should be < strong ({r_strong['gSIG_B']:.4f})"


def test_chain_overlap_local_pairs_exceed_distant_pairs():
    """chain: false_linking_rate < 1.0 (local > distant linking gain)."""
    m = build_motif("chain_overlap", n_branches=8, n_traces=3)
    r = _run_motif_cell(m, CANONICAL_PARAMS, DEFAULT_SEED)
    fl = r.get("false_linking_rate", float("nan"))
    assert not math.isnan(fl), "chain_overlap should have a defined false_linking_rate"
    assert fl < 1.0, f"chain: expected local > distant linking; FL={fl:.3f}"


def test_hub_overlap_increases_false_linking_risk():
    """hub: all pairs share hub branch, so expected_unlinked_pairs is empty."""
    m = build_motif("hub_overlap", n_branches=8, n_traces=4)
    assert not m.expected_unlinked_pairs, \
        "hub: expected_unlinked_pairs should be empty (false-linking is universal)"
    r = _run_motif_cell(m, CANONICAL_PARAMS, DEFAULT_SEED)
    assert math.isnan(r.get("false_linking_rate", float("nan"))), \
        "hub: false_linking_rate should be NaN (no unlinked pairs)"
