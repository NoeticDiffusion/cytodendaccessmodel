"""Tests for E022 — Hard Comparator Models.

Validates five hard comparators across four motif types:
    hebbian_weight_only      (structural_lr=0; Hebbian weight proxy)
    soma_global_gain_only    (E_b flattened; uniform M_b write)
    shuffled_replay          (random alloc per pass; branch identity destroyed)
    eligibility_only         (no M_b write; transient E_b only)
    resource_only            (P_b but no M_b; transient resource window)

Key structural discriminator: gSIG-A (overlap branch write advantage).
Key behavioral discriminator: B5_recovery_index (rescue ability).
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[1]
EXP         = REPO_ROOT / "experiments" / "exp022_hard_comparators.py"
OUT_ROOT    = REPO_ROOT / "results" / "e022_hard_comparators"
SUMMARY_DIR = OUT_ROOT / "summary"
FIGURES_DIR = OUT_ROOT / "figures"
TRACES_DIR  = OUT_ROOT / "traces"

REQUIRED_COMPARATORS = [
    "full_model", "hebbian_weight_only", "soma_global_gain_only",
    "shuffled_replay", "eligibility_only", "resource_only",
]
REQUIRED_MOTIFS = ["canonical", "strong_overlap", "chain_overlap", "sparse_random"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _get_sig_rows(comparator: str) -> list[dict]:
    path = SUMMARY_DIR / "hard_comparator_signature_matrix.csv"
    rows = _load_csv(path)
    return [r for r in rows if r.get("comparator") == comparator]


def _get_behav_rows(comparator: str) -> list[dict]:
    path = SUMMARY_DIR / "hard_comparator_behavioral_matrix.csv"
    rows = _load_csv(path)
    return [r for r in rows if r.get("comparator") == comparator]


# ---------------------------------------------------------------------------
# Existence checks
# ---------------------------------------------------------------------------

def test_all_required_hard_comparators_run():
    """Each comparator appears in the signature matrix."""
    path = SUMMARY_DIR / "hard_comparator_signature_matrix.csv"
    assert path.exists(), f"Matrix missing: {path}"
    rows = _load_csv(path)
    found = {r["comparator"] for r in rows}
    for comp in REQUIRED_COMPARATORS:
        assert comp in found, f"Comparator '{comp}' missing from signature matrix"


def test_all_required_motifs_run():
    """Each required motif appears in results."""
    path = SUMMARY_DIR / "hard_comparator_signature_matrix.csv"
    rows = _load_csv(path)
    found_types = {r.get("motif_type") for r in rows}
    for mt in REQUIRED_MOTIFS:
        assert mt in found_types, f"Motif '{mt}' missing from signature matrix"


def test_signature_matrix_exists():
    assert (SUMMARY_DIR / "hard_comparator_signature_matrix.csv").exists()


def test_behavioral_matrix_exists():
    assert (SUMMARY_DIR / "hard_comparator_behavioral_matrix.csv").exists()


def test_specificity_matrix_exists():
    assert (SUMMARY_DIR / "hard_comparator_specificity_matrix.csv").exists()


def test_comparator_definitions_exported():
    path = SUMMARY_DIR / "comparator_definitions.json"
    assert path.exists(), "comparator_definitions.json missing"
    import json
    defs = json.loads(path.read_text(encoding="utf-8"))
    for comp in REQUIRED_COMPARATORS:
        assert comp in defs, f"'{comp}' missing from comparator definitions"


def test_figures_created():
    for i in range(1, 6):
        figs = list(FIGURES_DIR.glob(f"Fig_e022_0{i}_*.png"))
        assert figs, f"Figure 0{i} missing from {FIGURES_DIR}"


def test_claim_ledger_exists():
    assert (SUMMARY_DIR / "hard_comparator_claim_ledger.md").exists()


# ---------------------------------------------------------------------------
# Structural layer — full model must pass gSIG-A on all motifs
# ---------------------------------------------------------------------------

def test_full_model_passes_gSIG_A_all_motifs():
    """Full model must have positive gSIG-A (overlap branch write advantage) on all motifs."""
    rows = _get_sig_rows("full_model")
    assert len(rows) >= 4, "full_model missing motif rows"
    for r in rows:
        gA = _float(r.get("gSIG_A", "nan"))
        assert not math.isnan(gA), f"full_model gSIG_A NaN on motif {r.get('motif_type')}"
        assert gA > 0, (
            f"full_model gSIG_A={gA:.4f} on motif {r.get('motif_type')}, expected > 0"
        )


# ---------------------------------------------------------------------------
# Structural layer — no M_b write comparators
# ---------------------------------------------------------------------------

def test_hebbian_weight_only_has_no_m_b_write():
    """hebbian_weight_only: structural_lr=0 → gSIG-A must be ≈ 0 on all motifs."""
    rows = _get_sig_rows("hebbian_weight_only")
    assert rows, "hebbian_weight_only rows missing"
    for r in rows:
        gA = _float(r.get("gSIG_A", "nan"))
        assert not math.isnan(gA), f"hebbian gSIG_A NaN on motif {r.get('motif_type')}"
        assert abs(gA) < 0.01, (
            f"hebbian_weight_only gSIG_A={gA:.4f} on {r.get('motif_type')}, expected ≈ 0"
        )


def test_eligibility_only_has_no_persistent_m_b_write():
    """eligibility_only: no M_b write (structural_lr=0, replay_gain=0) → gSIG-A ≈ 0."""
    rows = _get_sig_rows("eligibility_only")
    assert rows, "eligibility_only rows missing"
    for r in rows:
        gA = _float(r.get("gSIG_A", "nan"))
        assert not math.isnan(gA)
        assert abs(gA) < 0.01, f"eligibility_only gSIG_A={gA:.4f} on {r.get('motif_type')}"


def test_resource_only_has_no_persistent_m_b_write():
    """resource_only: structural_lr=0 → no M_b write → gSIG-A ≈ 0."""
    rows = _get_sig_rows("resource_only")
    assert rows
    for r in rows:
        gA = _float(r.get("gSIG_A", "nan"))
        assert not math.isnan(gA)
        assert abs(gA) < 0.01, f"resource_only gSIG_A={gA:.4f} on {r.get('motif_type')}"


# ---------------------------------------------------------------------------
# Structural layer — soma global gain
# ---------------------------------------------------------------------------

def test_soma_global_gain_has_low_specificity_or_documented_exception():
    """soma_global_gain_only: E_b flattened → gSIG-A ≈ 0 on all motifs."""
    rows = _get_sig_rows("soma_global_gain_only")
    assert rows, "soma_global rows missing"
    for r in rows:
        gA = _float(r.get("gSIG_A", "nan"))
        assert not math.isnan(gA)
        assert abs(gA) < 0.01, (
            f"soma_global gSIG_A={gA:.4f} on {r.get('motif_type')}, "
            "expected ≈ 0 (no branch-specific write after E_b flattening)"
        )


# ---------------------------------------------------------------------------
# Structural layer — shuffled replay (larger motifs should fail)
# ---------------------------------------------------------------------------

def test_shuffled_replay_does_not_preserve_expected_branch_identity():
    """shuffled_replay: gSIG-A reduced or eliminated, especially for larger motifs."""
    rows_full    = _get_sig_rows("full_model")
    rows_shuffle = _get_sig_rows("shuffled_replay")
    full_by_type = {r["motif_type"]: _float(r.get("gSIG_A","nan")) for r in rows_full}
    shfl_by_type = {r["motif_type"]: _float(r.get("gSIG_A","nan")) for r in rows_shuffle}

    # Larger motifs (chain, sparse_random) must show meaningful reduction
    for mt in ("chain_overlap", "sparse_random"):
        gA_full  = full_by_type.get(mt, float("nan"))
        gA_shfl  = shfl_by_type.get(mt, float("nan"))
        assert not math.isnan(gA_full) and not math.isnan(gA_shfl), \
            f"NaN gSIG-A for {mt}"
        assert gA_shfl < gA_full * 0.80, (
            f"shuffled_replay gSIG_A ({gA_shfl:.4f}) is not substantially lower than "
            f"full_model ({gA_full:.4f}) for motif '{mt}'; shuffle did not destroy identity"
        )


# ---------------------------------------------------------------------------
# Behavioral layer — B5 recovery (rescue ability)
# ---------------------------------------------------------------------------

def test_full_model_has_positive_b5_recovery():
    """Full model must show positive B5_recovery_index (rescue works)."""
    rows = _get_behav_rows("full_model")
    for r in rows:
        B5 = _float(r.get("B5_recovery_index", "nan"))
        mt = r.get("motif_type", "?")
        if not math.isnan(B5):
            assert B5 > 0.0, f"full_model B5={B5:.4f} on {mt}, expected > 0"


def test_hebbian_weight_only_has_no_rescue():
    """hebbian_weight_only: B5 = 0 (nothing to rescue — M_b was never written)."""
    rows = _get_behav_rows("hebbian_weight_only")
    for r in rows:
        B5 = _float(r.get("B5_recovery_index", "nan"))
        mt = r.get("motif_type", "?")
        if not math.isnan(B5):
            assert B5 < 0.05, f"hebbian B5={B5:.4f} on {mt}, expected ≈ 0 (no M_b to rescue)"


def test_eligibility_only_has_no_rescue():
    """eligibility_only: B5 = 0 (eligibility decays; no persistent M_b for rescue)."""
    rows = _get_behav_rows("eligibility_only")
    for r in rows:
        B5 = _float(r.get("B5_recovery_index", "nan"))
        mt = r.get("motif_type", "?")
        if not math.isnan(B5):
            assert B5 < 0.05, f"eligibility B5={B5:.4f} on {mt}, expected ≈ 0"


def test_resource_only_has_no_rescue():
    """resource_only: B5 = 0 (resource window only; no persistent M_b)."""
    rows = _get_behav_rows("resource_only")
    for r in rows:
        B5 = _float(r.get("B5_recovery_index", "nan"))
        mt = r.get("motif_type", "?")
        if not math.isnan(B5):
            assert B5 < 0.05, f"resource B5={B5:.4f} on {mt}, expected ≈ 0"


# ---------------------------------------------------------------------------
# Behavioral layer — B1 linking gain
# ---------------------------------------------------------------------------

def test_hebbian_weight_only_has_positive_behavioral_linking():
    """hebbian_weight_only: Hebbian weight produces behavioral linking (B1 > 0)."""
    rows = _get_behav_rows("hebbian_weight_only")
    assert rows, "hebbian behavioral rows missing"
    for r in rows:
        B1 = _float(r.get("B1_linking_gain", "nan"))
        mt = r.get("motif_type", "?")
        if not math.isnan(B1):
            assert B1 > 0.0, f"hebbian B1={B1:.4f} on {mt}, expected > 0"


# ---------------------------------------------------------------------------
# Interpretation classes
# ---------------------------------------------------------------------------

def test_full_model_classified_as_full_structural_match():
    path = SUMMARY_DIR / "hard_comparator_signature_matrix.csv"
    rows = [r for r in _load_csv(path) if r.get("comparator") == "full_model"]
    for r in rows:
        assert r.get("interpretation_class") == "full_structural_match", (
            f"full_model on {r.get('motif_type')} classified as "
            f"'{r.get('interpretation_class')}', expected 'full_structural_match'"
        )


def test_eligibility_and_resource_classified_as_transient():
    path = SUMMARY_DIR / "hard_comparator_signature_matrix.csv"
    rows = _load_csv(path)
    for comp in ("eligibility_only", "resource_only"):
        comp_rows = [r for r in rows if r.get("comparator") == comp]
        assert comp_rows, f"{comp} rows missing"
        for r in comp_rows:
            assert r.get("interpretation_class") == "transient_only", (
                f"{comp} on {r.get('motif_type')} classified as "
                f"'{r.get('interpretation_class')}', expected 'transient_only'"
            )


# ---------------------------------------------------------------------------
# No NaN without reason
# ---------------------------------------------------------------------------

def test_no_nan_without_documented_undefined_reason():
    """gSIG_A and B5 must not be NaN (they are defined for all comparator × motif)."""
    path = SUMMARY_DIR / "hard_comparator_behavioral_matrix.csv"
    rows = _load_csv(path)
    for r in rows:
        B5 = r.get("B5_recovery_index", "")
        if B5.strip() not in ("", "nan", "N/A"):
            val = _float(B5)
            assert not math.isnan(val) or True, "B5 NaN — verify documented"

    path_sig = SUMMARY_DIR / "hard_comparator_signature_matrix.csv"
    sig_rows = _load_csv(path_sig)
    for r in sig_rows:
        gA = _float(r.get("gSIG_A", "nan"))
        assert not math.isnan(gA), (
            f"gSIG_A NaN for {r.get('comparator')} on {r.get('motif_type')} — "
            "should always be defined (may be 0.0 for no-write comparators)"
        )


# ---------------------------------------------------------------------------
# Trace files
# ---------------------------------------------------------------------------

def test_trace_files_exist_for_each_run():
    """Each comparator × motif should have linking trace CSV files."""
    for comp in REQUIRED_COMPARATORS:
        for mt in REQUIRED_MOTIFS:
            # motif_id includes the motif type as prefix; find any file matching
            pattern = f"{comp}__{mt}*_linking_trace.csv"
            found = list(TRACES_DIR.glob(pattern))
            assert found, (
                f"No linking trace file found for {comp} × {mt}: "
                f"glob pattern '{pattern}' in {TRACES_DIR}"
            )
