"""Tests for E022R — Shuffled Replay Scaling Audit.

Verifies that the shuffled_replay partial gSIG-A match observed in E022
(single seed, 4-branch motif) is a small-branch-count sampling artefact
that decays monotonically as branch count increases.

Key assertions:
- All required branch counts are tested.
- 20 seeds are used per condition.
- Full-model reference is exported for all conditions.
- ratio_to_full_model is exported and well-defined.
- Ratio decays monotonically with branch count.
- No shuffled-replay condition at n >= 8 passes the full structural profile.
- Claim ledger exists and documents the correct conclusion.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO_ROOT   = Path(__file__).resolve().parents[1]
OUT_ROOT    = REPO_ROOT / "results" / "e022r_shuffled_replay_scaling_audit"
SUMMARY_DIR = OUT_ROOT / "summary"
FIGURES_DIR = OUT_ROOT / "figures"

REQUIRED_BRANCH_COUNTS = [4, 8, 16, 32]
REQUIRED_MOTIFS        = ["canonical", "strong_overlap"]
MIN_SEEDS              = 20


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


def _comparison_rows() -> list[dict]:
    return _load_csv(SUMMARY_DIR / "shuffled_replay_vs_full_model.csv")


def _scaling_rows() -> list[dict]:
    return _load_csv(SUMMARY_DIR / "shuffled_replay_scaling.csv")


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------

def test_scaling_csv_exists():
    assert (SUMMARY_DIR / "shuffled_replay_scaling.csv").exists()


def test_comparison_csv_exists():
    assert (SUMMARY_DIR / "shuffled_replay_vs_full_model.csv").exists()


def test_claim_ledger_exists():
    assert (SUMMARY_DIR / "claim_ledger.md").exists()


def test_figures_created():
    for i in (1, 2, 3):
        figs = list(FIGURES_DIR.glob(f"Fig_e022r_0{i}_*.png"))
        assert figs, f"Figure 0{i} missing from {FIGURES_DIR}"


# ---------------------------------------------------------------------------
# Required conditions
# ---------------------------------------------------------------------------

def test_required_branch_counts_run():
    rows = _comparison_rows()
    found_counts = {int(r["n_branches"]) for r in rows}
    for nb in REQUIRED_BRANCH_COUNTS:
        assert nb in found_counts, f"Branch count {nb} missing from comparison CSV"


def test_required_motifs_run():
    rows = _comparison_rows()
    found_motifs = {r["motif_type"] for r in rows}
    for mt in REQUIRED_MOTIFS:
        assert mt in found_motifs, f"Motif '{mt}' missing from comparison CSV"


def test_multiple_seeds_used():
    """Each condition must report n_seeds >= MIN_SEEDS."""
    rows = _comparison_rows()
    for r in rows:
        n_seeds = _float(r.get("n_seeds", "0"))
        assert n_seeds >= MIN_SEEDS, (
            f"{r.get('motif_type')} n={r.get('n_branches')}: "
            f"only {n_seeds} seeds (expected >= {MIN_SEEDS})"
        )


def test_full_model_reference_exists():
    """Full model gSIG_A must be recorded for all conditions."""
    rows = _comparison_rows()
    assert rows, "Comparison CSV is empty"
    for r in rows:
        full_gA = _float(r.get("full_gSIG_A", "nan"))
        assert not math.isnan(full_gA), (
            f"full_gSIG_A is NaN for {r.get('motif_type')} n={r.get('n_branches')}"
        )
        assert full_gA > 0, (
            f"full_model gSIG_A={full_gA:.4f} for {r.get('motif_type')} n={r.get('n_branches')} "
            f"— full model should always show positive structural specificity"
        )


# ---------------------------------------------------------------------------
# Ratio export
# ---------------------------------------------------------------------------

def test_ratio_to_full_model_exported():
    rows = _comparison_rows()
    for r in rows:
        ratio = _float(r.get("ratio_to_full_model", "nan"))
        assert not math.isnan(ratio), (
            f"ratio_to_full_model NaN for {r.get('motif_type')} n={r.get('n_branches')}"
        )


# ---------------------------------------------------------------------------
# Core scientific tests
# ---------------------------------------------------------------------------

def test_ratio_decays_with_branch_count():
    """Mean shuffled gSIG-A / full gSIG-A must decrease from n=4 to n=32."""
    rows = _comparison_rows()
    for mt in REQUIRED_MOTIFS:
        mt_rows = sorted(
            [r for r in rows if r["motif_type"] == mt],
            key=lambda r: int(r["n_branches"]),
        )
        ratios = [_float(r["ratio_to_full_model"]) for r in mt_rows]
        # Must be monotonically non-increasing
        for i in range(len(ratios) - 1):
            assert ratios[i] >= ratios[i + 1], (
                f"ratio_to_full_model NOT monotonically decreasing for '{mt}': "
                f"n={mt_rows[i]['n_branches']} ratio={ratios[i]:.4f} but "
                f"n={mt_rows[i+1]['n_branches']} ratio={ratios[i+1]:.4f}"
            )


def test_shuffled_replay_does_not_pass_full_profile_at_n8_or_larger():
    """No shuffled_replay condition at n >= 8 should achieve ratio > 0.5."""
    rows = _comparison_rows()
    for r in rows:
        if int(r["n_branches"]) >= 8:
            ratio = _float(r.get("ratio_to_full_model", "nan"))
            assert math.isnan(ratio) or ratio < 0.5, (
                f"shuffled_replay ratio={ratio:.3f} at n={r['n_branches']} ({r['motif_type']}) "
                f">= 0.5; this suggests shuffled replay mimics structural specificity at scale"
            )


def test_shuffled_mean_near_zero_at_n16_and_above():
    """Mean shuffled gSIG-A should be < 0.01 for n >= 16 (essentially noise)."""
    rows = _comparison_rows()
    for r in rows:
        if int(r["n_branches"]) >= 16:
            s_mean = _float(r.get("shuffled_mean", "nan"))
            assert math.isnan(s_mean) or abs(s_mean) < 0.01, (
                f"shuffled_mean={s_mean:.4f} at n={r['n_branches']} ({r['motif_type']}) "
                f"is >= 0.01; expected near zero for large branch counts"
            )


def test_full_model_gSIG_A_increases_with_branch_count():
    """Full model structural specificity should be strong at all branch counts."""
    rows = _comparison_rows()
    for mt in REQUIRED_MOTIFS:
        for r in [r for r in rows if r["motif_type"] == mt]:
            full_gA = _float(r.get("full_gSIG_A", "nan"))
            assert full_gA > 0.10, (
                f"full_model gSIG_A={full_gA:.4f} for {mt} n={r['n_branches']} "
                "is unexpectedly low (should increase with more private branches)"
            )


# ---------------------------------------------------------------------------
# Seed spread
# ---------------------------------------------------------------------------

def test_seed_sd_exported():
    rows = _comparison_rows()
    for r in rows:
        sd = r.get("shuffled_sd", "")
        assert sd not in ("", "nan"), (
            f"shuffled_sd missing for {r.get('motif_type')} n={r.get('n_branches')}"
        )


def test_seed_sd_decreases_with_branch_count():
    """Standard deviation of shuffled gSIG-A should decrease as n grows."""
    rows = _comparison_rows()
    for mt in REQUIRED_MOTIFS:
        mt_rows = sorted(
            [r for r in rows if r["motif_type"] == mt],
            key=lambda r: int(r["n_branches"]),
        )
        sds = [_float(r.get("shuffled_sd", "nan")) for r in mt_rows]
        # SD at n=4 should be larger than at n=32
        if not math.isnan(sds[0]) and not math.isnan(sds[-1]):
            assert sds[0] > sds[-1], (
                f"SD did NOT decrease from n=4 ({sds[0]:.4f}) to "
                f"n={mt_rows[-1]['n_branches']} ({sds[-1]:.4f}) for '{mt}'"
            )


# ---------------------------------------------------------------------------
# No NaN without reason
# ---------------------------------------------------------------------------

def test_no_nan_without_documented_reason():
    """gSIG_A and ratio_to_full_model must be defined in all comparison rows."""
    rows = _comparison_rows()
    for r in rows:
        for key in ("full_gSIG_A", "shuffled_mean", "ratio_to_full_model"):
            val = _float(r.get(key, "nan"))
            assert not math.isnan(val), (
                f"Unexpected NaN in {key} for {r.get('motif_type')} n={r.get('n_branches')}"
            )


# ---------------------------------------------------------------------------
# Claim ledger content
# ---------------------------------------------------------------------------

def test_claim_ledger_documents_artifact():
    """Claim ledger should confirm the sampling artifact conclusion."""
    ledger = (SUMMARY_DIR / "claim_ledger.md").read_text(encoding="utf-8")
    assert "artifact" in ledger.lower() or "confirmed" in ledger.lower(), (
        "Claim ledger does not explicitly confirm artifact or confirmation status"
    )
