"""Tests for E019R — Signature Protocol Lock and Rescue Audit.

Run with: pytest tests/test_e019r_signature_protocol_lock.py -v
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

OUT_ROOT    = REPO_ROOT / "results" / "e019r_signature_protocol_lock"
SUMMARY_DIR = OUT_ROOT / "summary"

# ---------------------------------------------------------------------------
# 1. Shared module exists and imports correctly
# ---------------------------------------------------------------------------

def test_shared_signature_module_exists():
    path = REPO_ROOT / "src" / "cytodend_accessmodel" / "signatures.py"
    assert path.exists(), "signatures.py not found in src/cytodend_accessmodel/"


def test_shared_module_exports_required_symbols():
    from cytodend_accessmodel.signatures import (
        compute_signature_profile,
        SignatureInputs,
        SignatureProfile,
        RescueConditionResult,
        DEFAULT_THRESHOLDS,
        inputs_from_run_dict,
    )
    assert callable(compute_signature_profile)
    assert DEFAULT_THRESHOLDS


def test_signature_thresholds_are_exported():
    path = SUMMARY_DIR / "signature_thresholds.json"
    assert path.exists(), "signature_thresholds.json not found"
    data = json.loads(path.read_text(encoding="utf-8"))
    for k in ["SIG_A", "SIG_B", "SIG_C", "SIG_D", "SIG_E_normalized", "SIG_E_raw"]:
        assert k in data, f"Missing threshold key: {k}"
        assert data[k] > 0


def test_signature_units_are_exported():
    path = SUMMARY_DIR / "signature_units.json"
    assert path.exists(), "signature_units.json not found"
    data = json.loads(path.read_text(encoding="utf-8"))
    for sig in ["SIG_A", "SIG_B", "SIG_C", "SIG_D", "SIG_E_raw", "SIG_E_normalized"]:
        assert sig in data, f"Missing unit for {sig}"
        val = data[sig]
        assert "percentage_point" not in val.lower() or sig == "SIG_D", (
            f"{sig}: units should not say 'percentage points' (E019R rule), got: {val}"
        )


# ---------------------------------------------------------------------------
# 2. SIG-E audit
# ---------------------------------------------------------------------------

def test_sig_e_audit_contains_required_rescue_conditions():
    path = SUMMARY_DIR / "sig_e_rescue_audit.csv"
    assert path.exists(), "sig_e_rescue_audit.csv not found"
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = {r["rescue_condition"] for r in rows}
    required = {
        "no_rescue",
        "targeted_overlap_rescue",
        "generic_plain_consolidation",
        "generic_all_branch_precue",
        "nonoverlap_branch_rescue",
    }
    missing = required - names
    assert not missing, f"Missing rescue conditions in audit: {missing}"


def test_no_percentage_point_label_for_unbounded_metric():
    """SIG-E normalized can exceed 1.0 — must NOT be labeled percentage points."""
    from cytodend_accessmodel.signatures import DEFAULT_THRESHOLDS
    # Verify the unit description in units.json does not say "pp" for SIG_E_normalized
    path = SUMMARY_DIR / "signature_units.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        norm_unit = data.get("SIG_E_normalized", "")
        assert "percentage_point" not in norm_unit.lower(), (
            f"SIG_E_normalized must not be labeled in percentage points, got: {norm_unit}"
        )


# ---------------------------------------------------------------------------
# 3. Canonical reproduction table
# ---------------------------------------------------------------------------

def test_canonical_reproduction_table_exists():
    path = SUMMARY_DIR / "canonical_reproduction_table.csv"
    assert path.exists(), "canonical_reproduction_table.csv not found"
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    experiments = {r["experiment"] for r in rows}
    assert "E019R" in experiments, "E019R row missing from reproduction table"


def test_e019r_passes_joint_profile():
    """E019R canonical must pass SIG-A through SIG-D (E not checked for threshold as NR-based)."""
    from cytodend_accessmodel.signatures import DEFAULT_THRESHOLDS
    path = SUMMARY_DIR / "canonical_reproduction_table.csv"
    if not path.exists():
        pytest.skip("canonical_reproduction_table.csv not found")
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    e019r = next((r for r in rows if r["experiment"] == "E019R"), None)
    assert e019r is not None
    for sig, thr_key in [
        ("SIG_A", "SIG_A"), ("SIG_B", "SIG_B"),
        ("SIG_C", "SIG_C"), ("SIG_D", "SIG_D"),
    ]:
        val = float(e019r[sig])
        assert val > DEFAULT_THRESHOLDS[thr_key], (
            f"E019R {sig} = {val:.4f} below threshold {DEFAULT_THRESHOLDS[thr_key]}"
        )


def test_e017_e018_sig_values_are_close():
    """E017 and E018 used identical protocols — their SIG-A to SIG-D should be identical."""
    path = SUMMARY_DIR / "canonical_reproduction_table.csv"
    if not path.exists():
        pytest.skip("canonical_reproduction_table.csv not found")
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    e017 = next((r for r in rows if r["experiment"] == "E017"), None)
    e018 = next((r for r in rows if r["experiment"] == "E018"), None)
    if e017 is None or e018 is None:
        pytest.skip("E017 or E018 row not found")
    for sig in ["SIG_A", "SIG_B", "SIG_C", "SIG_D"]:
        v17 = float(e017[sig])
        v18 = float(e018[sig])
        assert abs(v17 - v18) < 1e-6, (
            f"{sig}: E017={v17:.6f} vs E018={v18:.6f} — should be identical"
        )


# ---------------------------------------------------------------------------
# 4. Shared signature module correctness
# ---------------------------------------------------------------------------

def test_joint_pass_recomputed_from_shared_logic():
    """Compute signature profile from canonical values and verify joint pass."""
    from cytodend_accessmodel.signatures import (
        SignatureInputs, RescueConditionResult, compute_signature_profile, DEFAULT_THRESHOLDS
    )
    # Use rough canonical values known to pass
    targ = RescueConditionResult.compute("targeted_overlap_rescue", 0.645, 0.506, 0.647)
    ref  = RescueConditionResult.compute("generic_plain_consolidation", 0.465, 0.506, 0.647)

    inputs = SignatureInputs(
        mb_overlap_pre=0.500,
        mb_overlap_post_cons=0.799,
        mb_nonoverlap_mean_pre=0.500,
        mb_nonoverlap_mean_post_cons=0.671,
        L_pre=0.408,
        L_post_cons=0.647,
        context_separation=0.155,
        L_post_damage=0.506,
        recall_support_post_cons=1.42,
        recall_support_post_damage=1.41,
        rescue_conditions=[targ, ref],
        targeted_rescue_name="targeted_overlap_rescue",
        reference_rescue_name="generic_plain_consolidation",
    )
    profile = compute_signature_profile(inputs)
    assert profile.SIG_A > DEFAULT_THRESHOLDS["SIG_A"]
    assert profile.SIG_B > DEFAULT_THRESHOLDS["SIG_B"]
    assert profile.SIG_C > DEFAULT_THRESHOLDS["SIG_C"]
    assert profile.SIG_D > DEFAULT_THRESHOLDS["SIG_D"]
    assert profile.joint_protected_pass


def test_sig_e_overshoot_flag():
    """RescueConditionResult.overshoot is True when NR > 1.0."""
    from cytodend_accessmodel.signatures import RescueConditionResult
    r = RescueConditionResult.compute("test", L_post_rescue=0.80,
                                      L_post_damage=0.50, L_healthy=0.65)
    assert r.overshoot, f"NR = {r.normalized_recovery:.3f} should be > 1.0 (overshoot)"


def test_sig_e_negative_nr_when_recovery_worse():
    """NR < 0 when post-rescue L is below post-damage L (active decay scenario)."""
    from cytodend_accessmodel.signatures import RescueConditionResult
    r = RescueConditionResult.compute("test", L_post_rescue=0.45,
                                      L_post_damage=0.506, L_healthy=0.647)
    assert r.normalized_recovery < 0, (
        f"Expected NR < 0 for ongoing decay, got {r.normalized_recovery:.4f}"
    )


def test_e017_e018_e019_can_call_shared_signature_logic():
    """Verify that calling shared module with E017-style inputs produces consistent output."""
    from cytodend_accessmodel.signatures import (
        SignatureInputs, RescueConditionResult, compute_signature_profile, DEFAULT_THRESHOLDS
    )
    # E017/E018 canonical values from saved file
    targ = RescueConditionResult.compute(
        "targeted_overlap_rescue", 0.6552, 0.5058, 0.6472
    )
    ref  = RescueConditionResult.compute(
        "generic_plain_consolidation", 0.6081, 0.5058, 0.6472
    )
    inputs = SignatureInputs(
        mb_overlap_pre=0.50, mb_overlap_post_cons=0.799,
        mb_nonoverlap_mean_pre=0.50, mb_nonoverlap_mean_post_cons=0.671,
        L_pre=0.4075, L_post_cons=0.6472,
        context_separation=0.1545,
        L_post_damage=0.5058,
        recall_support_post_cons=1.424, recall_support_post_damage=1.411,
        rescue_conditions=[targ, ref],
        targeted_rescue_name="targeted_overlap_rescue",
        reference_rescue_name="generic_plain_consolidation",
        protocol_name="e017_e018_probe_cue_variant",
    )
    profile = compute_signature_profile(inputs)
    assert profile.SIG_A > DEFAULT_THRESHOLDS["SIG_A"]
    assert profile.SIG_B > DEFAULT_THRESHOLDS["SIG_B"]
    assert abs(profile.SIG_E_normalized - 0.3329) < 0.01, (
        f"SIG_E_normalized from E017 values should be ~0.333, got {profile.SIG_E_normalized:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. Article language file exists
# ---------------------------------------------------------------------------

def test_article_signature_language_exists():
    path = SUMMARY_DIR / "article_signature_language.md"
    assert path.exists(), "article_signature_language.md not found"
    content = path.read_text(encoding="utf-8")
    assert "SIG-A and SIG-B" in content
    assert "architectural" in content.lower()
    assert "is not" in content.lower() and "diagnostic" in content.lower()
    assert "percentage points" not in content.lower() or "SIG-D" in content
