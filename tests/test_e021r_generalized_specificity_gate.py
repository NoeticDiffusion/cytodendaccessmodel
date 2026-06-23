"""Tests for E021R — Generalized Specificity Gate and Hub-Failure Audit.

Run with: pytest tests/test_e021r_generalized_specificity_gate.py -v
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

OUT_ROOT    = REPO_ROOT / "results" / "e021r_generalized_specificity_gate"
SUMMARY_DIR = OUT_ROOT / "summary"

from exp021r_generalized_specificity_gate import (
    classify_row, CLASS_MAP, CLAIM_STATUS_MAP,
    SPECIFICITY_THRESHOLDS, _fl_band,
)


def _load_classification() -> list[dict]:
    path = SUMMARY_DIR / "motif_specificity_classification.csv"
    if not path.exists():
        pytest.skip("motif_specificity_classification.csv not found — run exp021r first")
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Output file tests
# ---------------------------------------------------------------------------

def test_specificity_classification_file_exists():
    path = SUMMARY_DIR / "motif_specificity_classification.csv"
    assert path.exists(), "motif_specificity_classification.csv not found"
    rows = _load_classification()
    assert rows, "Classification file is empty"


def test_specificity_thresholds_file_exists():
    path = SUMMARY_DIR / "specificity_thresholds.json"
    assert path.exists(), "specificity_thresholds.json not found"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "fl_rate_good"     in data
    assert "fl_rate_moderate" in data
    assert "fl_rate_poor"     in data


def test_article_interpretation_class_present():
    rows = _load_classification()
    for row in rows:
        assert "article_interpretation_class" in row, \
            f"Missing article_interpretation_class in row: {row.get('run_id')}"
        assert row["article_interpretation_class"] != "", \
            f"Empty article_interpretation_class: {row.get('run_id')}"


def test_all_required_columns_present():
    rows = _load_classification()
    if not rows: pytest.skip("No rows")
    cols = set(rows[0].keys())
    required = {
        "motif_type", "n_branches", "n_traces",
        "mechanistic_pass", "specificity_pass",
        "false_linking_rate", "specificity_index",
        "universal_linking_flag", "article_interpretation_class",
        "claim_status",
    }
    assert required <= cols, f"Missing columns: {required - cols}"


# ---------------------------------------------------------------------------
# Classification correctness tests
# ---------------------------------------------------------------------------

def test_hub_overlap_is_not_clean_specificity_success():
    rows = _load_classification()
    hub_rows = [r for r in rows if r["motif_type"] == "hub_overlap"]
    assert hub_rows, "No hub_overlap rows found"
    for r in hub_rows:
        assert r["article_interpretation_class"] == "hub_overlinking_boundary", \
            f"hub_overlap should be hub_overlinking_boundary, got: {r['article_interpretation_class']}"
        assert str(r["specificity_pass"]).lower() in ("false", "0"), \
            f"hub_overlap should fail specificity, got: {r['specificity_pass']}"


def test_universal_linking_flag_for_hub():
    rows = _load_classification()
    hub_rows = [r for r in rows if r["motif_type"] == "hub_overlap"]
    assert hub_rows, "No hub_overlap rows found"
    for r in hub_rows:
        assert str(r["universal_linking_flag"]).lower() in ("true", "1"), \
            f"hub_overlap should have universal_linking_flag=True: {r.get('run_id')}"


def test_weak_overlap_is_classified_as_failure():
    rows = _load_classification()
    weak_rows = [r for r in rows if r["motif_type"] == "weak_overlap"]
    assert weak_rows, "No weak_overlap rows found"
    for r in weak_rows:
        assert r["article_interpretation_class"] == "weak_overlap_failure", \
            f"Expected weak_overlap_failure, got {r['article_interpretation_class']}"
        assert str(r["mechanistic_pass"]).lower() in ("false", "0"), \
            f"weak_overlap should fail mechanistic_pass"
        assert str(r["universal_linking_flag"]).lower() in ("false", "0"), \
            f"weak_overlap should NOT have universal_linking_flag"


def test_canonical_has_correct_classification():
    rows = _load_classification()
    canon_rows = [r for r in rows if r["motif_type"] == "canonical"]
    assert canon_rows, "No canonical rows found"
    for r in canon_rows:
        assert r["article_interpretation_class"] == "canonical_reference"
        assert str(r["mechanistic_pass"]).lower() in ("true", "1"), \
            "canonical should pass mechanistic"
        assert str(r["universal_linking_flag"]).lower() in ("false", "0"), \
            "canonical (2-trace) should NOT have universal_linking_flag"


def test_chain_overlap_has_false_linking_rate_below_one():
    rows = _load_classification()
    chain_rows = [r for r in rows if r["motif_type"] == "chain_overlap"]
    assert chain_rows, "No chain_overlap rows found"
    for r in chain_rows:
        fl_raw = r.get("false_linking_rate", "nan")
        assert fl_raw not in ("nan", "", "N/A"), \
            f"chain_overlap should have a defined false_linking_rate: {r.get('run_id')}"
        fl = float(fl_raw)
        assert fl < 1.0, \
            f"chain_overlap FL={fl:.3f} should be < 1.0 (local > distant linking)"
        assert fl < SPECIFICITY_THRESHOLDS["fl_rate_moderate"], \
            f"chain_overlap FL={fl:.3f} should be < moderate threshold " \
            f"({SPECIFICITY_THRESHOLDS['fl_rate_moderate']})"


def test_sparse_random_has_specificity_metrics():
    rows = _load_classification()
    sparse_rows = [r for r in rows if r["motif_type"] == "sparse_random"]
    assert sparse_rows, "No sparse_random rows found"
    # At least some sparse runs should have a defined FL rate
    defined_fl = [r for r in sparse_rows
                  if r.get("false_linking_rate", "nan") not in ("nan", "", "N/A")]
    assert defined_fl, "At least one sparse_random run should have a defined FL rate"


# ---------------------------------------------------------------------------
# classify_row unit tests (directly test the function)
# ---------------------------------------------------------------------------

def test_classify_row_hub_gives_universal_linking_flag():
    fake_row = {
        "run_id": "hub_test", "motif_type": "hub_overlap",
        "n_branches": "8", "n_traces": "4", "seed": "42",
        "joint_pass": "True",
        "false_linking_rate": "nan", "specificity_index": "nan",
        "gSIG_A": "0.19", "gSIG_B": "0.23", "gSIG_D": "0.14", "gSIG_E": "1.58",
    }
    result = classify_row(fake_row)
    assert result["universal_linking_flag"] is True
    assert result["specificity_pass"] is False
    assert result["mechanistic_pass"] is True
    assert result["article_interpretation_class"] == "hub_overlinking_boundary"


def test_classify_row_canonical_2_trace_no_universal_flag():
    fake_row = {
        "run_id": "canon_test", "motif_type": "canonical",
        "n_branches": "4", "n_traces": "2", "seed": "42",
        "joint_pass": "True",
        "false_linking_rate": "nan", "specificity_index": "nan",
        "gSIG_A": "0.13", "gSIG_B": "0.24", "gSIG_D": "0.14", "gSIG_E": "1.57",
    }
    result = classify_row(fake_row)
    assert result["universal_linking_flag"] is False, \
        "2-trace canonical should NOT trigger universal_linking_flag"
    assert result["mechanistic_pass"] is True
    assert result["article_interpretation_class"] == "canonical_reference"


def test_classify_row_chain_good_specificity():
    fake_row = {
        "run_id": "chain_test", "motif_type": "chain_overlap",
        "n_branches": "8", "n_traces": "3", "seed": "42",
        "joint_pass": "True",
        "false_linking_rate": "0.193", "specificity_index": "0.14",
        "gSIG_A": "0.16", "gSIG_B": "0.18", "gSIG_D": "0.07", "gSIG_E": "1.41",
    }
    result = classify_row(fake_row)
    assert result["mechanistic_pass"] is True
    assert result["specificity_pass"] is True
    assert result["universal_linking_flag"] is False
    assert result["false_linking_band"] == "good"


def test_fl_band_thresholds():
    assert _fl_band(0.0)  == "good"
    assert _fl_band(0.24) == "good"
    assert _fl_band(0.25) == "moderate"
    assert _fl_band(0.49) == "moderate"
    assert _fl_band(0.50) == "poor"
    assert _fl_band(0.99) == "poor"
    assert _fl_band(1.00) == "very_poor"
    assert _fl_band(float("nan")) == "undefined"


def test_article_language_file_exists_and_contains_hub_warning():
    path = SUMMARY_DIR / "article_motif_language.md"
    assert path.exists(), "article_motif_language.md not found"
    content = path.read_text(encoding="utf-8").lower()
    assert "hub" in content and "over-linking" in content.replace("overlinking","over-linking"), \
        "article language should contain hub over-linking warning"
    assert "weak-overlap" in content or "weak overlap" in content, \
        "article language should mention weak-overlap failure"
