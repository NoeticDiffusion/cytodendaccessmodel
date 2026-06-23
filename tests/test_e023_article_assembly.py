"""Tests for E023 — Article Figure Assembly and Results Draft.

Checks that the article folder contains all required outputs:
    - 6 publication figures in figures2/
    - draft.md (article text source)
    - v2_claim_ledger.md (with 'not supported' for biological validation)
    - v2_abstract_stub.md
    - v2_figure_manifest.md (with all 6 figure names)
    - v2_next_gaps.md
    - Typst .typ article file
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTICLE_DIR = (
    REPO_ROOT
    / "article"
    / "Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking"
)
FIGURES2 = ARTICLE_DIR / "figures2"

REQUIRED_FIGURES = [
    "Fig_e023_01_model_concept.png",
    "Fig_e023_02_canonical_traces.png",
    "Fig_e023_03_comparator_matrix.png",
    "Fig_e023_04_robustness_landscape.png",
    "Fig_e023_05_motif_scaling.png",
    "Fig_e023_06_shuffled_replay_audit.png",
]


class TestFigures:
    def test_figures2_directory_exists(self):
        assert FIGURES2.exists(), f"figures2/ not found at {FIGURES2}"
        assert FIGURES2.is_dir()

    def test_all_required_figures_exist(self):
        missing = [f for f in REQUIRED_FIGURES if not (FIGURES2 / f).exists()]
        assert not missing, f"Missing figures: {missing}"

    def test_required_figure_count(self):
        pngs = list(FIGURES2.glob("Fig_e023_0*.png"))
        assert len(pngs) >= 6, f"Expected at least 6 figures, found {len(pngs)}"

    def test_figures_have_nonzero_size(self):
        for f in REQUIRED_FIGURES:
            path = FIGURES2 / f
            if path.exists():
                assert path.stat().st_size > 1000, \
                    f"{f} is suspiciously small ({path.stat().st_size} bytes)"


class TestArticleTextSources:
    def test_results_draft_exists(self):
        draft = ARTICLE_DIR / "draft.md"
        assert draft.exists(), f"draft.md not found at {draft}"

    def test_draft_contains_abstract(self):
        draft = ARTICLE_DIR / "draft.md"
        if draft.exists():
            text = draft.read_text(encoding="utf-8")
            assert "Abstract" in text, "draft.md should contain an Abstract section"

    def test_typst_article_exists(self):
        typ = ARTICLE_DIR / "Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking.typ"
        assert typ.exists(), f"Typst article not found at {typ}"

    def test_typst_article_contains_figures(self):
        typ = ARTICLE_DIR / "Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking.typ"
        if typ.exists():
            text = typ.read_text(encoding="utf-8")
            assert "figures2/" in text, "Typst article should reference figures2/"
            assert "Fig_e023_01" in text, "Typst article should reference Fig_e023_01"

    def test_typst_article_contains_equations(self):
        typ = ARTICLE_DIR / "Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking.typ"
        if typ.exists():
            text = typ.read_text(encoding="utf-8")
            assert "M_b" in text, "Typst article should contain M_b variable"
            assert "L_(mu nu)" in text or "L_{\\mu\\nu}" in text or "L_" in text, \
                "Typst article should contain linking equation"


class TestClaimLedger:
    def test_claim_ledger_exists(self):
        ledger = ARTICLE_DIR / "v2_claim_ledger.md"
        assert ledger.exists(), f"v2_claim_ledger.md not found at {ledger}"

    def test_claim_ledger_marks_biological_validation_as_not_supported(self):
        ledger = ARTICLE_DIR / "v2_claim_ledger.md"
        if ledger.exists():
            text = ledger.read_text(encoding="utf-8").lower()
            assert "not supported" in text, \
                "Claim ledger should contain 'not supported' for biological validation claims"

    def test_claim_ledger_has_claim_ids(self):
        ledger = ARTICLE_DIR / "v2_claim_ledger.md"
        if ledger.exists():
            text = ledger.read_text(encoding="utf-8")
            assert "C01" in text, "Claim ledger should contain claim ID C01"
            assert "C15" in text or "C16" in text, \
                "Claim ledger should contain boundary claims C15 or C16"

    def test_claim_ledger_has_supported_claims(self):
        ledger = ARTICLE_DIR / "v2_claim_ledger.md"
        if ledger.exists():
            text = ledger.read_text(encoding="utf-8")
            assert "**supported**" in text or "| supported |" in text or "supported" in text, \
                "Claim ledger should contain supported claims"


class TestAbstractStub:
    def test_abstract_stub_exists(self):
        stub = ARTICLE_DIR / "v2_abstract_stub.md"
        assert stub.exists(), f"v2_abstract_stub.md not found at {stub}"

    def test_abstract_stub_word_count(self):
        stub = ARTICLE_DIR / "v2_abstract_stub.md"
        if stub.exists():
            text = stub.read_text(encoding="utf-8")
            # Count words in the abstract text paragraph (not headers/metadata)
            lines = [ln for ln in text.splitlines()
                     if ln.strip() and not ln.startswith("#") and not ln.startswith("**")]
            words = sum(len(ln.split()) for ln in lines)
            assert 150 <= words <= 350, \
                f"Abstract stub should be 150–350 words; found ~{words}"

    def test_abstract_stub_no_dandi_reference(self):
        stub = ARTICLE_DIR / "v2_abstract_stub.md"
        if stub.exists():
            text = stub.read_text(encoding="utf-8")
            assert "DANDI" not in text or "no dandi" in text.lower() or "without dandi" in text.lower(), \
                "Abstract stub should not reference DANDI in the abstract text itself"


class TestFigureManifest:
    def test_figure_manifest_exists(self):
        manifest = ARTICLE_DIR / "v2_figure_manifest.md"
        assert manifest.exists(), f"v2_figure_manifest.md not found at {manifest}"

    def test_figures_have_manifest_entries(self):
        manifest = ARTICLE_DIR / "v2_figure_manifest.md"
        if manifest.exists():
            text = manifest.read_text(encoding="utf-8")
            for f in REQUIRED_FIGURES:
                assert f in text, f"Figure {f} not found in v2_figure_manifest.md"

    def test_manifest_has_key_messages(self):
        manifest = ARTICLE_DIR / "v2_figure_manifest.md"
        if manifest.exists():
            text = manifest.read_text(encoding="utf-8")
            assert "Key message" in text, "Manifest should have Key message sections"
            assert "Caption stub" in text, "Manifest should have Caption stub sections"


class TestNextGaps:
    def test_next_gaps_exists(self):
        gaps = ARTICLE_DIR / "v2_next_gaps.md"
        assert gaps.exists(), f"v2_next_gaps.md not found at {gaps}"

    def test_next_gaps_has_three_sections(self):
        gaps = ARTICLE_DIR / "v2_next_gaps.md"
        if gaps.exists():
            text = gaps.read_text(encoding="utf-8")
            assert "Before manuscript rewrite" in text, \
                "v2_next_gaps.md should have 'Before manuscript rewrite' section"
            assert "Before DANDI" in text, \
                "v2_next_gaps.md should have 'Before DANDI reintroduction' section"
            assert "Before submission" in text, \
                "v2_next_gaps.md should have 'Before submission' section"
