"""Tests for dandi_analysis.inventory — no NWB files required."""
from __future__ import annotations

import os
from pathlib import Path


def _make_nwb(root: Path, rel: str, size: int = 4096) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00" * size)
    return p


# ---------------------------------------------------------------------------
# discover_nwb_assets
# ---------------------------------------------------------------------------

def test_discover_empty_dir(tmp_path):
    from dandi_analysis.inventory import discover_nwb_assets
    assets = discover_nwb_assets(tmp_path)
    assert assets == []


def test_discover_nonexistent_dir(tmp_path):
    from dandi_analysis.inventory import discover_nwb_assets
    assets = discover_nwb_assets(tmp_path / "does_not_exist")
    assert assets == []


def test_discover_canonical_only(tmp_path):
    from dandi_analysis.inventory import discover_nwb_assets
    _make_nwb(tmp_path, "sub-mouse01/ses-01_ecephys.nwb")
    _make_nwb(tmp_path, "sub-mouse02/ses-02_ecephys.nwb")

    assets = discover_nwb_assets(tmp_path)
    assert len(assets) == 2
    assert all(a.is_canonical for a in assets)
    assert all(a.duplicate_of is None for a in assets)


def test_discover_loose_root_files_are_non_canonical(tmp_path):
    from dandi_analysis.inventory import discover_nwb_assets
    _make_nwb(tmp_path, "some_loose_file.nwb")
    assets = discover_nwb_assets(tmp_path)
    assert len(assets) == 1
    assert not assets[0].is_canonical


def test_discover_duplicate_matched_by_stem(tmp_path):
    from dandi_analysis.inventory import discover_nwb_assets
    canonical = _make_nwb(tmp_path, "sub-mouse01/ses-01.nwb")
    _make_nwb(tmp_path, "ses-01.nwb")  # loose duplicate with same stem

    assets = discover_nwb_assets(tmp_path)
    non_can = [a for a in assets if not a.is_canonical]
    assert len(non_can) == 1
    assert non_can[0].duplicate_of == canonical


def test_discover_canonical_first_in_ordering(tmp_path):
    from dandi_analysis.inventory import discover_nwb_assets
    _make_nwb(tmp_path, "loose.nwb")
    _make_nwb(tmp_path, "sub-a/session.nwb")

    assets = discover_nwb_assets(tmp_path)
    assert assets[0].is_canonical
    assert not assets[-1].is_canonical


def test_discover_non_nwb_ignored(tmp_path):
    from dandi_analysis.inventory import discover_nwb_assets
    (tmp_path / "data.h5").write_bytes(b"\x00" * 100)
    (tmp_path / "notes.txt").write_text("hello")
    assets = discover_nwb_assets(tmp_path)
    assert assets == []


# ---------------------------------------------------------------------------
# canonical_assets / duplicate_assets helpers
# ---------------------------------------------------------------------------

def test_canonical_assets_filter(tmp_path):
    from dandi_analysis.inventory import discover_nwb_assets, canonical_assets
    _make_nwb(tmp_path, "sub-a/s1.nwb")
    _make_nwb(tmp_path, "s1.nwb")
    assets = discover_nwb_assets(tmp_path)
    can = canonical_assets(assets)
    assert len(can) == 1
    assert can[0].is_canonical


def test_duplicate_assets_filter(tmp_path):
    from dandi_analysis.inventory import discover_nwb_assets, duplicate_assets
    _make_nwb(tmp_path, "sub-a/s1.nwb")
    _make_nwb(tmp_path, "s1.nwb")
    assets = discover_nwb_assets(tmp_path)
    dups = duplicate_assets(assets)
    assert len(dups) == 1
    assert not dups[0].is_canonical


# ---------------------------------------------------------------------------
# build_inventory_report
# ---------------------------------------------------------------------------

def test_inventory_report_contains_expected_strings(tmp_path):
    from dandi_analysis.inventory import discover_nwb_assets, build_inventory_report
    _make_nwb(tmp_path, "sub-mouse01/ses-01.nwb")
    assets = discover_nwb_assets(tmp_path)
    report = build_inventory_report(assets)
    assert "NWB Asset Inventory" in report
    assert "Canonical: 1" in report
    assert "ses-01.nwb" in report
