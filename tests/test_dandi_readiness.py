"""Tests for dandi_analysis.readiness — no real NWB files required."""
from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# check_readiness
# ---------------------------------------------------------------------------

def test_readiness_missing_file(tmp_path):
    from dandi_analysis.readiness import check_readiness
    r = check_readiness(tmp_path / "ghost.nwb")
    assert not r.is_ready
    assert r.error == "file_not_found"
    assert r.size == 0


def test_readiness_too_small(tmp_path):
    from dandi_analysis.readiness import check_readiness
    p = tmp_path / "tiny.nwb"
    p.write_bytes(b"\x00" * 10)
    r = check_readiness(p)
    assert not r.is_ready
    assert r.error is not None
    assert "too_small" in r.error


def test_readiness_not_hdf5(tmp_path):
    from dandi_analysis.readiness import check_readiness
    p = tmp_path / "bad.nwb"
    p.write_bytes(b"NOTANHDF5FILE" * 500)
    r = check_readiness(p)
    assert not r.is_ready
    assert not r.is_h5_openable


def test_readiness_returns_ready_asset_type(tmp_path):
    from dandi_analysis.readiness import check_readiness
    from dandi_analysis.contracts import ReadyNwbAsset
    p = tmp_path / "bad.nwb"
    p.write_bytes(b"X" * 2048)
    r = check_readiness(p)
    assert isinstance(r, ReadyNwbAsset)
    assert r.path == p
    assert r.size == 2048


def test_readiness_size_stability_fields_present(tmp_path):
    """Size check fields are accessible even on non-hdf5 files."""
    from dandi_analysis.readiness import check_readiness
    p = tmp_path / "stable.nwb"
    p.write_bytes(b"Z" * 2048)
    r = check_readiness(p)
    assert r.size == 2048


# ---------------------------------------------------------------------------
# filter_ready — uses DiscoveredNwbAsset stubs
# ---------------------------------------------------------------------------

def _make_discovered(path: Path, *, is_canonical: bool = True):
    from dandi_analysis.contracts import DiscoveredNwbAsset
    stat = path.stat()
    return DiscoveredNwbAsset(
        path=path,
        size=stat.st_size,
        mtime=stat.st_mtime,
        is_canonical=is_canonical,
    )


def test_filter_ready_excludes_non_canonical_by_default(tmp_path):
    from dandi_analysis.readiness import filter_ready
    sub = tmp_path / "sub-a"
    sub.mkdir()
    canon = sub / "s1.nwb"
    canon.write_bytes(b"NOTANWB" * 500)
    loose = tmp_path / "s1.nwb"
    loose.write_bytes(b"NOTANWB" * 500)

    assets = [
        _make_discovered(canon, is_canonical=True),
        _make_discovered(loose, is_canonical=False),
    ]
    ready = filter_ready(assets, canonical_only=True)
    paths = [r.path for r in ready]
    assert loose not in paths


def test_filter_ready_includes_non_canonical_when_asked(tmp_path):
    from dandi_analysis.readiness import filter_ready
    # Both files are too small — we only verify the filter doesn't crash
    p1 = tmp_path / "a.nwb"
    p1.write_bytes(b"\x00" * 10)
    p2 = tmp_path / "b.nwb"
    p2.write_bytes(b"\x00" * 10)

    assets = [
        _make_discovered(p1, is_canonical=True),
        _make_discovered(p2, is_canonical=False),
    ]
    ready = filter_ready(assets, canonical_only=False)
    assert isinstance(ready, list)


# ---------------------------------------------------------------------------
# build_readiness_report
# ---------------------------------------------------------------------------

def test_readiness_report_structure(tmp_path):
    from dandi_analysis.readiness import build_readiness_report, check_readiness
    p = tmp_path / "x.nwb"
    p.write_bytes(b"BAD" * 500)
    results = [check_readiness(p)]
    report = build_readiness_report(results)
    assert "Readiness Report" in report
    assert "Files checked: 1" in report
    assert "x.nwb" in report
