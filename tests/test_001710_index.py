"""Tests for dandi_analysis.dataset_001710.index — no NWB files required."""
from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# parse_subject_session
# ---------------------------------------------------------------------------

def test_subject_group_from_plain_subject_id():
    from dandi_analysis.dataset_001710.index import subject_group
    assert subject_group("SparseKO-7") == "SparseKO"
    assert subject_group("Cre-2") == "Cre"
    assert subject_group("Ctrl-9") == "Ctrl"


def test_subject_group_strips_sub_prefix():
    from dandi_analysis.dataset_001710.index import subject_group
    assert subject_group("sub-SparseKO-7") == "SparseKO"


def test_subject_group_unknown():
    from dandi_analysis.dataset_001710.index import subject_group
    assert subject_group("Mouse-1") == "Unknown"


def test_parse_canonical_path():
    from dandi_analysis.dataset_001710.index import parse_subject_session
    p = Path(
        "data/dandi/raw/001710/sub-Cre-1/"
        "sub-Cre-1_ses-ymaze-day3-scan0-novel-arm-1_behavior+ophys.nwb"
    )
    subject, label, day, novel_arm = parse_subject_session(p)
    assert subject == "Cre-1"
    assert day == 3
    assert novel_arm == 1
    assert "day3" in label


def test_parse_day_zero():
    from dandi_analysis.dataset_001710.index import parse_subject_session
    p = Path("sub-Cre-1/sub-Cre-1_ses-ymaze-day0-scan0-novel-arm-1_behavior+ophys.nwb")
    _, _, day, _ = parse_subject_session(p)
    assert day == 0


def test_parse_missing_day():
    from dandi_analysis.dataset_001710.index import parse_subject_session
    p = Path("sub-Mouse1/sub-Mouse1_ses-freerunning_behavior+ophys.nwb")
    _, label, day, novel_arm = parse_subject_session(p)
    assert day == -1


def test_parse_missing_sub_token():
    from dandi_analysis.dataset_001710.index import parse_subject_session
    p = Path("data/some_ymaze_file.nwb")
    subject, label, day, novel_arm = parse_subject_session(p)
    assert subject == "unknown"


def test_parse_novel_arm_extraction():
    from dandi_analysis.dataset_001710.index import parse_subject_session
    p = Path("sub-X/sub-X_ses-ymaze-day5-scan0-novel-arm-2_behavior+ophys.nwb")
    _, _, _, novel_arm = parse_subject_session(p)
    assert novel_arm == 2


# ---------------------------------------------------------------------------
# build_session_index (synthetic ReadyNwbAsset, no real NWB needed)
# ---------------------------------------------------------------------------

def _make_ready_asset(path: Path):
    from dandi_analysis.contracts import ReadyNwbAsset
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 4096)
    return ReadyNwbAsset(
        path=path,
        size=4096,
        is_h5_openable=True,
        is_nwb_openable=True,
    )


def test_build_session_index_returns_rows(tmp_path):
    from dandi_analysis.dataset_001710.index import build_session_index
    assets = [
        _make_ready_asset(
            tmp_path / "sub-Cre-1" /
            f"sub-Cre-1_ses-ymaze-day{d}-scan0-novel-arm-1_behavior+ophys.nwb"
        )
        for d in range(3)
    ]
    rows = build_session_index(assets, read_metadata=False)
    assert len(rows) == 3


def test_build_session_index_day_order(tmp_path):
    from dandi_analysis.dataset_001710.index import build_session_index
    days = [2, 0, 1]
    assets = [
        _make_ready_asset(
            tmp_path / "sub-Cre-1" /
            f"sub-Cre-1_ses-ymaze-day{d}-scan0-novel-arm-1_behavior+ophys.nwb"
        )
        for d in days
    ]
    rows = build_session_index(assets, read_metadata=False)
    row_days = [r.metadata["day"] for r in rows]
    assert row_days == sorted(row_days)


def test_build_session_index_skips_not_ready(tmp_path):
    from dandi_analysis.contracts import ReadyNwbAsset
    from dandi_analysis.dataset_001710.index import build_session_index
    bad = ReadyNwbAsset(
        path=tmp_path / "sub-X" / "sub-X_ses-ymaze-day0-scan0-novel-arm-1_behavior+ophys.nwb",
        size=0,
        is_h5_openable=False,
        is_nwb_openable=False,
        error="file_not_found",
    )
    rows = build_session_index([bad], read_metadata=False)
    assert rows == []


def test_build_session_index_fallback_without_annotation(tmp_path):
    """build_session_index must not crash when annotation blob is absent."""
    from dandi_analysis.dataset_001710.index import build_session_index
    asset = _make_ready_asset(
        tmp_path / "sub-Cre-1" /
        "sub-Cre-1_ses-ymaze-day1-scan0-novel-arm-1_behavior+ophys.nwb"
    )
    rows = build_session_index([asset], read_metadata=False)
    assert len(rows) == 1
    assert rows[0].metadata["day"] == 1
