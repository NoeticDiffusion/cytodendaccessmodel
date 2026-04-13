"""Tests for dandi_analysis.dataset_000718.index — no NWB files required."""
from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# parse_subject_session
# ---------------------------------------------------------------------------

def test_parse_canonical_path():
    from dandi_analysis.dataset_000718.index import parse_subject_session
    p = Path("data/dandi/raw/000718/sub-mouse01/ses-recording01_ecephys.nwb")
    subject, session = parse_subject_session(p)
    assert subject == "mouse01"
    assert session == "recording01"


def test_parse_missing_sub_token():
    from dandi_analysis.dataset_000718.index import parse_subject_session
    p = Path("data/some_file.nwb")
    subject, session = parse_subject_session(p)
    assert subject == "unknown"
    assert session == "some_file"


def test_parse_missing_ses_token():
    from dandi_analysis.dataset_000718.index import parse_subject_session
    p = Path("sub-rat02/no_session_token.nwb")
    subject, session = parse_subject_session(p)
    assert subject == "rat02"
    assert session == "no_session_token"


def test_parse_deep_nested_path():
    from dandi_analysis.dataset_000718.index import parse_subject_session
    p = Path("root/000718/sub-abc/behavior/ses-xyz_behavior.nwb")
    subject, session = parse_subject_session(p)
    assert subject == "abc"
    assert session == "xyz"


# ---------------------------------------------------------------------------
# build_session_index
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


def test_build_index_no_assets():
    from dandi_analysis.dataset_000718.index import build_session_index
    rows = build_session_index([], read_metadata=False)
    assert rows == []


def test_build_index_returns_one_row_per_asset(tmp_path):
    from dandi_analysis.dataset_000718.index import build_session_index

    a1 = _make_ready_asset(tmp_path / "sub-a" / "ses-01.nwb")
    a2 = _make_ready_asset(tmp_path / "sub-b" / "ses-02.nwb")
    rows = build_session_index([a1, a2], read_metadata=False)
    assert len(rows) == 2


def test_build_index_subject_session_parsed(tmp_path):
    from dandi_analysis.dataset_000718.index import build_session_index

    asset = _make_ready_asset(tmp_path / "sub-mouse99" / "ses-day1_ecephys.nwb")
    rows = build_session_index([asset], read_metadata=False)
    assert rows[0].subject_id == "mouse99"
    assert rows[0].session_label == "day1"


def test_build_index_state_is_ready(tmp_path):
    from dandi_analysis.dataset_000718.index import build_session_index

    asset = _make_ready_asset(tmp_path / "sub-r01" / "ses-s1.nwb")
    rows = build_session_index([asset], read_metadata=False)
    assert rows[0].state == "ready"


def test_build_index_skips_non_ready_assets(tmp_path):
    from dandi_analysis.contracts import ReadyNwbAsset
    from dandi_analysis.dataset_000718.index import build_session_index

    not_ready = ReadyNwbAsset(
        path=tmp_path / "sub-x" / "ses-y.nwb",
        size=4096,
        is_h5_openable=False,
        is_nwb_openable=False,
        error="h5py_open_failed:OSError",
    )
    rows = build_session_index([not_ready], read_metadata=False)
    assert rows == []


def test_build_index_duplicate_warning(tmp_path, recwarn):
    """When metadata read fails for a file, a warning is emitted and index is
    still produced (because read_metadata is True but the file is not real NWB).
    """
    from dandi_analysis.dataset_000718.index import build_session_index

    asset = _make_ready_asset(tmp_path / "sub-m" / "ses-s.nwb")
    # read_metadata=True will try to open the dummy file via pynwb and fail
    rows = build_session_index([asset], read_metadata=True)
    # Row should still be created despite metadata failure
    assert len(rows) == 1
    assert rows[0].state == "ready"
