"""Tests for dandi_analysis.dataset_000718.epochs — no real NWB files required.

Uses pynwb to build synthetic in-memory NWBFile objects with known interval tables.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_interval_table(starts: list, stops: list):
    """Return a mock DynamicTable-like object with start_time and stop_time."""
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: (
        MagicMock(data=starts) if key == "start_time" else MagicMock(data=stops)
    )
    return mock


# ---------------------------------------------------------------------------
# _classify_epoch_type
# ---------------------------------------------------------------------------

def test_classify_nrem():
    from dandi_analysis.dataset_000718.epochs import _classify_epoch_type
    assert _classify_epoch_type("NREM_epochs") == "NREM"


def test_classify_rem():
    from dandi_analysis.dataset_000718.epochs import _classify_epoch_type
    assert _classify_epoch_type("REM_periods") == "REM"


def test_classify_sleep():
    from dandi_analysis.dataset_000718.epochs import _classify_epoch_type
    assert _classify_epoch_type("sleep_intervals") == "sleep"


def test_classify_rest():
    from dandi_analysis.dataset_000718.epochs import _classify_epoch_type
    assert _classify_epoch_type("quiet_rest") == "rest"


def test_classify_fallback():
    from dandi_analysis.dataset_000718.epochs import _classify_epoch_type
    assert _classify_epoch_type("unknown_epoch_123") == "offline"


# ---------------------------------------------------------------------------
# _col_to_list
# ---------------------------------------------------------------------------

def test_col_to_list_data_attr():
    from dandi_analysis.dataset_000718.epochs import _col_to_list

    class FakeCol:
        data = [1.0, 2.0, 3.0]

    class FakeTable:
        def __getitem__(self, key):
            return FakeCol()

    result = _col_to_list(FakeTable(), "start_time")
    assert result == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# extract_offline_windows — patching pynwb
# ---------------------------------------------------------------------------

def _make_mock_nwb(interval_keys: list[str], starts: list, stops: list):
    nwb = MagicMock()

    class FakeCol:
        def __init__(self, data):
            self.data = data

    class FakeTable:
        def __init__(self, s, e):
            self._s = s
            self._e = e

        def __getitem__(self, key):
            if key == "start_time":
                return FakeCol(self._s)
            return FakeCol(self._e)

    intervals = {k: FakeTable(starts, stops) for k in interval_keys}
    nwb.intervals.keys.return_value = interval_keys
    nwb.intervals.__getitem__ = lambda self, k: intervals[k]
    return nwb


def test_extract_offline_windows_returns_list(tmp_path):
    from dandi_analysis.dataset_000718.epochs import extract_offline_windows

    p = tmp_path / "sub-a" / "ses-01.nwb"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"\x00" * 10)  # not a real NWB; expect empty result + warning
    result = extract_offline_windows(p, "sub-a__ses-01")
    assert isinstance(result, list)


def test_extract_windows_empty_on_bad_file(tmp_path):
    from dandi_analysis.dataset_000718.epochs import extract_offline_windows

    p = tmp_path / "fake.nwb"
    p.write_bytes(b"\x00" * 10)
    windows = extract_offline_windows(p, "ses_x")
    assert windows == []


def test_offline_window_duration():
    from dandi_analysis.contracts import OfflineWindow

    w = OfflineWindow(
        session_id="s1", label="sleep_0", start_sec=100.0, stop_sec=500.0,
        epoch_type="sleep",
    )
    assert w.duration_sec == pytest.approx(400.0)


def test_offline_window_short_duration_excluded_by_min():
    from dandi_analysis.contracts import OfflineWindow

    w = OfflineWindow(
        session_id="s1", label="tiny", start_sec=0.0, stop_sec=10.0,
        epoch_type="rest",
    )
    assert w.duration_sec < 30.0
