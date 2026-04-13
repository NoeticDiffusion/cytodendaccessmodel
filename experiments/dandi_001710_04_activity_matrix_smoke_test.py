"""Experiment 04: Activity matrix smoke test for DANDI 001710.

Goal
----
Extract one conservative dF matrix, confirm time alignment to 2P-aligned
behavior, and save a tiny QC export.

Success condition
-----------------
A real session can be turned into aligned behavior + activity tables without
manual hacking.

Usage
-----
    python experiments/dandi_001710_04_activity_matrix_smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from dandi_analysis.inventory import discover_nwb_assets
from dandi_analysis.readiness import check_readiness
from dandi_analysis.dataset_001710.index import parse_subject_session
from dandi_analysis.dataset_001710.behavior import load_behavior_table
from dandi_analysis.dataset_001710.ophys import load_ophys_matrix
from dandi_analysis.dataset_001710.qc import (
    check_behavior_channels,
    check_frame_count_consistency,
    check_roi_count,
    format_qc_report,
)
from dandi_analysis.dataset_001710.exports import export_dff_matrix, export_qc_report

DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "001710"
OUTPUT_DIR = ROOT / "data" / "dandi" / "triage" / "001710"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    assets = discover_nwb_assets(DATA_ROOT)
    if not assets:
        print(f"No NWB files found under {DATA_ROOT}.")
        return

    canonical_ready = [
        a for a in assets if a.is_canonical and check_readiness(a.path).is_ready
    ]
    if not canonical_ready:
        print("No ready sessions found.")
        return

    # Use the first ready session as the smoke-test target
    target = canonical_ready[0]
    _, _, day, _ = parse_subject_session(target.path)
    print(f"Smoke test target: {target.path.name}  (day {day})")

    # Load behavior
    print("  Loading 2P-aligned behavior ...")
    beh = load_behavior_table(target.path, source="2p")
    if beh is None:
        print("  ERROR: could not load behavior table.")
        return
    print(f"    {beh.n_frames} frames, channels: {beh.keys()}")

    # Load dF/F
    print("  Loading dF/F matrix ...")
    ophys = load_ophys_matrix(target.path, signal="dff")
    if ophys is None:
        print("  ERROR: could not load dF/F matrix.")
        return
    print(f"    shape: ({ophys.n_frames}, {ophys.n_rois}), rate: {ophys.sampling_rate:.2f} Hz")

    # Alignment check
    n_common = min(ophys.n_frames, beh.n_frames)
    if ophys.timestamps is not None and beh.timestamps is not None:
        ts_diff = float(np.nanmean(np.abs(ophys.timestamps[:n_common] - beh.timestamps[:n_common])))
        print(f"    Mean timestamp offset (ophys vs behavior): {ts_diff:.4f} s")

    # QC
    issues: list = []
    issues.extend(check_behavior_channels(beh))
    issues.extend(check_roi_count(ophys))
    issues.extend(check_frame_count_consistency(ophys, beh))
    qc_text = format_qc_report(issues, session_label=f"day{day}_smoke_test")
    print(qc_text)

    # Export a tiny slice (first 500 frames) as QC artifact
    tiny_ophys = type(ophys)(
        session_path=ophys.session_path,
        signal=ophys.signal,
        data=ophys.data[:500, :],
        timestamps=ophys.timestamps[:500],
        roi_ids=ophys.roi_ids,
        sampling_rate=ophys.sampling_rate,
        n_frames=min(500, ophys.n_frames),
        n_rois=ophys.n_rois,
        metadata={"day": day, "slice": "first_500_frames"},
    )
    npz_path = export_dff_matrix(tiny_ophys, OUTPUT_DIR, filename=f"dff_smoke_day{day:02d}.npz")
    qc_path = export_qc_report(qc_text, OUTPUT_DIR, filename=f"qc_smoke_day{day:02d}.md")
    print(f"  Exported: {npz_path.name}, {qc_path.name}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
