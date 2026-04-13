"""Experiment 05: Within-day place tuning for DANDI 001710.

Goal
----
Compute simple ROI tuning curves by position and arm, compare against embedded
place-cell metadata, and summarise within-day context sensitivity.

Success condition
-----------------
We have the first transparent place-code observable from the dataset.

Usage
-----
    python experiments/dandi_001710_05_within_day_place_tuning.py
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
from dandi_analysis.dataset_001710.trials import build_trial_table
from dandi_analysis.dataset_001710.placecode import (
    arm_tuning,
    compute_tuning_curves,
    split_half_reliability,
)
from dandi_analysis.dataset_001710.remapping import within_day_arm_separation
from dandi_analysis.dataset_001710.exports import export_tuning_summary

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

    print(f"Processing {len(canonical_ready)} ready session(s)...")

    for asset in canonical_ready:
        _, _, day, _ = parse_subject_session(asset.path)
        print(f"\n  Day {day}: {asset.path.name}")

        beh = load_behavior_table(asset.path, source="2p")
        if beh is None:
            print("    Skipping: behavior not available.")
            continue

        ophys = load_ophys_matrix(asset.path, signal="dff")
        if ophys is None:
            print("    Skipping: ophys not available.")
            continue

        trials = build_trial_table(asset.path, day=day)

        # Full-session tuning curves
        try:
            tc_all = compute_tuning_curves(ophys, beh, n_bins=50)
            print(f"    Tuning curves: {tc_all.n_rois} ROIs x {tc_all.n_bins} bins")
            print(f"    Occupancy (non-zero bins): {int(np.sum(tc_all.occupancy > 0))}")

            # Split-half reliability
            rel = split_half_reliability(tc_all.tuning_curves, ophys, beh, n_splits=5)
            finite_rel = rel[np.isfinite(rel)]
            if len(finite_rel) > 0:
                print(f"    Split-half reliability: mean={np.mean(finite_rel):.3f}  "
                      f"median={np.median(finite_rel):.3f}  "
                      f"n_valid={len(finite_rel)}")

            export_tuning_summary(tc_all, OUTPUT_DIR, label=f"day{day:02d}_all")
        except Exception as exc:
            print(f"    Tuning curve error: {exc}")
            continue

        # Arm-separated tuning
        try:
            arm_tc = arm_tuning(ophys, beh, trials, n_bins=50)
            if arm_tc.left is not None and arm_tc.right is not None:
                sep = within_day_arm_separation(arm_tc.left, arm_tc.right)
                print(f"    Arm separation (pop. vector r): {sep.similarity:.4f}  "
                      f"(n_rois={sep.n_rois})")
                export_tuning_summary(arm_tc.left, OUTPUT_DIR, label=f"day{day:02d}_left")
                export_tuning_summary(arm_tc.right, OUTPUT_DIR, label=f"day{day:02d}_right")
            else:
                print("    Arm tuning: insufficient trials for one or both arms.")
        except Exception as exc:
            print(f"    Arm tuning error: {exc}")

    print("\nDone.  Tuning summaries written to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
