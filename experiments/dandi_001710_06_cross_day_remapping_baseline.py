"""Experiment 06: Cross-day remapping baseline for DANDI 001710.

Goal
----
Compute day-to-day similarity for conservative population or ROI-level
observables, report left/right arm and block-sensitive shifts, and compare
to cheap nulls.

Success condition
-----------------
We have the first actual data-side remapping observable aligned with the
article's contextual retrieval story.

Usage
-----
    python experiments/dandi_001710_06_cross_day_remapping_baseline.py
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
from dandi_analysis.dataset_001710.placecode import compute_tuning_curves, arm_tuning
from dandi_analysis.dataset_001710.remapping import (
    build_day_similarity_matrix,
    cross_day_tuning_correlation,
    within_day_arm_separation,
)
from dandi_analysis.dataset_001710.nulls import circular_time_shift
from dandi_analysis.dataset_001710.exports import export_similarity_matrix

DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "001710"
OUTPUT_DIR = ROOT / "data" / "dandi" / "triage" / "001710"
N_NULLS = 20


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

    print(f"Computing tuning curves for {len(canonical_ready)} session(s)...")

    from dandi_analysis.dataset_001710.placecode import TuningCurveSet
    tc_by_day: dict[str, TuningCurveSet] = {}
    arm_sep_by_day: dict[str, float] = {}

    for asset in canonical_ready:
        _, _, day, _ = parse_subject_session(asset.path)
        label = f"day{day}"
        print(f"  {label}: {asset.path.name}")

        beh = load_behavior_table(asset.path, source="2p")
        ophys = load_ophys_matrix(asset.path, signal="dff")
        if beh is None or ophys is None:
            print(f"    Skipping {label}: data unavailable.")
            continue

        trials = build_trial_table(asset.path, day=day)

        try:
            tc = compute_tuning_curves(ophys, beh, n_bins=50)
            tc_by_day[label] = tc
        except Exception as exc:
            print(f"    Tuning curve error for {label}: {exc}")
            continue

        # Within-day arm separation
        try:
            arm_tc = arm_tuning(ophys, beh, trials, n_bins=50)
            if arm_tc.left is not None and arm_tc.right is not None:
                sep = within_day_arm_separation(arm_tc.left, arm_tc.right)
                arm_sep_by_day[label] = sep.similarity
                print(f"    Arm separation: {sep.similarity:.4f}")
        except Exception:
            pass

    # Cross-day similarity matrix
    if len(tc_by_day) < 2:
        print("Not enough days with valid tuning curves for cross-day analysis.")
        return

    print("\nBuilding day-to-day similarity matrix ...")
    sim_mat = build_day_similarity_matrix(tc_by_day)
    print("  Labels:", sim_mat.labels)
    print("  Matrix:")
    for i, label in enumerate(sim_mat.labels):
        row_str = "  " + label + "  " + "  ".join(
            f"{v:.3f}" if np.isfinite(v) else " nan"
            for v in sim_mat.matrix[i]
        )
        print(row_str)

    mat_path = export_similarity_matrix(sim_mat, OUTPUT_DIR)
    print(f"\n  Similarity matrix written to {mat_path}")

    # Null comparison via circular time shift
    print(f"\nComputing {N_NULLS} circular-shift nulls ...")
    if len(canonical_ready) >= 2:
        asset_a = canonical_ready[0]
        asset_b = canonical_ready[1]
        _, _, day_a, _ = parse_subject_session(asset_a.path)
        _, _, day_b, _ = parse_subject_session(asset_b.path)
        label_a, label_b = f"day{day_a}", f"day{day_b}"

        if label_a in tc_by_day and label_b in tc_by_day:
            observed = cross_day_tuning_correlation(
                tc_by_day[label_a], tc_by_day[label_b],
                label_a=label_a, label_b=label_b,
            ).similarity

            # Load the second session once for null generation
            beh_b = load_behavior_table(asset_b.path, source="2p")
            ophys_b = load_ophys_matrix(asset_b.path, signal="dff")
            if beh_b is not None and ophys_b is not None:
                null_sims: list[float] = []
                for k in range(N_NULLS):
                    shifted_data = circular_time_shift(ophys_b.data, seed=k)
                    import copy
                    null_ophys = copy.copy(ophys_b)
                    null_ophys.data = shifted_data
                    try:
                        null_tc = compute_tuning_curves(null_ophys, beh_b, n_bins=50)
                        null_res = cross_day_tuning_correlation(
                            tc_by_day[label_a], null_tc,
                            label_a=label_a, label_b=f"null_{k}",
                        )
                        null_sims.append(null_res.similarity)
                    except Exception:
                        pass

                if null_sims:
                    null_arr = np.array(null_sims)
                    null_mean = float(np.nanmean(null_arr))
                    null_std = float(np.nanstd(null_arr))
                    z = (observed - null_mean) / null_std if null_std > 0 else np.nan
                    print(f"\n  {label_a} vs {label_b}:")
                    print(f"    Observed similarity : {observed:.4f}")
                    print(f"    Null mean ± std     : {null_mean:.4f} ± {null_std:.4f}")
                    print(f"    Z-score             : {z:.2f}")

    # Arm separation summary
    if arm_sep_by_day:
        print("\nWithin-day arm separation (population vector r) across days:")
        for label, sep in sorted(arm_sep_by_day.items()):
            print(f"  {label}: {sep:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
