"""Experiment 07: Group-aware replication-bundle pass for DANDI 001710.

Runs the complete analysis pipeline (inventory → trial reconstruction →
activity matrix → within-day tuning → cross-day remapping) on all discovered
001710 subjects and summarizes them by genotype group: Cre, Ctrl, SparseKO.

Cross-day similarity is still computed within subject, because ROI matching is
only meaningful across days from the same animal. Group-level reporting is
therefore built from subject-level summaries rather than pooled cross-subject
matrices.

Results are reported per subject and per group.

Usage
-----
    python experiments/dandi_001710_07_replication_bundle_full_pass.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from dandi_analysis.inventory import discover_nwb_assets
from dandi_analysis.readiness import check_readiness
from dandi_analysis.dataset_001710.index import parse_subject_session, subject_group
from dandi_analysis.dataset_001710.io import list_session_channels
from dandi_analysis.dataset_001710.behavior import load_behavior_table
from dandi_analysis.dataset_001710.ophys import load_ophys_matrix, load_all_channel_matrices
from dandi_analysis.dataset_001710.trials import build_trial_table
from dandi_analysis.dataset_001710.placecode import compute_tuning_curves, split_half_reliability, arm_tuning
from dandi_analysis.dataset_001710.remapping import (
    build_day_similarity_matrix,
    within_day_arm_separation,
)
from dandi_analysis.dataset_001710.qc import (
    check_behavior_channels,
    check_frame_count_consistency,
    check_multichannel_structure,
    check_roi_count,
    check_trial_table,
    format_qc_report,
)
from dandi_analysis.dataset_001710.exports import (
    export_robustness_csv,
    export_similarity_matrix,
    export_trial_table,
    export_tuning_summary,
    export_qc_report,
)

DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "001710"
OUTPUT_DIR = ROOT / "data" / "dandi" / "triage" / "001710" / "replication_bundle"

GROUPS = ["Cre", "Ctrl", "SparseKO"]


def run_session(
    path: Path,
    day: int,
    subject_id: str,
    group: str,
    out_dir: Path,
) -> dict:
    """Run the full single-session pipeline; return a result dict."""
    result = {
        "path": str(path),
        "day": day,
        "subject_id": subject_id,
        "group": group,
        "channels": [],
        "qc_issues": [],
    }

    channels = list_session_channels(path)
    is_multi = any(c.is_multichannel for c in channels)

    for ch in channels:
        ch_id = ch.channel_id
        ch_label = f"ch{ch_id}" if is_multi else "single"
        ch_dir = out_dir / ch_label
        ch_dir.mkdir(parents=True, exist_ok=True)

        ch_result: dict = {"channel_id": ch_id}

        # ---- Behavior ------------------------------------------------
        beh = load_behavior_table(path, channel=ch)
        if beh is None:
            ch_result["error"] = "behavior_not_loaded"
            result["channels"].append(ch_result)
            continue

        ch_result["n_frames"] = beh.n_frames
        ch_result["behavior_channels"] = beh.keys()

        # ---- Ophys ---------------------------------------------------
        ophys = load_ophys_matrix(path, channel=ch)
        if ophys is None:
            ch_result["error"] = "ophys_not_loaded"
            result["channels"].append(ch_result)
            continue

        ch_result["n_rois"] = ophys.n_rois
        ch_result["sampling_rate"] = ophys.sampling_rate

        # ---- Trials --------------------------------------------------
        trials = build_trial_table(path, day=day)
        ch_result["n_trials"] = len(trials)
        ch_result["n_valid_trials"] = len(trials.valid_trials())
        ch_result["n_left"] = len(trials.by_arm("left"))
        ch_result["n_right"] = len(trials.by_arm("right"))

        # ---- QC ------------------------------------------------------
        issues = []
        issues.extend(check_behavior_channels(beh))
        issues.extend(check_roi_count(ophys))
        issues.extend(check_frame_count_consistency(ophys, beh))
        issues.extend(check_trial_table(trials))
        if is_multi:
            all_mats = load_all_channel_matrices(path)
            issues.extend(check_multichannel_structure(channels, all_mats, path))
        ch_result["n_qc_issues"] = len(issues)
        ch_result["qc_issues"] = [
            f"[{i.severity}] {i.issue_type}: {i.message}" for i in issues
        ]
        qc_text = format_qc_report(
            issues,
            session_label=f"{subject_id}_day{day}_{ch_label}",
        )
        export_qc_report(qc_text, ch_dir, filename=f"qc_day{day:02d}.md")

        # ---- Tuning curves -------------------------------------------
        try:
            tc = compute_tuning_curves(ophys, beh, n_bins=50)
            rel = split_half_reliability(tc.tuning_curves, ophys, beh, n_splits=5)
            finite_rel = rel[np.isfinite(rel)]
            ch_result["split_half_median"] = float(np.median(finite_rel)) if len(finite_rel) else float("nan")
            ch_result["occ_bins"] = int(np.sum(tc.occupancy > 0))
            export_tuning_summary(tc, ch_dir, label=f"day{day:02d}")
        except Exception as exc:
            ch_result["tuning_error"] = str(exc)
            tc = None

        # ---- Arm separation ------------------------------------------
        try:
            arm_tc = arm_tuning(ophys, beh, trials, n_bins=50)
            if arm_tc.left is not None and arm_tc.right is not None:
                sep = within_day_arm_separation(arm_tc.left, arm_tc.right)
                ch_result["arm_separation"] = round(sep.similarity, 4)
            else:
                ch_result["arm_separation"] = None
        except Exception as exc:
            ch_result["arm_separation"] = None

        # ---- Trial table export --------------------------------------
        export_trial_table(trials, ch_dir, filename=f"trials_day{day:02d}.csv")

        result["channels"].append(ch_result)

    return result


def print_group_summary(group: str, session_results: list[dict]) -> None:
    print(f"\n{'='*60}")
    print(f"  GROUP: {group}")
    print(f"{'='*60}")

    subject_ids = sorted({r["subject_id"] for r in session_results})
    for subject_id in subject_ids:
        print(f"\n  Subject: {subject_id}")
        subject_sessions = sorted(
            [r for r in session_results if r["subject_id"] == subject_id],
            key=lambda x: x["day"],
        )
        for r in subject_sessions:
            day = r["day"]
            print(f"\n    Day {day}:")
            for ch in r["channels"]:
                ch_id = ch.get("channel_id", "?")
                tag = f"    [ch{ch_id}]" if ch_id != "single" else "            "
                if "error" in ch:
                    print(f"{tag} ERROR: {ch['error']}")
                    continue
                print(
                    f"{tag} ROIs={ch.get('n_rois','?'):4d}  "
                    f"frames={ch.get('n_frames','?'):6d}  "
                    f"trials={ch.get('n_valid_trials','?'):3d}  "
                    f"L={ch.get('n_left','?'):3d} R={ch.get('n_right','?'):3d}  "
                    f"sph_r={ch.get('split_half_median', float('nan')):.3f}  "
                    f"arm_sep={ch.get('arm_separation') or 'N/A'}"
                )
                if ch.get("n_qc_issues", 0) > 0:
                    for msg in ch.get("qc_issues", []):
                        print(f"              QC: {msg}")


def _subject_output_dir(group: str, subject_id: str) -> Path:
    return OUTPUT_DIR / group.lower() / subject_id.lower().replace("-", "_")


def _off_diagonal_values(mat: np.ndarray) -> list[float]:
    n = mat.shape[0]
    return [
        float(mat[i, j])
        for i in range(n)
        for j in range(i + 1, n)
        if np.isfinite(mat[i, j])
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    assets = discover_nwb_assets(DATA_ROOT)
    if not assets:
        print(f"No NWB files found under {DATA_ROOT}.")
        return

    canonical_ready = [
        a for a in assets if a.is_canonical and check_readiness(a.path).is_ready
    ]
    print(f"Found {len(canonical_ready)} ready sessions across all groups.")

    group_results: dict[str, list[dict]] = {g: [] for g in GROUPS}
    tc_by_subject_day: dict[str, dict[str, object]] = {}
    subject_to_group: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Pass 1: per-session analysis
    # ------------------------------------------------------------------ #
    for asset in canonical_ready:
        subject_id, _, day, _ = parse_subject_session(asset.path)
        group = subject_group(subject_id)
        if group == "Unknown":
            print(f"  Skipping unknown subject group: {asset.path.name}")
            continue

        subject_to_group[subject_id] = group
        subject_dir = _subject_output_dir(group, subject_id)
        r = run_session(asset.path, day, subject_id, group, subject_dir)
        group_results[group].append(r)

        # Collect tuning curves for subject-level cross-day matrices (use first channel)
        if r["channels"] and "tuning_error" not in r["channels"][0]:
            try:
                from dandi_analysis.dataset_001710.io import list_session_channels
                chs = list_session_channels(asset.path)
                from dandi_analysis.dataset_001710.behavior import load_behavior_table
                from dandi_analysis.dataset_001710.ophys import load_ophys_matrix
                from dandi_analysis.dataset_001710.placecode import compute_tuning_curves
                beh = load_behavior_table(asset.path, channel=chs[0])
                oph = load_ophys_matrix(asset.path, channel=chs[0])
                if beh and oph:
                    tc = compute_tuning_curves(oph, beh, n_bins=50)
                    tc_by_subject_day.setdefault(subject_id, {})[f"day{day}"] = tc
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Pass 2: per-subject cross-day similarity matrices + group summaries
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("  CROSS-DAY SIMILARITY MATRICES (PER SUBJECT)")
    print("="*60)

    subject_summary_rows: list[dict] = []

    for subject_id in sorted(tc_by_subject_day):
        tc_dict = tc_by_subject_day[subject_id]
        if len(tc_dict) < 2:
            print(f"\n  {subject_id}: insufficient days for matrix.")
            continue
        try:
            sim_mat = build_day_similarity_matrix(tc_dict)
            off_diag_vals = _off_diagonal_values(sim_mat.matrix)
            group = subject_to_group.get(subject_id, "Unknown")
            labels_str = "  ".join(f"{l:>5}" for l in sim_mat.labels)
            print(f"\n  {subject_id} ({group}):")
            print(f"         {labels_str}")
            for i, label in enumerate(sim_mat.labels):
                row_str = "  ".join(
                    f"{v:5.3f}" if np.isfinite(v) else "  nan" for v in sim_mat.matrix[i]
                )
                print(f"  {label:>5}  {row_str}")
            subject_dir = _subject_output_dir(group, subject_id)
            subject_dir.mkdir(parents=True, exist_ok=True)
            export_similarity_matrix(
                sim_mat,
                subject_dir,
                filename=f"day_similarity_{subject_id}.csv",
            )
            subject_summary_rows.append(
                {
                    "group": group,
                    "subject_id": subject_id,
                    "n_days": len(sim_mat.labels),
                    "n_pairs": len(off_diag_vals),
                    "off_diagonal_mean": round(float(np.mean(off_diag_vals)), 4)
                    if off_diag_vals
                    else float("nan"),
                }
            )
        except Exception as exc:
            print(f"\n  {subject_id}: matrix error — {exc}")

    if subject_summary_rows:
        export_robustness_csv(
            subject_summary_rows,
            OUTPUT_DIR,
            filename="subject_cross_day_summary.csv",
        )
        for group in GROUPS:
            group_rows = [row for row in subject_summary_rows if row["group"] == group]
            if not group_rows:
                continue
            export_robustness_csv(
                group_rows,
                OUTPUT_DIR / group.lower(),
                filename=f"subject_cross_day_summary_{group}.csv",
            )

        print("\n" + "="*60)
        print("  GROUP-LEVEL SUBJECT SUMMARIES")
        print("="*60)
        for group in GROUPS:
            group_rows = [row for row in subject_summary_rows if row["group"] == group]
            if not group_rows:
                continue
            print(f"\n  {group}:")
            for row in sorted(group_rows, key=lambda x: x["subject_id"]):
                print(
                    f"    {row['subject_id']}: off_diag_mean={row['off_diagonal_mean']:.4f}"
                )
            vals = [
                float(row["off_diagonal_mean"])
                for row in group_rows
                if np.isfinite(float(row["off_diagonal_mean"]))
            ]
            if vals:
                print(
                    f"    group_mean={np.mean(vals):.4f}  "
                    f"n_subjects={len(vals)}"
                )

    # ------------------------------------------------------------------ #
    # Print per-group summaries
    # ------------------------------------------------------------------ #
    for group in GROUPS:
        if group_results[group]:
            print_group_summary(group, group_results[group])

    print(f"\n\nAll outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
