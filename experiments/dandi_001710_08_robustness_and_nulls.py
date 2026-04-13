"""Experiment 08: Group-aware robustness and null-test package for DANDI 001710.

Delivers four analyses to strengthen claims about SparseKO subjects' lower
cross-day stability and preserved within-day separation:

  1. Arm-label audit — verifies left/right labels against NWB annotation blob.
  2. SparseKO channel comparison — checks whether the effect is channel-
     independent (ch0 vs ch1 metrics side-by-side).
  3. Group null test — subject-level permutation test for cross-day similarity
     differences between SparseKO and non-KO groups.
  4. Day-lag robustness — group-aggregated mean cross-day similarity per lag.

Additionally runs a QC sweep across all sessions and channels.

All outputs are written to::

    data/dandi/triage/001710/robustness/

Usage
-----
    python experiments/dandi_001710_08_robustness_and_nulls.py
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
from dandi_analysis.dataset_001710.io import list_session_channels, read_trial_annotation_blob
from dandi_analysis.dataset_001710.behavior import load_behavior_table
from dandi_analysis.dataset_001710.ophys import load_ophys_matrix, load_all_channel_matrices
from dandi_analysis.dataset_001710.trials import build_trial_table
from dandi_analysis.dataset_001710.placecode import (
    compute_tuning_curves,
    split_half_reliability,
    arm_tuning,
)
from dandi_analysis.dataset_001710.remapping import (
    build_day_similarity_matrix,
    within_day_arm_separation,
)
from dandi_analysis.dataset_001710.qc import (
    run_all_checks,
    format_qc_report,
    check_multichannel_structure,
)
from dandi_analysis.dataset_001710.robustness import (
    aggregate_group_day_lag,
    arm_label_audit,
    compare_sparseko_channels,
    day_lag_similarity,
    group_null_tests,
    plot_day_lag_curves,
)
from dandi_analysis.dataset_001710.exports import (
    export_qc_report,
    export_robustness_csv,
    export_null_json,
    export_similarity_matrix,
)

DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "001710"
OUTPUT_DIR = ROOT / "data" / "dandi" / "triage" / "001710" / "robustness"

GROUPS = ["Cre", "Ctrl", "SparseKO"]


# ---------------------------------------------------------------------------
# Analysis 1 — Arm label audit
# ---------------------------------------------------------------------------

def run_arm_label_audit(assets: list) -> None:
    print("\n" + "=" * 60)
    print("  ANALYSIS 1: ARM LABEL AUDIT")
    print("=" * 60)

    audit_rows: list[dict] = []

    for asset in assets:
        subject_id, _, day, _ = parse_subject_session(asset.path)
        group = subject_group(subject_id)
        if group == "Unknown":
            continue

        result = arm_label_audit(asset.path, day)
        result["group"] = group
        result["subject_id"] = subject_id
        result["session"] = asset.path.name
        audit_rows.append(result)

        flag_str = " *** FLAGGED ***" if result["flagged"] else ""
        vr_str = "vr_trial_info=YES" if result["vr_trial_info_found"] else "vr_trial_info=NO"
        print(
            f"  {group:9s}  {subject_id:12s}  day{day:02d}  "
            f"L={result['n_left']:3d}  R={result['n_right']:3d}  "
            f"mismatch={result['mismatch_rate']!r:6}  {vr_str}{flag_str}"
        )
        if result["note"]:
            print(f"           note: {result['note']}")

    if audit_rows:
        export_robustness_csv(
            audit_rows,
            OUTPUT_DIR,
            filename="arm_label_audit.csv",
        )
        _write_md_table(
            audit_rows,
            OUTPUT_DIR / "arm_label_audit.md",
            title="# Arm Label Audit",
            cols=["group", "subject_id", "session", "day", "n_valid", "n_left", "n_right",
                  "mismatch_rate", "vr_trial_info_found", "flagged", "note"],
        )
        print(f"\n  -> arm_label_audit.csv / .md")


# ---------------------------------------------------------------------------
# Analysis 2 — SparseKO channel comparison
# ---------------------------------------------------------------------------

def run_sparseko_channel_comparison(assets: list) -> None:
    print("\n" + "=" * 60)
    print("  ANALYSIS 2: SPARSEKO MULTICHANNEL COMPARISON (ch0 vs ch1)")
    print("=" * 60)

    sko_paths_by_subject: dict[str, dict[int, Path]] = {}
    for asset in assets:
        subject_id, _, day, _ = parse_subject_session(asset.path)
        group = subject_group(subject_id)
        if group != "SparseKO":
            continue
        channels = list_session_channels(asset.path)
        if not any(ch.is_multichannel for ch in channels):
            continue
        sko_paths_by_subject.setdefault(subject_id, {})[day] = asset.path

    if not sko_paths_by_subject:
        print("  No multichannel SparseKO subjects found — skipping.")
        return

    combined_summary_rows: list[dict] = []

    for subject_id in sorted(sko_paths_by_subject):
        result = compare_sparseko_channels(sko_paths_by_subject[subject_id])

        print(f"\n  Subject: {subject_id}")
        print(f"  Days processed: {sorted(result['days_processed'])}")
        for ch_id in ("0", "1"):
            ch_key = f"ch{ch_id}"
            info = result[ch_key]
            print(f"\n  {ch_key}:")
            print(f"    off_diag_mean = {info.get('off_diagonal_mean', 'N/A')}")
            print(f"    arm_sep_mean  = {info.get('arm_separation_mean', 'N/A')}")
            print(f"    sph_mean      = {info.get('split_half_mean', 'N/A')}")

        print("\n  ch0 - ch1 differences:")
        for metric, val in result["ch0_minus_ch1"].items():
            print(f"    {metric}: {val}")

        per_day = result.get("per_day_metrics", [])
        if per_day:
            export_rows = [{"subject_id": subject_id, **row} for row in per_day]
            export_robustness_csv(
                export_rows,
                OUTPUT_DIR,
                filename=f"sparseko_channel_comparison_{subject_id}.csv",
            )

        summary_rows = []
        for ch_id in ("0", "1"):
            ch_key = f"ch{ch_id}"
            info = result[ch_key]
            row = {
                "subject_id": subject_id,
                "channel": ch_key,
                "off_diagonal_mean": info.get("off_diagonal_mean", float("nan")),
                "arm_separation_mean": info.get("arm_separation_mean", float("nan")),
                "split_half_mean": info.get("split_half_mean", float("nan")),
            }
            summary_rows.append(row)
            combined_summary_rows.append(row)
        diff_row = {
            "subject_id": subject_id,
            "channel": "ch0-ch1",
            **result["ch0_minus_ch1"],
        }
        summary_rows.append(diff_row)
        combined_summary_rows.append(diff_row)

        export_robustness_csv(
            summary_rows,
            OUTPUT_DIR,
            filename=f"sparseko_channel_summary_{subject_id}.csv",
        )

    if combined_summary_rows:
        export_robustness_csv(
            combined_summary_rows,
            OUTPUT_DIR,
            filename="sparseko_channel_summary.csv",
        )
        _write_md_table(
            combined_summary_rows,
            OUTPUT_DIR / "sparseko_channel_comparison.md",
            title="# SparseKO Channel Comparison (ch0 vs ch1)",
            cols=["subject_id", "channel", "off_diagonal_mean", "arm_separation_mean", "split_half_mean"],
        )
        print(f"\n  -> sparseko_channel_summary.csv / .md")


# ---------------------------------------------------------------------------
# Analysis 3 — Cohort null test + Analysis 4 — Day-lag
# (shared pipeline pass: rebuild minimal tuning curves for all cohorts)
# ---------------------------------------------------------------------------

def run_null_and_lag_analyses(assets: list) -> None:
    print("\n" + "=" * 60)
    print("  BUILDING SUBJECT SIMILARITY MATRICES (for analyses 3 & 4)")
    print("=" * 60)

    tc_by_subject_day: dict[str, dict[str, object]] = {}
    subject_groups: dict[str, str] = {}

    for asset in assets:
        subject_id, _, day, _ = parse_subject_session(asset.path)
        group = subject_group(subject_id)
        if group == "Unknown":
            continue

        try:
            chs = list_session_channels(asset.path)
            if not chs:
                continue
            ch = chs[0]
            beh = load_behavior_table(asset.path, channel=ch)
            oph = load_ophys_matrix(asset.path, channel=ch)
            if beh is None or oph is None:
                continue
            tc = compute_tuning_curves(oph, beh, n_bins=50)
            tc_by_subject_day.setdefault(subject_id, {})[f"day{day}"] = tc
            subject_groups[subject_id] = group
            print(f"  {group:9s}  {subject_id:12s}  day{day:02d}  ROIs={oph.n_rois}")
        except Exception as exc:
            print(f"  {group:9s}  {subject_id:12s}  day{day:02d}  ERROR: {exc}")

    subject_sim_matrices: dict[str, object] = {}
    subject_summary_rows: list[dict] = []
    for subject_id, tc_dict in sorted(tc_by_subject_day.items()):
        if len(tc_dict) < 2:
            print(f"  {subject_id}: only {len(tc_dict)} day(s) — skipping matrix.")
            continue
        try:
            sim_mat = build_day_similarity_matrix(tc_dict)
            subject_sim_matrices[subject_id] = sim_mat
            export_similarity_matrix(
                sim_mat,
                OUTPUT_DIR / "subject_matrices",
                filename=f"day_similarity_{subject_id}.csv",
            )
            off_diag = [
                float(sim_mat.matrix[i, j])
                for i in range(len(sim_mat.labels))
                for j in range(i + 1, len(sim_mat.labels))
                if np.isfinite(sim_mat.matrix[i, j])
            ]
            subject_summary_rows.append(
                {
                    "group": subject_groups.get(subject_id, "Unknown"),
                    "subject_id": subject_id,
                    "n_days": len(sim_mat.labels),
                    "n_pairs": len(off_diag),
                    "off_diagonal_mean": round(float(np.mean(off_diag)), 4)
                    if off_diag
                    else float("nan"),
                }
            )
            print(
                f"  {subject_id}: {len(sim_mat.labels)}-day matrix built; "
                f"off_diag_mean={subject_summary_rows[-1]['off_diagonal_mean']}"
            )
        except Exception as exc:
            print(f"  {subject_id}: matrix error — {exc}")

    if subject_summary_rows:
        export_robustness_csv(
            subject_summary_rows,
            OUTPUT_DIR,
            filename="subject_cross_day_summary.csv",
        )

    # --- Analysis 3: Group null tests ---
    print("\n" + "=" * 60)
    print("  ANALYSIS 3: GROUP PERMUTATION NULL TESTS")
    print("=" * 60)

    if len(subject_sim_matrices) < 2 or "SparseKO" not in set(subject_groups.values()):
        print("  Insufficient group data for null tests.")
    else:
        null_results = group_null_tests(
            subject_sim_matrices,
            subject_groups,
            target_group="SparseKO",
            n_perms=1000,
            seed=0,
        )

        for entry in null_results:
            if "error" in entry:
                print(f"  {entry.get('comparison', '?')}: ERROR — {entry['error']}")
                continue
            print(
                f"\n  {entry['comparison']}:"
                f"\n    SparseKO mean={entry['target_mean']:.4f}  "
                f"other mean={entry['other_mean']:.4f}"
                f"\n    obs_diff={entry['observed_diff']:.4f}  "
                f"z={entry['z']!r}  p={entry['p_empirical']:.4f}"
                f"\n    n_subjects={entry['target_n_subjects']} vs {entry['other_n_subjects']}"
                f"\n    -> {entry['claim']}"
            )
            if entry.get("note"):
                print(f"    note: {entry['note']}")

        # Strip null_distribution from export (large array)
        export_rows = [
            {k: v for k, v in e.items() if k != "null_distribution"}
            for e in null_results
        ]
        export_robustness_csv(
            export_rows,
            OUTPUT_DIR,
            filename="group_null_tests.csv",
        )
        export_null_json(
            null_results,
            OUTPUT_DIR,
            filename="group_null_tests.json",
        )
        _write_md_table(
            export_rows,
            OUTPUT_DIR / "group_null_tests.md",
            title="# Group Null Tests (subject-level permutation, n=1000)",
            cols=["comparison", "target_mean", "other_mean", "target_n_subjects",
                  "other_n_subjects", "observed_diff", "z", "p_empirical", "claim", "note"],
        )
        print(f"\n  -> group_null_tests.csv / .md / .json")

        # Claim-boundary summary
        _print_claim_boundary(null_results)

    # --- Analysis 4: Day-lag ---
    print("\n" + "=" * 60)
    print("  ANALYSIS 4: DAY-LAG SIMILARITY")
    print("=" * 60)

    subject_lag_data = {
        subject_id: day_lag_similarity(sim_mat)
        for subject_id, sim_mat in subject_sim_matrices.items()
    }
    lag_data_by_group = aggregate_group_day_lag(subject_lag_data, subject_groups)
    lag_rows: list[dict] = []

    for group in GROUPS:
        lag_data = lag_data_by_group.get(group)
        if not lag_data:
            continue
        for lag, stats in lag_data.items():
            lag_rows.append({
                "group": group,
                "lag": lag,
                **stats,
            })
        means_str = "  ".join(
            f"lag{l}={v['mean']:.3f}" if np.isfinite(v["mean"]) else f"lag{l}=nan"
            for l, v in sorted(lag_data.items())
        )
        print(f"  {group:12s}  {means_str}")

    if lag_rows:
        export_robustness_csv(
            lag_rows,
            OUTPUT_DIR,
            filename="day_lag_similarity.csv",
        )
        _write_md_table(
            lag_rows,
            OUTPUT_DIR / "day_lag_similarity.md",
            title="# Day-Lag Similarity",
            cols=["group", "lag", "mean", "median", "n_subjects"],
        )

    if lag_data_by_group:
        plot_day_lag_curves(
            lag_data_by_group,
            OUTPUT_DIR / "day_lag_similarity.png",
        )
        print(f"\n  -> day_lag_similarity.csv / .md / .png")


# ---------------------------------------------------------------------------
# QC sweep
# ---------------------------------------------------------------------------

def run_qc_sweep(assets: list) -> None:
    print("\n" + "=" * 60)
    print("  QC SWEEP")
    print("=" * 60)

    qc_rows: list[dict] = []

    for asset in assets:
        subject_id, _, day, _ = parse_subject_session(asset.path)
        group = subject_group(subject_id)
        if group == "Unknown":
            continue

        channels = list_session_channels(asset.path)
        is_multi = any(c.is_multichannel for c in channels)
        blob = read_trial_annotation_blob(asset.path)

        # Load all matrices for multi-channel check
        all_mats = load_all_channel_matrices(asset.path) if is_multi else None

        for ch in channels:
            ch_id = ch.channel_id
            ch_label = f"ch{ch_id}" if is_multi else "single"

            row: dict = {
                "group": group,
                "subject_id": subject_id,
                "session": asset.path.name,
                "day": day,
                "channel": ch_label,
                "n_rois": "",
                "n_frames": "",
                "n_valid_trials": "",
                "n_left": "",
                "n_right": "",
                "arm_balance": "",
                "split_half_median": "",
                "n_qc_issues": 0,
                "used_in_cross_day": True,
            }

            beh = load_behavior_table(asset.path, channel=ch)
            ophys = load_ophys_matrix(asset.path, channel=ch)
            trials = build_trial_table(asset.path, day=day)

            if beh is not None:
                row["n_frames"] = beh.n_frames
            if ophys is not None:
                row["n_rois"] = ophys.n_rois

            valid = trials.valid_trials()
            row["n_valid_trials"] = len(valid)
            row["n_left"] = len(trials.by_arm("left"))
            row["n_right"] = len(trials.by_arm("right"))

            nl, nr = row["n_left"], row["n_right"]
            if nl > 0 and nr > 0:
                row["arm_balance"] = round(max(nl, nr) / min(nl, nr), 2)

            if beh is not None and ophys is not None:
                try:
                    tc = compute_tuning_curves(ophys, beh, n_bins=50)
                    rel = split_half_reliability(tc.tuning_curves, ophys, beh, n_splits=3)
                    finite_rel = rel[np.isfinite(rel)]
                    if len(finite_rel) > 0:
                        row["split_half_median"] = round(float(np.median(finite_rel)), 3)
                except Exception:
                    pass

            issues = run_all_checks(
                behavior=beh,
                ophys=ophys,
                trials=trials,
                blob=blob,
                path=asset.path,
                channels=channels if is_multi else None,
                all_matrices=all_mats if is_multi else None,
            )
            row["n_qc_issues"] = len(issues)

            qc_text = format_qc_report(issues, session_label=f"{subject_id}_day{day}_{ch_label}")
            export_qc_report(
                qc_text,
                OUTPUT_DIR / "qc_per_session",
                filename=f"qc_{subject_id}_day{day:02d}_{ch_label}.md",
            )

            qc_rows.append(row)

            issues_str = f"  {len(issues)} QC issue(s)" if issues else ""
            print(
                f"  {group:9s}  {subject_id:12s}  day{day:02d}  [{ch_label}]  "
                f"ROIs={row['n_rois']:4}  frames={row['n_frames']:6}  "
                f"valid={row['n_valid_trials']:3}  sph={row.get('split_half_median', 'N/A')}"
                f"{issues_str}"
            )

    if qc_rows:
        export_robustness_csv(
            qc_rows,
            OUTPUT_DIR,
            filename="qc_summary_001710.csv",
        )
        _write_md_table(
            qc_rows,
            OUTPUT_DIR / "qc_summary_001710.md",
            title="# QC Summary — DANDI 001710",
            cols=["group", "subject_id", "session", "day", "channel", "n_rois", "n_frames",
                  "n_valid_trials", "n_left", "n_right", "arm_balance",
                  "split_half_median", "n_qc_issues"],
        )
        print(f"\n  -> qc_summary_001710.csv / .md  ({len(qc_rows)} rows)")


# ---------------------------------------------------------------------------
# Claim-boundary summary
# ---------------------------------------------------------------------------

def _print_claim_boundary(null_results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("  CLAIM-BOUNDARY SUMMARY")
    print("=" * 60)

    has_null_sep = any(
        "null-separated" in e.get("claim", "") for e in null_results if "error" not in e
    )
    has_directional = any(
        "directionally consistent" in e.get("claim", "").lower()
        for e in null_results
        if "error" not in e
    )

    if has_null_sep:
        print(
            "\n  The SparseKO group showed lower subject-level cross-day similarity\n"
            "  than at least one non-KO group under permutation-based null testing\n"
            "  (z < -1.96, p < 0.05). This supports a broader genotype-level deficit."
        )
    elif has_directional:
        print(
            "\n  The SparseKO group showed directionally lower subject-level\n"
            "  cross-day similarity than non-KO groups, but this did not reach\n"
            "  null-separation (|z| < 1.96). Interpret as 'directionally\n"
            "  consistent, not yet null-separated' pending larger N."
        )
    else:
        print(
            "\n  No clear cross-day similarity difference detected between SparseKO\n"
            "  and non-KO groups under current subject-level permutation testing."
        )

    cb_path = OUTPUT_DIR / "claim_boundary.md"
    cb_path.write_text(
        _claim_boundary_text(null_results),
        encoding="utf-8",
    )
    print(f"\n  -> claim_boundary.md")


def _claim_boundary_text(null_results: list[dict]) -> str:
    lines = [
        "# Claim Boundary — 001710 Group Cross-Day Stability (Experiment 08)",
        "",
        "Generated by `dandi_001710_08_robustness_and_nulls.py`.",
        "",
        "## Null test results",
        "",
    ]
    for entry in null_results:
        if "error" in entry:
            lines.append(f"- **{entry.get('comparison', '?')}**: {entry['error']}")
            continue
        lines.append(
            f"- **{entry['comparison']}**: "
            f"obs_diff={entry['observed_diff']:.4f}, "
            f"z={entry['z']!r}, "
            f"p={entry['p_empirical']:.4f} — "
            f"*{entry['claim']}*"
        )
        if entry.get("note"):
            lines.append(f"  - note: {entry['note']}")

    lines += [
        "",
        "## Policy",
        "",
        "- z < −1.96 **and** p < 0.05 → full claim: "
        '"SparseKO shows lower subject-level cross-day similarity than the comparison group."',
        "- z directionally negative but |z| < 1.96 → "
        '"directionally consistent, not yet null-separated."',
        "- ch0/ch1 diverge → "
        '"the effect is channel-sensitive; interpret per channel."',
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Markdown table helper
# ---------------------------------------------------------------------------

def _write_md_table(
    rows: list[dict],
    path: Path,
    *,
    title: str = "",
    cols: list[str] | None = None,
) -> None:
    """Write a simple Markdown table to *path*."""
    if not rows:
        path.write_text(f"{title}\n\n*(no data)*\n", encoding="utf-8")
        return
    if cols is None:
        cols = list(rows[0].keys())

    lines = []
    if title:
        lines += [title, ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for row in rows:
        cells = []
        for c in cols:
            val = row.get(c, "")
            if isinstance(val, float) and not np.isfinite(val):
                val = ""
            cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    assets = discover_nwb_assets(DATA_ROOT)
    if not assets:
        print(f"No NWB files found under {DATA_ROOT}.")
        return

    canonical_ready = [
        a for a in assets if a.is_canonical and check_readiness(a.path).is_ready
    ]
    print(f"Found {len(canonical_ready)} ready sessions.")

    # --- 1. Arm-label audit ---
    run_arm_label_audit(canonical_ready)

    # --- 2. SparseKO channel comparison ---
    run_sparseko_channel_comparison(canonical_ready)

    # --- 3 & 4. Null test + day-lag (shared matrix build) ---
    run_null_and_lag_analyses(canonical_ready)

    # --- QC sweep ---
    run_qc_sweep(canonical_ready)

    print(f"\n\nAll outputs written to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
