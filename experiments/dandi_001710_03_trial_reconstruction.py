"""Experiment 03: Trial reconstruction for DANDI 001710.

Goal
----
Derive a canonical trial table from behavior series plus trial_cell_data.

Success condition
-----------------
Each ready session yields a clean per-trial table or an explicit QC failure
reason.

Usage
-----
    python experiments/dandi_001710_03_trial_reconstruction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dandi_analysis.inventory import discover_nwb_assets
from dandi_analysis.readiness import check_readiness
from dandi_analysis.dataset_001710.index import parse_subject_session
from dandi_analysis.dataset_001710.trials import build_trial_table
from dandi_analysis.dataset_001710.qc import check_trial_table, format_qc_report
from dandi_analysis.dataset_001710.exports import export_trial_table, export_qc_report

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
    print(f"Processing {len(canonical_ready)} ready sessions ...")

    for asset in canonical_ready:
        _, _, day, _ = parse_subject_session(asset.path)
        print(f"\n  Day {day}: {asset.path.name}")

        trials = build_trial_table(asset.path, day=day)
        issues = check_trial_table(trials)

        n_valid = len(trials.valid_trials())
        n_total = len(trials)
        n_left = len(trials.by_arm("left"))
        n_right = len(trials.by_arm("right"))

        print(f"    trials total: {n_total}  valid: {n_valid}")
        print(f"    arm breakdown: left={n_left}  right={n_right}")

        if issues:
            print(f"    QC issues: {len(issues)}")
            for issue in issues:
                print(f"      [{issue.severity}] {issue.issue_type}: {issue.message}")

        # Export
        trial_csv = export_trial_table(
            trials, OUTPUT_DIR, filename=f"trials_day{day:02d}.csv"
        )
        qc_md = export_qc_report(
            format_qc_report(issues, session_label=f"day{day}"),
            OUTPUT_DIR,
            filename=f"qc_trials_day{day:02d}.md",
        )
        print(f"    exported: {trial_csv.name}, {qc_md.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
