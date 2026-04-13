"""Experiment 04 — Activity matrix smoke test.

Extracts one activity matrix from the first ready NWB file that contains
enough units, prints shape and timing metadata, and writes a QC note.

Usage:
    python experiments/dandi_000718_04_activity_matrix_smoke_test.py

Outputs:
    data/dandi/triage/000718/activity_smoke.npz
    data/dandi/triage/000718/activity_smoke_qc.md
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "000718"
TRIAGE_ROOT = ROOT / "data" / "dandi" / "triage" / "000718"

sys.path.insert(0, str(ROOT / "src"))

_MIN_UNITS = 5


def main() -> None:
    from dandi_analysis.inventory import canonical_assets, discover_nwb_assets
    from dandi_analysis.readiness import check_readiness
    from dandi_analysis.dataset_000718.index import parse_subject_session
    from dandi_analysis.dataset_000718.activity import build_activity_matrix
    from dandi_analysis.dataset_000718.qc import run_qc
    from dandi_analysis.dataset_000718.exports import write_activity_npz, write_qc_report

    print("=" * 60)
    print("Experiment 04 - Activity Matrix Smoke Test: DANDI 000718")
    print("=" * 60)

    if not DATA_ROOT.exists():
        print(f"WARNING: data root not found: {DATA_ROOT}")
        print("         Skipping activity matrix test.")
        return

    assets = discover_nwb_assets(DATA_ROOT)
    can = canonical_assets(assets)

    target_mat = None
    target_session_id = None

    for asset in can:
        r = check_readiness(asset.path)
        if not r.is_ready:
            print(f"  SKIP (not ready): {asset.path.name}")
            continue

        subject_id, session_label = parse_subject_session(asset.path)
        session_id = f"{subject_id}__{session_label}"

        print(f"\n  Trying: {asset.path.name}")
        mat = build_activity_matrix(asset.path, session_id)
        if mat is None:
            print("    Could not build activity matrix.")
            continue

        print(f"    Matrix shape: ({mat.n_time} time bins x {mat.n_units} units)")
        if mat.n_units < _MIN_UNITS:
            print(f"    Too few units ({mat.n_units} < {_MIN_UNITS}), skipping.")
            continue

        target_mat = mat
        target_session_id = session_id
        break

    if target_mat is None:
        print("\nNo suitable activity matrix found in any ready file.")
        return

    import numpy as np

    data = np.array(target_mat.data)
    ts = np.array(target_mat.timestamps)

    print(f"\nSelected: {target_session_id}")
    print(f"  Shape      : {data.shape}")
    print(f"  Unit count : {target_mat.n_units}")
    print(f"  Duration   : {ts[-1] - ts[0]:.2f}s  "
          f"({ts[0]:.2f}s to {ts[-1]:.2f}s)")
    print(f"  Sampling   : {target_mat.sampling_rate:.2f} Hz")
    print(f"  Data range : [{data.min():.3f}, {data.max():.3f}]")
    print(f"  Window     : '{target_mat.window_label or 'full session'}'")

    TRIAGE_ROOT.mkdir(parents=True, exist_ok=True)

    npz_out = TRIAGE_ROOT / "activity_smoke.npz"
    write_activity_npz(target_mat, npz_out)
    print(f"\nActivity matrix written: {npz_out.relative_to(ROOT)}")

    from dandi_analysis.contracts import SessionIndexRow, QcIssue
    qc_issues = run_qc([], [target_mat])
    qc_out = TRIAGE_ROOT / "activity_smoke_qc.md"
    write_qc_report(qc_issues, qc_out)
    print(f"QC report written     : {qc_out.relative_to(ROOT)}")
    if qc_issues:
        print(f"\nQC issues ({len(qc_issues)}):")
        for issue in qc_issues:
            print(f"  [{issue.severity.upper()}] {issue.issue_type}: {issue.message}")
    else:
        print("  QC: No issues found.")


if __name__ == "__main__":
    main()
