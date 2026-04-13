"""Experiment 03 — Offline epoch candidates: extract rest/sleep windows.

Usage:
    python experiments/dandi_000718_03_offline_epoch_candidates.py

Outputs:
    data/dandi/triage/000718/epochs.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "000718"
TRIAGE_ROOT = ROOT / "data" / "dandi" / "triage" / "000718"

sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from dandi_analysis.inventory import canonical_assets, discover_nwb_assets
    from dandi_analysis.readiness import check_readiness
    from dandi_analysis.dataset_000718.index import parse_subject_session
    from dandi_analysis.dataset_000718.epochs import extract_offline_windows
    from dandi_analysis.dataset_000718.exports import write_epoch_csv

    print("=" * 60)
    print("Experiment 03 - Offline Epoch Candidates: DANDI 000718")
    print("=" * 60)

    if not DATA_ROOT.exists():
        print(f"WARNING: data root not found: {DATA_ROOT}")
        print("         Skipping epoch extraction.")
        return

    assets = discover_nwb_assets(DATA_ROOT)
    can = canonical_assets(assets)
    print(f"Canonical NWB files found: {len(can)}")

    all_windows = []
    for asset in can:
        r = check_readiness(asset.path)
        if not r.is_ready:
            print(f"  SKIP (not ready): {asset.path.name}  [{r.error}]")
            continue

        subject_id, session_label = parse_subject_session(asset.path)
        session_id = f"{subject_id}__{session_label}"
        windows = extract_offline_windows(asset.path, session_id)
        all_windows.extend(windows)

        if windows:
            print(f"\n  {asset.path.name}  ({len(windows)} windows)")
            for w in windows:
                print(f"    [{w.epoch_type:8s}] {w.label}: "
                      f"{w.start_sec:.1f}s - {w.stop_sec:.1f}s "
                      f"({w.duration_sec:.1f}s)")
        else:
            print(f"  {asset.path.name}: no offline windows found")

    print(f"\nTotal offline windows: {len(all_windows)}")

    out = TRIAGE_ROOT / "epochs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_epoch_csv(all_windows, out)
    print(f"Epoch table written: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
