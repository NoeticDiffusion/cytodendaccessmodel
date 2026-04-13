"""Experiment 02 — Header probe: open each ready NWB file and record metadata.

Usage:
    python experiments/dandi_000718_02_header_probe.py

Outputs:
    data/dandi/triage/000718/headers.json
    (also prints per-file summary to stdout)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "000718"
TRIAGE_ROOT = ROOT / "data" / "dandi" / "triage" / "000718"

sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from dandi_analysis.inventory import canonical_assets, discover_nwb_assets
    from dandi_analysis.readiness import check_readiness, build_readiness_report
    from dandi_analysis.dataset_000718.metadata import extract_nwb_metadata

    print("=" * 60)
    print("Experiment 02 - Header Probe: DANDI 000718")
    print("=" * 60)

    if not DATA_ROOT.exists():
        print(f"WARNING: data root not found: {DATA_ROOT}")
        print("         Skipping header probe.")
        return

    assets = discover_nwb_assets(DATA_ROOT)
    can = canonical_assets(assets)
    print(f"Canonical NWB files found: {len(can)}")

    all_metadata: dict[str, dict] = {}
    ready_count = 0

    for asset in can:
        print(f"\n  Checking: {asset.path.name}")
        r = check_readiness(asset.path)
        if not r.is_ready:
            print(f"    NOT READY: {r.error}")
            all_metadata[str(asset.path)] = {"path": str(asset.path), "ready": False, "error": r.error}
            continue

        ready_count += 1
        meta = extract_nwb_metadata(asset.path)
        all_metadata[str(asset.path)] = meta

        print(f"    Size              : {r.size:,} bytes")
        print(f"    Session           : {meta.get('session_description', '(none)')[:80]}")
        print(f"    Start time        : {meta.get('session_start_time', 'N/A')}")
        print(f"    Subject           : {meta.get('subject', {})}")
        print(f"    Acquisitions      : {meta.get('acquisitions', [])}")
        print(f"    Processing modules: {meta.get('processing_keys', [])}")
        print(f"    Interval names    : {meta.get('interval_names', [])}")
        print(f"    Imaging planes    : {meta.get('imaging_planes', [])}")
        units_count = meta.get("units_count")
        if units_count is not None:
            print(f"    Units             : {units_count}")
            print(f"    Unit columns      : {meta.get('units_columns', [])}")

    print(f"\nReady files: {ready_count} / {len(can)}")

    out = TRIAGE_ROOT / "headers.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(all_metadata, fh, indent=2, default=str)
    print(f"Headers written: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
