"""Experiment 02 — Header probe: open each ready 000336 NWB file and record metadata.

000336 is the canonical Allen Institute OpenScope dendritic-coupling dataset
(published version of draft dandiset 000871).

For the dendritic coupling dataset the key items are:
  - imaging plane descriptions (somatic vs dendritic depth)
  - ROI response series names and shapes
  - interval / trial table structure
  - processing module keys

Usage:
    python experiments/dandi_000336_02_header_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "000336"
TRIAGE_ROOT = ROOT / "data" / "dandi" / "triage" / "000336"

sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from dandi_analysis.inventory import canonical_assets, discover_nwb_assets
    from dandi_analysis.readiness import check_readiness, build_readiness_report
    from dandi_analysis.dataset_000336.index import parse_subject_session
    from dandi_analysis.dataset_000336.metadata import extract_nwb_metadata

    print("=" * 60)
    print("Experiment 02 - Header Probe: DANDI 000336")
    print("=" * 60)

    if not DATA_ROOT.exists():
        print(f"WARNING: data root not found: {DATA_ROOT}")
        return

    assets = discover_nwb_assets(DATA_ROOT)
    can = canonical_assets(assets)
    print(f"Canonical NWB files found: {len(can)}")

    all_metadata: dict[str, dict] = {}
    ready_count = 0

    for asset in can:
        subject_id, session_label = parse_subject_session(asset.path)
        print(f"\n  [{subject_id}] {session_label}")
        print(f"  Checking readiness ...")

        r = check_readiness(asset.path)
        if not r.is_ready:
            print(f"  NOT READY: {r.error}")
            all_metadata[str(asset.path)] = {"ready": False, "error": r.error}
            continue

        ready_count += 1
        meta = extract_nwb_metadata(asset.path)
        all_metadata[str(asset.path)] = meta

        print(f"  Size              : {r.size / 1e9:.2f} GB")
        print(f"  Session           : {str(meta.get('session_description', ''))[:80]}")
        print(f"  Start time        : {meta.get('session_start_time', 'N/A')}")
        print(f"  Subject           : {meta.get('subject', {})}")
        print(f"  Processing modules: {meta.get('processing_keys', [])}")
        print(f"  Acquisitions      : {meta.get('acquisitions', [])}")
        print(f"  Interval names    : {meta.get('interval_names', [])}")
        print(f"  Imaging planes    : {meta.get('imaging_planes', [])}")

        plane_info = meta.get("imaging_plane_info", {})
        if plane_info:
            print(f"  Plane details:")
            for pname, pdata in plane_info.items():
                print(f"    {pname}: loc={pdata.get('location')}  "
                      f"rate={pdata.get('imaging_rate')}  "
                      f"desc={str(pdata.get('description', ''))[:50]}")

        rrs_info = meta.get("roi_response_series", [])
        if rrs_info:
            print(f"  ROI response series:")
            for rrs in rrs_info:
                print(f"    [{rrs['interface']}] {rrs['name']}: shape={rrs['shape']}")

        interval_info = meta.get("interval_tables", {})
        if interval_info:
            print(f"  Interval tables:")
            for iname, idata in interval_info.items():
                print(f"    {iname}: {idata['n_rows']} rows  cols={idata['columns']}")

    print(f"\nReady files: {ready_count} / {len(can)}")

    out = TRIAGE_ROOT / "headers.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(all_metadata, fh, indent=2, default=str)
    print(f"Headers written: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
