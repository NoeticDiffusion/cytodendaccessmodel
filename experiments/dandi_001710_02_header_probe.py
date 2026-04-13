"""Experiment 02: Header and signal survey for DANDI 001710.

Goal
----
Record behavior channels, ophys interfaces, matrix shapes, sampling rate, ROI
counts, and embedded annotation payload keys for each ready session.

Success condition
-----------------
We know exactly which fields can support trial parsing and remapping analyses.

Usage
-----
    python experiments/dandi_001710_02_header_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dandi_analysis.inventory import discover_nwb_assets
from dandi_analysis.readiness import check_readiness
from dandi_analysis.dataset_001710.metadata import extract_nwb_metadata
from dandi_analysis.dataset_001710.exports import export_metadata_json

DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "001710"
OUTPUT_DIR = ROOT / "data" / "dandi" / "triage" / "001710"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    assets = discover_nwb_assets(DATA_ROOT)
    if not assets:
        print(f"No NWB files found under {DATA_ROOT}.")
        return

    canonical = [a for a in assets if a.is_canonical]
    ready = [a for a in canonical if check_readiness(a.path).is_ready]
    print(f"Found {len(ready)} ready canonical sessions.")

    all_meta: list[dict] = []
    for asset in ready:
        print(f"  Probing {asset.path.name} ...")
        meta = extract_nwb_metadata(asset.path)
        all_meta.append(meta)

        print(f"    processing keys : {meta.get('processing_keys', [])}")
        print(f"    behavior channels: {list((meta.get('behavior_channels') or {}).keys())}")
        print(f"    ophys RRS shapes : {[(r['interface'], r['name'], r['shape']) for r in meta.get('roi_response_series', [])]}")
        print(f"    annotation blob  : {meta.get('annotation_blob_keys', [])}")
        print(f"    day (path)       : {meta.get('blob_day')}")
        print(f"    novel_arm        : {meta.get('blob_novel_arm')}")

    summary_path = export_metadata_json(
        {"sessions": all_meta},
        OUTPUT_DIR,
        filename="header_probe_001710.json",
    )
    print(f"\nFull metadata written to {summary_path}")


if __name__ == "__main__":
    main()
