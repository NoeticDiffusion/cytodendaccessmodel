"""Experiment 01 — Inventory: discover all local DANDI 000336 NWB files.

000336 is the canonical Allen Institute OpenScope dendritic-coupling dataset
(published version of draft dandiset 000871).

Usage:
    python experiments/dandi_000336_01_inventory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "000336"
TRIAGE_ROOT = ROOT / "data" / "dandi" / "triage" / "000336"

sys.path.insert(0, str(ROOT / "src"))

_EXPECTED_PATHS = [
    "sub-656228/sub-656228_ses-1245548523-acq-1245937727_ophys.nwb",
    "sub-656228/sub-656228_ses-1245548523-acq-1245937736_ophys.nwb",
    "sub-644972/sub-644972_ses-1237338784-acq-1237809219_ophys.nwb",
    "sub-644972/sub-644972_ses-1237338784-acq-1237809217_ophys.nwb",
    "sub-656228/sub-656228_ses-1247233186-acq-1247385130_ophys.nwb",
    "sub-656228/sub-656228_ses-1247233186-acq-1247385128_ophys.nwb",
]

_ROLES = {
    "sub-656228_ses-1245548523-acq-1245937727": "pair_c_plane_a",
    "sub-656228_ses-1245548523-acq-1245937736": "proof_of_access",
    "sub-644972_ses-1237338784-acq-1237809219": "pair_a_plane1",
    "sub-644972_ses-1237338784-acq-1237809217": "pair_a_plane2",
    "sub-656228_ses-1247233186-acq-1247385130": "pair_b_plane1",
    "sub-656228_ses-1247233186-acq-1247385128": "pair_b_plane2",
}


def main() -> None:
    from dandi_analysis.inventory import (
        build_inventory_report,
        canonical_assets,
        discover_nwb_assets,
    )

    print("=" * 60)
    print("Experiment 01 - NWB Inventory: DANDI 000336")
    print("=" * 60)
    print(f"Scanning: {DATA_ROOT}")

    if not DATA_ROOT.exists():
        print(f"WARNING: data root not found: {DATA_ROOT}")
        print("         Download first: python -m dandi_io.cli download ...")
        assets = []
    else:
        assets = discover_nwb_assets(DATA_ROOT)

    can = canonical_assets(assets)
    print(f"\nTotal NWB files : {len(assets)}")
    print(f"  Canonical     : {len(can)}")

    present = {str(a.path.relative_to(DATA_ROOT).as_posix()): a for a in can}
    print(f"\nExpected file status:")
    for ep in _EXPECTED_PATHS:
        role = _ROLES.get(Path(ep).stem.rsplit("_ophys", 1)[0], "unknown")
        exists = ep in present
        size = f"{present[ep].size / 1e9:.2f} GB" if exists else "missing"
        mark = "OK" if exists else "--"
        print(f"  [{mark}] ({role:<22}) {Path(ep).name}  {size}")

    report = build_inventory_report(assets)
    out = TRIAGE_ROOT / "inventory.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nReport written: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
