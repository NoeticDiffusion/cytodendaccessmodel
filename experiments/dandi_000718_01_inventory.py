"""Experiment 01 — Inventory: discover all local DANDI 000718 NWB files.

Usage:
    python experiments/dandi_000718_01_inventory.py

Outputs:
    data/dandi/triage/000718/inventory.md
    (also prints a summary to stdout)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "000718"
TRIAGE_ROOT = ROOT / "data" / "dandi" / "triage" / "000718"

sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from dandi_analysis.inventory import (
        build_inventory_report,
        canonical_assets,
        discover_nwb_assets,
        duplicate_assets,
    )

    print("=" * 60)
    print("Experiment 01 - NWB Inventory: DANDI 000718")
    print("=" * 60)
    print(f"Scanning: {DATA_ROOT}")

    if not DATA_ROOT.exists():
        print(f"WARNING: data root not found: {DATA_ROOT}")
        print("         Download data first with: python -m dandi_io.cli download ...")
        assets = []
    else:
        assets = discover_nwb_assets(DATA_ROOT)

    can = canonical_assets(assets)
    dup = duplicate_assets(assets)

    print(f"\nTotal NWB files : {len(assets)}")
    print(f"  Canonical     : {len(can)}")
    print(f"  Non-canonical : {len(dup)}")

    if can:
        print("\nCanonical paths:")
        for a in can:
            print(f"  {a.path.relative_to(ROOT)}  ({a.size:,} bytes)")

    if dup:
        print("\nNon-canonical (loose/duplicate) paths:")
        for a in dup:
            dup_note = f" -> {a.duplicate_of.relative_to(ROOT)}" if a.duplicate_of else ""
            print(f"  {a.path.relative_to(ROOT)}{dup_note}")

    report = build_inventory_report(assets)
    out = TRIAGE_ROOT / "inventory.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nReport written: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
