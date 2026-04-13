"""Experiment 01: Inventory and readiness for DANDI 001710.

Goal
----
Discover all local 001710 NWB files, mark readiness, and emit a clean session
inventory with day labels and dimensions.

Success condition
-----------------
We know exactly which sessions are analysable and how they differ in size.

Usage
-----
    python experiments/dandi_001710_01_inventory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dandi_analysis.inventory import build_inventory_report, discover_nwb_assets
from dandi_analysis.readiness import build_readiness_report, check_readiness

DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "001710"
OUTPUT_DIR = ROOT / "data" / "dandi" / "triage" / "001710"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {DATA_ROOT} ...")
    assets = discover_nwb_assets(DATA_ROOT)
    if not assets:
        print(f"No NWB files found under {DATA_ROOT}.")
        print("Download the dataset first with:")
        print("  python -m dandi_io.cli download --config configs/dandi/dataset_001710.yaml")
        return

    inv_report = build_inventory_report(assets)
    inv_path = OUTPUT_DIR / "inventory.md"
    inv_path.write_text(inv_report, encoding="utf-8")
    print(f"Inventory written to {inv_path}")
    print(inv_report)

    print("Checking readiness ...")
    readiness_results = [check_readiness(a.path) for a in assets if a.is_canonical]
    ready_report = build_readiness_report(readiness_results)
    ready_path = OUTPUT_DIR / "readiness.md"
    ready_path.write_text(ready_report, encoding="utf-8")
    print(f"Readiness report written to {ready_path}")
    print(ready_report)

    n_ready = sum(1 for r in readiness_results if r.is_ready)
    print(f"\nSummary: {n_ready}/{len(readiness_results)} canonical sessions are ready.")


if __name__ == "__main__":
    main()
