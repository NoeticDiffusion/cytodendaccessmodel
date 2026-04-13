"""Experiment 07 — Bootstrap confidence intervals on simulator bridge metrics.

Runs the canonical two-trace linking scenario with mild parameter jitter
(overlap_weight ± 0.05, consolidation_passes ± 10%) over N repeats and
reports 95% CIs for linking_index_model and context_margin_model across all
five model configurations.

Usage:
    python experiments/dandi_simulator_07_bootstrap_ci.py

Outputs:
    data/dandi/triage/model/simulator_bootstrap_ci.json
    data/dandi/triage/model/simulator_bootstrap_ci.md
    data/dandi/triage/model/simulator_bootstrap_ci.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRIAGE_ROOT = ROOT / "data" / "dandi" / "triage" / "model"

sys.path.insert(0, str(ROOT / "src"))

_N_REPEATS = 100
_CI_ALPHA = 0.05


def _emit(log_lines: list[str], message: str = "") -> None:
    print(message)
    log_lines.append(message)


def main() -> None:
    from dandi_analysis.simulator_bridge import run_bootstrap_scenarios

    TRIAGE_ROOT.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    _emit(log_lines, "=" * 60)
    _emit(log_lines, "Exp 07 - Bootstrap CI: Simulator Bridge Metrics")
    _emit(log_lines, f"n_repeats={_N_REPEATS}  ci_alpha={_CI_ALPHA}")
    _emit(log_lines, "=" * 60)

    _emit(log_lines, "\nRunning bootstrap scenarios...")
    results = run_bootstrap_scenarios(
        n_repeats=_N_REPEATS,
        ci_alpha=_CI_ALPHA,
    )

    # Print table
    _emit(log_lines, "")
    header = f"{'Scenario':<26} {'LI mean':>9} {'LI ci_lo':>9} {'LI ci_hi':>9} {'CM mean':>9} {'CM ci_lo':>9} {'CM ci_hi':>9}"
    _emit(log_lines, header)
    _emit(log_lines, "-" * len(header))

    for name, stats in results.items():
        li_mean = stats.get("linking_index_model_mean", float("nan"))
        li_lo   = stats.get("linking_index_model_ci_lo", float("nan"))
        li_hi   = stats.get("linking_index_model_ci_hi", float("nan"))
        cm_mean = stats.get("context_margin_model_mean", float("nan"))
        cm_lo   = stats.get("context_margin_model_ci_lo", float("nan"))
        cm_hi   = stats.get("context_margin_model_ci_hi", float("nan"))
        _emit(
            log_lines,
            f"  {name:<24} {li_mean:>9.4f} {li_lo:>9.4f} {li_hi:>9.4f}"
            f" {cm_mean:>9.4f} {cm_lo:>9.4f} {cm_hi:>9.4f}",
        )

    # Separation check
    full_li_lo = results["full_model"]["linking_index_model_ci_lo"]
    comparators = ["fast_context_only", "replay_no_structure", "random_slow_drift", "fixed_allocation_only"]
    _emit(log_lines, "\n  Separation check (full_model LI CI_lo vs comparator CI_hi):")
    for comp in comparators:
        comp_hi = results[comp]["linking_index_model_ci_hi"]
        sep = full_li_lo - comp_hi
        status = "SEPARATED" if sep > 0 else "OVERLAP"
        _emit(log_lines, f"    full_model vs {comp:<26}: gap={sep:+.4f}  {status}")

    # Write outputs
    json_path = TRIAGE_ROOT / "simulator_bootstrap_ci.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _emit(log_lines, f"\nJSON -> {json_path}")

    # Markdown
    md_lines = [
        "# Simulator Bootstrap CI — Model Anchor\n",
        f"n_repeats={_N_REPEATS}  CI alpha={_CI_ALPHA} (95%)\n",
        "| Scenario | LI mean | LI [lo, hi] | CM mean | CM [lo, hi] |",
        "|---|---|---|---|---|",
    ]
    for name, stats in results.items():
        li_mean = stats["linking_index_model_mean"]
        li_lo   = stats["linking_index_model_ci_lo"]
        li_hi   = stats["linking_index_model_ci_hi"]
        cm_mean = stats["context_margin_model_mean"]
        cm_lo   = stats["context_margin_model_ci_lo"]
        cm_hi   = stats["context_margin_model_ci_hi"]
        md_lines.append(
            f"| {name} | {li_mean:.4f} | [{li_lo:.4f}, {li_hi:.4f}] "
            f"| {cm_mean:.4f} | [{cm_lo:.4f}, {cm_hi:.4f}] |"
        )

    md_path = TRIAGE_ROOT / "simulator_bootstrap_ci.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    _emit(log_lines, f"MD   -> {md_path}")

    log_path = TRIAGE_ROOT / "simulator_bootstrap_ci.log"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    _emit(log_lines, f"LOG  -> {log_path}")

    _emit(log_lines, "\nDone.")


if __name__ == "__main__":
    main()
