"""Experiment 09 — Factorial simulator sensitivity per parameter.

Runs the canonical two-trace linking scenario while sweeping each parameter
independently (all others held at baseline), producing a sensitivity profile
for linking_index_model and context_margin_model.

This is more diagnostic than exp07 (joint jitter bootstrap), because it
attributes variance to individual parameters rather than the combined jitter.

Usage:
    python experiments/dandi_simulator_09_sensitivity.py

Outputs:
    data/dandi/triage/model/simulator_sensitivity.json
    data/dandi/triage/model/simulator_sensitivity.md
    data/dandi/triage/model/simulator_sensitivity.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRIAGE_ROOT = ROOT / "data" / "dandi" / "triage" / "model"

sys.path.insert(0, str(ROOT / "src"))


def _emit(log_lines: list[str], msg: str = "") -> None:
    print(msg)
    log_lines.append(msg)


def main() -> None:
    import numpy as np
    from dandi_analysis.simulator_bridge import run_linking_scenario, BASE_PARAMS
    from cytodend_accessmodel.contracts import DynamicsParameters
    import dataclasses

    TRIAGE_ROOT.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    _emit(log_lines, "=" * 60)
    _emit(log_lines, "Exp 09 - Factorial Simulator Sensitivity")
    _emit(log_lines, "=" * 60)

    # Baseline result (reference)
    base_result = run_linking_scenario()
    base_li = base_result["linking_index_model"]
    base_cm = base_result["context_margin_model"]
    _emit(log_lines, f"\nBaseline: LI={base_li:.4f}  CM={base_cm:.4f}")

    # Parameters to sweep and their ranges (low, mid=baseline, high)
    base_dict = dataclasses.asdict(BASE_PARAMS)

    param_sweeps: dict[str, list[float]] = {
        "structural_gain":          [0.5, 1.0, 2.0, 3.0, 4.0],
        "structural_lr":            [0.05, 0.10, 0.15, 0.20, 0.25],
        "structural_decay":         [0.001, 0.005, 0.01, 0.02, 0.05],
        "replay_gain":              [0.0, 0.5, 1.0, 1.5, 2.0],
        "sleep_gain":               [0.0, 0.5, 1.0, 1.5, 2.0],
        "context_mismatch_penalty": [0.0, 0.125, 0.25, 0.375, 0.50],
        "fast_gain":                [0.5, 1.0, 2.0, 3.0, 4.0],
        "eligibility_decay":        [0.05, 0.075, 0.10, 0.15, 0.20],
    }

    # Also sweep structural parameters controlling the scenario
    scenario_sweeps: dict[str, list[float]] = {
        "overlap_weight":          [0.2, 0.4, 0.6, 0.8, 0.95],
        "n_consolidation_passes":  [2,   5,   8,  12,  16],
    }

    results: dict[str, list[dict]] = {}

    # Model parameter sweeps (one at a time, others at baseline)
    for param, values in param_sweeps.items():
        _emit(log_lines, f"\n  {param}:")
        sweep: list[dict] = []
        for v in values:
            kwargs = dict(base_dict)
            kwargs[param] = v
            p = DynamicsParameters(**kwargs)
            r = run_linking_scenario(params=p)
            li = r["linking_index_model"]
            cm = r["context_margin_model"]
            delta_li = li - base_li
            _emit(log_lines, f"    {param}={v:.4g}  LI={li:.4f}  CM={cm:.4f}  dLI={delta_li:+.4f}")
            sweep.append({"value": v, "linking_index": li, "context_margin": cm, "delta_li": delta_li})
        results[param] = sweep

    # Scenario parameter sweeps (overlap_weight, n_consolidation_passes)
    for param, values in scenario_sweeps.items():
        _emit(log_lines, f"\n  {param} (scenario):")
        sweep = []
        for v in values:
            kwargs: dict = {}
            if param == "overlap_weight":
                kwargs["overlap_weight"] = v
            elif param == "n_consolidation_passes":
                kwargs["n_consolidation_passes"] = int(v)
            r = run_linking_scenario(**kwargs)
            li = r["linking_index_model"]
            cm = r["context_margin_model"]
            delta_li = li - base_li
            _emit(log_lines, f"    {param}={v}  LI={li:.4f}  CM={cm:.4f}  dLI={delta_li:+.4f}")
            sweep.append({"value": float(v), "linking_index": li, "context_margin": cm, "delta_li": delta_li})
        results[param] = sweep

    # Compute sensitivity magnitude (range of LI across sweep)
    _emit(log_lines, "\n  Sensitivity ranking (LI range over sweep):")
    sensitivity: list[tuple[str, float]] = []
    for param, sweep in results.items():
        li_values = [s["linking_index"] for s in sweep]
        li_range = max(li_values) - min(li_values)
        sensitivity.append((param, li_range))
    sensitivity.sort(key=lambda x: -x[1])
    for rank, (param, rng) in enumerate(sensitivity, 1):
        _emit(log_lines, f"    {rank:2d}. {param:<30} LI range={rng:.4f}")

    output = {
        "baseline": {"linking_index_model": base_li, "context_margin_model": base_cm},
        "sensitivity_ranking": [{"param": p, "li_range": r} for p, r in sensitivity],
        "sweeps": results,
    }

    # Write outputs
    json_path = TRIAGE_ROOT / "simulator_sensitivity.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _emit(log_lines, f"\nJSON -> {json_path}")

    # Markdown
    md_lines = [
        "# Factorial Simulator Sensitivity\n",
        f"Baseline: LI={base_li:.4f}  CM={base_cm:.4f}\n",
        "## Sensitivity Ranking (LI range across sweep)\n",
        "| Rank | Parameter | LI range |",
        "|---|---|---|",
    ]
    for rank, (param, rng) in enumerate(sensitivity, 1):
        md_lines.append(f"| {rank} | {param} | {rng:.4f} |")

    md_lines.append("\n## Per-Parameter Sweeps\n")
    for param, sweep in results.items():
        md_lines.append(f"\n### {param}\n")
        md_lines.append("| Value | LI | CM | delta LI |")
        md_lines.append("|---|---|---|---|")
        for s in sweep:
            md_lines.append(f"| {s['value']} | {s['linking_index']:.4f} | {s['context_margin']:.4f} | {s['delta_li']:+.4f} |")

    md_path = TRIAGE_ROOT / "simulator_sensitivity.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    _emit(log_lines, f"MD   -> {md_path}")

    log_path = TRIAGE_ROOT / "simulator_sensitivity.log"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    _emit(log_lines, f"LOG  -> {log_path}")

    _emit(log_lines, "\nDone.")


if __name__ == "__main__":
    main()
