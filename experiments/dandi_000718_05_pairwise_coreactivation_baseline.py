"""Experiment 05 — Pairwise co-reactivation baseline (full multi-session).

Runs on all offline sessions in DANDI 000718, reports per-session co-reactivation
statistics, and compares against the simulator baseline family.

Usage:
    python experiments/dandi_000718_05_pairwise_coreactivation_baseline.py

Outputs:
    data/dandi/triage/000718/coreactivation_baseline.json
    data/dandi/triage/000718/coreactivation_baseline.md
    data/dandi/triage/000718/coreactivation_baseline.log
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "000718"
TRIAGE_ROOT = ROOT / "data" / "dandi" / "triage" / "000718"

sys.path.insert(0, str(ROOT / "src"))

_N_NULL_SAMPLES = 20
_MIN_UNITS = 5
_MAX_PAIRS_PER_WINDOW = 25


def _emit(log_lines: list[str], message: str = "") -> None:
    print(message)
    log_lines.append(message)


def _stable_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _sample_pairs(unit_ids: list[str], session_id: str, limit: int):
    import numpy as np

    all_pairs = [(unit_ids[i], unit_ids[j]) for i in range(len(unit_ids)) for j in range(i + 1, len(unit_ids))]
    seed = _stable_seed(session_id)
    if len(all_pairs) <= limit:
        return all_pairs, seed, len(all_pairs)

    rng = np.random.default_rng(seed)
    selected_idx = rng.choice(len(all_pairs), size=limit, replace=False)
    sampled = [all_pairs[int(idx)] for idx in selected_idx]
    return sampled, seed, len(all_pairs)


def _run_data_side() -> dict:
    from dandi_analysis.inventory import canonical_assets, discover_nwb_assets
    from dandi_analysis.readiness import check_readiness
    from dandi_analysis.dataset_000718.index import parse_subject_session
    from dandi_analysis.dataset_000718.epochs import extract_offline_windows
    from dandi_analysis.dataset_000718.activity import build_activity_matrix
    from dandi_analysis.dataset_000718.observables import offline_coreactivation_score
    from dandi_analysis.dataset_000718.nulls import (
        circular_time_shift,
        matched_count_shuffle,
        unit_label_permutation,
    )
    import numpy as np

    if not DATA_ROOT.exists():
        return {"status": "no_data", "sessions": []}

    assets = discover_nwb_assets(DATA_ROOT)
    can = canonical_assets(assets)

    session_summaries = []

    for asset in can:
        r = check_readiness(asset.path)
        if not r.is_ready:
            continue

        subject_id, session_label = parse_subject_session(asset.path)
        session_id = f"{subject_id}__{session_label}"

        windows = extract_offline_windows(asset.path, session_id)
        if not windows:
            continue

        # Use the longest quiet_wake window for the primary analysis
        primary_window = max(windows, key=lambda w: w.duration_sec)

        mat = build_activity_matrix(asset.path, session_id, window=primary_window)
        if mat is None or mat.n_units < _MIN_UNITS:
            continue

        data_arr = __import__("numpy").array(mat.data)
        good_idx = [i for i, s in enumerate(data_arr.std(axis=0)) if s > 1e-6]
        if len(good_idx) < 2:
            continue

        uid_list = [list(mat.unit_ids)[i] for i in good_idx]

        nulls = (
            [circular_time_shift(mat, shift=s) for s in range(10, _N_NULL_SAMPLES * 10 + 1, 10)]
            + [unit_label_permutation(mat, np.random.default_rng(i)) for i in range(_N_NULL_SAMPLES)]
            + [matched_count_shuffle(mat, np.random.default_rng(i + 100)) for i in range(_N_NULL_SAMPLES)]
        )

        sampled_pairs, pair_seed, n_possible_pairs = _sample_pairs(uid_list, session_id, _MAX_PAIRS_PER_WINDOW)

        pair_results = []
        for trace_i, trace_j in sampled_pairs:
            result = offline_coreactivation_score(
                mat, primary_window, trace_i, trace_j,
                null_matrices=nulls,
            )
            pair_results.append({
                "trace_i": trace_i,
                "trace_j": trace_j,
                "co_reactivation_score": result.co_reactivation_score,
                "null_mean": result.null_mean,
                "null_std": result.null_std,
                "z_score": result.z_score,
            })

        valid = [p for p in pair_results if not math.isnan(p["co_reactivation_score"])]
        valid_z = [p["z_score"] for p in valid if not math.isnan(p["z_score"])]
        above_threshold = [p for p in valid if p.get("z_score", float("nan")) > 1.65]

        session_summaries.append({
            "session_id": session_id,
            "subject_id": subject_id,
            "session_label": session_label,
            "window_label": primary_window.label,
            "window_duration_sec": primary_window.duration_sec,
            "n_good_units": len(good_idx),
            "n_possible_pairs": n_possible_pairs,
            "pair_sampling_mode": "random_without_replacement",
            "pair_sampling_seed": pair_seed,
            "n_pairs_tested": len(valid),
            "mean_score": sum(p["co_reactivation_score"] for p in valid) / max(1, len(valid)),
            "mean_z": sum(valid_z) / max(1, len(valid_z)),
            "n_above_threshold": len(above_threshold),
            "pairs": pair_results[:10],
        })

    return {
        "status": "ok" if session_summaries else "no_windows",
        "sessions": session_summaries,
    }


def _run_model_side() -> dict:
    from dandi_analysis.simulator_bridge import run_baseline_scenarios
    return run_baseline_scenarios()


def _write_summary(path: Path, model_results: dict, data_result: dict) -> None:
    lines = [
        "# Pairwise Co-reactivation Baseline Summary: DANDI 000718",
        "",
        "## Model Side",
        "",
        "| Baseline | Linking index | Context margin |",
        "| --- | ---: | ---: |",
    ]
    for name, res in model_results.items():
        lines.append(
            f"| {name} | {res['linking_index_model']:.4f} | {res['context_margin_model']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Data Side",
            "",
            "| Session | Window (s) | Good units | Pairs tested | Mean score | Mean z | N > 1.65 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for session in data_result.get("sessions", []):
        lines.append(
            f"| {session['session_label']} | {session['window_duration_sec']:.0f} | "
            f"{session['n_good_units']} | {session['n_pairs_tested']} | "
            f"{session['mean_score']:.4f} | {session['mean_z']:.3f} | {session['n_above_threshold']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    log_lines: list[str] = []
    _emit(log_lines, "=" * 60)
    _emit(log_lines, "Experiment 05 - Pairwise Co-reactivation Baseline")
    _emit(log_lines, "=" * 60)

    _emit(log_lines, "\n[MODEL SIDE] Running simulator baseline scenarios ...")
    model_results = _run_model_side()

    _emit(log_lines, f"\n  {'Baseline':<25}  {'Linking index':>14}  {'Context margin':>15}")
    _emit(log_lines, f"  {'-'*25}  {'-'*14}  {'-'*15}")
    for name, res in model_results.items():
        _emit(log_lines, f"  {name:<25}  {res['linking_index_model']:>14.4f}  "
                         f"{res['context_margin_model']:>15.4f}")

    _emit(log_lines, "\n[DATA SIDE] Processing all offline sessions in DANDI 000718 ...")
    data_result = _run_data_side()
    status = data_result["status"]

    if status == "no_data":
        _emit(log_lines, f"  No local NWB data found at: {DATA_ROOT}")
    elif status == "no_windows":
        _emit(log_lines, "  NWB files found but no usable offline windows extracted.")
    else:
        sessions = data_result["sessions"]
        _emit(log_lines, f"\n  Sessions with offline windows: {len(sessions)}")
        _emit(log_lines)
        _emit(log_lines, f"  {'Session':<35} {'Win dur':>8} {'Good units':>10} {'Pairs':>6} "
                         f"{'Mean score':>11} {'Mean z':>8} {'N>1.65':>7}")
        _emit(log_lines, f"  {'-'*35} {'-'*8} {'-'*10} {'-'*6} {'-'*11} {'-'*8} {'-'*7}")
        for s in sessions:
            _emit(log_lines, f"  {s['session_label']:<35} {s['window_duration_sec']:>8.0f}s "
                             f"{s['n_good_units']:>10} {s['n_pairs_tested']:>6} "
                             f"{s['mean_score']:>11.4f} {s['mean_z']:>8.3f} "
                             f"{s['n_above_threshold']:>7}")

        # Cross-session comparison for Ca-EEG3-4 (has both Day1 and Day2)
        day1 = next((s for s in sessions if "OfflineDay1" in s["session_label"]
                     and "Ca-EEG3-4" in s["session_id"]), None)
        day2_3_4 = next((s for s in sessions if "OfflineDay2" in s["session_label"]
                         and "Ca-EEG3-4" in s["session_id"]), None)

        _emit(log_lines, "\n  --- Cross-session (Ca-EEG3-4 Day1 vs Day2) ---")
        if day1 and day2_3_4:
            delta_z = day2_3_4["mean_z"] - day1["mean_z"]
            delta_score = day2_3_4["mean_score"] - day1["mean_score"]
            _emit(log_lines, f"  Day1 mean z: {day1['mean_z']:.3f}   Day2 mean z: {day2_3_4['mean_z']:.3f}  "
                             f"delta: {delta_z:+.3f}")
            _emit(log_lines, f"  Day1 mean score: {day1['mean_score']:.4f}   Day2 mean score: {day2_3_4['mean_score']:.4f}  "
                             f"delta: {delta_score:+.4f}")
            if delta_z > 0:
                _emit(log_lines, "  DIRECTION: Day2 co-reactivation HIGHER than Day1 (consistent with linking)")
            else:
                _emit(log_lines, "  DIRECTION: Day2 co-reactivation not higher than Day1")
        else:
            _emit(log_lines, "  (not enough sessions for comparison)")

    # Directional comparison
    full_li = model_results["full_model"]["linking_index_model"]
    best_baseline_li = max(
        v["linking_index_model"] for k, v in model_results.items() if k != "full_model"
    )
    _emit(log_lines, f"\n  Full model LI: {full_li:.4f}  Best baseline LI: {best_baseline_li:.4f}  "
                     f"{'PASS' if full_li > best_baseline_li else 'NOTE'}")

    output = {"model": model_results, "data": data_result}
    TRIAGE_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = TRIAGE_ROOT / "coreactivation_baseline.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, default=str)
    _write_summary(TRIAGE_ROOT / "coreactivation_baseline.md", model_results, data_result)
    _emit(log_lines, f"\nResults written: {out_path.relative_to(ROOT)}")
    _emit(log_lines, f"Summary written: {(TRIAGE_ROOT / 'coreactivation_baseline.md').relative_to(ROOT)}")
    _emit(log_lines, f"Log written: {(TRIAGE_ROOT / 'coreactivation_baseline.log').relative_to(ROOT)}")
    (TRIAGE_ROOT / "coreactivation_baseline.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
