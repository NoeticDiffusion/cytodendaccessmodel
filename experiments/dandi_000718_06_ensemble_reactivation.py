"""Experiment 06 — NMF ensemble extraction and offline reactivation (DANDI 000718).

For each available session, NMF components (candidate memory ensembles) are
extracted from the largest offline window.  The temporal reactivation of each
component is then scored against a circular-shift null and reported.

Cross-session: for subject/session pairs where a matching "encoding" session
(NeutralExposure Day1) and an "offline" session (Day2 offline) share an
overlapping unit roster, we project the encoding ensembles onto the offline
data and compare against within-session null ensembles.

Usage:
    python experiments/dandi_000718_06_ensemble_reactivation.py

Outputs:
    data/dandi/triage/000718/ensemble_reactivation.json
    data/dandi/triage/000718/ensemble_reactivation.md
    data/dandi/triage/000718/ensemble_reactivation.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "000718"
TRIAGE_ROOT = ROOT / "data" / "dandi" / "triage" / "000718"

sys.path.insert(0, str(ROOT / "src"))

_N_COMPONENTS = 8
_MIN_UNITS = 10
_NULL_N = 200


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
def _emit(log_lines: list[str], message: str = "") -> None:
    print(message)
    log_lines.append(message)


# ---------------------------------------------------------------------------
# Single-session ensemble extraction
# ---------------------------------------------------------------------------
def _process_session(
    asset_path: Path,
    session_id: str,
    log_lines: list[str],
) -> dict | None:
    from dandi_analysis.readiness import check_readiness
    from dandi_analysis.dataset_000718.epochs import extract_offline_windows
    from dandi_analysis.dataset_000718.activity import build_activity_matrix
    from dandi_analysis.dataset_000718.ensembles import (
        extract_ensembles,
        offline_ensemble_reactivation,
    )
    import numpy as np

    status = check_readiness(asset_path)
    if status.error or not status.is_nwb_openable:
        _emit(log_lines, f"  SKIP {session_id}: {status.error or 'not openable'}")
        return None

    windows = extract_offline_windows(asset_path, session_id, min_duration_sec=30.0)
    if not windows:
        _emit(log_lines, f"  SKIP {session_id}: no offline windows")
        return None

    # Use the longest window for ensemble extraction
    primary = max(windows, key=lambda w: w.duration_sec)
    _emit(log_lines, f"  window: {primary.label}  ({primary.duration_sec:.0f}s)")

    mat = build_activity_matrix(asset_path, session_id, window=primary)
    if mat is None or mat.n_units < _MIN_UNITS:
        _emit(log_lines, f"  SKIP {session_id}: activity matrix too small")
        return None

    data_arr = np.array(mat.data)
    # Filter silent units
    good = np.array(data_arr.std(axis=0)) > 1e-6
    if good.sum() < _MIN_UNITS:
        _emit(log_lines, f"  SKIP {session_id}: too few active units")
        return None
    data_arr = data_arr[:, good]
    unit_ids = tuple(u for u, g in zip(mat.unit_ids, good) if g)

    n_comp = min(_N_COMPONENTS, data_arr.shape[1], data_arr.shape[0] // 4)
    if n_comp < 2:
        _emit(log_lines, f"  SKIP {session_id}: not enough data for NMF")
        return None

    result = extract_ensembles(
        data_arr,
        unit_ids,
        session_id=session_id,
        window_label=primary.label,
        n_components=n_comp,
    )
    _emit(
        log_lines,
        f"  NMF: k={n_comp}  reconstruction_error={result.reconstruction_error:.4f}"
        f"  total_var_explained={result.total_variance_explained:.3f}",
    )

    # For each ensemble compute self-reactivation z-score (temporal autocorrelation proxy)
    # Split window in half: use first half to define "encoding", second for "reactivation"
    T = data_arr.shape[0]
    half = T // 2
    enc_data = data_arr[:half]
    off_data = data_arr[half:]

    enc_result = extract_ensembles(
        enc_data,
        unit_ids,
        session_id=session_id,
        window_label="encoding_half",
        n_components=n_comp,
    )

    comp_scores: list[dict] = []
    for ens in enc_result.ensembles:
        score = offline_ensemble_reactivation(off_data, ens, null_n=_NULL_N, rng_seed=42 + ens.component_idx)
        top_n = int((ens.unit_weights > 0.3 * ens.unit_weights.max()).sum())
        comp_scores.append(
            {
                "component": ens.component_idx,
                "z_score": round(score["z_score"], 3) if not (score["z_score"] != score["z_score"]) else None,
                "mean_activation": round(score.get("burst_score", score.get("mean_activation", 0.0)), 6),
                "null_mean": round(score["null_mean"], 6),
                "null_std": round(score["null_std"], 6),
                "top_units_n": top_n,
                "explained_variance_ratio": round(ens.explained_variance_ratio, 4),
            }
        )

    # Summary: how many components have z > 1.96
    n_sig = sum(1 for s in comp_scores if s["z_score"] is not None and s["z_score"] > 1.96)
    _emit(log_lines, f"  significant reactivations (z>1.96): {n_sig}/{n_comp}")

    return {
        "session_id": session_id,
        "window_label": primary.label,
        "window_duration_sec": round(primary.duration_sec, 1),
        "n_units": int(good.sum()),
        "n_components": n_comp,
        "total_var_explained": round(result.total_variance_explained, 4),
        "component_scores": comp_scores,
        "n_significant": n_sig,
    }


# ---------------------------------------------------------------------------
# Cross-session matching
# ---------------------------------------------------------------------------
def _cross_session_projection(
    session_records: list[dict],
    asset_map: dict[str, Path],
    log_lines: list[str],
) -> list[dict]:
    """Project encoding-session ensembles onto offline-session data.

    The dataset 000718 has a NeutralExposure session (Day1) and an offline
    session (Day2). If they share unit_ids we can project Day1 ensembles onto
    Day2 offline data.
    """
    from dandi_analysis.dataset_000718.epochs import extract_offline_windows
    from dandi_analysis.dataset_000718.activity import build_activity_matrix
    from dandi_analysis.dataset_000718.ensembles import (
        extract_ensembles,
        ensemble_overlap,
        offline_ensemble_reactivation,
    )
    import numpy as np

    cross_results: list[dict] = []

    # Group sessions by subject
    by_subject: dict[str, list[dict]] = {}
    for rec in session_records:
        subj = rec["session_id"].split("_")[0]
        by_subject.setdefault(subj, []).append(rec)

    for subj, recs in by_subject.items():
        if len(recs) < 2:
            continue
        _emit(log_lines, f"\n  [cross-session] subject={subj}  n_sessions={len(recs)}")

        # Build activity matrices for all sessions of this subject
        sess_data: list[tuple[str, np.ndarray, tuple[str, ...]]] = []
        for rec in recs:
            path = asset_map.get(rec["session_id"])
            if path is None:
                continue
            windows = extract_offline_windows(path, rec["session_id"], min_duration_sec=30.0)
            if not windows:
                continue
            primary = max(windows, key=lambda w: w.duration_sec)
            mat = build_activity_matrix(path, rec["session_id"], window=primary)
            if mat is None or mat.n_units < _MIN_UNITS:
                continue
            data_arr = np.array(mat.data)
            good = np.array(data_arr.std(axis=0)) > 1e-6
            if good.sum() < _MIN_UNITS:
                continue
            unit_ids = tuple(u for u, g in zip(mat.unit_ids, good) if g)
            sess_data.append((rec["session_id"], data_arr[:, good], unit_ids))

        if len(sess_data) < 2:
            _emit(log_lines, "    not enough sessions with usable data")
            continue

        # For each pair: use first as encoding, second as offline
        for i in range(len(sess_data)):
            for j in range(i + 1, len(sess_data)):
                enc_id, enc_data, enc_units = sess_data[i]
                off_id, off_data, off_units = sess_data[j]

                # Find shared units
                shared = sorted(set(enc_units) & set(off_units))
                if len(shared) < _MIN_UNITS:
                    _emit(log_lines, f"    {enc_id} -> {off_id}: only {len(shared)} shared units")
                    continue

                enc_idx = [enc_units.index(u) for u in shared]
                off_idx = [off_units.index(u) for u in shared]

                enc_sub = enc_data[:, enc_idx]
                off_sub = off_data[:, off_idx]

                n_comp = min(_N_COMPONENTS, len(shared), enc_sub.shape[0] // 4)
                if n_comp < 2:
                    continue

                enc_ens_res = extract_ensembles(
                    enc_sub,
                    tuple(shared),
                    session_id=enc_id,
                    window_label="encoding",
                    n_components=n_comp,
                )

                pair_scores: list[dict] = []
                for ens in enc_ens_res.ensembles:
                    score = offline_ensemble_reactivation(
                        off_sub, ens, null_n=_NULL_N, rng_seed=42 + ens.component_idx
                    )
                    pair_scores.append(
                        {
                            "component": ens.component_idx,
                            "z_score": round(score["z_score"], 3) if score["z_score"] == score["z_score"] else None,
                            "mean_activation": round(score.get("burst_score", score.get("mean_activation", 0.0)), 6),
                        }
                    )

                n_sig = sum(1 for s in pair_scores if s["z_score"] is not None and s["z_score"] > 1.96)
                mean_z_vals = [s["z_score"] for s in pair_scores if s["z_score"] is not None]
                mean_z = round(float(np.mean(mean_z_vals)), 3) if mean_z_vals else None

                _emit(
                    log_lines,
                    f"    {enc_id} -> {off_id}: n_shared={len(shared)}"
                    f"  n_sig={n_sig}/{n_comp}  mean_z={mean_z}",
                )

                cross_results.append(
                    {
                        "encoding_session": enc_id,
                        "offline_session": off_id,
                        "subject": subj,
                        "n_shared_units": len(shared),
                        "n_components": n_comp,
                        "n_significant": n_sig,
                        "mean_z": mean_z,
                        "component_scores": pair_scores,
                    }
                )

    return cross_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    from dandi_analysis.inventory import discover_nwb_assets
    from dandi_analysis.dataset_000718.index import parse_subject_session

    TRIAGE_ROOT.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    _emit(log_lines, "=" * 60)
    _emit(log_lines, "Exp 06 — NMF Ensemble Extraction + Offline Reactivation")
    _emit(log_lines, "Dataset: DANDI 000718")
    _emit(log_lines, "=" * 60)

    assets = discover_nwb_assets(DATA_ROOT)
    if not assets:
        _emit(log_lines, "No NWB files found — check DATA_ROOT")
        return

    session_results: list[dict] = []
    asset_map: dict[str, Path] = {}

    for asset in assets:
        if not asset.is_canonical:
            continue
        subj, sess = parse_subject_session(asset.path)
        session_id = f"{subj}_{sess}" if sess else subj
        _emit(log_lines, f"\n--- {session_id} ---")
        rec = _process_session(asset.path, session_id, log_lines)
        if rec is not None:
            session_results.append(rec)
            asset_map[session_id] = asset.path

    _emit(log_lines, "\n" + "=" * 60)
    _emit(log_lines, f"Sessions processed: {len(session_results)}")

    # Cross-session analysis
    _emit(log_lines, "\n--- Cross-session ensemble projection ---")
    cross_results = _cross_session_projection(session_results, asset_map, log_lines)

    # Aggregate summary
    all_z: list[float] = []
    for rec in session_results:
        for cs in rec["component_scores"]:
            if cs["z_score"] is not None:
                all_z.append(cs["z_score"])

    import numpy as np

    summary = {
        "n_sessions": len(session_results),
        "n_total_components": sum(r["n_components"] for r in session_results),
        "n_significant_total": sum(r["n_significant"] for r in session_results),
        "mean_z_across_components": round(float(np.mean(all_z)), 3) if all_z else None,
        "cross_session": cross_results,
        "sessions": session_results,
    }

    # Write outputs
    json_path = TRIAGE_ROOT / "ensemble_reactivation.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _emit(log_lines, f"\nJSON -> {json_path}")

    # Markdown report
    lines: list[str] = ["# Ensemble Reactivation — DANDI 000718\n"]
    lines.append(f"Sessions processed: {summary['n_sessions']}\n")
    lines.append(f"Total NMF components: {summary['n_total_components']}\n")
    lines.append(f"Significant reactivations (z>1.96): {summary['n_significant_total']}\n")
    if summary["mean_z_across_components"] is not None:
        lines.append(f"Mean z across all components: {summary['mean_z_across_components']}\n")
    lines.append("\n## Per-Session Summary\n")
    lines.append("| Session | Duration (s) | N units | N comp | N sig | Var expl |")
    lines.append("|---|---|---|---|---|---|")
    for rec in session_results:
        lines.append(
            f"| {rec['session_id']} | {rec['window_duration_sec']} "
            f"| {rec['n_units']} | {rec['n_components']} "
            f"| {rec['n_significant']} | {rec['total_var_explained']:.3f} |"
        )
    if cross_results:
        lines.append("\n## Cross-Session Projection\n")
        lines.append("| Encoding -> Offline | N shared | N sig | Mean z |")
        lines.append("|---|---|---|---|")
        for cr in cross_results:
            lines.append(
                f"| {cr['encoding_session']} -> {cr['offline_session']} "
                f"| {cr['n_shared_units']} | {cr['n_significant']}/{cr['n_components']} "
                f"| {cr['mean_z']} |"
            )

    md_path = TRIAGE_ROOT / "ensemble_reactivation.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    _emit(log_lines, f"MD   -> {md_path}")

    log_path = TRIAGE_ROOT / "ensemble_reactivation.log"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    _emit(log_lines, f"LOG  -> {log_path}")

    _emit(log_lines, "\nDone.")


if __name__ == "__main__":
    main()
