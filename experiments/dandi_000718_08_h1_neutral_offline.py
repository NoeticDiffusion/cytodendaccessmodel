"""Experiment 08 — H1 pipeline: NeutralExposure full-session ensembles
projected onto OfflineDay activity (DANDI 000718, Ca-EEG3-4).

This is the biologically correct H1 test:
  1. Extract NMF ensembles from the *full* NeutralExposure (encoding) session.
  2. Project each ensemble's spatial weight vector onto the *full* OfflineDay
     (consolidation) session's activity matrix.
  3. Score reactivation using the burst score (top-10% frames), which is
     sensitive to replay-like events rather than the shift-invariant plain mean.
  4. Compare against a circular-shift null applied to the projection timeseries.

Run NMF robustness sweep: k in {8, 16, 32}, multiple random restarts per k.

Usage:
    python experiments/dandi_000718_08_h1_neutral_offline.py

Outputs:
    data/dandi/triage/000718/h1_neutral_offline.json
    data/dandi/triage/000718/h1_neutral_offline.md
    data/dandi/triage/000718/h1_neutral_offline.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "000718"
TRIAGE_ROOT = ROOT / "data" / "dandi" / "triage" / "000718"

sys.path.insert(0, str(ROOT / "src"))

_MIN_UNITS = 10
_NMF_K_VALUES = [8, 16, 32]
_NMF_N_RESTARTS = 3
_NULL_N = 300
_TOP_FRAC = 0.10


def _emit(log_lines: list[str], msg: str = "") -> None:
    print(msg)
    log_lines.append(msg)


# ---------------------------------------------------------------------------
# Session discovery helpers
# ---------------------------------------------------------------------------

def _find_session_paths(subject: str) -> dict[str, Path | None]:
    """Return {'neutral': path, 'offline_d1': path, 'offline_d2': path} for subject.

    The NWB tree uses ``sub-<subject>`` subdirectories.
    """
    mapping: dict[str, Path | None] = {"neutral": None, "offline_d1": None, "offline_d2": None}
    # Try both bare and prefixed directory names
    for dir_candidate in (DATA_ROOT / f"sub-{subject}", DATA_ROOT / subject):
        if dir_candidate.exists():
            subject_dir = dir_candidate
            break
    else:
        return mapping

    for nwb_path in sorted(subject_dir.rglob("*.nwb")):
        name = nwb_path.name.lower()
        if "neutralexposure" in name:
            mapping["neutral"] = nwb_path
        elif "offlineday1" in name:
            mapping["offline_d1"] = nwb_path
        elif "offlineday2" in name:
            mapping["offline_d2"] = nwb_path

    return mapping


# ---------------------------------------------------------------------------
# NMF with multiple restarts, return best reconstruction
# ---------------------------------------------------------------------------

def _fit_nmf_best(data: np.ndarray, unit_ids: tuple[str, ...], session_id: str, k: int):
    """Fit NMF with multiple restarts and return the run with lowest recon error."""
    from dandi_analysis.dataset_000718.ensembles import extract_ensembles

    best = None
    for restart in range(_NMF_N_RESTARTS):
        result = extract_ensembles(
            data, unit_ids, session_id=session_id,
            n_components=k, random_state=42 + restart * 13,
        )
        if best is None or result.reconstruction_error < best.reconstruction_error:
            best = result
    return best


# ---------------------------------------------------------------------------
# Core H1 test for one (neutral, offline) pair
# ---------------------------------------------------------------------------

def _run_h1_pair(
    neutral_path: Path,
    offline_path: Path,
    subject: str,
    pair_label: str,
    log_lines: list[str],
) -> dict | None:
    import numpy as np
    from dandi_analysis.dataset_000718.activity import (
        build_full_session_activity_matrix,
        build_activity_matrix,
    )
    from dandi_analysis.dataset_000718.epochs import extract_offline_windows
    from dandi_analysis.dataset_000718.ensembles import offline_ensemble_reactivation

    _emit(log_lines, f"\n  [{pair_label}]")
    _emit(log_lines, f"    Encoding (NeutralExposure): {neutral_path.name}")
    _emit(log_lines, f"    Offline:                    {offline_path.name}")

    # ---- Load encoding session (full, no window restriction) ----
    enc_mat = build_full_session_activity_matrix(neutral_path, f"{subject}_NE")
    if enc_mat is None or enc_mat.n_units < _MIN_UNITS:
        _emit(log_lines, "    SKIP: NeutralExposure matrix too small or failed")
        return None

    enc_data = np.array(enc_mat.data)
    enc_good = np.array(enc_data.std(axis=0)) > 1e-6
    if enc_good.sum() < _MIN_UNITS:
        _emit(log_lines, "    SKIP: too few active units in NeutralExposure")
        return None

    enc_data = enc_data[:, enc_good]
    enc_units = tuple(u for u, g in zip(enc_mat.unit_ids, enc_good) if g)
    _emit(log_lines, f"    NE active units: {len(enc_units)}  frames: {enc_data.shape[0]}")

    # ---- Load offline session (best available: use largest offline window if
    #      sleep annotations exist, else use the full session) ----
    offline_sid = f"{subject}_OD"
    windows = extract_offline_windows(offline_path, offline_sid, min_duration_sec=30.0)
    if windows:
        primary_window = max(windows, key=lambda w: w.duration_sec)
        off_mat = build_activity_matrix(offline_path, offline_sid, window=primary_window)
        _emit(log_lines, f"    Offline window: {primary_window.label} ({primary_window.duration_sec:.0f}s)")
    else:
        off_mat = build_full_session_activity_matrix(offline_path, offline_sid)
        _emit(log_lines, "    Offline: no sleep annotations, using full session")

    if off_mat is None or off_mat.n_units < _MIN_UNITS:
        _emit(log_lines, "    SKIP: offline matrix too small or failed")
        return None

    off_data = np.array(off_mat.data)

    # ---- Shared units ----
    shared_units = sorted(set(enc_units) & set(off_mat.unit_ids))
    if len(shared_units) < _MIN_UNITS:
        _emit(log_lines, f"    SKIP: only {len(shared_units)} shared units")
        return None

    enc_idx = [list(enc_units).index(u) for u in shared_units]
    off_idx = [list(off_mat.unit_ids).index(u) for u in shared_units]
    enc_sub = enc_data[:, enc_idx]
    off_sub = off_data[:, off_idx]
    _emit(log_lines, f"    Shared units: {len(shared_units)}  off frames: {off_sub.shape[0]}")

    # ---- NMF robustness sweep ----
    pair_k_results: list[dict] = []
    for k in _NMF_K_VALUES:
        actual_k = min(k, len(shared_units), enc_sub.shape[0] // 4)
        if actual_k < 2:
            _emit(log_lines, f"    k={k}: skip (data too small)")
            continue

        enc_result = _fit_nmf_best(enc_sub, tuple(shared_units), f"{subject}_NE", actual_k)
        if enc_result is None:
            continue

        comp_scores: list[dict] = []
        for ens in enc_result.ensembles:
            score = offline_ensemble_reactivation(
                off_sub, ens,
                null_n=_NULL_N, rng_seed=42 + ens.component_idx,
                top_frac=_TOP_FRAC,
            )
            comp_scores.append({
                "component": ens.component_idx,
                "z_score": round(score["z_score"], 3) if score["z_score"] == score["z_score"] else None,
                "burst_score": round(score["burst_score"], 6),
                "null_mean": round(score["null_mean"], 6),
                "null_std": round(score["null_std"], 6),
                "n_top_units": int((ens.unit_weights > 0.3 * ens.unit_weights.max()).sum()),
                "ev_ratio": round(ens.explained_variance_ratio, 4),
            })

        n_sig = sum(1 for s in comp_scores if s["z_score"] is not None and s["z_score"] > 1.96)
        valid_z = [s["z_score"] for s in comp_scores if s["z_score"] is not None]
        mean_z = round(float(np.mean(valid_z)), 3) if valid_z else None

        _emit(log_lines, (
            f"    k={actual_k}  restarts={_NMF_N_RESTARTS}"
            f"  recon_err={enc_result.reconstruction_error:.2f}"
            f"  var_expl={enc_result.total_variance_explained:.3f}"
            f"  n_sig={n_sig}/{actual_k}  mean_z={mean_z}"
        ))

        pair_k_results.append({
            "k": actual_k,
            "reconstruction_error": round(enc_result.reconstruction_error, 4),
            "total_var_explained": round(enc_result.total_variance_explained, 4),
            "n_significant": n_sig,
            "mean_z": mean_z,
            "component_scores": comp_scores,
        })

    if not pair_k_results:
        return None

    return {
        "subject": subject,
        "pair_label": pair_label,
        "n_encoding_units": int(enc_good.sum()),
        "n_shared_units": len(shared_units),
        "n_offline_frames": int(off_sub.shape[0]),
        "top_frac": _TOP_FRAC,
        "k_sweep": pair_k_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

import numpy as np


def main() -> None:
    from dandi_analysis.inventory import discover_nwb_assets
    from dandi_analysis.dataset_000718.index import parse_subject_session

    TRIAGE_ROOT.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    _emit(log_lines, "=" * 60)
    _emit(log_lines, "Exp 08 - H1: NeutralExposure -> OfflineDay")
    _emit(log_lines, "Dataset: DANDI 000718  (Ca-EEG3-4 primary)")
    _emit(log_lines, f"NMF k sweep: {_NMF_K_VALUES}  restarts: {_NMF_N_RESTARTS}")
    _emit(log_lines, f"Null: circular-shift, n={_NULL_N}  top_frac={_TOP_FRAC}")
    _emit(log_lines, "=" * 60)

    # Discover subjects with NeutralExposure files
    assets = discover_nwb_assets(DATA_ROOT)
    subjects_with_ne: set[str] = set()
    for a in assets:
        if not a.is_canonical:
            continue
        if "NeutralExposure" in a.path.name:
            subj, _ = parse_subject_session(a.path)
            subjects_with_ne.add(subj)

    if not subjects_with_ne:
        _emit(log_lines, "No NeutralExposure files found.")
        return

    _emit(log_lines, f"\nSubjects with NeutralExposure: {sorted(subjects_with_ne)}")

    all_results: list[dict] = []

    for subject in sorted(subjects_with_ne):
        paths = _find_session_paths(subject)
        neutral = paths["neutral"]

        if neutral is None:
            _emit(log_lines, f"\n{subject}: NeutralExposure path not found, skipping")
            continue

        for label, offline_key in [("NE->OfflineD1", "offline_d1"), ("NE->OfflineD2", "offline_d2")]:
            offline = paths[offline_key]
            if offline is None:
                _emit(log_lines, f"\n{subject}: {label} — offline path not found")
                continue

            result = _run_h1_pair(neutral, offline, subject, label, log_lines)
            if result is not None:
                all_results.append(result)

    # ---- Aggregate summary ----
    _emit(log_lines, "\n" + "=" * 60)
    _emit(log_lines, "SUMMARY")
    _emit(log_lines, "=" * 60)

    for res in all_results:
        _emit(log_lines, f"\n  {res['subject']} {res['pair_label']}")
        _emit(log_lines, f"    shared units: {res['n_shared_units']}  offline frames: {res['n_offline_frames']}")
        for ks in res["k_sweep"]:
            _emit(log_lines, (
                f"    k={ks['k']}: n_sig={ks['n_significant']}/{ks['k']}"
                f"  mean_z={ks['mean_z']}  var_expl={ks['total_var_explained']:.3f}"
            ))

    # ---- Write outputs ----
    json_path = TRIAGE_ROOT / "h1_neutral_offline.json"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    _emit(log_lines, f"\nJSON -> {json_path}")

    # Markdown
    md_lines = ["# H1 Pipeline: NeutralExposure → OfflineDay\n"]
    md_lines.append(f"Score: burst-mean top {_TOP_FRAC*100:.0f}% frames, circular-shift null (n={_NULL_N})\n")
    for res in all_results:
        md_lines.append(f"\n## {res['subject']} {res['pair_label']}\n")
        md_lines.append(f"Shared units: {res['n_shared_units']}  Offline frames: {res['n_offline_frames']}\n")
        md_lines.append("| k | Var expl | N sig | Mean z |")
        md_lines.append("|---|---|---|---|")
        for ks in res["k_sweep"]:
            md_lines.append(
                f"| {ks['k']} | {ks['total_var_explained']:.3f}"
                f" | {ks['n_significant']}/{ks['k']} | {ks['mean_z']} |"
            )

    md_path = TRIAGE_ROOT / "h1_neutral_offline.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    _emit(log_lines, f"MD   -> {md_path}")

    log_path = TRIAGE_ROOT / "h1_neutral_offline.log"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    _emit(log_lines, f"LOG  -> {log_path}")

    _emit(log_lines, "\nDone.")


if __name__ == "__main__":
    main()
