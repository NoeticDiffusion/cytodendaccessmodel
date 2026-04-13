"""Experiment 10 — H1: Registration-aware, event-based NeutralExposure -> OfflineDay.

This is the main deliverable from the H1 event-bridge implementer brief:

  WP1: spatial footprint registration (centroid + Dice + shape + neighbourhood
       consistency) to establish cross-session unit correspondence with confidence.
  WP2: high-synchrony event detection in the offline period followed by
       ensemble recruitment scoring using event-local projection nulls.
  WP3: NMF / ICA / graph assembly comparison + cross-restart stability.
  WP4: interpretive stop-rule report — does any method converge on a positive
       or does the negative boundary hold under the improved observable?

Stops when one of:
  (a) an event-based, registration-aware analysis yields a reproducible above-null
      result across multiple assembly methods, or
  (b) all three assembly methods converge on the same negative boundary.

Usage:
    python experiments/dandi_000718_10_h1_event_registration.py

Outputs:
    data/dandi/triage/000718/h1_event_registration.json
    data/dandi/triage/000718/h1_event_registration.md
    data/dandi/triage/000718/h1_event_registration.log
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "000718"
TRIAGE_ROOT = ROOT / "data" / "dandi" / "triage" / "000718"

sys.path.insert(0, str(ROOT / "src"))

_MIN_REG_UNITS = 20         # minimum accepted registered units to proceed
_NMF_K = 8
_STABILITY_RESTARTS = 4
_EVENT_THRESHOLD_SIGMA = 2.0
_NULL_N = 300


def _emit(log_lines: list[str], msg: str = "") -> None:
    print(msg)
    log_lines.append(msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_session_paths(subject: str) -> dict[str, Path | None]:
    mapping: dict[str, Path | None] = {"neutral": None, "offline_d1": None, "offline_d2": None}
    for dir_candidate in (DATA_ROOT / f"sub-{subject}", DATA_ROOT / subject):
        if dir_candidate.exists():
            subject_dir = dir_candidate
            break
    else:
        return mapping
    for p in sorted(subject_dir.rglob("*.nwb")):
        name = p.name.lower()
        if "neutralexposure" in name:
            mapping["neutral"] = p
        elif "offlineday1" in name:
            mapping["offline_d1"] = p
        elif "offlineday2" in name:
            mapping["offline_d2"] = p
    return mapping


# ---------------------------------------------------------------------------
# WP1 — Registration
# ---------------------------------------------------------------------------

def _run_registration(
    ne_path: Path,
    off_path: Path,
    session_ne: str,
    session_off: str,
    log_lines: list[str],
) -> tuple | None:
    """Register sessions, return (reg_result, ne_unit_ids, off_unit_ids) or None."""
    from dandi_analysis.dataset_000718.registration import register_sessions

    _emit(log_lines, f"  Registration: {session_ne} -> {session_off}")
    reg = register_sessions(
        ne_path, off_path, session_ne, session_off,
        max_centroid_dist_px=12.0,
        min_dice=0.05,
        confidence_threshold=0.25,
    )
    summ = reg.summary_dict()
    _emit(log_lines, (
        f"    ROIs: A={summ['n_rois_a']}  B={summ['n_rois_b']}"
        f"  candidates={summ['n_candidates']}  accepted={summ['n_accepted']}"
        f"  frac_A={summ['fraction_a_matched']}  mean_conf={summ['mean_confidence']}"
        f"  mean_dice={summ['mean_dice']}"
    ))

    if summ["n_accepted"] < _MIN_REG_UNITS:
        _emit(log_lines, f"    SKIP: only {summ['n_accepted']} registered units (min {_MIN_REG_UNITS})")
        return None

    src_idx, tgt_idx = reg.matched_indices()
    ne_unit_ids = tuple(str(i) for i in src_idx)
    off_unit_ids = tuple(str(i) for i in tgt_idx)
    return reg, ne_unit_ids, off_unit_ids, summ


# ---------------------------------------------------------------------------
# WP2 + WP3 — Event scoring + assembly benchmark
# ---------------------------------------------------------------------------

def _run_assembly_and_events(
    ne_path: Path,
    off_path: Path,
    session_ne: str,
    session_off: str,
    ne_unit_ids: tuple[str, ...],
    off_unit_ids: tuple[str, ...],
    n_registered: int,
    log_lines: list[str],
) -> dict:
    import numpy as np
    from dandi_analysis.dataset_000718.activity import build_full_session_activity_matrix, build_activity_matrix
    from dandi_analysis.dataset_000718.epochs import extract_offline_windows
    from dandi_analysis.dataset_000718.ensembles import benchmark_assembly_methods
    from dandi_analysis.dataset_000718.events import run_event_h1

    # ---- Load NE data (registered ROI subset) ----
    ne_mat = build_full_session_activity_matrix(ne_path, session_ne)
    if ne_mat is None:
        _emit(log_lines, "    SKIP: NE matrix failed")
        return {}

    ne_data = _select_units(ne_mat, ne_unit_ids)
    if ne_data is None or ne_data.shape[1] < _MIN_REG_UNITS:
        _emit(log_lines, f"    SKIP: NE registered subset too small ({ne_data.shape[1] if ne_data is not None else 0})")
        return {}
    _emit(log_lines, f"  NE data: {ne_data.shape[0]} frames x {ne_data.shape[1]} registered units")

    # ---- Load Offline data (registered ROI subset) ----
    windows = extract_offline_windows(off_path, session_off, min_duration_sec=30.0)
    if windows:
        primary = max(windows, key=lambda w: w.duration_sec)
        off_mat = build_activity_matrix(off_path, session_off, window=primary)
        _emit(log_lines, f"  Offline window: {primary.label} ({primary.duration_sec:.0f}s)")
    else:
        off_mat = build_full_session_activity_matrix(off_path, session_off)
        _emit(log_lines, "  Offline: full session (no sleep annotations)")

    if off_mat is None:
        _emit(log_lines, "    SKIP: offline matrix failed")
        return {}

    off_data = _select_units(off_mat, off_unit_ids)
    if off_data is None or off_data.shape[1] < _MIN_REG_UNITS:
        _emit(log_lines, f"    SKIP: offline registered subset too small")
        return {}
    _emit(log_lines, f"  Offline data: {off_data.shape[0]} frames x {off_data.shape[1]} registered units")

    method_results: dict[str, dict] = {}

    for signal_type, data_ne, data_off in [
        ("deconvolved", ne_data, off_data),
    ]:
        _emit(log_lines, f"\n  === {signal_type} ===")

        # ---- WP3: Assembly benchmark ----
        _emit(log_lines, f"  Assembly benchmark (k={_NMF_K}, {_STABILITY_RESTARTS} restarts)...")
        bench = benchmark_assembly_methods(
            data_ne, ne_unit_ids, session_ne,
            n_components=_NMF_K,
            n_stability_restarts=_STABILITY_RESTARTS,
        )

        for method_name, method_data in bench.items():
            ens_res = method_data["ensembles"]
            stab = method_data["stability"]
            _emit(log_lines, (
                f"    {method_name}: var_expl={ens_res.total_variance_explained:.3f}"
                f"  stability_mean={stab['mean_stability']}  stability_min={stab['min_stability']}"
            ))

            # ---- WP2: Event-based H1 ----
            if not ens_res.ensembles:
                continue
            ensemble_weights = [e.unit_weights for e in ens_res.ensembles]
            h1_result = run_event_h1(
                data_off, ensemble_weights,
                session_ne=session_ne,
                session_offline=session_off,
                n_registered_units=n_registered,
                threshold_sigma=_EVENT_THRESHOLD_SIGMA,
                null_n=_NULL_N,
                rng_seed=42,
            )
            summ = h1_result.session_summary
            _emit(log_lines, (
                f"      events={summ['n_events']}"
                f"  n_sig={summ['n_significant_z196']}/{summ['total_scored_pairs']}"
                f"  frac_sig={summ['fraction_significant']}"
                f"  max_z={summ['max_norm_score']}"
            ))

            method_results[f"{signal_type}_{method_name}"] = {
                "assembly": {
                    "method": method_name,
                    "signal_type": signal_type,
                    "var_explained": round(ens_res.total_variance_explained, 4)
                        if ens_res.total_variance_explained == ens_res.total_variance_explained else None,
                    "reconstruction_error": round(ens_res.reconstruction_error, 4)
                        if ens_res.reconstruction_error == ens_res.reconstruction_error else None,
                    "stability": stab,
                },
                "h1_event": {
                    "n_events": summ["n_events"],
                    "n_ensembles": h1_result.n_ensembles,
                    "total_pairs": summ["total_scored_pairs"],
                    "n_significant": summ["n_significant_z196"],
                    "fraction_significant": summ["fraction_significant"],
                    "max_norm_score": summ["max_norm_score"],
                    "threshold_sigma": _EVENT_THRESHOLD_SIGMA,
                },
            }

    return method_results


def _select_units(mat, unit_ids: tuple[str, ...]):
    """Select a subset of columns from an ActivityMatrix by unit_ids."""
    import numpy as np
    all_ids = list(mat.unit_ids)
    idx = []
    for uid in unit_ids:
        if uid in all_ids:
            idx.append(all_ids.index(uid))
    if not idx:
        return None
    data = np.array(mat.data)[:, idx]
    # Re-zscore the subset
    mu = data.mean(axis=0, keepdims=True)
    sd = data.std(axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (data - mu) / sd


# ---------------------------------------------------------------------------
# Stop-rule evaluation
# ---------------------------------------------------------------------------

def _stop_rule(method_results: dict[str, dict], log_lines: list[str]) -> str:
    """Evaluate WP4 stop rule. Returns 'positive', 'negative', or 'inconclusive'."""
    import numpy as np

    if not method_results:
        return "inconclusive"

    max_z_vals = [
        r["h1_event"]["max_norm_score"]
        for r in method_results.values()
        if r["h1_event"]["max_norm_score"] is not None
    ]
    frac_sig_vals = [
        r["h1_event"]["fraction_significant"]
        for r in method_results.values()
    ]

    n_above_null = sum(1 for z in max_z_vals if z is not None and z > 1.96)
    mean_frac_sig = float(np.mean(frac_sig_vals)) if frac_sig_vals else 0.0

    _emit(log_lines, f"\n  Stop-rule: n_methods_above_null={n_above_null}/{len(method_results)}")
    _emit(log_lines, f"  Mean frac significant across methods: {mean_frac_sig:.3f}")

    if n_above_null >= 2:
        verdict = "positive"
        _emit(log_lines, "  VERDICT: POSITIVE — event-based signal in >= 2 assembly methods")
    elif n_above_null == 0 and mean_frac_sig < 0.05:
        verdict = "negative"
        _emit(log_lines, "  VERDICT: NEGATIVE BOUNDARY — all assembly methods converge below null")
    else:
        verdict = "inconclusive"
        _emit(log_lines, "  VERDICT: INCONCLUSIVE — mixed result, see per-method breakdown")

    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from dandi_analysis.inventory import discover_nwb_assets
    from dandi_analysis.dataset_000718.index import parse_subject_session

    TRIAGE_ROOT.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    _emit(log_lines, "=" * 60)
    _emit(log_lines, "Exp 10 - H1: Registration-Aware Event-Based Pipeline")
    _emit(log_lines, "Dataset: DANDI 000718")
    _emit(log_lines, f"WP1 registration: centroid+Dice+shape+neighbourhood")
    _emit(log_lines, f"WP2 events: threshold_sigma={_EVENT_THRESHOLD_SIGMA}  null_n={_NULL_N}")
    _emit(log_lines, f"WP3 assemblies: NMF + ICA + Graph  k={_NMF_K}  restarts={_STABILITY_RESTARTS}")
    _emit(log_lines, "=" * 60)

    assets = discover_nwb_assets(DATA_ROOT)
    subjects_with_ne: set[str] = set()
    for a in assets:
        if a.is_canonical and "NeutralExposure" in a.path.name:
            subj, _ = parse_subject_session(a.path)
            subjects_with_ne.add(subj)

    if not subjects_with_ne:
        _emit(log_lines, "No NeutralExposure files found.")
        return

    all_pair_results: list[dict] = []

    for subject in sorted(subjects_with_ne):
        paths = _find_session_paths(subject)
        ne_path = paths["neutral"]
        if ne_path is None:
            continue

        for label, off_key in [("NE->OffD1", "offline_d1"), ("NE->OffD2", "offline_d2")]:
            off_path = paths[off_key]
            if off_path is None:
                _emit(log_lines, f"\n{subject} {label}: offline not found")
                continue

            session_ne = f"{subject}_NE"
            session_off = f"{subject}_{off_key}"
            _emit(log_lines, f"\n{'='*40}")
            _emit(log_lines, f"{subject}  {label}")
            _emit(log_lines, f"{'='*40}")

            reg_out = _run_registration(ne_path, off_path, session_ne, session_off, log_lines)
            if reg_out is None:
                continue

            reg, ne_uids, off_uids, reg_summ = reg_out

            method_results = _run_assembly_and_events(
                ne_path, off_path, session_ne, session_off,
                ne_uids, off_uids, reg_summ["n_accepted"], log_lines,
            )

            verdict = _stop_rule(method_results, log_lines)

            all_pair_results.append({
                "subject": subject,
                "pair_label": label,
                "registration": reg_summ,
                "method_results": method_results,
                "stop_rule_verdict": verdict,
            })

    # ---- Final summary ----
    _emit(log_lines, "\n" + "=" * 60)
    _emit(log_lines, "FINAL SUMMARY")
    _emit(log_lines, "=" * 60)
    for res in all_pair_results:
        _emit(log_lines, (
            f"  {res['subject']} {res['pair_label']}: "
            f"registered={res['registration']['n_accepted']}  "
            f"verdict={res['stop_rule_verdict']}"
        ))

    # Write artifacts
    json_path = TRIAGE_ROOT / "h1_event_registration.json"
    json_path.write_text(json.dumps(all_pair_results, indent=2, default=str), encoding="utf-8")
    _emit(log_lines, f"\nJSON -> {json_path}")

    # Markdown
    md_lines = [
        "# H1: Registration-Aware Event-Based Pipeline\n",
        f"WP1 registration: centroid+Dice+shape+neighbourhood confidence\n",
        f"WP2 events: threshold_sigma={_EVENT_THRESHOLD_SIGMA}\n",
        f"WP3 assemblies: NMF + ICA + Graph  k={_NMF_K}\n",
        "\n## Results\n",
        "| Subject | Pair | Registered units | Verdict |",
        "|---|---|---|---|",
    ]
    for res in all_pair_results:
        md_lines.append(
            f"| {res['subject']} | {res['pair_label']} "
            f"| {res['registration']['n_accepted']} | **{res['stop_rule_verdict']}** |"
        )

    for res in all_pair_results:
        md_lines.append(f"\n### {res['subject']} {res['pair_label']}\n")
        md_lines.append(f"Registration: {res['registration']}\n")
        md_lines.append("| Method | Events | N sig | Frac sig | Max z | Stab mean |")
        md_lines.append("|---|---|---|---|---|---|")
        for mname, mr in res.get("method_results", {}).items():
            h1 = mr.get("h1_event", {})
            stab = mr.get("assembly", {}).get("stability", {})
            md_lines.append(
                f"| {mname} | {h1.get('n_events')} | {h1.get('n_significant')} "
                f"| {h1.get('fraction_significant')} | {h1.get('max_norm_score')} "
                f"| {stab.get('mean_stability')} |"
            )

    md_path = TRIAGE_ROOT / "h1_event_registration.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    _emit(log_lines, f"MD   -> {md_path}")

    log_path = TRIAGE_ROOT / "h1_event_registration.log"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    _emit(log_lines, f"LOG  -> {log_path}")

    _emit(log_lines, "\nDone.")


if __name__ == "__main__":
    main()
