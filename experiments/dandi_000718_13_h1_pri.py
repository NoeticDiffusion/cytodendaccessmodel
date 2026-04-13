"""Experiment 13 — H1 Selective Replay: Preferential Reactivation Index (PRI).

Tests whether offline high-synchrony events selectively reactivate the same
units that defined NeutralExposure encoding ensembles, using a binary
unit-level metric rather than a global projection score.

PRI asks: of the k units that most strongly define an encoding ensemble,
how many are active during each offline event? Is that count above what
would be expected for a random k-subset of the same size?

This is the critical test that projection scores (exp10) could not pass,
because projection scores are blind to population-burst confounds.

Controls (same philosophy as exp12):
  C1. registration shuffle — break unit identity, preserve event structure
  C3. event-time shuffle — break event timing, preserve unit identity

A strong H1 result: real >> C1 and real >> C3.

Usage:
    python experiments/dandi_000718_13_h1_pri.py

Outputs:
    data/dandi/triage/000718/h1_pri.json
    data/dandi/triage/000718/h1_pri.md
    data/dandi/triage/000718/h1_pri.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "000718"
TRIAGE_ROOT = ROOT / "data" / "dandi" / "triage" / "000718"

sys.path.insert(0, str(ROOT / "src"))

# Primary pair: Ca-EEG3-4 (largest window, best registration)
_SUBJECTS = [
    ("Ca-EEG3-4", "sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-NeutralExposure_image+ophys.nwb",
                  "sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-OfflineDay1Session1_ophys.nwb", "OffD1"),
    ("Ca-EEG3-4", "sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-NeutralExposure_image+ophys.nwb",
                  "sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-OfflineDay2Session1_ophys.nwb", "OffD2"),
    ("Ca-EEG2-1", "sub-Ca-EEG2-1/sub-Ca-EEG2-1_ses-NeutralExposure_image+ophys.nwb",
                  "sub-Ca-EEG2-1/sub-Ca-EEG2-1_ses-OfflineDay2Session1_ophys.nwb", "OffD2"),
]

_NMF_K          = 8
_NULL_N         = 500
_SIGMA          = 2.0
_TOP_FRAC       = 0.15   # top 15% of registered units define ensemble core
_ACT_THRESHOLD  = 0.0    # "active" = above session z-score mean
_N_SHUFFLES     = 20
_MIN_REG        = 20


def _emit(log_lines: list[str], msg: str = "") -> None:
    print(msg)
    log_lines.append(msg)


# ---------------------------------------------------------------------------
# Shared data loading helpers
# ---------------------------------------------------------------------------

def _load_z(path: Path, window=None, prefer: str = "Deconvolved") -> "np.ndarray":
    import pynwb, numpy as np
    with pynwb.NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
        nwb = io.read()
        fluor = nwb.processing["ophys"]["Fluorescence"]
        keys = list(fluor.roi_response_series.keys())
        key = next((k for k in [prefer, prefer.lower()] if k in keys), keys[0])
        rrs = fluor.roi_response_series[key]
        data = np.array(rrs.data[:], dtype=float)
        try:
            ts = np.array(rrs.timestamps[:], dtype=float)
        except Exception:
            ts = np.arange(data.shape[0]) / float(getattr(rrs, "rate", 1.0) or 1.0)
        if window is not None:
            mask = (ts >= window.start_sec) & (ts <= window.stop_sec)
            data = data[mask]
    mu, sd = data.mean(0), data.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    return (data - mu) / sd


# ---------------------------------------------------------------------------
# PRI helper: one condition (real or shuffled mapping)
# ---------------------------------------------------------------------------

def _run_pri(
    ne_sub: "np.ndarray",
    off_sub: "np.ndarray",
    events: list,
    session_ne: str,
    session_offline: str,
    n_reg: int,
    log_lines: list[str],
    label: str = "real",
) -> dict:
    from dandi_analysis.dataset_000718.ensembles import (
        extract_ensembles, extract_ensembles_ica, extract_ensembles_graph,
    )
    from dandi_analysis.dataset_000718.pri import run_pri_session
    import numpy as np

    uid = tuple(str(i) for i in range(ne_sub.shape[1]))
    method_summaries: dict[str, dict] = {}

    for mname, extractor in [
        ("nmf",   lambda: extract_ensembles(ne_sub, uid, session_ne, n_components=_NMF_K, random_state=42)),
        ("ica",   lambda: extract_ensembles_ica(ne_sub, uid, session_ne, n_components=_NMF_K, random_state=42)),
        ("graph", lambda: extract_ensembles_graph(ne_sub, uid, session_ne, n_components=_NMF_K)),
    ]:
        ens_res = extractor()
        if not ens_res.ensembles:
            continue
        weights = [e.unit_weights for e in ens_res.ensembles]
        pri = run_pri_session(
            ne_sub, off_sub, weights, events,
            session_ne=session_ne,
            session_offline=session_offline,
            n_registered=n_reg,
            top_frac=_TOP_FRAC,
            activity_threshold=_ACT_THRESHOLD,
            null_n=_NULL_N,
            rng_seed=42,
        )
        d = pri.to_dict()
        _emit(log_lines, (
            f"    [{label}] {mname:6s}: events={d['n_events']:3d}"
            f"  n_sig={d['n_significant']:4d}/{d['total_scored_pairs']:4d}"
            f"  frac={d['fraction_significant']:.3f}"
            f"  mean_z={d['mean_z']}  max_z={d['max_z']}"
        ))
        method_summaries[mname] = d

    return method_summaries


# ---------------------------------------------------------------------------
# One pair: real + C1 shuffle + C3 event-shuffle
# ---------------------------------------------------------------------------

def _run_pair(
    ne_path: Path,
    od_path: Path,
    subject: str,
    pair_label: str,
    log_lines: list[str],
) -> dict | None:
    import numpy as np
    from dandi_analysis.dataset_000718.registration import register_sessions
    from dandi_analysis.dataset_000718.epochs import extract_offline_windows
    from dandi_analysis.dataset_000718.events import (
        detect_synchrony_events, SynchronyEvent, EventDetectionResult,
    )
    from dandi_analysis.dataset_000718.pri import run_pri_session
    from dandi_analysis.dataset_000718.ensembles import extract_ensembles

    session_ne  = f"{subject}_NE"
    session_off = f"{subject}_{pair_label}"

    _emit(log_lines, f"\n{'='*50}")
    _emit(log_lines, f"{subject}  {pair_label}")
    _emit(log_lines, f"{'='*50}")

    # ---- Registration ----
    reg = register_sessions(
        ne_path, od_path, session_ne, session_off,
        max_centroid_dist_px=12.0, min_dice=0.05, confidence_threshold=0.25,
    )
    reg_summ = reg.summary_dict()
    _emit(log_lines, (
        f"  Registration: accepted={reg_summ['n_accepted']}"
        f"  conf={reg_summ['mean_confidence']:.3f}  dice={reg_summ['mean_dice']:.3f}"
    ))
    if reg_summ["n_accepted"] < _MIN_REG:
        _emit(log_lines, "  SKIP: too few registered units")
        return None

    src_idx, tgt_idx = reg.matched_indices()
    n_reg = len(src_idx)

    # ---- Load data ----
    windows = extract_offline_windows(od_path, session_off, min_duration_sec=30.0)
    primary = max(windows, key=lambda w: w.duration_sec) if windows else None
    ne_data  = _load_z(ne_path, None, prefer="Deconvolved")
    off_data = _load_z(od_path, primary, prefer="Deconvolved")
    ne_sub   = ne_data[:, src_idx]
    off_sub  = off_data[:, tgt_idx]
    _emit(log_lines, f"  NE: {ne_sub.shape}  OD: {off_sub.shape}")

    # ---- Detect events ----
    detection = detect_synchrony_events(off_sub, threshold_sigma=_SIGMA)
    _emit(log_lines, f"  Events detected: {detection.n_events}")
    if detection.n_events == 0:
        _emit(log_lines, "  SKIP: no events")
        return None
    events = detection.events

    # ---- REAL result ----
    _emit(log_lines, "\n  --- REAL ---")
    real_res = _run_pri(ne_sub, off_sub, events, session_ne, session_off, n_reg, log_lines, "real")
    real_frac = _mean_frac(real_res)
    _emit(log_lines, f"  REAL mean_frac_sig={real_frac:.3f}")

    # ---- C1: Registration shuffle ----
    _emit(log_lines, "\n  --- C1: Registration shuffle ---")
    rng = np.random.default_rng(42)
    c1_fracs: list[float] = []
    for rep in range(_N_SHUFFLES):
        shuffled_tgt = list(rng.permutation(tgt_idx))
        off_shuf = off_data[:, shuffled_tgt]
        res_shuf = _run_pri(ne_sub, off_shuf, events, session_ne, session_off, n_reg,
                            log_lines if rep == 0 else [], f"shuf{rep}")
        c1_fracs.append(_mean_frac(res_shuf))
    c1_mean = round(float(np.mean(c1_fracs)), 3)
    c1_std  = round(float(np.std(c1_fracs)),  3)
    _emit(log_lines, f"  C1 mean_frac_sig={c1_mean}  std={c1_std}  ({_N_SHUFFLES} reps)")

    # ---- C3: Event-time shuffle ----
    _emit(log_lines, "\n  --- C3: Event-time shuffle ---")
    T = off_sub.shape[0]
    event_durs = [ev.duration_frames for ev in events]
    uid = tuple(str(i) for i in range(ne_sub.shape[1]))
    ne_ens = extract_ensembles(ne_sub, uid, session_ne, n_components=_NMF_K, random_state=42)
    enc_weights = [e.unit_weights for e in ne_ens.ensembles]

    c3_fracs: list[float] = []
    for rep in range(_N_SHUFFLES):
        rep_rng = np.random.default_rng(rep * 31 + 7)
        fake_events = []
        for k, dur in enumerate(event_durs):
            start = int(rep_rng.integers(0, max(1, T - dur)))
            fake_events.append(SynchronyEvent(
                event_idx=k, start_frame=start, stop_frame=start + dur,
                duration_frames=dur, peak_population_activity=0.0,
                mean_population_activity=0.0, n_active_units=0,
            ))
        from dandi_analysis.dataset_000718.pri import run_pri_session
        pri_c3 = run_pri_session(
            ne_sub, off_sub, enc_weights, fake_events,
            session_ne=session_ne, session_offline=session_off,
            n_registered=n_reg, top_frac=_TOP_FRAC,
            activity_threshold=_ACT_THRESHOLD, null_n=_NULL_N, rng_seed=42,
        )
        c3_fracs.append(pri_c3.fraction_significant)

    c3_mean = round(float(np.mean(c3_fracs)), 3)
    c3_std  = round(float(np.std(c3_fracs)),  3)
    _emit(log_lines, f"  C3 mean_frac_sig={c3_mean}  std={c3_std}  ({_N_SHUFFLES} reps)")

    # ---- Verdict ----
    c1_ok = real_frac > c1_mean + 2 * c1_std
    c3_ok = real_frac > c3_mean + 2 * c3_std
    verdict = "SELECTIVE" if (c1_ok and c3_ok) else ("PARTIAL" if (c1_ok or c3_ok) else "NOT_SELECTIVE")
    _emit(log_lines, f"\n  Verdict: {verdict}  (C1={'PASS' if c1_ok else 'FAIL'}  C3={'PASS' if c3_ok else 'FAIL'})")

    return {
        "subject": subject,
        "pair_label": pair_label,
        "n_registered": n_reg,
        "n_events": detection.n_events,
        "registration": reg_summ,
        "real": {"methods": real_res, "mean_frac": real_frac},
        "c1_reg_shuffle": {"mean_frac": c1_mean, "std_frac": c1_std},
        "c3_event_shuffle": {"mean_frac": c3_mean, "std_frac": c3_std},
        "c1_ok": c1_ok,
        "c3_ok": c3_ok,
        "verdict": verdict,
    }


def _mean_frac(method_dict: dict) -> float:
    import numpy as np
    vals = [v.get("fraction_significant", 0.0) for v in method_dict.values()]
    return round(float(np.mean(vals)), 3) if vals else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    TRIAGE_ROOT.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    _emit(log_lines, "=" * 60)
    _emit(log_lines, "Exp 13 — H1 Selective Replay: PRI")
    _emit(log_lines, f"top_frac={_TOP_FRAC}  act_threshold={_ACT_THRESHOLD}"
                     f"  sigma={_SIGMA}  null_n={_NULL_N}")
    _emit(log_lines, "=" * 60)

    all_results: list[dict] = []

    for subject, ne_rel, od_rel, pair_label in _SUBJECTS:
        ne_path = DATA_ROOT / ne_rel
        od_path = DATA_ROOT / od_rel
        if not ne_path.exists() or not od_path.exists():
            _emit(log_lines, f"\n{subject} {pair_label}: files not found — skip")
            continue
        res = _run_pair(ne_path, od_path, subject, pair_label, log_lines)
        if res is not None:
            all_results.append(res)

    # ---- Final summary ----
    _emit(log_lines, "\n" + "=" * 60)
    _emit(log_lines, "PRI SUMMARY")
    _emit(log_lines, "=" * 60)
    _emit(log_lines, f"{'Pair':<30} {'Real':>6} {'C1':>6} {'C3':>6} {'Verdict'}")
    _emit(log_lines, "-" * 60)
    for r in all_results:
        _emit(log_lines, (
            f"  {r['subject']} {r['pair_label']:<20}"
            f"  {r['real']['mean_frac']:>6.3f}"
            f"  {r['c1_reg_shuffle']['mean_frac']:>6.3f}"
            f"  {r['c3_event_shuffle']['mean_frac']:>6.3f}"
            f"  {r['verdict']}"
        ))

    n_selective = sum(1 for r in all_results if r["verdict"] == "SELECTIVE")
    n_partial   = sum(1 for r in all_results if r["verdict"] == "PARTIAL")
    n_not       = sum(1 for r in all_results if r["verdict"] == "NOT_SELECTIVE")
    _emit(log_lines, f"\nSELECTIVE={n_selective}  PARTIAL={n_partial}  NOT_SELECTIVE={n_not}")

    if n_selective >= 2:
        overall = "H1 SELECTIVE — article-stable selective replay signal"
    elif n_selective + n_partial >= 2:
        overall = "H1 PARTIAL — some selectivity, bridge refinement needed"
    else:
        overall = "H1 NOT YET SELECTIVE — negative boundary on current bridge"
    _emit(log_lines, f"Overall: {overall}")

    # ---- Write outputs ----
    json_path = TRIAGE_ROOT / "h1_pri.json"
    json_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    _emit(log_lines, f"\nJSON -> {json_path}")

    md_lines = [
        "# H1 Selective Replay: PRI\n",
        f"top_frac={_TOP_FRAC}  activity_threshold={_ACT_THRESHOLD}  sigma={_SIGMA}\n",
        "| Pair | Real frac | C1 shuffle | C3 event-shuffle | Verdict |",
        "|---|---|---|---|---|",
    ]
    for r in all_results:
        c1 = r["c1_reg_shuffle"]
        c3 = r["c3_event_shuffle"]
        md_lines.append((
            f"| {r['subject']} {r['pair_label']} "
            f"| {r['real']['mean_frac']:.3f} "
            f"| {c1['mean_frac']:.3f} ± {c1['std_frac']:.3f} "
            f"| {c3['mean_frac']:.3f} ± {c3['std_frac']:.3f} "
            f"| **{r['verdict']}** |"
        ))

    md_lines.append(f"\n**Overall: {overall}**\n")

    md_lines.append("\n## Per-Method PRI (real result)\n")
    for r in all_results:
        md_lines.append(f"\n### {r['subject']} {r['pair_label']}\n")
        md_lines.append(f"n_registered={r['n_registered']}  n_events={r['n_events']}\n")
        md_lines.append("| Method | n_sig | total | frac_sig | mean_z | max_z |")
        md_lines.append("|---|---|---|---|---|---|")
        for mname, d in r["real"]["methods"].items():
            md_lines.append((
                f"| {mname} | {d.get('n_significant','?')} | {d.get('total_scored_pairs','?')} "
                f"| {d.get('fraction_significant','?')} | {d.get('mean_z','?')} | {d.get('max_z','?')} |"
            ))

    (TRIAGE_ROOT / "h1_pri.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (TRIAGE_ROOT / "h1_pri.log").write_text("\n".join(log_lines), encoding="utf-8")
    _emit(log_lines, "Done.")


if __name__ == "__main__":
    main()
