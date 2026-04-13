"""Experiment 14 — H1 Intra-vs-Inter-Event PRI Enrichment.

Directly tests whether high-synchrony offline events are privileged moments
for reactivation of NeutralExposure-defined core units, relative to
duration-matched inter-event windows from the same offline session.

This addresses the exp13 C3 failure: PRI was unit-specific (C1 PASS) but
not event-locked (C3 FAIL). The enrichment ratio isolates whether events
are actually better reactivation windows than general offline baseline —
even for the same session and same units.

Analysis logic:
  For each (event, ensemble) pair:
    - event_frac: fraction of core units active DURING the event
    - inter_frac: fraction of core units active in duration-matched
                  inter-event windows (average of n_inter_samples draws)
    - enrichment = event_frac − inter_frac

  Session-level summary:
    - mean_enrichment: expected to be > 0 if events are privileged
    - enrichment_z: t-like statistic (mean / SE) over all pairs
    - frac_positive: fraction of pairs with enrichment > 0

Activity threshold sweep: 0.0, 0.5, 1.0
  Tests whether the result is stable as we tighten the "active" definition.

Controls:
  C1: registration shuffle — enrichment should collapse to ≈ 0
  C3: event-time shuffle — enrichment is trivially 0 by definition
      (shuffled events come from the same inter-event pool)

Usage:
    python experiments/dandi_000718_14_h1_pri_enrichment.py

Outputs:
    data/dandi/triage/000718/h1_pri_enrichment.json
    data/dandi/triage/000718/h1_pri_enrichment.md
    data/dandi/triage/000718/h1_pri_enrichment.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "000718"
TRIAGE_ROOT = ROOT / "data" / "dandi" / "triage" / "000718"
sys.path.insert(0, str(ROOT / "src"))

_SUBJECTS = [
    ("Ca-EEG3-4",
     "sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-NeutralExposure_image+ophys.nwb",
     "sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-OfflineDay1Session1_ophys.nwb",
     "OffD1"),
    ("Ca-EEG3-4",
     "sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-NeutralExposure_image+ophys.nwb",
     "sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-OfflineDay2Session1_ophys.nwb",
     "OffD2"),
    ("Ca-EEG2-1",
     "sub-Ca-EEG2-1/sub-Ca-EEG2-1_ses-NeutralExposure_image+ophys.nwb",
     "sub-Ca-EEG2-1/sub-Ca-EEG2-1_ses-OfflineDay2Session1_ophys.nwb",
     "OffD2"),
]

_NMF_K          = 8
_SIGMA          = 2.0
_TOP_FRAC       = 0.15
_N_INTER        = 10     # inter-event samples per event
_THRESH_SWEEP   = [0.0, 0.5, 1.0]
_N_C1_REPS      = 15
_MIN_REG        = 20


def _emit(log_lines: list[str], msg: str = "") -> None:
    print(msg)
    log_lines.append(msg)


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
# Enrichment for one threshold / one mapping
# ---------------------------------------------------------------------------

def _run_enrichment(
    ne_sub: "np.ndarray",
    off_sub: "np.ndarray",
    events: list,
    session_ne: str,
    session_off: str,
    n_reg: int,
    threshold: float,
    log_lines: list[str],
    label: str = "real",
) -> dict:
    from dandi_analysis.dataset_000718.ensembles import extract_ensembles
    from dandi_analysis.dataset_000718.pri import compute_pri_enrichment_session

    uid = tuple(str(i) for i in range(ne_sub.shape[1]))
    ens_res = extract_ensembles(ne_sub, uid, session_ne, n_components=_NMF_K, random_state=42)
    weights_list = [e.unit_weights for e in ens_res.ensembles]
    if not weights_list:
        return {}

    enr = compute_pri_enrichment_session(
        off_sub, weights_list, events,
        session_ne=session_ne, session_offline=session_off,
        n_registered=n_reg,
        top_frac=_TOP_FRAC,
        activity_threshold=threshold,
        n_inter_samples=_N_INTER,
        rng_seed=42,
    )
    d = enr.to_dict()
    _emit(log_lines, (
        f"    [{label}] thr={threshold:.1f}: "
        f"mean_enr={d['mean_enrichment']:+.4f}  "
        f"enr_z={d['enrichment_z']:+.3f}  "
        f"frac_pos={d['frac_positive']:.3f}  "
        f"n_pairs={d['n_scored_pairs']}"
    ))
    return d


# ---------------------------------------------------------------------------
# One session pair
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
    from dandi_analysis.dataset_000718.events import detect_synchrony_events

    session_ne  = f"{subject}_NE"
    session_off = f"{subject}_{pair_label}"

    _emit(log_lines, f"\n{'='*55}")
    _emit(log_lines, f"{subject}  {pair_label}")
    _emit(log_lines, f"{'='*55}")

    reg = register_sessions(
        ne_path, od_path, session_ne, session_off,
        max_centroid_dist_px=12.0, min_dice=0.05, confidence_threshold=0.25,
    )
    reg_summ = reg.summary_dict()
    _emit(log_lines, (
        f"  Registration: accepted={reg_summ['n_accepted']}"
        f"  conf={reg_summ['mean_confidence']:.3f}"
    ))
    if reg_summ["n_accepted"] < _MIN_REG:
        _emit(log_lines, "  SKIP: too few registered units")
        return None

    src_idx, tgt_idx = reg.matched_indices()
    n_reg = len(src_idx)

    windows = extract_offline_windows(od_path, session_off, min_duration_sec=30.0)
    primary = max(windows, key=lambda w: w.duration_sec) if windows else None
    ne_data  = _load_z(ne_path, None)
    off_data = _load_z(od_path, primary)
    ne_sub   = ne_data[:, src_idx]
    off_sub  = off_data[:, tgt_idx]
    _emit(log_lines, f"  NE: {ne_sub.shape}  OD: {off_sub.shape}")

    detection = detect_synchrony_events(off_sub, threshold_sigma=_SIGMA)
    events    = detection.events
    _emit(log_lines, f"  Events: {detection.n_events}")
    if detection.n_events == 0:
        _emit(log_lines, "  SKIP: no events")
        return None

    # ---- Threshold sweep: real mapping ----
    _emit(log_lines, "\n  --- REAL mapping ---")
    real_by_thresh: dict[float, dict] = {}
    for thr in _THRESH_SWEEP:
        real_by_thresh[thr] = _run_enrichment(
            ne_sub, off_sub, events, session_ne, session_off, n_reg, thr, log_lines, "real"
        )

    # ---- C1: Registration shuffle (at default threshold 0.0) ----
    _emit(log_lines, "\n  --- C1: Registration shuffle (thr=0.0, 15 reps) ---")
    rng = np.random.default_rng(123)
    c1_enr: list[float] = []
    for rep in range(_N_C1_REPS):
        shuf_tgt = list(rng.permutation(tgt_idx))
        off_shuf = off_data[:, shuf_tgt]
        d_shuf = _run_enrichment(
            ne_sub, off_shuf, events, session_ne, session_off, n_reg,
            0.0, log_lines if rep == 0 else [], f"shuf{rep}"
        )
        if d_shuf:
            c1_enr.append(d_shuf.get("mean_enrichment", 0.0))
    c1_mean = round(float(np.mean(c1_enr)), 4) if c1_enr else float("nan")
    c1_std  = round(float(np.std(c1_enr)),  4) if c1_enr else float("nan")
    _emit(log_lines, f"  C1 mean_enrichment={c1_mean}  std={c1_std}")

    real_enr_0 = real_by_thresh.get(0.0, {}).get("mean_enrichment", float("nan"))
    c1_ok = (real_enr_0 > c1_mean + 2 * c1_std) if c1_std == c1_std else False
    real_z = real_by_thresh.get(0.0, {}).get("enrichment_z", float("nan"))
    _emit(log_lines, (
        f"\n  SUMMARY: real_enr={real_enr_0:+.4f}  real_z={real_z:+.3f}"
        f"  C1={'PASS' if c1_ok else 'FAIL'}  (c1={c1_mean:+.4f}±{c1_std:.4f})"
    ))

    verdict = "ENRICHED" if (c1_ok and real_z > 1.96) else (
              "MARGINAL" if (c1_ok or real_z > 1.96) else "NOT_ENRICHED")
    _emit(log_lines, f"  Verdict: {verdict}")

    return {
        "subject": subject,
        "pair_label": pair_label,
        "n_registered": n_reg,
        "n_events": detection.n_events,
        "registration": reg_summ,
        "real_by_threshold": {str(k): v for k, v in real_by_thresh.items()},
        "c1_reg_shuffle": {"mean_enrichment": c1_mean, "std_enrichment": c1_std, "n_reps": _N_C1_REPS},
        "c1_ok": c1_ok,
        "real_enrichment_z": real_z,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import numpy as np
    TRIAGE_ROOT.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    _emit(log_lines, "=" * 60)
    _emit(log_lines, "Exp 14 — H1 Intra-vs-Inter-Event PRI Enrichment")
    _emit(log_lines, f"top_frac={_TOP_FRAC}  sigma={_SIGMA}  n_inter={_N_INTER}")
    _emit(log_lines, f"threshold sweep: {_THRESH_SWEEP}")
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
    _emit(log_lines, "ENRICHMENT SUMMARY (thr=0.0)")
    _emit(log_lines, "=" * 60)
    _emit(log_lines, f"{'Pair':<30} {'Real enr':>9} {'Real z':>8} {'C1 enr':>9} {'C1':>5} {'Verdict'}")
    _emit(log_lines, "-" * 75)
    for r in all_results:
        c1 = r["c1_reg_shuffle"]
        thr0 = r["real_by_threshold"].get("0.0", {})
        _emit(log_lines, (
            f"  {r['subject']} {r['pair_label']:<20}"
            f"  {thr0.get('mean_enrichment', float('nan')):+9.4f}"
            f"  {r['real_enrichment_z']:+8.3f}"
            f"  {c1['mean_enrichment']:+9.4f}"
            f"  {'PASS' if r['c1_ok'] else 'FAIL':>5}"
            f"  {r['verdict']}"
        ))

    _emit(log_lines, "\nThreshold sweep (real mapping):")
    _emit(log_lines, f"{'Pair':<30} " + "  ".join(f"thr={t:.1f}" for t in _THRESH_SWEEP))
    for r in all_results:
        enr_row = "  ".join(
            f"{r['real_by_threshold'].get(str(t), {}).get('mean_enrichment', float('nan')):+.4f}"
            for t in _THRESH_SWEEP
        )
        _emit(log_lines, f"  {r['subject']} {r['pair_label']:<20}  {enr_row}")

    n_enr     = sum(1 for r in all_results if r["verdict"] == "ENRICHED")
    n_marg    = sum(1 for r in all_results if r["verdict"] == "MARGINAL")
    n_not_enr = sum(1 for r in all_results if r["verdict"] == "NOT_ENRICHED")
    _emit(log_lines, f"\nENRICHED={n_enr}  MARGINAL={n_marg}  NOT_ENRICHED={n_not_enr}")

    if n_enr >= 2:
        overall = "H1 EVENT-ENRICHED — events are privileged reactivation windows"
    elif n_enr + n_marg >= 2:
        overall = "H1 MARGINAL ENRICHMENT — weak event preference, bridge still limited"
    else:
        overall = "H1 NOT EVENT-ENRICHED — offline elevation is session-wide, not event-locked"
    _emit(log_lines, f"Overall: {overall}")

    # ---- Write outputs ----
    json_path = TRIAGE_ROOT / "h1_pri_enrichment.json"
    json_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")

    md_lines = [
        "# H1 PRI Enrichment: Intra-vs-Inter-Event\n",
        f"top_frac={_TOP_FRAC}  sigma={_SIGMA}  n_inter={_N_INTER}\n",
        "## Summary (thr=0.0)\n",
        "| Pair | Real enr | Real z | C1 enr | C1 | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in all_results:
        c1 = r["c1_reg_shuffle"]
        thr0 = r["real_by_threshold"].get("0.0", {})
        md_lines.append((
            f"| {r['subject']} {r['pair_label']} "
            f"| {thr0.get('mean_enrichment', '?'):+.4f} "
            f"| {r['real_enrichment_z']:+.3f} "
            f"| {c1['mean_enrichment']:+.4f} ± {c1['std_enrichment']:.4f} "
            f"| {'PASS' if r['c1_ok'] else 'FAIL'} "
            f"| **{r['verdict']}** |"
        ))
    md_lines += ["", f"**Overall: {overall}**\n", "## Threshold sweep\n",
                 "| Pair | thr=0.0 enr | thr=0.5 enr | thr=1.0 enr |",
                 "|---|---|---|---|"]
    for r in all_results:
        row_vals = [r["real_by_threshold"].get(str(t), {}).get("mean_enrichment", float("nan"))
                    for t in _THRESH_SWEEP]
        md_lines.append(
            f"| {r['subject']} {r['pair_label']} | " +
            " | ".join(f"{v:+.4f}" if v == v else "nan" for v in row_vals) + " |"
        )

    (TRIAGE_ROOT / "h1_pri_enrichment.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (TRIAGE_ROOT / "h1_pri_enrichment.log").write_text("\n".join(log_lines), encoding="utf-8")
    _emit(log_lines, f"\nJSON -> {json_path}")
    _emit(log_lines, "Done.")


if __name__ == "__main__":
    main()
