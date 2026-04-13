"""Experiment 12 — H1 Specificity Controls.

Three control conditions against the real Ca-EEG3-4 NE->OfflineDay1 result:

  C1. Registration shuffle — break unit identity while keeping the correct
      number of matched units, event structure, and assembly profiles.
      Expect: n_sig near chance (5%).

  C2. Mismatched session pair — use NE from Ca-EEG3-4 projected onto offline
      from Ca-EEG2-1 (different animal). Registration will fail or be weak.
      Expect: few matched units or chance-level signal.

  C3. Event shuffle — keep real registration and real ensembles, but randomly
      relocate detected event windows in the offline period.
      Expect: shuffled events score at chance.

If all three controls fall below the real result and show n_sig near chance,
the positive H1 result is specific rather than an artefact of general synchrony.

Usage:
    python experiments/dandi_000718_12_h1_specificity.py

Outputs:
    data/dandi/triage/000718/h1_specificity.json
    data/dandi/triage/000718/h1_specificity.md
    data/dandi/triage/000718/h1_specificity.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "000718"
TRIAGE_ROOT = ROOT / "data" / "dandi" / "triage" / "000718"

sys.path.insert(0, str(ROOT / "src"))

_NE_34_REL  = "sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-NeutralExposure_image+ophys.nwb"
_OD1_34_REL = "sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-OfflineDay1Session1_ophys.nwb"
_NE_21_REL  = "sub-Ca-EEG2-1/sub-Ca-EEG2-1_ses-NeutralExposure_image+ophys.nwb"
_OD2_21_REL = "sub-Ca-EEG2-1/sub-Ca-EEG2-1_ses-OfflineDay2Session1_ophys.nwb"

_NMF_K     = 8
_NULL_N    = 300
_SIGMA     = 2.0
_MIN_REG   = 20
_N_SHUFFLES = 20   # repetitions for shuffle controls


def _emit(log_lines: list[str], msg: str = "") -> None:
    print(msg)
    log_lines.append(msg)


def _load_z(path: Path, window=None, prefer: str = "Deconvolved"):
    """Load z-scored activity matrix (returns numpy array)."""
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
            rate = float(getattr(rrs, "rate", 1.0) or 1.0)
            ts = np.arange(data.shape[0]) / rate
        if window is not None:
            mask = (ts >= window.start_sec) & (ts <= window.stop_sec)
            data = data[mask]
        mu, sd = data.mean(0), data.std(0)
        sd = np.where(sd == 0, 1.0, sd)
        return (data - mu) / sd


def _run_h1(ne_sub, off_sub, threshold_sigma=_SIGMA) -> dict:
    """Run NMF + event H1 and return summary dict."""
    from dandi_analysis.dataset_000718.ensembles import (
        extract_ensembles, extract_ensembles_ica, extract_ensembles_graph,
    )
    from dandi_analysis.dataset_000718.events import run_event_h1
    import numpy as np

    uid = tuple(str(i) for i in range(ne_sub.shape[1]))
    out: dict[str, dict] = {}
    for mn, extractor in [
        ("nmf",   lambda: extract_ensembles(ne_sub, uid, "NE", n_components=_NMF_K, random_state=42)),
        ("ica",   lambda: extract_ensembles_ica(ne_sub, uid, "NE", n_components=_NMF_K, random_state=42)),
        ("graph", lambda: extract_ensembles_graph(ne_sub, uid, "NE", n_components=_NMF_K)),
    ]:
        ens_res = extractor()
        if not ens_res.ensembles:
            continue
        h1 = run_event_h1(
            off_sub, [e.unit_weights for e in ens_res.ensembles],
            "NE", "OD", n_registered_units=ne_sub.shape[1],
            threshold_sigma=threshold_sigma, null_n=_NULL_N, rng_seed=42,
        )
        s = h1.session_summary
        out[mn] = {
            "n_events": s["n_events"],
            "n_significant": s["n_significant_z196"],
            "total_pairs": s["total_scored_pairs"],
            "fraction_significant": s["fraction_significant"],
            "max_norm_score": s["max_norm_score"],
        }
    return out


def _mean_frac(method_dict: dict) -> float:
    import numpy as np
    vals = [v.get("fraction_significant", 0.0) for v in method_dict.values()]
    return round(float(np.mean(vals)), 3) if vals else 0.0


def main() -> None:
    import numpy as np
    from dandi_analysis.dataset_000718.registration import register_sessions
    from dandi_analysis.dataset_000718.epochs import extract_offline_windows
    from dandi_analysis.dataset_000718.events import detect_synchrony_events

    TRIAGE_ROOT.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    ne_34  = DATA_ROOT / _NE_34_REL
    od1_34 = DATA_ROOT / _OD1_34_REL
    ne_21  = DATA_ROOT / _NE_21_REL
    od2_21 = DATA_ROOT / _OD2_21_REL

    _emit(log_lines, "=" * 60)
    _emit(log_lines, "Exp 12 — H1 Specificity Controls")
    _emit(log_lines, "=" * 60)

    # ---- Real registration ----
    _emit(log_lines, "\nReal registration (Ca-EEG3-4 NE -> OD1)...")
    reg = register_sessions(
        ne_34, od1_34, "NE_34", "OD_34",
        max_centroid_dist_px=12.0, min_dice=0.05, confidence_threshold=0.25,
    )
    reg_summ = reg.summary_dict()
    _emit(log_lines, f"  accepted={reg_summ['n_accepted']}  conf={reg_summ['mean_confidence']:.3f}")
    src_idx, tgt_idx = reg.matched_indices()
    n_reg = len(src_idx)

    # Offline window
    windows = extract_offline_windows(od1_34, "OD_34", min_duration_sec=30.0)
    primary = max(windows, key=lambda w: w.duration_sec) if windows else None

    # Load data
    ne_data  = _load_z(ne_34,  None,    prefer="Deconvolved")
    off_data = _load_z(od1_34, primary, prefer="Deconvolved")
    ne_sub   = ne_data[:, src_idx]
    off_sub  = off_data[:, tgt_idx]
    _emit(log_lines, f"  NE subset: {ne_sub.shape}  OD subset: {off_sub.shape}")

    # ---- REAL result (reference) ----
    _emit(log_lines, "\n--- REAL result (reference) ---")
    real_res = _run_h1(ne_sub, off_sub)
    for mn, v in real_res.items():
        _emit(log_lines, f"  {mn}: n_sig={v['n_significant']}/{v['total_pairs']}  frac={v['fraction_significant']}  max_z={v['max_norm_score']}")
    real_frac = _mean_frac(real_res)
    _emit(log_lines, f"  mean_frac_significant={real_frac}")

    results: dict[str, dict] = {"real": {"methods": real_res, "mean_frac": real_frac, "registration": reg_summ}}

    # ---- C1: Registration shuffle ----
    _emit(log_lines, "\n--- C1: Registration shuffle ---")
    rng = np.random.default_rng(42)
    shuffle_fracs: list[float] = []
    for rep in range(_N_SHUFFLES):
        # Shuffle the mapping of offline unit indices while keeping count
        shuffled_tgt = list(rng.permutation(tgt_idx))
        off_shuf = off_data[:, shuffled_tgt]
        res_shuf = _run_h1(ne_sub, off_shuf)
        f = _mean_frac(res_shuf)
        shuffle_fracs.append(f)
        if rep < 3 or rep == _N_SHUFFLES - 1:
            _emit(log_lines, f"  rep {rep:2d}: mean_frac={f:.3f}")
    c1_mean = round(float(np.mean(shuffle_fracs)), 3)
    c1_std  = round(float(np.std(shuffle_fracs)), 3)
    _emit(log_lines, f"  C1 summary: mean_frac={c1_mean}  std={c1_std}  n_reps={_N_SHUFFLES}")
    results["c1_reg_shuffle"] = {"mean_frac": c1_mean, "std_frac": c1_std, "n_reps": _N_SHUFFLES}

    # ---- C2: Mismatched pair ----
    _emit(log_lines, "\n--- C2: Mismatched pair (Ca-EEG3-4 NE -> Ca-EEG2-1 OD) ---")
    can_run_c2 = ne_21.exists() and od2_21.exists()
    if can_run_c2:
        reg_mm = register_sessions(
            ne_34, od2_21, "NE_34", "OD_21",
            max_centroid_dist_px=12.0, min_dice=0.05, confidence_threshold=0.25,
        )
        mm_summ = reg_mm.summary_dict()
        _emit(log_lines, f"  mismatch registration: accepted={mm_summ['n_accepted']}  conf={mm_summ['mean_confidence']:.3f}")
        if mm_summ["n_accepted"] >= _MIN_REG:
            src_mm, tgt_mm = reg_mm.matched_indices()
            ne_sub_mm  = ne_data[:, src_mm]
            off_data_mm = _load_z(od2_21, None, prefer="Deconvolved")
            off_sub_mm = off_data_mm[:, tgt_mm]
            res_mm = _run_h1(ne_sub_mm, off_sub_mm)
            mm_frac = _mean_frac(res_mm)
            for mn, v in res_mm.items():
                _emit(log_lines, f"  {mn}: n_sig={v['n_significant']}/{v['total_pairs']}  frac={v['fraction_significant']}")
            _emit(log_lines, f"  C2 mean_frac={mm_frac}")
            results["c2_mismatch"] = {"methods": res_mm, "mean_frac": mm_frac, "registration": mm_summ}
        else:
            _emit(log_lines, f"  C2: too few cross-animal matched units ({mm_summ['n_accepted']}) — expected for different animals")
            results["c2_mismatch"] = {"mean_frac": None, "n_accepted": mm_summ["n_accepted"], "note": "too few matched units across animals"}
    else:
        _emit(log_lines, "  C2: Ca-EEG2-1 files not found — skipping")
        results["c2_mismatch"] = {"mean_frac": None, "note": "files not found"}

    # ---- C3: Event shuffle ----
    _emit(log_lines, "\n--- C3: Event-time shuffle ---")
    # Detect real events to get their durations
    detection = detect_synchrony_events(off_sub, threshold_sigma=_SIGMA)
    _emit(log_lines, f"  Real events detected: {detection.n_events}")
    if detection.n_events == 0:
        _emit(log_lines, "  C3: No events to shuffle — skipping")
        results["c3_event_shuffle"] = {"mean_frac": None, "note": "no events detected"}
    else:
        from dandi_analysis.dataset_000718.ensembles import extract_ensembles
        from dandi_analysis.dataset_000718.events import (
            SynchronyEvent, EventDetectionResult, run_event_h1,
        )

        # Build real NMF ensembles once
        uid = tuple(str(i) for i in range(ne_sub.shape[1]))
        enc_ens = extract_ensembles(ne_sub, uid, "NE", n_components=_NMF_K, random_state=42)
        enc_weights = [e.unit_weights for e in enc_ens.ensembles]

        T = off_sub.shape[0]
        event_durs = [ev.duration_frames for ev in detection.events]

        shuffle_fracs_c3: list[float] = []
        for rep in range(_N_SHUFFLES):
            rep_rng = np.random.default_rng(rep * 17 + 3)
            # Build fake detection with same event durations at random start times
            shuffled_events = []
            for k, dur in enumerate(event_durs):
                max_start = max(1, T - dur)
                start = int(rep_rng.integers(0, max_start))
                shuffled_events.append(SynchronyEvent(
                    event_idx=k,
                    start_frame=start,
                    stop_frame=start + dur,
                    duration_frames=dur,
                    peak_population_activity=0.0,
                    mean_population_activity=0.0,
                    n_active_units=0,
                ))
            fake_detection = EventDetectionResult(
                session_id="shuffled",
                n_frames=T, n_units=off_sub.shape[1],
                threshold_sigma=_SIGMA, threshold_value=0.0,
                n_events=len(shuffled_events), events=shuffled_events,
            )
            from dandi_analysis.dataset_000718.events import score_event_recruitment
            all_scores = []
            for k, w in enumerate(enc_weights):
                all_scores.extend(score_event_recruitment(off_sub, fake_detection, w, k, null_n=_NULL_N, rng_seed=42))
            valid = [s for s in all_scores if s.norm_score == s.norm_score]
            n_sig = sum(1 for s in valid if s.norm_score > 1.96)
            frac = n_sig / max(1, len(valid))
            shuffle_fracs_c3.append(frac)
            if rep < 3 or rep == _N_SHUFFLES - 1:
                _emit(log_lines, f"  rep {rep:2d}: frac_sig={frac:.3f}")

        c3_mean = round(float(np.mean(shuffle_fracs_c3)), 3)
        c3_std  = round(float(np.std(shuffle_fracs_c3)), 3)
        _emit(log_lines, f"  C3 summary: mean_frac={c3_mean}  std={c3_std}")
        results["c3_event_shuffle"] = {"mean_frac": c3_mean, "std_frac": c3_std, "n_reps": _N_SHUFFLES}

    # ---- Summary ----
    _emit(log_lines, "\n" + "=" * 60)
    _emit(log_lines, "SPECIFICITY SUMMARY")
    _emit(log_lines, "=" * 60)
    _emit(log_lines, f"  Real result          mean_frac={real_frac:.3f}")
    _emit(log_lines, f"  C1 reg shuffle       mean_frac={results['c1_reg_shuffle']['mean_frac']:.3f}  (expected ~0.05)")
    if results["c2_mismatch"].get("mean_frac") is not None:
        _emit(log_lines, f"  C2 mismatch pair     mean_frac={results['c2_mismatch']['mean_frac']:.3f}  (expected ~real or lower)")
    else:
        _emit(log_lines, f"  C2 mismatch pair     {results['c2_mismatch'].get('note','N/A')}")
    if results["c3_event_shuffle"].get("mean_frac") is not None:
        _emit(log_lines, f"  C3 event shuffle     mean_frac={results['c3_event_shuffle']['mean_frac']:.3f}  (expected ~0.05)")
    else:
        _emit(log_lines, f"  C3 event shuffle     {results['c3_event_shuffle'].get('note','N/A')}")

    c1_ok = results["c1_reg_shuffle"]["mean_frac"] < real_frac * 0.5
    c3_ok = results["c3_event_shuffle"].get("mean_frac") is not None and results["c3_event_shuffle"]["mean_frac"] < real_frac * 0.5
    _emit(log_lines, f"\n  Specificity verdict: C1={'PASS' if c1_ok else 'FAIL'}  C3={'PASS' if c3_ok else 'FAIL'}")

    # ---- Write ----
    json_path = TRIAGE_ROOT / "h1_specificity.json"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    _emit(log_lines, f"\nJSON -> {json_path}")

    md_lines = [
        "# H1 Specificity Controls\n",
        "| Condition | Mean frac sig | vs Real | Verdict |",
        "|---|---|---|---|",
        f"| Real (Ca-EEG3-4 NE->OD1) | {real_frac:.3f} | — | Reference |",
        f"| C1 Registration shuffle | {results['c1_reg_shuffle']['mean_frac']:.3f} ± {results['c1_reg_shuffle']['std_frac']:.3f} | {'below' if c1_ok else 'NOT below'} | {'PASS' if c1_ok else 'FAIL'} |",
    ]
    if results["c2_mismatch"].get("mean_frac") is not None:
        md_lines.append(f"| C2 Mismatched pair | {results['c2_mismatch']['mean_frac']:.3f} | — | Info |")
    else:
        md_lines.append(f"| C2 Mismatched pair | {results['c2_mismatch'].get('note','N/A')} | — | Info |")
    if results["c3_event_shuffle"].get("mean_frac") is not None:
        md_lines.append(f"| C3 Event shuffle | {results['c3_event_shuffle']['mean_frac']:.3f} ± {results['c3_event_shuffle']['std_frac']:.3f} | {'below' if c3_ok else 'NOT below'} | {'PASS' if c3_ok else 'FAIL'} |")

    (TRIAGE_ROOT / "h1_specificity.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (TRIAGE_ROOT / "h1_specificity.log").write_text("\n".join(log_lines), encoding="utf-8")
    _emit(log_lines, "Done.")


if __name__ == "__main__":
    main()
