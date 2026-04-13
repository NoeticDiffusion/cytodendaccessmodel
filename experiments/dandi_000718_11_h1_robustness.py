"""Experiment 11 — H1 Robustness: threshold_sigma sweep + raw vs deconvolved.

Tests whether the positive H1 result from exp10 holds across:
  1. threshold_sigma in {1.5, 2.0, 2.5, 3.0}  (event detection sensitivity)
  2. signal_type in {deconvolved, denoised}     (Deconvolved vs raw Denoised traces)

Primary test case: Ca-EEG3-4 NE -> OfflineDay1 (621 registered units, 5539 offline frames)

Usage:
    python experiments/dandi_000718_11_h1_robustness.py

Outputs:
    data/dandi/triage/000718/h1_robustness.json
    data/dandi/triage/000718/h1_robustness.md
    data/dandi/triage/000718/h1_robustness.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "000718"
TRIAGE_ROOT = ROOT / "data" / "dandi" / "triage" / "000718"

sys.path.insert(0, str(ROOT / "src"))

_SUBJECT = "Ca-EEG3-4"
_NE_REL  = "sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-NeutralExposure_image+ophys.nwb"
_OD1_REL = "sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-OfflineDay1Session1_ophys.nwb"

_SIGMA_VALUES   = [1.5, 2.0, 2.5, 3.0]
_SIGNAL_TYPES   = ["deconvolved", "denoised"]   # denoised = raw Ca trace
_NMF_K          = 8
_NULL_N         = 300
_MIN_REG        = 20


def _emit(log_lines: list[str], msg: str = "") -> None:
    print(msg)
    log_lines.append(msg)


def _load_signal(path: Path, session_id: str, window, signal_type: str):
    """Load activity matrix with the requested signal type."""
    import pynwb, numpy as np

    with pynwb.NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
        nwb = io.read()
        fluor = nwb.processing["ophys"]["Fluorescence"]
        rrs_keys = list(fluor.roi_response_series.keys())
        if signal_type == "denoised":
            key = next((k for k in ["Denoised", "denoised"] if k in rrs_keys), None)
        else:
            key = next((k for k in ["Deconvolved", "deconvolved"] if k in rrs_keys), None)
        if key is None:
            key = rrs_keys[0]

        rrs = fluor.roi_response_series[key]
        data = np.array(rrs.data[:], dtype=float)
        try:
            ts = np.array(rrs.timestamps[:], dtype=float)
        except Exception:
            rate = float(getattr(rrs, "rate", 1.0) or 1.0)
            start = float(getattr(rrs, "starting_time", 0.0) or 0.0)
            ts = start + np.arange(data.shape[0]) / rate

        if window is not None:
            mask = (ts >= window.start_sec) & (ts <= window.stop_sec)
            data = data[mask]

        # z-score per unit
        mu = data.mean(axis=0, keepdims=True)
        sd = data.std(axis=0, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        return (data - mu) / sd, key


def _run_condition(
    ne_data: "np.ndarray",
    off_data: "np.ndarray",
    src_idx: list[int],
    tgt_idx: list[int],
    threshold_sigma: float,
    signal_label: str,
    log_lines: list[str],
) -> dict:
    import numpy as np
    from dandi_analysis.dataset_000718.ensembles import (
        extract_ensembles, extract_ensembles_ica, extract_ensembles_graph,
    )
    from dandi_analysis.dataset_000718.events import run_event_h1

    ne_sub  = ne_data[:, src_idx]
    off_sub = off_data[:, tgt_idx]
    unit_ids = tuple(str(i) for i in range(ne_sub.shape[1]))

    results: dict[str, dict] = {}
    for method_name, extractor in [
        ("nmf",   lambda: extract_ensembles(ne_sub, unit_ids, "NE", n_components=_NMF_K, random_state=42)),
        ("ica",   lambda: extract_ensembles_ica(ne_sub, unit_ids, "NE", n_components=_NMF_K, random_state=42)),
        ("graph", lambda: extract_ensembles_graph(ne_sub, unit_ids, "NE", n_components=_NMF_K)),
    ]:
        ens_res = extractor()
        if not ens_res.ensembles:
            continue
        weights = [e.unit_weights for e in ens_res.ensembles]
        h1 = run_event_h1(
            off_sub, weights, "NE", "OD",
            n_registered_units=len(src_idx),
            threshold_sigma=threshold_sigma,
            null_n=_NULL_N,
            rng_seed=42,
        )
        s = h1.session_summary
        _emit(log_lines, (
            f"    {method_name:6s}: events={s['n_events']:3d}"
            f"  n_sig={s['n_significant_z196']:4d}/{s['total_scored_pairs']:4d}"
            f"  frac={s['fraction_significant']:.3f}"
            f"  max_z={s['max_norm_score']}"
        ))
        results[method_name] = {
            "n_events": s["n_events"],
            "n_significant": s["n_significant_z196"],
            "total_pairs": s["total_scored_pairs"],
            "fraction_significant": s["fraction_significant"],
            "max_norm_score": s["max_norm_score"],
        }
    return results


def main() -> None:
    import numpy as np
    from dandi_analysis.dataset_000718.registration import register_sessions
    from dandi_analysis.dataset_000718.epochs import extract_offline_windows

    TRIAGE_ROOT.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    ne_path  = DATA_ROOT / _NE_REL
    od_path  = DATA_ROOT / _OD1_REL

    _emit(log_lines, "=" * 60)
    _emit(log_lines, "Exp 11 - H1 Robustness: sigma sweep + signal type")
    _emit(log_lines, f"Subject: {_SUBJECT}  NE->OfflineDay1")
    _emit(log_lines, f"sigma values: {_SIGMA_VALUES}")
    _emit(log_lines, f"signal types: {_SIGNAL_TYPES}")
    _emit(log_lines, "=" * 60)

    # ---- Registration (once, reused for all conditions) ----
    _emit(log_lines, "\nRegistering sessions...")
    reg = register_sessions(
        ne_path, od_path, f"{_SUBJECT}_NE", f"{_SUBJECT}_OD",
        max_centroid_dist_px=12.0, min_dice=0.05, confidence_threshold=0.25,
    )
    summ = reg.summary_dict()
    _emit(log_lines, (
        f"  accepted={summ['n_accepted']}  conf={summ['mean_confidence']}"
        f"  dice={summ['mean_dice']}  dist={summ['mean_centroid_dist_px']}"
    ))
    if summ["n_accepted"] < _MIN_REG:
        _emit(log_lines, "  Too few registered units — abort.")
        return
    src_idx, tgt_idx = reg.matched_indices()

    # ---- Offline window ----
    windows = extract_offline_windows(od_path, f"{_SUBJECT}_OD", min_duration_sec=30.0)
    primary = max(windows, key=lambda w: w.duration_sec) if windows else None
    _emit(log_lines, f"  Offline window: {primary.label if primary else 'full session'}")

    all_results: list[dict] = []

    # ---- Signal type sweep ----
    for sig_type in _SIGNAL_TYPES:
        _emit(log_lines, f"\n{'='*40}\nSignal type: {sig_type}\n{'='*40}")

        ne_data, ne_key = _load_signal(ne_path, f"{_SUBJECT}_NE", None, sig_type)
        off_data, off_key = _load_signal(od_path, f"{_SUBJECT}_OD", primary, sig_type)
        _emit(log_lines, f"  NE series: {ne_key}  shape: {ne_data.shape}")
        _emit(log_lines, f"  OD series: {off_key}  shape: {off_data.shape}")

        # ---- Sigma sweep ----
        for sigma in _SIGMA_VALUES:
            _emit(log_lines, f"\n  sigma={sigma}")
            cond = _run_condition(
                ne_data, off_data, src_idx, tgt_idx,
                sigma, sig_type, log_lines,
            )
            all_results.append({
                "signal_type": sig_type,
                "threshold_sigma": sigma,
                "methods": cond,
                "n_registered": summ["n_accepted"],
            })

    # ---- Aggregate: is result stable across conditions? ----
    _emit(log_lines, "\n" + "=" * 60)
    _emit(log_lines, "ROBUSTNESS SUMMARY")
    _emit(log_lines, "=" * 60)
    _emit(log_lines, f"{'Signal':12s} {'Sigma':6s} {'NMF n_sig':12s} {'ICA n_sig':12s} {'Graph n_sig':12s} {'Verdict':8s}")
    _emit(log_lines, "-" * 70)
    for r in all_results:
        m = r["methods"]
        n_pos = sum(
            1 for mname in ["nmf", "ica", "graph"]
            if m.get(mname, {}).get("fraction_significant", 0) > 0.05
        )
        verdict = "POSITIVE" if n_pos >= 2 else ("WEAK" if n_pos == 1 else "NEG")
        def _fmt(mname):
            d = m.get(mname, {})
            n, tot = d.get("n_significant", "?"), d.get("total_pairs", "?")
            return f"{n}/{tot}"
        _emit(log_lines, (
            f"  {r['signal_type']:12s} {r['threshold_sigma']:6.1f}"
            f"  {_fmt('nmf'):12s} {_fmt('ica'):12s} {_fmt('graph'):12s} {verdict}"
        ))

    # ---- Write ----
    json_path = TRIAGE_ROOT / "h1_robustness.json"
    json_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    _emit(log_lines, f"\nJSON -> {json_path}")

    md_lines = [
        "# H1 Robustness: Sigma Sweep + Signal Type\n",
        f"Subject: {_SUBJECT}  NE->OfflineDay1\n",
        "| Signal | Sigma | NMF n_sig | ICA n_sig | Graph n_sig | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in all_results:
        m = r["methods"]
        n_pos = sum(1 for mn in ["nmf","ica","graph"] if m.get(mn,{}).get("fraction_significant",0) > 0.05)
        v = "**POSITIVE**" if n_pos >= 2 else ("WEAK" if n_pos == 1 else "NEG")
        def _fmt(mn): d=m.get(mn,{}); return f"{d.get('n_significant','?')}/{d.get('total_pairs','?')}"
        md_lines.append(f"| {r['signal_type']} | {r['threshold_sigma']} | {_fmt('nmf')} | {_fmt('ica')} | {_fmt('graph')} | {v} |")

    (TRIAGE_ROOT / "h1_robustness.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (TRIAGE_ROOT / "h1_robustness.log").write_text("\n".join(log_lines), encoding="utf-8")
    _emit(log_lines, "Done.")


if __name__ == "__main__":
    main()
