"""Experiment 04 â€” Cross-plane coupling replication: sub-656228 (DANDI 000336).

Exact replication of exp03 (sub-644972) analysis on a second subject.
Same methodological guardrails, same condition handling, same null construction.
Writes to subject-specific output files so sub-644972 artifacts are untouched.

Subject file mapping (confirmed by probe):
  sub-656228_ses-1247233186-acq-1247385128  â†’  shallow plane, 4 ROIs
  sub-656228_ses-1247233186-acq-1247385130  â†’  deep plane,    25 ROIs

Main question:
  Does sub-656228 reproduce the qualitative H3 signature from sub-644972?
  - cross-plane coupling above null?
  - cross-plane coupling < within-plane coupling (access-constraint signature)?
  - pattern visible across the same conditions?

Usage:
    python experiments/dandi_000336_04_sub656228_replication.py

Outputs:
    data/dandi/triage/000336/crossplane_coupling_sub656228.json
    data/dandi/triage/000336/crossplane_coupling_sub656228.md
    data/dandi/triage/000336/crossplane_coupling_sub656228.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_ROOT  = ROOT / "data" / "dandi" / "raw" / "000336"
TRIAGE_ROOT = ROOT / "data" / "dandi" / "triage" / "000336"

sys.path.insert(0, str(ROOT / "src"))

_SUBJECT     = "sub-656228"
_SESSION     = "ses-1247233186"
_SHALLOW_REL = f"{_SUBJECT}/{_SUBJECT}_{_SESSION}-acq-1247385128_ophys.nwb"
_DEEP_REL    = f"{_SUBJECT}/{_SUBJECT}_{_SESSION}-acq-1247385130_ophys.nwb"

_N_NULL             = 200
_MIN_BINS_PER_WINDOW = 20
_ALIGN_DECIMALS     = 6


def _emit(log_lines: list[str], message: str = "") -> None:
    print(message)
    log_lines.append(message)


# ---------------------------------------------------------------------------
# Data loading â€” identical to exp03
# ---------------------------------------------------------------------------

def _load_dff_and_intervals(path: Path):
    import pynwb, numpy as np
    with pynwb.NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
        nwb = io.read()
        ophys = nwb.processing["ophys"]
        rrs = ophys["dff"].roi_response_series["traces"]
        data = np.array(rrs.data[:])
        ts   = np.array(rrs.timestamps[:])

        intervals: dict[str, tuple[list, list]] = {}
        for name in nwb.intervals.keys():
            table = nwb.intervals[name]
            try:
                starts = list(table["start_time"].data[:])
                stops  = list(table["stop_time"].data[:])
                intervals[name] = (starts, stops)
            except Exception:
                pass
    return data, ts, intervals


# ---------------------------------------------------------------------------
# Window helpers â€” identical to exp03
# ---------------------------------------------------------------------------

def _epoch_windows(starts, stops, min_duration=1.0):
    windows = []
    for t0, t1 in zip(starts, stops):
        if t1 - t0 >= min_duration:
            windows.append((float(t0), float(t1)))
    return windows


def _merge_into_blocks(starts, stops, max_gap_sec: float = 5.0, min_block_dur: float = 10.0):
    if not starts:
        return []
    pairs = sorted(zip(starts, stops))
    blocks: list[tuple[float, float]] = []
    block_start = float(pairs[0][0])
    block_end   = float(pairs[0][1])
    for t0, t1 in pairs[1:]:
        gap = float(t0) - block_end
        if gap <= max_gap_sec:
            block_end = max(block_end, float(t1))
        else:
            if block_end - block_start >= min_block_dur:
                blocks.append((block_start, block_end))
            block_start = float(t0)
            block_end   = float(t1)
    if block_end - block_start >= min_block_dur:
        blocks.append((block_start, block_end))
    return blocks


def _epoch_mask(ts, start, stop):
    import numpy as np
    return (ts >= start) & (ts <= stop)


def _align_by_timestamp(s_chunk, s_ts, d_chunk, d_ts, *, decimals=_ALIGN_DECIMALS):
    import numpy as np
    if len(s_chunk) == 0 or len(d_chunk) == 0:
        return s_chunk[:0], d_chunk[:0], 0.0
    s_key = np.round(np.asarray(s_ts, dtype=float), decimals)
    d_key = np.round(np.asarray(d_ts, dtype=float), decimals)
    _, s_idx, d_idx = np.intersect1d(s_key, d_key, assume_unique=False, return_indices=True)
    if len(s_idx) == 0:
        return s_chunk[:0], d_chunk[:0], 0.0
    s_aligned = s_chunk[s_idx]
    d_aligned = d_chunk[d_idx]
    max_delta = float(np.max(np.abs(np.asarray(s_ts)[s_idx] - np.asarray(d_ts)[d_idx])))
    return s_aligned, d_aligned, max_delta


def _zscore(chunk):
    import numpy as np
    mu = chunk.mean(axis=0, keepdims=True)
    sd = chunk.std(axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (chunk - mu) / sd


def _cross_block(s_chunk, d_chunk):
    import numpy as np
    denom = max(1, len(s_chunk) - 1)
    return (s_chunk.T @ d_chunk) / denom


def _within_block(chunk):
    import numpy as np
    denom = max(1, len(chunk) - 1)
    corr = (chunk.T @ chunk) / denom
    np.fill_diagonal(corr, np.nan)
    return corr


def _window_cross_coupling(s_chunk, d_chunk, n_null=_N_NULL, seed=42):
    import numpy as np
    s_norm = _zscore(np.asarray(s_chunk, dtype=float))
    d_norm = _zscore(np.asarray(d_chunk, dtype=float))
    cross_block = _cross_block(s_norm, d_norm)

    rng = np.random.default_rng(seed)
    null_means = []
    T = len(d_norm)
    min_shift = max(1, min(10, max(1, T // 10)))
    max_shift = max(min_shift + 1, T - min_shift)
    for _ in range(n_null):
        shift = rng.integers(min_shift, max_shift)
        d_shifted = np.roll(d_norm, shift, axis=0)
        c_null = _cross_block(s_norm, d_shifted)
        null_means.append(c_null.mean())

    null_arr = np.array(null_means)
    z = (cross_block.mean() - null_arr.mean()) / null_arr.std()

    return {
        "n_bins": int(T),
        "mean_r": float(cross_block.mean()),
        "max_r": float(cross_block.max()),
        "min_r": float(cross_block.min()),
        "std_r": float(cross_block.std()),
        "n_pairs": int(cross_block.size),
        "n_above_02": int((abs(cross_block) > 0.2).sum()),
        "n_above_03": int((abs(cross_block) > 0.3).sum()),
        "null_mean": float(null_arr.mean()),
        "null_std": float(null_arr.std()),
        "z_vs_null": float(z),
        "per_shallow_roi_mean": cross_block.mean(axis=1).tolist(),
        "null_means": null_arr.tolist(),
    }


def _within_coupling(chunk):
    corr = _within_block(_zscore(chunk))
    return {
        "mean_r": float(__import__("numpy").nanmean(corr)),
        "max_r": float(__import__("numpy").nanmax(corr)),
    }


def _aggregate_condition(window_stats):
    import numpy as np
    weights  = np.array([w["n_bins"] for w in window_stats], dtype=float)
    observed = np.array([w["cross_plane"]["mean_r"] for w in window_stats], dtype=float)
    null_matrix = np.array([w["cross_plane"]["null_means"] for w in window_stats], dtype=float)
    agg_null = np.average(null_matrix, axis=0, weights=weights)
    weighted_observed = float(np.average(observed, weights=weights))
    null_mean = float(np.mean(agg_null))
    null_std  = float(np.std(agg_null))
    z = (weighted_observed - null_mean) / null_std if null_std > 0 else float("nan")

    shallow_means = np.array([w["within_shallow"]["mean_r"] for w in window_stats], dtype=float)
    deep_means    = np.array([w["within_deep"]["mean_r"] for w in window_stats], dtype=float)
    per_shallow   = np.array([w["cross_plane"]["per_shallow_roi_mean"] for w in window_stats], dtype=float)

    return {
        "n_windows_used": len(window_stats),
        "n_bins_total": int(weights.sum()),
        "max_alignment_delta_sec": float(max(w["max_alignment_delta_sec"] for w in window_stats)),
        "cross_plane": {
            "mean_r": weighted_observed,
            "max_r": float(max(w["cross_plane"]["max_r"] for w in window_stats)),
            "min_r": float(min(w["cross_plane"]["min_r"] for w in window_stats)),
            "std_r": float(np.average(
                np.array([w["cross_plane"]["std_r"] for w in window_stats], dtype=float),
                weights=weights,
            )),
            "n_pairs_per_window": int(window_stats[0]["cross_plane"]["n_pairs"]),
            "n_above_02_window_sum": int(sum(w["cross_plane"]["n_above_02"] for w in window_stats)),
            "n_above_03_window_sum": int(sum(w["cross_plane"]["n_above_03"] for w in window_stats)),
            "null_mean": null_mean,
            "null_std": null_std,
            "z_vs_null": float(z),
            "per_shallow_roi_mean": np.average(per_shallow, axis=0, weights=weights).tolist(),
        },
        "within_shallow": {
            "mean_r": float(np.average(shallow_means, weights=weights)),
            "max_r": float(max(w["within_shallow"]["max_r"] for w in window_stats)),
        },
        "within_deep": {
            "mean_r": float(np.average(deep_means, weights=weights)),
            "max_r": float(max(w["within_deep"]["max_r"] for w in window_stats)),
        },
        "window_stats": window_stats,
    }


def _write_summary(path: Path, results: dict, subject: str) -> None:
    lines = [
        f"# Cross-plane Coupling Summary: DANDI 000336 â€” {subject}",
        "",
        "| Condition | Windows | Bins | Mean r | z vs null | Within shallow | Within deep |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['n_windows_used']} | {result['n_bins_total']} | "
            f"{result['cross_plane']['mean_r']:.4f} | {result['cross_plane']['z_vs_null']:.2f} | "
            f"{result['within_shallow']['mean_r']:.4f} | {result['within_deep']['mean_r']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log_lines: list[str] = []
    TRIAGE_ROOT.mkdir(parents=True, exist_ok=True)

    _emit(log_lines, "=" * 60)
    _emit(log_lines, f"Exp 04 â€” Cross-plane Coupling: DANDI 000336 ({_SUBJECT})")
    _emit(log_lines, f"Replication of exp03 (sub-644972) with identical method")
    _emit(log_lines, "=" * 60)

    shallow_path = DATA_ROOT / _SHALLOW_REL
    deep_path    = DATA_ROOT / _DEEP_REL

    for label, p in [("Shallow", shallow_path), ("Deep", deep_path)]:
        if not p.exists():
            _emit(log_lines, f"MISSING: {label}: {p}")
            return

    _emit(log_lines, f"\nLoading shallow plane ({shallow_path.name}) ...")
    s_data, s_ts, s_ivals = _load_dff_and_intervals(shallow_path)
    _emit(log_lines, f"  dff shape: {s_data.shape}  active ROIs: {(s_data.std(axis=0)>0.01).sum()}/{s_data.shape[1]}")

    _emit(log_lines, f"Loading deep plane ({deep_path.name}) ...")
    d_data, d_ts, d_ivals = _load_dff_and_intervals(deep_path)
    _emit(log_lines, f"  dff shape: {d_data.shape}  active ROIs: {(d_data.std(axis=0)>0.01).sum()}/{d_data.shape[1]}")

    shared_ivals = {k: v for k, v in s_ivals.items() if k in d_ivals}
    _emit(log_lines, f"\nShared interval tables: {sorted(shared_ivals.keys())}")

    spont_ivals   = shared_ivals.get("spontaneous_presentations", ([], []))
    spont_windows = _epoch_windows(spont_ivals[0], spont_ivals[1], min_duration=1.0)

    def _stim_blocks(key: str) -> list[tuple[float, float]]:
        ivals = shared_ivals.get(key, ([], []))
        return _merge_into_blocks(ivals[0], ivals[1], max_gap_sec=5.0, min_block_dur=10.0)

    conditions = {
        "spontaneous":      spont_windows,
        "gratings":         _stim_blocks("gratings_presentations"),
        "fixed_gabors":     _stim_blocks("fixed_gabors_presentations"),
        "movie_flower_fwd": _stim_blocks("movie_flower_fwd_presentations"),
    }

    _emit(log_lines, f"\n{'Condition':<22} {'Win':>5} {'N bins':>7} {'Mean r':>8} "
                     f"{'z vs null':>10} {'Max dt':>8}")
    _emit(log_lines, f"{'-'*22} {'-'*5} {'-'*7} {'-'*8} {'-'*10} {'-'*8}")

    results = {}
    for cond_name, windows in conditions.items():
        if not windows:
            _emit(log_lines, f"  {cond_name:<22}: no windows")
            continue

        window_stats = []
        for idx, (t0, t1) in enumerate(windows):
            s_mask = _epoch_mask(s_ts, t0, t1)
            d_mask = _epoch_mask(d_ts, t0, t1)
            s_chunk = s_data[s_mask]
            d_chunk = d_data[d_mask]
            s_chunk, d_chunk, max_delta = _align_by_timestamp(
                s_chunk, s_ts[s_mask], d_chunk, d_ts[d_mask]
            )
            if len(s_chunk) < _MIN_BINS_PER_WINDOW or len(d_chunk) < _MIN_BINS_PER_WINDOW:
                continue
            cross    = _window_cross_coupling(s_chunk, d_chunk, seed=42 + idx)
            within_s = _within_coupling(s_chunk)
            within_d = _within_coupling(d_chunk)
            window_stats.append({
                "window_index": idx,
                "start_sec": float(t0),
                "stop_sec": float(t1),
                "duration_sec": float(t1 - t0),
                "n_bins": int(len(s_chunk)),
                "max_alignment_delta_sec": float(max_delta),
                "cross_plane": cross,
                "within_shallow": within_s,
                "within_deep": within_d,
            })

        if not window_stats:
            _emit(log_lines, f"  {cond_name:<22}: no aligned windows above {_MIN_BINS_PER_WINDOW} bins")
            continue

        aggregated = _aggregate_condition(window_stats)
        results[cond_name] = aggregated
        _emit(log_lines, (
            f"  {cond_name:<22} {aggregated['n_windows_used']:>5} {aggregated['n_bins_total']:>7} "
            f"{aggregated['cross_plane']['mean_r']:>8.4f} {aggregated['cross_plane']['z_vs_null']:>10.2f} "
            f"{aggregated['max_alignment_delta_sec']:>8.5f}"
        ))

    # ---- Qualitative H3 check ----
    _emit(log_lines, "\n--- H3 qualitative checks ---")

    if "spontaneous" in results and "gratings" in results:
        sp = results["spontaneous"]["cross_plane"]["mean_r"]
        gr = results["gratings"]["cross_plane"]["mean_r"]
        _emit(log_lines, f"  Spontaneous cross-plane mean r: {sp:.4f}")
        _emit(log_lines, f"  Gratings    cross-plane mean r: {gr:.4f}")
        if sp < gr:
            _emit(log_lines, "  -> Gratings > Spontaneous -- stimulus drives coupling")
        else:
            _emit(log_lines, "  -> Spontaneous >= Gratings -- resting access structure maintained")

    if "spontaneous" in results:
        sp_res = results["spontaneous"]
        wsh = sp_res["within_shallow"]["mean_r"]
        wde = sp_res["within_deep"]["mean_r"]
        crs = sp_res["cross_plane"]["mean_r"]
        z   = sp_res["cross_plane"]["z_vs_null"]
        _emit(log_lines, f"\n  Spontaneous coupling structure:")
        _emit(log_lines, f"    Within-shallow mean r: {wsh:.4f}")
        _emit(log_lines, f"    Within-deep    mean r: {wde:.4f}")
        _emit(log_lines, f"    Cross-plane    mean r: {crs:.4f}  z={z:.2f}")
        if crs > 0 and z > 1.96:
            if crs < min(wsh, wde):
                verdict = "H3 POSITIVE â€” cross-plane above null AND below within-plane (access-constraint signature)"
            else:
                verdict = "H3 PARTIAL â€” cross-plane above null but not below within-plane"
        elif crs > 0:
            verdict = "H3 WEAK â€” cross-plane positive but z < 1.96"
        else:
            verdict = "H3 NEGATIVE â€” cross-plane not above null"
        _emit(log_lines, f"    H3 verdict: {verdict}")

    _emit(log_lines, "\n--- All conditions z vs null ---")
    for cname, res in results.items():
        z  = res["cross_plane"]["z_vs_null"]
        mr = res["cross_plane"]["mean_r"]
        _emit(log_lines, f"  {cname:<22} mean_r={mr:.4f}  z={z:.2f}  {'ABOVE NULL' if z > 1.96 else 'not significant'}")

    # ---- Compare to sub-644972 reference ----
    _emit(log_lines, "\n--- Comparison target (sub-644972 reference, exp03) ---")
    _ref = {
        "spontaneous":      {"mean_r": None, "z": None},
        "gratings":         {"mean_r": None, "z": None},
        "fixed_gabors":     {"mean_r": None, "z": None},
        "movie_flower_fwd": {"mean_r": None, "z": None},
    }
    ref_json = TRIAGE_ROOT / "crossplane_coupling.json"
    if ref_json.exists():
        import json as _json
        ref_data = _json.loads(ref_json.read_text(encoding="utf-8"))
        for cname in _ref:
            if cname in ref_data:
                _ref[cname]["mean_r"] = ref_data[cname]["cross_plane"]["mean_r"]
                _ref[cname]["z"]      = ref_data[cname]["cross_plane"]["z_vs_null"]
        _emit(log_lines, f"  {'Condition':<22} {'644972 mean_r':>14} {'656228 mean_r':>14} {'644972 z':>9} {'656228 z':>9}")
        _emit(log_lines, f"  {'-'*22} {'-'*14} {'-'*14} {'-'*9} {'-'*9}")
        for cname in ["spontaneous", "gratings", "fixed_gabors", "movie_flower_fwd"]:
            ref_mr = _ref[cname]["mean_r"]
            ref_z  = _ref[cname]["z"]
            cur_mr = results.get(cname, {}).get("cross_plane", {}).get("mean_r", None)
            cur_z  = results.get(cname, {}).get("cross_plane", {}).get("z_vs_null", None)
            _emit(log_lines, (
                f"  {cname:<22} "
                f"  {(f'{ref_mr:.4f}' if ref_mr is not None else 'N/A'):>14}"
                f"  {(f'{cur_mr:.4f}' if cur_mr is not None else 'N/A'):>14}"
                f"  {(f'{ref_z:.2f}' if ref_z is not None else 'N/A'):>9}"
                f"  {(f'{cur_z:.2f}' if cur_z is not None else 'N/A'):>9}"
            ))
    else:
        _emit(log_lines, "  (sub-644972 reference JSON not found â€” run exp03 first)")

    # ---- Write outputs ----
    out_json = TRIAGE_ROOT / "crossplane_coupling_sub656228.json"
    out_md   = TRIAGE_ROOT / "crossplane_coupling_sub656228.md"
    out_log  = TRIAGE_ROOT / "crossplane_coupling_sub656228.log"

    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    _write_summary(out_md, results, _SUBJECT)

    _emit(log_lines, f"\nResults  -> {out_json.relative_to(ROOT)}")
    _emit(log_lines, f"Summary  -> {out_md.relative_to(ROOT)}")
    _emit(log_lines, f"Log      -> {out_log.relative_to(ROOT)}")
    out_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


