"""Experiment 05 â€” Cross-plane coupling: sub-656228 / ses-1245548523.

Second session replication for sub-656228.  Same conservative H3 analysis as
exp03 (sub-644972) and exp04 (sub-656228 / ses-1247233186).

Session background
------------------
ses-1245548523 is a multi-plane session with 8 simultaneous acquisitions.
We pair the lowest-ID acquisition (acq-1245937725, likely shallowest plane)
against the already-downloaded acq-1245937736 (144 Âµm depth, 27 ROIs).

Depth metadata is read from the NWB file at runtime and reported in the log,
so the actual depths are always visible in the output.

Key question
------------
Does ses-1245548523 reproduce the H3 access-constraint signature, and does it
recover any conditions missing from ses-1247233186 (notably movie_flower_fwd)?

Usage:
    python experiments/dandi_000336_05_ses1245548523.py

Outputs:
    data/dandi/triage/000336/crossplane_coupling_sub656228_ses1245548523.json
    data/dandi/triage/000336/crossplane_coupling_sub656228_ses1245548523.md
    data/dandi/triage/000336/crossplane_coupling_sub656228_ses1245548523.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT       = Path(__file__).parent.parent
DATA_ROOT  = ROOT / "data" / "dandi" / "raw" / "000336"
TRIAGE_ROOT = ROOT / "data" / "dandi" / "triage" / "000336"

sys.path.insert(0, str(ROOT / "src"))

_SUBJECT     = "sub-656228"
_SESSION     = "ses-1245548523"
# acq-1245937727: 53 ROIs, VISp,  148 Âµm
_PLANE_A_REL = f"{_SUBJECT}/{_SUBJECT}_{_SESSION}-acq-1245937727_ophys.nwb"
# acq-1245937736: 27 ROIs, VISpm, 144 Âµm
_PLANE_B_REL = f"{_SUBJECT}/{_SUBJECT}_{_SESSION}-acq-1245937736_ophys.nwb"

_N_NULL              = 200
_MIN_BINS_PER_WINDOW = 20
_ALIGN_DECIMALS      = 6
# For interleaved multi-plane sessions: bin both planes at this resolution (s)
# before alignment. Set to 0 to use exact timestamp matching (ses-1247233186 style).
_BIN_SIZE_SEC        = 0.5


def _emit(log_lines: list[str], message: str = "") -> None:
    print(message)
    log_lines.append(message)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_dff_and_intervals(path: Path):
    import pynwb, numpy as np
    with pynwb.NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
        nwb = io.read()
        rrs = nwb.processing["ophys"]["dff"].roi_response_series["traces"]
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


def _read_depth_desc(path: Path) -> str:
    import h5py
    try:
        with h5py.File(str(path), "r") as hf:
            desc = hf["general/optophysiology/imaging_plane_1/description"][()]
            return desc.decode() if isinstance(desc, bytes) else str(desc)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Window helpers â€” identical to exp03/exp04
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


def _bin_align(a_chunk, a_ts, b_chunk, b_ts, bin_size_sec: float):
    """Bin both data arrays at bin_size_sec resolution and return aligned bins.

    Used for interleaved multi-plane sessions where exact timestamp matching fails.
    Within each bin, all frames are averaged before the coupling computation.
    """
    import numpy as np
    if len(a_chunk) == 0 or len(b_chunk) == 0:
        return a_chunk[:0], b_chunk[:0], 0.0

    t_start = max(float(a_ts[0]), float(b_ts[0]))
    t_stop  = min(float(a_ts[-1]), float(b_ts[-1]))
    if t_stop <= t_start:
        return a_chunk[:0], b_chunk[:0], 0.0

    edges = np.arange(t_start, t_stop + bin_size_sec, bin_size_sec)
    n_bins = len(edges) - 1
    if n_bins < 1:
        return a_chunk[:0], b_chunk[:0], 0.0

    def _bin_data(data, ts):
        out = np.zeros((n_bins, data.shape[1]), dtype=float)
        for i in range(n_bins):
            mask = (ts >= edges[i]) & (ts < edges[i + 1])
            if mask.sum() > 0:
                out[i] = data[mask].mean(axis=0)
        return out

    a_binned = _bin_data(a_chunk, np.asarray(a_ts, dtype=float))
    b_binned = _bin_data(b_chunk, np.asarray(b_ts, dtype=float))
    max_delta = bin_size_sec  # report bin size as alignment resolution
    return a_binned, b_binned, max_delta


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
        null_means.append(_cross_block(s_norm, d_shifted).mean())
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
        "per_plane_a_roi_mean": cross_block.mean(axis=1).tolist(),
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
    pa_means = np.array([w["within_a"]["mean_r"] for w in window_stats], dtype=float)
    pb_means = np.array([w["within_b"]["mean_r"] for w in window_stats], dtype=float)
    per_a    = np.array([w["cross_plane"]["per_plane_a_roi_mean"] for w in window_stats], dtype=float)
    return {
        "n_windows_used": len(window_stats),
        "n_bins_total": int(weights.sum()),
        "max_alignment_delta_sec": float(max(w["max_alignment_delta_sec"] for w in window_stats)),
        "cross_plane": {
            "mean_r": weighted_observed,
            "max_r": float(max(w["cross_plane"]["max_r"] for w in window_stats)),
            "min_r": float(min(w["cross_plane"]["min_r"] for w in window_stats)),
            "std_r": float(np.average(
                [w["cross_plane"]["std_r"] for w in window_stats], weights=weights)),
            "n_pairs_per_window": int(window_stats[0]["cross_plane"]["n_pairs"]),
            "n_above_02_window_sum": int(sum(w["cross_plane"]["n_above_02"] for w in window_stats)),
            "n_above_03_window_sum": int(sum(w["cross_plane"]["n_above_03"] for w in window_stats)),
            "null_mean": null_mean,
            "null_std": null_std,
            "z_vs_null": float(z),
            "per_plane_a_roi_mean": np.average(per_a, axis=0, weights=weights).tolist(),
        },
        "within_a": {
            "mean_r": float(np.average(pa_means, weights=weights)),
            "max_r": float(max(w["within_a"]["max_r"] for w in window_stats)),
        },
        "within_b": {
            "mean_r": float(np.average(pb_means, weights=weights)),
            "max_r": float(max(w["within_b"]["max_r"] for w in window_stats)),
        },
        "window_stats": window_stats,
    }


def _write_summary(path: Path, results: dict, subject: str, session: str) -> None:
    lines = [
        f"# Cross-plane Coupling Summary: DANDI 000336 â€” {subject} / {session}",
        "",
        "| Condition | Windows | Bins | Mean r | z vs null | Within A | Within B |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['n_windows_used']} | {result['n_bins_total']} | "
            f"{result['cross_plane']['mean_r']:.4f} | {result['cross_plane']['z_vs_null']:.2f} | "
            f"{result['within_a']['mean_r']:.4f} | {result['within_b']['mean_r']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log_lines: list[str] = []
    TRIAGE_ROOT.mkdir(parents=True, exist_ok=True)

    _emit(log_lines, "=" * 65)
    _emit(log_lines, f"Exp 05 -- Cross-plane Coupling: {_SUBJECT} / {_SESSION}")
    _emit(log_lines, "Same conservative H3 method as exp03/exp04")
    _emit(log_lines, "=" * 65)

    path_a = DATA_ROOT / _PLANE_A_REL
    path_b = DATA_ROOT / _PLANE_B_REL

    for label, p in [("Plane A (candidate shallow)", path_a),
                     ("Plane B (144 um known)", path_b)]:
        if not p.exists():
            _emit(log_lines, f"MISSING: {label}: {p}")
            _emit(log_lines, "Run experiments/_download_ses1245548523_shallow.py first.")
            return

    # Read depth metadata before full load
    depth_a = _read_depth_desc(path_a)
    depth_b = _read_depth_desc(path_b)
    _emit(log_lines, f"\nPlane A: {depth_a}")
    _emit(log_lines, f"Plane B: {depth_b}")
    _emit(log_lines, "NOTE: Both planes at similar depth (~144-148 um).")
    _emit(log_lines, "      This session records from VISp (plane A) and VISpm (plane B)")
    _emit(log_lines, "      -- a cross-AREA pairing, not cross-DEPTH as in ses-1247233186.")
    _emit(log_lines, f"      Timestamps are interleaved; using {_BIN_SIZE_SEC}s bins for alignment.")

    _emit(log_lines, f"\nLoading Plane A ({path_a.name}) ...")
    a_data, a_ts, a_ivals = _load_dff_and_intervals(path_a)
    _emit(log_lines, f"  shape: {a_data.shape}  active: {(a_data.std(axis=0)>0.01).sum()}/{a_data.shape[1]}")

    _emit(log_lines, f"Loading Plane B ({path_b.name}) ...")
    b_data, b_ts, b_ivals = _load_dff_and_intervals(path_b)
    _emit(log_lines, f"  shape: {b_data.shape}  active: {(b_data.std(axis=0)>0.01).sum()}/{b_data.shape[1]}")

    # Use A as "shallow" and B as "deep" based on ROI count if depths are similar
    if a_data.shape[1] > b_data.shape[1]:
        _emit(log_lines, "  NOTE: Plane A has more ROIs than B; swapping labels for cross-plane analysis.")
        a_data, a_ts, a_ivals, b_data, b_ts, b_ivals = b_data, b_ts, b_ivals, a_data, a_ts, a_ivals
        depth_a, depth_b = depth_b, depth_a

    shared_ivals = {k: v for k, v in a_ivals.items() if k in b_ivals}
    _emit(log_lines, f"\nShared interval tables ({len(shared_ivals)}): {sorted(shared_ivals.keys())}")

    spont_windows = _epoch_windows(
        *shared_ivals.get("spontaneous_presentations", ([], [])), min_duration=1.0
    )

    def _stim_blocks(key: str) -> list[tuple[float, float]]:
        ivals = shared_ivals.get(key, ([], []))
        return _merge_into_blocks(ivals[0], ivals[1], max_gap_sec=5.0, min_block_dur=10.0)

    conditions = {
        "spontaneous":      spont_windows,
        "gratings":         _stim_blocks("gratings_presentations"),
        "fixed_gabors":     _stim_blocks("fixed_gabors_presentations"),
        "movie_flower_fwd": _stim_blocks("movie_flower_fwd_presentations"),
        "movie_worms_fwd":  _stim_blocks("movie_worms_fwd_presentations"),
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
            a_mask = _epoch_mask(a_ts, t0, t1)
            b_mask = _epoch_mask(b_ts, t0, t1)
            a_chunk = a_data[a_mask]
            b_chunk = b_data[b_mask]
            if _BIN_SIZE_SEC > 0:
                a_chunk, b_chunk, max_delta = _bin_align(
                    a_chunk, a_ts[a_mask], b_chunk, b_ts[b_mask], _BIN_SIZE_SEC
                )
            else:
                a_chunk, b_chunk, max_delta = _align_by_timestamp(
                    a_chunk, a_ts[a_mask], b_chunk, b_ts[b_mask]
                )
            if len(a_chunk) < _MIN_BINS_PER_WINDOW or len(b_chunk) < _MIN_BINS_PER_WINDOW:
                continue
            cross    = _window_cross_coupling(a_chunk, b_chunk, seed=42 + idx)
            within_a = _within_coupling(a_chunk)
            within_b = _within_coupling(b_chunk)
            window_stats.append({
                "window_index": idx,
                "start_sec": float(t0), "stop_sec": float(t1),
                "duration_sec": float(t1 - t0),
                "n_bins": int(len(a_chunk)),
                "max_alignment_delta_sec": float(max_delta),
                "cross_plane": cross,
                "within_a": within_a,
                "within_b": within_b,
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

    # ---- H3 qualitative checks ----
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
        wa  = sp_res["within_a"]["mean_r"]
        wb  = sp_res["within_b"]["mean_r"]
        crs = sp_res["cross_plane"]["mean_r"]
        z   = sp_res["cross_plane"]["z_vs_null"]
        _emit(log_lines, f"\n  Spontaneous coupling structure:")
        _emit(log_lines, f"    Within-A (shallower) mean r: {wa:.4f}  depth: {depth_a}")
        _emit(log_lines, f"    Within-B (deeper)    mean r: {wb:.4f}  depth: {depth_b}")
        _emit(log_lines, f"    Cross-plane          mean r: {crs:.4f}  z={z:.2f}")
        if crs > 0 and z > 1.96:
            if crs < min(wa, wb):
                verdict = "H3 POSITIVE -- cross-plane above null AND below within-plane"
            else:
                verdict = "H3 PARTIAL -- cross-plane above null but not below within-plane"
        elif crs > 0:
            verdict = "H3 WEAK -- cross-plane positive but z < 1.96"
        else:
            verdict = "H3 NEGATIVE -- cross-plane not above null"
        _emit(log_lines, f"    H3 verdict: {verdict}")

    # ---- Conditions summary ----
    _emit(log_lines, "\n--- All conditions z vs null ---")
    for cname, res in results.items():
        z  = res["cross_plane"]["z_vs_null"]
        mr = res["cross_plane"]["mean_r"]
        _emit(log_lines, (
            f"  {cname:<22} mean_r={mr:.4f}  z={z:.2f}  "
            f"{'ABOVE NULL' if z > 1.96 else 'not significant'}"
        ))

    # Check movie_flower_fwd availability vs ses-1247233186
    _emit(log_lines, "\n--- New conditions vs ses-1247233186 ---")
    for cname in ["movie_flower_fwd", "movie_worms_fwd"]:
        status = "PRESENT" if cname in results else "absent"
        _emit(log_lines, f"  {cname}: {status}")

    # ---- Cross-reference to previous results ----
    ref_json_a = TRIAGE_ROOT / "crossplane_coupling.json"
    ref_json_b = TRIAGE_ROOT / "crossplane_coupling_sub656228.json"
    _emit(log_lines, "\n--- Cross-subject/session comparison (spontaneous) ---")
    _emit(log_lines, (
        f"  {'Source':<40} {'mean_r':>8} {'z':>8}"
    ))
    _emit(log_lines, f"  {'-'*40} {'-'*8} {'-'*8}")
    for label, ref_path in [
        ("sub-644972 (exp03)", ref_json_a),
        ("sub-656228 ses-1247233186 (exp04)", ref_json_b),
        (f"{_SUBJECT} {_SESSION} (exp05)", None),
    ]:
        if ref_path is not None and ref_path.exists():
            ref = json.loads(ref_path.read_text(encoding="utf-8"))
            sp  = ref.get("spontaneous", {})
            mr  = sp.get("cross_plane", {}).get("mean_r", float("nan"))
            z   = sp.get("cross_plane", {}).get("z_vs_null", float("nan"))
        elif ref_path is None and "spontaneous" in results:
            mr = results["spontaneous"]["cross_plane"]["mean_r"]
            z  = results["spontaneous"]["cross_plane"]["z_vs_null"]
        else:
            mr, z = float("nan"), float("nan")
        _emit(log_lines, f"  {label:<40} {mr:>8.4f} {z:>8.2f}")

    # ---- Write outputs ----
    out_stem = f"crossplane_coupling_{_SUBJECT}_{_SESSION}"
    out_json = TRIAGE_ROOT / f"{out_stem}.json"
    out_md   = TRIAGE_ROOT / f"{out_stem}.md"
    out_log  = TRIAGE_ROOT / f"{out_stem}.log"

    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    _write_summary(out_md, results, _SUBJECT, _SESSION)

    _emit(log_lines, f"\nResults  -> {out_json.relative_to(ROOT)}")
    _emit(log_lines, f"Summary  -> {out_md.relative_to(ROOT)}")
    _emit(log_lines, f"Log      -> {out_log.relative_to(ROOT)}")
    out_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


