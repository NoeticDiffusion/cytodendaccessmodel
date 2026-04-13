"""Experiment 03 — Cross-plane coupling: shallow (20 µm) vs deep (295 µm) plane.

Tests H3 (dendritic access signature): is cross-plane coupling structured and
non-random? Compares spontaneous, grating, and movie epochs.

Key methodological guardrails:
- align the two planes on shared timestamps within each epoch window,
- compute correlations window-by-window rather than on one concatenated series,
- build nulls within each window before aggregating to a condition-level z-score,
- for stimulus conditions (gratings, movies) use block-level windows rather than
  individual short presentations: consecutive presentations within max_gap_sec are
  merged into one block so that there are enough imaging frames per window,
- write both JSON and human-readable summary artifacts to disk.

Requires both pair_a files:
  sub-644972_ses-1237338784-acq-1237809217_image+ophys.nwb  (shallow, 6 ROIs)
  sub-644972_ses-1237338784-acq-1237809219_image+ophys.nwb  (deep,    62 ROIs)

Usage:
    python experiments/dandi_000871_03_crossplane_coupling.py

Outputs:
    data/dandi/triage/000871/crossplane_coupling.json
    data/dandi/triage/000871/crossplane_coupling.md
    data/dandi/triage/000871/crossplane_coupling.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "dandi" / "raw" / "000871"
TRIAGE_ROOT = ROOT / "data" / "dandi" / "triage" / "000871"

sys.path.insert(0, str(ROOT / "src"))

_SHALLOW_REL = "sub-644972/sub-644972_ses-1237338784-acq-1237809217_image+ophys.nwb"
_DEEP_REL    = "sub-644972/sub-644972_ses-1237338784-acq-1237809219_image+ophys.nwb"

_N_NULL = 200
_MIN_BINS_PER_WINDOW = 20
_ALIGN_DECIMALS = 6


def _emit(log_lines: list[str], message: str = "") -> None:
    print(message)
    log_lines.append(message)


def _load_dff_and_intervals(path: Path):
    """Return (dff_array, timestamps, interval_dict) from a 000871 NWB file."""
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


def _epoch_windows(starts, stops, min_duration=1.0):
    windows = []
    for t0, t1 in zip(starts, stops):
        if t1 - t0 >= min_duration:
            windows.append((float(t0), float(t1)))
    return windows


def _merge_into_blocks(starts, stops, max_gap_sec: float = 5.0, min_block_dur: float = 10.0):
    """Merge consecutive short presentations into blocks for stimulus conditions.

    Consecutive windows where the gap between stop_i and start_{i+1} is less
    than *max_gap_sec* are merged into one block window.  Only blocks with
    total duration >= *min_block_dur* are returned.

    This converts 600+ individual 0.3s grating presentations into a handful of
    long blocks that contain enough imaging frames for meaningful correlation
    analysis.
    """
    if not starts:
        return []

    pairs = sorted(zip(starts, stops))
    blocks: list[tuple[float, float]] = []
    block_start = float(pairs[0][0])
    block_end = float(pairs[0][1])

    for t0, t1 in pairs[1:]:
        gap = float(t0) - block_end
        if gap <= max_gap_sec:
            block_end = max(block_end, float(t1))
        else:
            if block_end - block_start >= min_block_dur:
                blocks.append((block_start, block_end))
            block_start = float(t0)
            block_end = float(t1)

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
    """Compute window-local cross-plane coupling and a within-window null."""
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

    weights = np.array([w["n_bins"] for w in window_stats], dtype=float)
    observed = np.array([w["cross_plane"]["mean_r"] for w in window_stats], dtype=float)
    null_matrix = np.array([w["cross_plane"]["null_means"] for w in window_stats], dtype=float)
    agg_null = np.average(null_matrix, axis=0, weights=weights)
    weighted_observed = float(np.average(observed, weights=weights))
    null_mean = float(np.mean(agg_null))
    null_std = float(np.std(agg_null))
    z = (weighted_observed - null_mean) / null_std if null_std > 0 else float("nan")

    shallow_means = np.array([w["within_shallow"]["mean_r"] for w in window_stats], dtype=float)
    deep_means = np.array([w["within_deep"]["mean_r"] for w in window_stats], dtype=float)
    per_shallow = np.array([w["cross_plane"]["per_shallow_roi_mean"] for w in window_stats], dtype=float)

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


def _write_summary(path: Path, results: dict) -> None:
    lines = [
        "# Cross-plane Coupling Summary: DANDI 000871",
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


def main() -> None:
    log_lines: list[str] = []

    _emit(log_lines, "=" * 60)
    _emit(log_lines, "Experiment 03 - Cross-plane Coupling: DANDI 000871")
    _emit(log_lines, "=" * 60)

    shallow_path = DATA_ROOT / _SHALLOW_REL
    deep_path    = DATA_ROOT / _DEEP_REL

    for label, p in [("Shallow (20 µm)", shallow_path), ("Deep (295 µm)", deep_path)]:
        if not p.exists():
            _emit(log_lines, f"MISSING: {label}: {p}")
            return

    _emit(log_lines, f"\nLoading shallow plane ({shallow_path.name}) ...")
    s_data, s_ts, s_ivals = _load_dff_and_intervals(shallow_path)
    _emit(log_lines, f"  dff shape: {s_data.shape}  active: {(s_data.std(axis=0)>0.01).sum()}/{s_data.shape[1]}")

    _emit(log_lines, f"Loading deep plane ({deep_path.name}) ...")
    d_data, d_ts, d_ivals = _load_dff_and_intervals(deep_path)
    _emit(log_lines, f"  dff shape: {d_data.shape}  active: {(d_data.std(axis=0)>0.01).sum()}/{d_data.shape[1]}")

    # Shared intervals (both files should have identical interval tables)
    shared_ivals = {k: v for k, v in s_ivals.items() if k in d_ivals}
    _emit(log_lines, f"\nShared interval tables: {list(shared_ivals.keys())}")

    # Epoch conditions to analyse.
    # Spontaneous: use individual long windows (already long enough).
    # Stimulus conditions (gratings, movies): merge short presentations into
    # blocks so each analysis window spans enough imaging frames.
    spont_ivals = shared_ivals.get("spontaneous_presentations", ([], []))
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

    results = {}
    _emit(log_lines, f"\n{'Condition':<22} {'Win':>5} {'N bins':>7} {'Mean r':>8} "
                     f"{'z vs null':>10} {'Max dt':>8}")
    _emit(log_lines, f"{'-'*22} {'-'*5} {'-'*7} {'-'*8} {'-'*10} {'-'*8}")

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

            cross = _window_cross_coupling(s_chunk, d_chunk, seed=42 + idx)
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

        _emit(
            log_lines,
            f"  {cond_name:<22} {aggregated['n_windows_used']:>5} {aggregated['n_bins_total']:>7} "
            f"{aggregated['cross_plane']['mean_r']:>8.4f} {aggregated['cross_plane']['z_vs_null']:>10.2f} "
            f"{aggregated['max_alignment_delta_sec']:>8.5f}",
        )

    # Summary
    if "spontaneous" in results and "gratings" in results:
        sp = results["spontaneous"]["cross_plane"]["mean_r"]
        gr = results["gratings"]["cross_plane"]["mean_r"]
        _emit(log_lines, "\n  Spontaneous vs Gratings cross-plane:")
        _emit(log_lines, f"    Spontaneous mean r: {sp:.4f}")
        _emit(log_lines, f"    Gratings    mean r: {gr:.4f}")
        if sp < gr:
            _emit(log_lines, "    Gratings coupling > Spontaneous — stimulus drives coupling")
        else:
            _emit(log_lines, "    Spontaneous coupling >= Gratings — resting access structure not dominated by stimulus drive")

    _emit(log_lines, "\n  Within-plane coupling (spontaneous):")
    if "spontaneous" in results:
        _emit(log_lines, f"    Within-shallow mean r: {results['spontaneous']['within_shallow']['mean_r']:.4f}")
        _emit(log_lines, f"    Within-deep    mean r: {results['spontaneous']['within_deep']['mean_r']:.4f}")
        _emit(log_lines, f"    Cross-plane    mean r: {results['spontaneous']['cross_plane']['mean_r']:.4f}")
        _emit(log_lines, "    (cross < within = access-restricted signature)")

    TRIAGE_ROOT.mkdir(parents=True, exist_ok=True)
    out = TRIAGE_ROOT / "crossplane_coupling.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    _write_summary(TRIAGE_ROOT / "crossplane_coupling.md", results)
    _emit(log_lines, f"\nResults written: {out.relative_to(ROOT)}")
    _emit(log_lines, f"Summary written: {(TRIAGE_ROOT / 'crossplane_coupling.md').relative_to(ROOT)}")
    _emit(log_lines, f"Log written: {(TRIAGE_ROOT / 'crossplane_coupling.log').relative_to(ROOT)}")
    (TRIAGE_ROOT / "crossplane_coupling.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
