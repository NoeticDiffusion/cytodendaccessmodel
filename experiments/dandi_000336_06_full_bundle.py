"""Experiment 06 â€” Full bundle: all three 000336 plane pairs, unified pipeline.

Runs the same cross-plane coupling analysis across all 6 NWB files organised
as three within-session pairs, using only the 7 conditions present in every file:
  spontaneous, gratings, fixed_gabors,
  movie_flower_fwd, movie_touch_of_evil_fwd, movie_worms_fwd, rotate_gabors.

Pairs:
  pair_a  sub-644972  ses-1237338784  20 um  (6 ROIs)  vs 295 um (62 ROIs)
  pair_b  sub-656228  ses-1247233186  42 um  (4 ROIs)  vs 152 um (25 ROIs)
  pair_c  sub-656228  ses-1245548523 148 um (53 ROIs)  vs 144 um (27 ROIs)  [cross-area]

Usage:
    python experiments/dandi_000336_06_full_bundle.py

Outputs:
    data/dandi/triage/000336/full_bundle_coupling.json
    data/dandi/triage/000336/full_bundle_coupling.md
    data/dandi/triage/000336/full_bundle_coupling.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_ROOT   = ROOT / "data" / "dandi" / "raw"   / "000336"
TRIAGE_ROOT = ROOT / "data" / "dandi" / "triage" / "000336"

sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------------
# Pair definitions
# ---------------------------------------------------------------------------

_PAIRS: list[dict] = [
    {
        "id": "pair_a",
        "subject": "sub-644972",
        "session": "ses-1237338784",
        "plane_a": "sub-644972/sub-644972_ses-1237338784-acq-1237809217_ophys.nwb",
        "plane_b": "sub-644972/sub-644972_ses-1237338784-acq-1237809219_ophys.nwb",
        "label_a": "shallow (20 um, 6 ROIs)",
        "label_b": "deep (295 um, 62 ROIs)",
        "pairing": "cross_depth",
        "bin_size_sec": 0.0,   # exact timestamp matching
    },
    {
        "id": "pair_b",
        "subject": "sub-656228",
        "session": "ses-1247233186",
        "plane_a": "sub-656228/sub-656228_ses-1247233186-acq-1247385128_ophys.nwb",
        "plane_b": "sub-656228/sub-656228_ses-1247233186-acq-1247385130_ophys.nwb",
        "label_a": "shallow (42 um, 4 ROIs)",
        "label_b": "deep (152 um, 25 ROIs)",
        "pairing": "cross_depth",
        "bin_size_sec": 0.0,
    },
    {
        "id": "pair_c",
        "subject": "sub-656228",
        "session": "ses-1245548523",
        "plane_a": "sub-656228/sub-656228_ses-1245548523-acq-1245937736_ophys.nwb",
        "plane_b": "sub-656228/sub-656228_ses-1245548523-acq-1245937727_ophys.nwb",
        "label_a": "VISpm 144 um (27 ROIs)",
        "label_b": "VISp  148 um (53 ROIs)",
        "pairing": "cross_area",
        "bin_size_sec": 0.5,   # interleaved timestamps -> bin alignment
    },
]

# Only conditions present in ALL six files
_COMMON_CONDITIONS = [
    "spontaneous_presentations",
    "gratings_presentations",
    "fixed_gabors_presentations",
    "movie_flower_fwd_presentations",
    "movie_touch_of_evil_fwd_presentations",
    "movie_worms_fwd_presentations",
    "rotate_gabors_presentations",
]

_N_NULL              = 200
_MIN_BINS_PER_WINDOW = 20
_ALIGN_DECIMALS      = 6


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _emit(log_lines, msg=""):
    print(msg)
    log_lines.append(msg)


def _load_dff_and_intervals(path: Path):
    import pynwb, numpy as np
    with pynwb.NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
        nwb = io.read()
        rrs  = nwb.processing["ophys"]["dff"].roi_response_series["traces"]
        data = np.array(rrs.data[:])
        ts   = np.array(rrs.timestamps[:])
        intervals = {}
        for name in nwb.intervals.keys():
            try:
                t = nwb.intervals[name]
                intervals[name] = (list(t["start_time"].data[:]), list(t["stop_time"].data[:]))
            except Exception:
                pass
    return data, ts, intervals


def _epoch_windows(starts, stops, min_dur=1.0):
    return [(float(t0), float(t1)) for t0, t1 in zip(starts, stops) if t1 - t0 >= min_dur]


def _merge_blocks(starts, stops, max_gap=5.0, min_dur=10.0):
    if not starts:
        return []
    pairs = sorted(zip(starts, stops))
    out, bs, be = [], float(pairs[0][0]), float(pairs[0][1])
    for t0, t1 in pairs[1:]:
        if float(t0) - be <= max_gap:
            be = max(be, float(t1))
        else:
            if be - bs >= min_dur: out.append((bs, be))
            bs, be = float(t0), float(t1)
    if be - bs >= min_dur: out.append((bs, be))
    return out


def _epoch_mask(ts, t0, t1):
    import numpy as np
    return (ts >= t0) & (ts <= t1)


def _zscore(x):
    import numpy as np
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / np.where(sd == 0, 1.0, sd)


def _cross_block(a, b):
    return (a.T @ b) / max(1, len(a) - 1)


def _within_block(x):
    import numpy as np
    c = (x.T @ x) / max(1, len(x) - 1)
    np.fill_diagonal(c, np.nan)
    return c


def _align_exact(a_chunk, a_ts, b_chunk, b_ts):
    import numpy as np
    if not len(a_chunk) or not len(b_chunk):
        return a_chunk[:0], b_chunk[:0], 0.0
    ak = np.round(np.asarray(a_ts, float), _ALIGN_DECIMALS)
    bk = np.round(np.asarray(b_ts, float), _ALIGN_DECIMALS)
    _, ai, bi = np.intersect1d(ak, bk, assume_unique=False, return_indices=True)
    if not len(ai):
        return a_chunk[:0], b_chunk[:0], 0.0
    dt = float(np.max(np.abs(np.asarray(a_ts)[ai] - np.asarray(b_ts)[bi])))
    return a_chunk[ai], b_chunk[bi], dt


def _align_bins(a_chunk, a_ts, b_chunk, b_ts, bin_sec):
    import numpy as np
    if not len(a_chunk) or not len(b_chunk):
        return a_chunk[:0], b_chunk[:0], 0.0
    t0 = max(float(a_ts[0]), float(b_ts[0]))
    t1 = min(float(a_ts[-1]), float(b_ts[-1]))
    if t1 <= t0:
        return a_chunk[:0], b_chunk[:0], 0.0
    edges = np.arange(t0, t1 + bin_sec, bin_sec)
    n = len(edges) - 1
    if n < 1:
        return a_chunk[:0], b_chunk[:0], 0.0
    def _bin(data, ts):
        out = np.zeros((n, data.shape[1]), float)
        for i in range(n):
            m = (ts >= edges[i]) & (ts < edges[i+1])
            if m.sum(): out[i] = data[m].mean(0)
        return out
    return _bin(a_chunk, np.asarray(a_ts, float)), _bin(b_chunk, np.asarray(b_ts, float)), bin_sec


def _window_coupling(a_chunk, b_chunk, seed=42):
    import numpy as np
    an = _zscore(np.asarray(a_chunk, float))
    bn = _zscore(np.asarray(b_chunk, float))
    cb = _cross_block(an, bn)
    rng = np.random.default_rng(seed)
    T = len(bn)
    lo = max(1, min(10, max(1, T // 10)))
    hi = max(lo + 1, T - lo)
    nulls = [_cross_block(an, np.roll(bn, rng.integers(lo, hi), 0)).mean() for _ in range(_N_NULL)]
    null = np.array(nulls)
    z = (cb.mean() - null.mean()) / null.std()
    return {
        "n_bins": T, "mean_r": float(cb.mean()), "max_r": float(cb.max()),
        "min_r": float(cb.min()), "std_r": float(cb.std()),
        "n_pairs": int(cb.size),
        "n_above_02": int((abs(cb) > 0.2).sum()), "n_above_03": int((abs(cb) > 0.3).sum()),
        "null_mean": float(null.mean()), "null_std": float(null.std()),
        "z_vs_null": float(z), "null_means": null.tolist(),
    }


def _within_stat(chunk):
    import numpy as np
    c = _within_block(_zscore(chunk))
    return {"mean_r": float(np.nanmean(c)), "max_r": float(np.nanmax(c))}


def _aggregate(window_stats):
    import numpy as np
    w  = np.array([s["n_bins"] for s in window_stats], float)
    ob = np.array([s["cross"]["mean_r"] for s in window_stats], float)
    nm = np.array([s["cross"]["null_means"] for s in window_stats], float)
    agg_null = np.average(nm, axis=0, weights=w)
    wm = float(np.average(ob, weights=w))
    nm_m, nm_s = float(np.mean(agg_null)), float(np.std(agg_null))
    z = (wm - nm_m) / nm_s if nm_s > 0 else float("nan")
    wa = np.average([s["within_a"]["mean_r"] for s in window_stats], weights=w)
    wb = np.average([s["within_b"]["mean_r"] for s in window_stats], weights=w)
    return {
        "n_windows": len(window_stats), "n_bins": int(w.sum()),
        "max_dt": float(max(s["max_dt"] for s in window_stats)),
        "cross": {"mean_r": wm, "null_mean": nm_m, "null_std": nm_s, "z_vs_null": float(z),
                  "max_r": float(max(s["cross"]["max_r"] for s in window_stats)),
                  "null_means": agg_null.tolist()},
        "within_a": {"mean_r": float(wa)},
        "within_b": {"mean_r": float(wb)},
        "window_stats": window_stats,
    }


def _h3_verdict(res):
    if "spontaneous" not in res:
        return "no_spontaneous"
    sp = res["spontaneous"]
    crs = sp["cross"]["mean_r"]
    z   = sp["cross"]["z_vs_null"]
    wa  = sp["within_a"]["mean_r"]
    wb  = sp["within_b"]["mean_r"]
    if crs > 0 and z > 1.96:
        return "H3_POSITIVE" if crs < min(wa, wb) else "H3_PARTIAL"
    elif crs > 0:
        return "H3_WEAK"
    return "H3_NEGATIVE"


# ---------------------------------------------------------------------------
# Per-pair runner
# ---------------------------------------------------------------------------

def _run_pair(pair: dict, log_lines: list) -> dict:
    import numpy as np
    pid = pair["id"]
    _emit(log_lines, f"\n{'='*60}")
    _emit(log_lines, f"  {pid}  {pair['subject']} / {pair['session']}")
    _emit(log_lines, f"  Pairing : {pair['pairing']}")
    _emit(log_lines, f"  Plane A : {pair['label_a']}")
    _emit(log_lines, f"  Plane B : {pair['label_b']}")
    _emit(log_lines, f"{'='*60}")

    pa = DATA_ROOT / pair["plane_a"]
    pb = DATA_ROOT / pair["plane_b"]
    for lbl, p in [("A", pa), ("B", pb)]:
        if not p.exists():
            _emit(log_lines, f"  MISSING plane {lbl}: {p}")
            return {"id": pid, "error": "missing_files"}

    _emit(log_lines, f"  Loading A ({pa.name}) ...")
    a_data, a_ts, a_ivals = _load_dff_and_intervals(pa)
    _emit(log_lines, f"    shape={a_data.shape}  active={int((a_data.std(0)>0.01).sum())}/{a_data.shape[1]}")

    _emit(log_lines, f"  Loading B ({pb.name}) ...")
    b_data, b_ts, b_ivals = _load_dff_and_intervals(pb)
    _emit(log_lines, f"    shape={b_data.shape}  active={int((b_data.std(0)>0.01).sum())}/{b_data.shape[1]}")

    shared = {k: v for k, v in a_ivals.items() if k in b_ivals}
    _emit(log_lines, f"  Shared intervals: {len(shared)}  (common-conditions subset: {[c for c in _COMMON_CONDITIONS if c in shared]})")

    bin_sec = pair["bin_size_sec"]
    results = {}

    _emit(log_lines, f"\n  {'Condition':<30} {'Win':>4} {'Bins':>6} {'Mean r':>8} {'z':>7} {'Within A':>9} {'Within B':>9}")
    _emit(log_lines, f"  {'-'*30} {'-'*4} {'-'*6} {'-'*8} {'-'*7} {'-'*9} {'-'*9}")

    for cond_key in _COMMON_CONDITIONS:
        if cond_key not in shared:
            _emit(log_lines, f"  {cond_key:<30}: absent")
            continue
        starts, stops = shared[cond_key]
        # Use long windows for spontaneous, blocks for stimuli
        if "spontaneous" in cond_key:
            windows = _epoch_windows(starts, stops, min_dur=1.0)
        else:
            windows = _merge_blocks(starts, stops, max_gap=5.0, min_dur=10.0)
        if not windows:
            _emit(log_lines, f"  {cond_key:<30}: no windows")
            continue

        window_stats = []
        for idx, (t0, t1) in enumerate(windows):
            am = _epoch_mask(a_ts, t0, t1)
            bm = _epoch_mask(b_ts, t0, t1)
            ac, bc = a_data[am], b_data[bm]
            if bin_sec > 0:
                ac, bc, dt = _align_bins(ac, a_ts[am], bc, b_ts[bm], bin_sec)
            else:
                ac, bc, dt = _align_exact(ac, a_ts[am], bc, b_ts[bm])
            if len(ac) < _MIN_BINS_PER_WINDOW or len(bc) < _MIN_BINS_PER_WINDOW:
                continue
            cross  = _window_coupling(ac, bc, seed=42 + idx)
            w_a    = _within_stat(ac)
            w_b    = _within_stat(bc)
            window_stats.append({"n_bins": len(ac), "max_dt": dt,
                                  "cross": cross, "within_a": w_a, "within_b": w_b})

        if not window_stats:
            _emit(log_lines, f"  {cond_key:<30}: no aligned windows >= {_MIN_BINS_PER_WINDOW} bins")
            continue

        agg = _aggregate(window_stats)
        results[cond_key.replace("_presentations", "")] = agg
        _emit(log_lines, (
            f"  {cond_key.replace('_presentations',''):<30} {agg['n_windows']:>4} {agg['n_bins']:>6} "
            f"{agg['cross']['mean_r']:>8.4f} {agg['cross']['z_vs_null']:>7.2f} "
            f"{agg['within_a']['mean_r']:>9.4f} {agg['within_b']['mean_r']:>9.4f}"
        ))

    verdict = _h3_verdict(results)
    _emit(log_lines, f"\n  H3 verdict: {verdict}")
    if "spontaneous" in results:
        sp = results["spontaneous"]
        _emit(log_lines, f"    spontaneous: cross r={sp['cross']['mean_r']:.4f}  z={sp['cross']['z_vs_null']:.2f}"
              f"  within_a={sp['within_a']['mean_r']:.4f}  within_b={sp['within_b']['mean_r']:.4f}")

    return {"id": pid, "subject": pair["subject"], "session": pair["session"],
            "pairing": pair["pairing"], "label_a": pair["label_a"], "label_b": pair["label_b"],
            "h3_verdict": verdict, "conditions": results}


# ---------------------------------------------------------------------------
# Summary table writer
# ---------------------------------------------------------------------------

def _write_summary(path: Path, bundle: list[dict]) -> None:
    lines = [
        "# Full-Bundle Cross-Plane Coupling: DANDI 000336",
        "",
        "## Spontaneous condition summary",
        "",
        "| Pair | Subject | Pairing | Plane A | Plane B | Cross r | z vs null | Within A | Within B | H3 verdict |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for res in bundle:
        if "error" in res:
            lines.append(f"| {res['id']} | â€” | â€” | â€” | â€” | ERROR | â€” | â€” | â€” | â€” |")
            continue
        sp = res["conditions"].get("spontaneous", {})
        cr = sp.get("cross", {})
        wa = sp.get("within_a", {})
        wb = sp.get("within_b", {})
        lines.append(
            f"| {res['id']} | {res['subject']} | {res['pairing']} | "
            f"{res['label_a']} | {res['label_b']} | "
            f"{cr.get('mean_r', float('nan')):.4f} | {cr.get('z_vs_null', float('nan')):.2f} | "
            f"{wa.get('mean_r', float('nan')):.4f} | {wb.get('mean_r', float('nan')):.4f} | "
            f"{res['h3_verdict']} |"
        )
    lines += [
        "",
        "## All-conditions overview",
        "",
        "| Pair | Condition | Windows | Bins | Cross r | z vs null |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for res in bundle:
        if "error" in res: continue
        for cname, cdata in res["conditions"].items():
            lines.append(
                f"| {res['id']} | {cname} | {cdata['n_windows']} | {cdata['n_bins']} | "
                f"{cdata['cross']['mean_r']:.4f} | {cdata['cross']['z_vs_null']:.2f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log_lines: list[str] = []
    TRIAGE_ROOT.mkdir(parents=True, exist_ok=True)

    _emit(log_lines, "=" * 60)
    _emit(log_lines, "Exp 06 â€” Full-Bundle Coupling: DANDI 000336")
    _emit(log_lines, "All 6 NWB files, 3 pairs, 7 common conditions")
    _emit(log_lines, "=" * 60)

    bundle = []
    for pair in _PAIRS:
        res = _run_pair(pair, log_lines)
        bundle.append(res)

    # Cross-pair spontaneous comparison
    _emit(log_lines, "\n" + "="*60)
    _emit(log_lines, "CROSS-PAIR SUMMARY (spontaneous)")
    _emit(log_lines, "="*60)
    _emit(log_lines, f"  {'Pair':<8} {'Subject':<12} {'Pairing':<12} {'Cross r':>8} {'z':>7} {'Within A':>9} {'Within B':>9} {'H3':>15}")
    _emit(log_lines, f"  {'-'*8} {'-'*12} {'-'*12} {'-'*8} {'-'*7} {'-'*9} {'-'*9} {'-'*15}")
    for res in bundle:
        if "error" in res: continue
        sp = res["conditions"].get("spontaneous", {})
        cr = sp.get("cross", {}); wa = sp.get("within_a", {}); wb = sp.get("within_b", {})
        _emit(log_lines, (
            f"  {res['id']:<8} {res['subject']:<12} {res['pairing']:<12} "
            f"{cr.get('mean_r', float('nan')):>8.4f} {cr.get('z_vs_null', float('nan')):>7.2f} "
            f"{wa.get('mean_r', float('nan')):>9.4f} {wb.get('mean_r', float('nan')):>9.4f} "
            f"{res['h3_verdict']:>15}"
        ))

    out_json = TRIAGE_ROOT / "full_bundle_coupling.json"
    out_md   = TRIAGE_ROOT / "full_bundle_coupling.md"
    out_log  = TRIAGE_ROOT / "full_bundle_coupling.log"

    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2, default=str)
    _write_summary(out_md, bundle)
    _emit(log_lines, f"\nJSON    -> {out_json.relative_to(ROOT)}")
    _emit(log_lines, f"Summary -> {out_md.relative_to(ROOT)}")
    _emit(log_lines, f"Log     -> {out_log.relative_to(ROOT)}")
    out_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
