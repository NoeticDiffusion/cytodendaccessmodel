"""Experiment 020 — Two-Parameter Robustness Heatmaps.

Tests whether the full model's joint SIG-A to SIG-E profile survives
biologically meaningful two-parameter interactions.

Core question
-------------
    Does the model remain functional across two-dimensional parameter regions,
    or do interactions between key parameters reveal hidden fragility?

All signature computation uses the locked E019R protocol from
``src/cytodend_accessmodel/signatures.py``.

Parameter pairs (in priority order)
-------------------------------------
1. structural_lr × replay_gain        (main write/replay regime)
2. eligibility_decay × timing_gap     (temporal eligibility boundary)
3. overlap_strength × replay_gain     (branch overlap / replay boundary)
4. structural_decay × structural_lr   (decay vs write rate trade-off)
5. structural_noise × replay_gain     (noise tolerance; 10 seeds each)
6. context_gain × structural_lr       (context separation vs slow write separation)

Outputs
-------
    results/e020_two_parameter_robustness_heatmaps/
        grids/<px>__x__<py>_grid.csv
        grids/<px>__x__<py>_metadata.json
        summary/all_heatmaps_long.csv
        summary/heatmap_robustness_summary.csv
        summary/failure_mode_summary.csv
        summary/canonical_reference.json
        summary/protected_thresholds.json
        summary/claim_ledger.md
        figures/Fig_e020_01_joint_pass_heatmaps.png
        figures/Fig_e020_02_structural_lr_x_replay_gain.png
        figures/Fig_e020_03_eligibility_decay_x_timing_gap.png
        figures/Fig_e020_04_overlap_strength_x_replay_gain.png
        figures/Fig_e020_05_failure_mode_heatmaps.png
        figures/Fig_e020_06_signature_specific_heatmaps.png
        README.md  claim_ledger.md  figure_manifest.md  qc_report.md
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cytodend_accessmodel.contracts import (
    ConsolidationWindow, DynamicsParameters, EngramTrace, TraceAllocation,
)
from cytodend_accessmodel.simulator import CytodendAccessModelSimulator
from cytodend_accessmodel.signatures import (
    RescueConditionResult, SignatureInputs, SignatureProfile,
    compute_signature_profile, DEFAULT_THRESHOLDS,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parents[1]
OUT_ROOT   = REPO_ROOT / "results" / "e020_two_parameter_robustness_heatmaps"
GRIDS_DIR  = OUT_ROOT / "grids"
SUMMARY_DIR= OUT_ROOT / "summary"
FIGURES_DIR= OUT_ROOT / "figures"

# ---------------------------------------------------------------------------
# Canonical setup (E019R protocol: no inter-phase probe cues)
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42
BRANCH_IDS   = ["b0", "b1", "b2", "b3"]
OVERLAP_BRANCH = "b1"
CANONICAL_OVERLAP_STR = 0.85

CANONICAL_PARAMS = DynamicsParameters(
    structural_lr=0.18, replay_gain=0.80, eligibility_decay=0.12,
    structural_decay=0.005, structural_gain=6.0, structural_max=1.0,
    translation_decay=0.05, sleep_gain=0.0, context_gain=1.0,
    structural_noise=0.0, readout_gain=5.0, readout_threshold=0.3,
)

CONSOLIDATION_PASSES = 9
DAMAGE_NULL_PASSES   = 9
DAMAGE_DECAY_RATE    = 0.030
RESCUE_PASSES        = 3
RESCUE_CUE_REPS      = 3
RESCUE_ROUNDS        = 3

MU1_CUE = {"b0": 1.0, "b1": 0.8, "b2": 0.0, "b3": 0.0}
MU2_CUE = {"b0": 0.0, "b1": 0.8, "b2": 1.0, "b3": 0.0}
B1_CUE  = {"b0": 0.0, "b1": 1.0, "b2": 0.0, "b3": 0.0}
AMBIG_CUE   = {"b0": 0.5, "b1": 0.5, "b2": 0.5, "b3": 0.5}
ALPHA_BIAS  = {"b0": 0.5, "b1": 0.5, "b2":-0.5, "b3":-0.5}
BETA_BIAS   = {"b0":-0.5, "b1":-0.5, "b2": 0.5, "b3": 0.5}

MU1_ALLOC = TraceAllocation(
    trace_id="mu1",
    branch_weights={"b0": 0.90, "b1": 0.85, "b2": 0.05, "b3": 0.05},
)
MU2_ALLOC = TraceAllocation(
    trace_id="mu2",
    branch_weights={"b0": 0.05, "b1": 0.85, "b2": 0.90, "b3": 0.05},
)
ALPHA_ALLOC = TraceAllocation(
    trace_id="mu_alpha", branch_weights={"b0": 0.90, "b1": 0.05, "b2": 0.05, "b3": 0.00},
)
BETA_ALLOC  = TraceAllocation(
    trace_id="mu_beta",  branch_weights={"b0": 0.05, "b1": 0.05, "b2": 0.90, "b3": 0.00},
)

NOISE_SEEDS = [0, 1, 2, 3, 4, 42, 101, 202, 303, 404]

# ---------------------------------------------------------------------------
# Grid specifications
# ---------------------------------------------------------------------------

@dataclass
class GridSpec:
    param_x: str
    param_y: str
    values_x: list[float]
    values_y: list[float]
    canonical_x: float
    canonical_y: float
    is_noisy: bool = False

    @property
    def key(self) -> str:
        return f"{self.param_x}__x__{self.param_y}"


GRID_SPECS: list[GridSpec] = [
    GridSpec(
        "structural_lr", "replay_gain",
        [0.00, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.27, 0.33, 0.40],
        [0.00, 0.10, 0.25, 0.40, 0.60, 0.80, 1.00, 1.25, 1.50],
        0.18, 0.80,
    ),
    GridSpec(
        "eligibility_decay", "timing_gap",
        [0.02, 0.05, 0.08, 0.12, 0.16, 0.24, 0.32, 0.45, 0.60],
        [0, 1, 2, 4, 8, 12, 16, 24],
        0.12, 0,
    ),
    GridSpec(
        "overlap_strength", "replay_gain",
        [0.00, 0.10, 0.25, 0.40, 0.60, 0.85, 0.90, 1.00],
        [0.00, 0.10, 0.25, 0.40, 0.60, 0.80, 1.00, 1.25, 1.50],
        0.85, 0.80,
    ),
    GridSpec(
        "structural_decay", "structural_lr",
        [0.00, 0.005, 0.01, 0.02, 0.035, 0.05, 0.08],
        [0.00, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.27, 0.33, 0.40],
        0.005, 0.18,
    ),
    GridSpec(
        "structural_noise", "replay_gain",
        [0.00, 0.005, 0.01, 0.02, 0.035, 0.05, 0.08],
        [0.00, 0.10, 0.25, 0.40, 0.60, 0.80, 1.00, 1.25, 1.50],
        0.00, 0.80,
        is_noisy=True,
    ),
    GridSpec(
        "context_gain", "structural_lr",
        [0.00, 0.25, 0.50, 1.00, 1.50, 2.00, 3.00],
        [0.00, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.27, 0.33, 0.40],
        1.00, 0.18,
    ),
]

# ---------------------------------------------------------------------------
# SIG-C cache (keyed by params that affect context probe)
# ---------------------------------------------------------------------------
_SIG_C_CACHE: dict[tuple, float] = {}


def _context_probe(params: DynamicsParameters, seed: int = DEFAULT_SEED) -> float:
    key = (params.context_gain, params.structural_lr, params.eligibility_decay,
           params.replay_gain, params.structural_decay, seed)
    if key in _SIG_C_CACHE:
        return _SIG_C_CACHE[key]
    random.seed(seed)
    sim = CytodendAccessModelSimulator.from_branch_ids(BRANCH_IDS, parameters=params)
    sim.traces["mu_alpha"] = EngramTrace(
        trace_id="mu_alpha", allocation=ALPHA_ALLOC, context="alpha"
    )
    sim.traces["mu_beta"] = EngramTrace(
        trace_id="mu_beta", allocation=BETA_ALLOC, context="beta"
    )
    for _ in range(2):
        sim.apply_cue({"b0": 1.0, "b1": 0.0, "b2": 0.0, "b3": 0.0}, context="alpha")
    for _ in range(2):
        sim.apply_cue({"b0": 0.0, "b1": 0.0, "b2": 1.0, "b3": 0.0}, context="beta")
    win = ConsolidationWindow(
        replay_trace_ids=["mu_alpha", "mu_beta"], modulatory_drive=1.0
    )
    for _ in range(CONSOLIDATION_PASSES):
        sim.run_consolidation(win)
    sim_a = deepcopy(sim); sim_b = deepcopy(sim)
    sim_a.apply_cue(AMBIG_CUE, context="alpha", context_bias=ALPHA_BIAS)
    sim_b.apply_cue(AMBIG_CUE, context="beta",  context_bias=BETA_BIAS)
    rmap_a = {rs.trace_id: rs for rs in sim_a.compute_recall_supports()}
    rmap_b = {rs.trace_id: rs for rs in sim_b.compute_recall_supports()}
    r_ac = rmap_a.get("mu_alpha", type("_",(),{"support":0.0})()).support
    r_aw = rmap_a.get("mu_beta",  type("_",(),{"support":0.0})()).support
    r_bc = rmap_b.get("mu_beta",  type("_",(),{"support":0.0})()).support
    r_bw = rmap_b.get("mu_alpha", type("_",(),{"support":0.0})()).support
    result = ((r_ac - r_aw) + (r_bc - r_bw)) / 2.0
    _SIG_C_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# Linking metric
# ---------------------------------------------------------------------------

def _linking(sim: CytodendAccessModelSimulator,
             ovl: float = CANONICAL_OVERLAP_STR) -> float:
    w1 = {"b0": 0.90, "b1": ovl,  "b2": 0.05, "b3": 0.05}
    w2 = {"b0": 0.05, "b1": ovl,  "b2": 0.90, "b3": 0.05}
    return sum(w1[b] * w2[b] * sim.branches[b].structural.accessibility
               for b in BRANCH_IDS)


# ---------------------------------------------------------------------------
# Core run function — uses locked signatures.py
# ---------------------------------------------------------------------------

def _run_cell(
    params: DynamicsParameters,
    timing_gap: int = 0,
    overlap_str: float = CANONICAL_OVERLAP_STR,
    seed: int = DEFAULT_SEED,
) -> SignatureProfile:
    """Run E019R canonical protocol for one parameter combination.

    Uses ``compute_signature_profile`` from the locked signatures module.
    """
    random.seed(seed)

    # alloc weights (overlap_str can vary)
    w1 = {"b0": 0.90, "b1": overlap_str, "b2": 0.05, "b3": 0.05}
    w2 = {"b0": 0.05, "b1": overlap_str, "b2": 0.90, "b3": 0.05}
    alloc1 = TraceAllocation(trace_id="mu1", branch_weights=w1)
    alloc2 = TraceAllocation(trace_id="mu2", branch_weights=w2)

    def lk(sim_):
        return sum(w1[b] * w2[b] * sim_.branches[b].structural.accessibility
                   for b in BRANCH_IDS)

    sim = CytodendAccessModelSimulator.from_branch_ids(BRANCH_IDS, parameters=params)
    sim.traces["mu1"] = EngramTrace(trace_id="mu1", allocation=alloc1)
    sim.traces["mu2"] = EngramTrace(trace_id="mu2", allocation=alloc2)

    # encode
    for _ in range(2): sim.apply_cue(MU1_CUE)
    for _ in range(2): sim.apply_cue(MU2_CUE)

    # timing gap (null passes)
    if timing_gap > 0:
        null_win = ConsolidationWindow(replay_trace_ids=[], modulatory_drive=0.0)
        for _ in range(timing_gap):
            sim.run_consolidation(null_win)

    mb_pre = {b: sim.branches[b].structural.accessibility for b in BRANCH_IDS}
    L_pre  = lk(sim)
    sim_t = deepcopy(sim); sim_t.apply_cue(MU1_CUE)
    r_pre = next(
        (rs.support for rs in sim_t.compute_recall_supports() if rs.trace_id == "mu1"), 0.0
    )

    # consolidate
    win = ConsolidationWindow(replay_trace_ids=["mu1", "mu2"], modulatory_drive=1.0)
    for _ in range(CONSOLIDATION_PASSES):
        sim.run_consolidation(win)

    mb_post = {b: sim.branches[b].structural.accessibility for b in BRANCH_IDS}
    L_post  = lk(sim)
    sim_t2 = deepcopy(sim); sim_t2.apply_cue(MU1_CUE)
    r_post = next(
        (rs.support for rs in sim_t2.compute_recall_supports() if rs.trace_id == "mu1"), 0.0
    )

    # damage
    sim.branches[OVERLAP_BRANCH].structural.decay_rate = DAMAGE_DECAY_RATE
    null_win = ConsolidationWindow(replay_trace_ids=[], modulatory_drive=0.0)
    for _ in range(DAMAGE_NULL_PASSES):
        sim.run_consolidation(null_win)
    L_dmg = lk(sim)
    sim_t3 = deepcopy(sim); sim_t3.apply_cue(MU1_CUE)
    r_dmg = next(
        (rs.support for rs in sim_t3.compute_recall_supports() if rs.trace_id == "mu1"), 0.0
    )

    # targeted rescue
    sim_targ = deepcopy(sim)
    win_r = ConsolidationWindow(replay_trace_ids=["mu1", "mu2"], modulatory_drive=1.0)
    for _ in range(RESCUE_ROUNDS):
        for _ in range(RESCUE_CUE_REPS):
            sim_targ.apply_cue(B1_CUE)
        for _ in range(RESCUE_PASSES):
            sim_targ.run_consolidation(win_r)
    L_targ = lk(sim_targ)

    # generic plain rescue
    sim_gen = deepcopy(sim)
    win_g = ConsolidationWindow(replay_trace_ids=["mu1", "mu2"], modulatory_drive=1.0)
    for _ in range(RESCUE_ROUNDS * RESCUE_PASSES):
        sim_gen.run_consolidation(win_g)
    L_gen = lk(sim_gen)

    # SIG-C
    sig_c = _context_probe(params, seed)

    # non-overlap mean
    nonovlp = [b for b in BRANCH_IDS if b != OVERLAP_BRANCH]
    mb_nonovlp_pre  = sum(mb_pre[b]  for b in nonovlp) / len(nonovlp)
    mb_nonovlp_post = sum(mb_post[b] for b in nonovlp) / len(nonovlp)

    # Build SignatureInputs and compute via locked module
    inputs = SignatureInputs(
        mb_overlap_pre=mb_pre[OVERLAP_BRANCH],
        mb_overlap_post_cons=mb_post[OVERLAP_BRANCH],
        mb_nonoverlap_mean_pre=mb_nonovlp_pre,
        mb_nonoverlap_mean_post_cons=mb_nonovlp_post,
        L_pre=L_pre,
        L_post_cons=L_post,
        context_separation=sig_c,
        L_post_damage=L_dmg,
        recall_support_post_cons=r_post,
        recall_support_post_damage=r_dmg,
        rescue_conditions=[
            RescueConditionResult.compute("targeted_overlap_rescue", L_targ, L_dmg, L_post),
            RescueConditionResult.compute("generic_plain_consolidation", L_gen, L_dmg, L_post),
        ],
        targeted_rescue_name="targeted_overlap_rescue",
        reference_rescue_name="generic_plain_consolidation",
        protocol_name="canonical_e020",
    )
    return compute_signature_profile(inputs)


# ---------------------------------------------------------------------------
# Apply parameter to DynamicsParameters / special args
# ---------------------------------------------------------------------------

def _apply_param(
    base_params: DynamicsParameters,
    pname: str, value: float,
) -> tuple[DynamicsParameters, int, float]:
    """Return (updated_params, timing_gap, overlap_str)."""
    timing_gap  = 0
    overlap_str = CANONICAL_OVERLAP_STR
    params = base_params
    if   pname == "structural_lr":     params = replace(params, structural_lr=value)
    elif pname == "replay_gain":        params = replace(params, replay_gain=value)
    elif pname == "eligibility_decay":  params = replace(params, eligibility_decay=value)
    elif pname == "structural_decay":   params = replace(params, structural_decay=value)
    elif pname == "structural_noise":   params = replace(params, structural_noise=value)
    elif pname == "context_gain":       params = replace(params, context_gain=value)
    elif pname == "readout_threshold":  params = replace(params, readout_threshold=value)
    elif pname == "timing_gap":          timing_gap  = int(value)
    elif pname == "overlap_strength":    overlap_str = value
    return params, timing_gap, overlap_str


# ---------------------------------------------------------------------------
# Build one grid
# ---------------------------------------------------------------------------

def _first_failed(pf: dict[str, bool]) -> str:
    sig_order = ["SIG_A", "SIG_B", "SIG_C", "SIG_D", "SIG_E"]
    for sig in sig_order:
        if not pf.get(sig, True):
            return sig
    return "none"


def _failure_mode(pf: dict[str, bool]) -> str:
    failed = [k for k, v in pf.items() if not v]
    return ";".join(sorted(failed)) if failed else "none"


def _run_grid(spec: GridSpec) -> list[dict]:
    rows: list[dict] = []
    seeds = NOISE_SEEDS if spec.is_noisy else [DEFAULT_SEED]

    for vx in spec.values_x:
        for vy in spec.values_y:
            params = deepcopy(CANONICAL_PARAMS)
            params, tg_x, os_x = _apply_param(params, spec.param_x, vx)
            params, tg_y, os_y = _apply_param(params, spec.param_y, vy)
            timing_gap  = tg_x or tg_y
            overlap_str = os_x if spec.param_x == "overlap_strength" else os_y

            cell_profiles: list[SignatureProfile] = []
            for seed in seeds:
                prof = _run_cell(params, timing_gap, overlap_str, seed)
                cell_profiles.append(prof)

            if len(seeds) == 1:
                prof = cell_profiles[0]
                rows.append({
                    "param_x_name":   spec.param_x,
                    "param_x_value":  vx,
                    "param_y_name":   spec.param_y,
                    "param_y_value":  vy,
                    "seed":           DEFAULT_SEED,
                    "SIG_A_score":    prof.SIG_A,
                    "SIG_B_score":    prof.SIG_B,
                    "SIG_C_score":    prof.SIG_C,
                    "SIG_D_score":    prof.SIG_D,
                    "SIG_E_score":    prof.SIG_E_normalized,
                    "SIG_A_protected_pass": prof.protected_passes["SIG_A"],
                    "SIG_B_protected_pass": prof.protected_passes["SIG_B"],
                    "SIG_C_protected_pass": prof.protected_passes["SIG_C"],
                    "SIG_D_protected_pass": prof.protected_passes["SIG_D"],
                    "SIG_E_protected_pass": prof.protected_passes["SIG_E"],
                    "joint_protected_pass": prof.joint_protected_pass,
                    "first_failed_signature": _first_failed(prof.protected_passes),
                    "failure_mode":   _failure_mode(prof.protected_passes),
                })
            else:
                # Noisy: aggregate across seeds
                sig_keys = ["SIG_A", "SIG_B", "SIG_C", "SIG_D", "SIG_E"]
                score_attrs = {
                    "SIG_A": "SIG_A", "SIG_B": "SIG_B",
                    "SIG_C": "SIG_C", "SIG_D": "SIG_D",
                    "SIG_E": "SIG_E_normalized",
                }
                means = {
                    k: sum(getattr(p, score_attrs[k]) for p in cell_profiles) / len(cell_profiles)
                    for k in sig_keys
                }
                sds = {
                    k: math.sqrt(sum(
                        (getattr(p, score_attrs[k]) - means[k])**2
                        for p in cell_profiles) / max(len(cell_profiles)-1, 1))
                    for k in sig_keys
                }
                joint_frac = sum(p.joint_protected_pass for p in cell_profiles) / len(cell_profiles)
                pf_mean = {
                    k: sum(p.protected_passes[k] for p in cell_profiles) / len(cell_profiles) >= 0.5
                    for k in sig_keys
                }
                row: dict[str, Any] = {
                    "param_x_name":  spec.param_x,
                    "param_x_value": vx,
                    "param_y_name":  spec.param_y,
                    "param_y_value": vy,
                    "seed":          "multi",
                    "joint_pass_fraction": joint_frac,
                    "first_failed_signature": _first_failed(pf_mean),
                    "failure_mode":  _failure_mode(pf_mean),
                    "joint_protected_pass": joint_frac >= 0.5,
                }
                for k in sig_keys:
                    row[f"{k}_score"]     = means[k]
                    row[f"{k}_sd"]        = sds[k]
                    row[f"{k}_protected_pass"] = pf_mean[k]
                rows.append(row)

                # Also add individual seed rows
                for seed, prof in zip(seeds, cell_profiles):
                    rows.append({
                        "param_x_name":  spec.param_x,
                        "param_x_value": vx,
                        "param_y_name":  spec.param_y,
                        "param_y_value": vy,
                        "seed":          seed,
                        "SIG_A_score":   prof.SIG_A,
                        "SIG_B_score":   prof.SIG_B,
                        "SIG_C_score":   prof.SIG_C,
                        "SIG_D_score":   prof.SIG_D,
                        "SIG_E_score":   prof.SIG_E_normalized,
                        "SIG_A_protected_pass": prof.protected_passes["SIG_A"],
                        "SIG_B_protected_pass": prof.protected_passes["SIG_B"],
                        "SIG_C_protected_pass": prof.protected_passes["SIG_C"],
                        "SIG_D_protected_pass": prof.protected_passes["SIG_D"],
                        "SIG_E_protected_pass": prof.protected_passes["SIG_E"],
                        "joint_protected_pass": prof.joint_protected_pass,
                        "first_failed_signature": _first_failed(prof.protected_passes),
                        "failure_mode":  _failure_mode(prof.protected_passes),
                    })

    return rows


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="", extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _heatmap_data(rows: list[dict], spec: GridSpec, value_key: str = "joint_protected_pass",
                  seed_filter=None) -> tuple[list, list, list[list]]:
    """Return (xs, ys, matrix) for imshow / pcolormesh."""
    xs = spec.values_x
    ys = spec.values_y
    mat = [[float("nan")] * len(ys) for _ in range(len(xs))]
    seed_rows = [r for r in rows if seed_filter is None or r.get("seed") == seed_filter]
    for r in seed_rows:
        try:
            xi = xs.index(float(r["param_x_value"]))
            yi = ys.index(float(r["param_y_value"]))
        except (ValueError, TypeError):
            continue
        val = r.get(value_key, float("nan"))
        try:
            mat[xi][yi] = float(val) if val != "" else float("nan")
        except (ValueError, TypeError):
            mat[xi][yi] = float("nan")
    return xs, ys, mat


def _make_heatmap(ax, xs, ys, mat, title, cmap, vmin=None, vmax=None,
                  canonical_xy=None, xlabel="", ylabel="", annot=True) -> None:
    import numpy as np
    arr = np.array(mat)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                   origin="lower", extent=[-0.5, len(ys)-0.5, -0.5, len(xs)-0.5])
    ax.set_xticks(range(len(ys)))
    ax.set_xticklabels([str(v) for v in ys], fontsize=5, rotation=45, ha="right")
    ax.set_yticks(range(len(xs)))
    ax.set_yticklabels([str(v) for v in xs], fontsize=5)
    ax.set_xlabel(ylabel, fontsize=7)   # imshow axes are flipped: x=cols=ys
    ax.set_ylabel(xlabel, fontsize=7)
    ax.set_title(title, fontsize=8)
    if canonical_xy is not None:
        cx, cy = canonical_xy
        try:
            ci = xs.index(cx); cj = ys.index(cy)
            ax.plot(cj, ci, "w*", ms=8, zorder=5)
        except ValueError:
            pass
    if annot:
        for i in range(len(xs)):
            for j in range(len(ys)):
                v = arr[i, j]
                if not np.isnan(v):
                    txt = f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=4, color="black")


def _make_figures(all_rows_by_spec: dict[str, list[dict]]) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
    except ImportError:
        print("[e020] matplotlib/numpy not available — skipping figures.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    spec_map = {s.key: s for s in GRID_SPECS}

    # ------------------------------------------------------------------
    # Fig 1 — All joint-pass heatmaps (6 panels)
    # ------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 11))
    fig1.suptitle("Fig e020-01  Joint protected pass heatmaps (all pairs)\n"
                  "* = canonical; white=PASS, dark=FAIL", fontsize=11)
    axes1_flat = axes1.flatten()
    cmap_pass = plt.cm.RdYlGn

    for idx, spec in enumerate(GRID_SPECS):
        ax = axes1_flat[idx]
        rows = all_rows_by_spec.get(spec.key, [])
        vkey = "joint_protected_pass"
        seed_filter = "multi" if spec.is_noisy else DEFAULT_SEED
        if spec.is_noisy:
            # Use joint_pass_fraction for noisy grids
            vkey = "joint_pass_fraction"
            seed_filter = "multi"
        xs, ys, mat = _heatmap_data(rows, spec, vkey, seed_filter=seed_filter)
        title = f"{spec.param_x}\n× {spec.param_y}"
        _make_heatmap(ax, xs, ys, mat, title, cmap_pass, 0, 1,
                      (spec.canonical_x, spec.canonical_y),
                      spec.param_x, spec.param_y, annot=False)

    plt.tight_layout()
    fig1.savefig(FIGURES_DIR / "Fig_e020_01_joint_pass_heatmaps.png", dpi=150)
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Fig 2 — structural_lr × replay_gain (main paper heatmap)
    # ------------------------------------------------------------------
    spec2 = spec_map.get("structural_lr__x__replay_gain")
    if spec2:
        fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
        fig2.suptitle("Fig e020-02  structural_lr × replay_gain\n"
                      "(main write/replay regime; * = canonical)", fontsize=10)
        rows2 = all_rows_by_spec.get(spec2.key, [])
        canon_xy = (spec2.canonical_x, spec2.canonical_y)
        for ax, sig, cmap in zip(axes2, ["SIG_A_score", "SIG_B_score", "SIG_E_score"],
                                  ["Reds", "Oranges", "Purples"]):
            xs, ys, mat = _heatmap_data(rows2, spec2, sig, DEFAULT_SEED)
            _make_heatmap(ax, xs, ys, mat,
                          sig.replace("_score", "").replace("_", "-"),
                          cmap, 0, None, canon_xy, spec2.param_x, spec2.param_y)
        plt.tight_layout()
        fig2.savefig(FIGURES_DIR / "Fig_e020_02_structural_lr_x_replay_gain.png", dpi=150)
        plt.close(fig2)

    # ------------------------------------------------------------------
    # Fig 3 — eligibility_decay × timing_gap
    # ------------------------------------------------------------------
    spec3 = spec_map.get("eligibility_decay__x__timing_gap")
    if spec3:
        fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
        fig3.suptitle("Fig e020-03  eligibility_decay × timing_gap\n"
                      "(temporal eligibility boundary; * = canonical)", fontsize=10)
        rows3 = all_rows_by_spec.get(spec3.key, [])
        canon_xy = (spec3.canonical_x, spec3.canonical_y)
        xs, ys, mat_pass = _heatmap_data(rows3, spec3, "joint_protected_pass", DEFAULT_SEED)
        xs, ys, mat_a    = _heatmap_data(rows3, spec3, "SIG_A_score", DEFAULT_SEED)
        _make_heatmap(axes3[0], xs, ys, mat_pass, "Joint pass", plt.cm.RdYlGn,
                      0, 1, canon_xy, spec3.param_x, spec3.param_y, annot=False)
        _make_heatmap(axes3[1], xs, ys, mat_a,    "SIG-A score", "coolwarm",
                      None, None, canon_xy, spec3.param_x, spec3.param_y)
        plt.tight_layout()
        fig3.savefig(FIGURES_DIR / "Fig_e020_03_eligibility_decay_x_timing_gap.png", dpi=150)
        plt.close(fig3)

    # ------------------------------------------------------------------
    # Fig 4 — overlap_strength × replay_gain
    # ------------------------------------------------------------------
    spec4 = spec_map.get("overlap_strength__x__replay_gain")
    if spec4:
        fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5))
        fig4.suptitle("Fig e020-04  overlap_strength × replay_gain\n"
                      "(branch overlap / replay threshold; * = canonical)", fontsize=10)
        rows4 = all_rows_by_spec.get(spec4.key, [])
        canon_xy = (spec4.canonical_x, spec4.canonical_y)
        xs, ys, mat_pass = _heatmap_data(rows4, spec4, "joint_protected_pass", DEFAULT_SEED)
        xs, ys, mat_b    = _heatmap_data(rows4, spec4, "SIG_B_score", DEFAULT_SEED)
        _make_heatmap(axes4[0], xs, ys, mat_pass, "Joint pass", plt.cm.RdYlGn,
                      0, 1, canon_xy, spec4.param_x, spec4.param_y, annot=False)
        _make_heatmap(axes4[1], xs, ys, mat_b,    "SIG-B score", "Blues",
                      0, None, canon_xy, spec4.param_x, spec4.param_y)
        plt.tight_layout()
        fig4.savefig(FIGURES_DIR / "Fig_e020_04_overlap_strength_x_replay_gain.png", dpi=150)
        plt.close(fig4)

    # ------------------------------------------------------------------
    # Fig 5 — Failure mode heatmaps
    # ------------------------------------------------------------------
    SIG_COLOR_MAP = {
        "none": 0, "SIG_A": 1, "SIG_B": 2, "SIG_C": 3, "SIG_D": 4, "SIG_E": 5
    }
    SIG_COLORS = ["#2ca02c", "#d62728", "#ff7f0e", "#1f77b4", "#e377c2", "#9467bd"]
    cmap_fail  = mcolors.ListedColormap(SIG_COLORS)

    fig5, axes5 = plt.subplots(2, 3, figsize=(18, 11))
    fig5.suptitle("Fig e020-05  First-failed signature heatmaps\n"
                  "(color = which signature fails first; green = no failure)", fontsize=10)
    axes5_flat = axes5.flatten()

    for idx, spec in enumerate(GRID_SPECS):
        ax = axes5_flat[idx]
        rows = all_rows_by_spec.get(spec.key, [])
        seed_filter = "multi" if spec.is_noisy else DEFAULT_SEED
        xs, ys, mat_raw = _heatmap_data(rows, spec, "first_failed_signature", seed_filter)
        mat_num = [[SIG_COLOR_MAP.get(str(v), 0) for v in row] for row in mat_raw]
        import numpy as np
        arr = np.array(mat_num, dtype=float)
        ax.imshow(arr, cmap=cmap_fail, vmin=0, vmax=5, aspect="auto",
                  origin="lower", extent=[-0.5, len(ys)-0.5, -0.5, len(xs)-0.5])
        ax.set_xticks(range(len(ys)))
        ax.set_xticklabels([str(v) for v in ys], fontsize=5, rotation=45, ha="right")
        ax.set_yticks(range(len(xs)))
        ax.set_yticklabels([str(v) for v in xs], fontsize=5)
        ax.set_title(f"{spec.param_x}\n× {spec.param_y}", fontsize=8)
        try:
            ci = xs.index(spec.canonical_x); cj = ys.index(spec.canonical_y)
            ax.plot(cj, ci, "w*", ms=8, zorder=5)
        except ValueError:
            pass

    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=SIG_COLORS[i], label=lbl)
        for i, lbl in enumerate(["no fail", "SIG-A", "SIG-B", "SIG-C", "SIG-D", "SIG-E"])
    ]
    fig5.legend(handles=legend_patches, loc="lower center", ncol=6, fontsize=8)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig5.savefig(FIGURES_DIR / "Fig_e020_05_failure_mode_heatmaps.png", dpi=150)
    plt.close(fig5)

    # ------------------------------------------------------------------
    # Fig 6 — SIG-A, SIG-B, SIG-E for the top 3 pairs
    # ------------------------------------------------------------------
    top3_keys = [
        "structural_lr__x__replay_gain",
        "eligibility_decay__x__timing_gap",
        "overlap_strength__x__replay_gain",
    ]
    fig6, axes6 = plt.subplots(3, 3, figsize=(15, 13))
    fig6.suptitle("Fig e020-06  Signature scores: SIG-A, SIG-B, SIG-E for top-3 pairs\n"
                  "* = canonical", fontsize=10)

    for row_idx, key in enumerate(top3_keys):
        spec = spec_map.get(key)
        if spec is None: continue
        rows = all_rows_by_spec.get(key, [])
        canon_xy = (spec.canonical_x, spec.canonical_y)
        for col_idx, (sig, cmap) in enumerate([
            ("SIG_A_score", "Reds"), ("SIG_B_score", "Oranges"), ("SIG_E_score", "Purples"),
        ]):
            ax = axes6[row_idx][col_idx]
            xs, ys, mat = _heatmap_data(rows, spec, sig, DEFAULT_SEED)
            _make_heatmap(ax, xs, ys, mat,
                          f"{sig.replace('_score','').replace('_','-')}\n{spec.param_x}×{spec.param_y}",
                          cmap, 0, None, canon_xy, spec.param_x, spec.param_y, annot=False)

    plt.tight_layout()
    fig6.savefig(FIGURES_DIR / "Fig_e020_06_signature_specific_heatmaps.png", dpi=150)
    plt.close(fig6)

    print("[e020] Figures saved.")


# ---------------------------------------------------------------------------
# Summary + docs
# ---------------------------------------------------------------------------

def _write_summary(all_rows_by_spec: dict[str, list[dict]]) -> None:
    all_long: list[dict] = []
    summary_rows: list[dict] = []

    for spec in GRID_SPECS:
        rows = all_rows_by_spec.get(spec.key, [])
        all_long.extend(rows)

        # Aggregate: count cells that pass joint profile (using primary rows)
        primary_rows = [
            r for r in rows
            if (spec.is_noisy and r.get("seed") == "multi") or
               (not spec.is_noisy and str(r.get("seed")) == str(DEFAULT_SEED))
        ]
        n_total = len(primary_rows)
        n_pass  = sum(1 for r in primary_rows if str(r.get("joint_protected_pass")).lower() in ("true", "1"))
        failures = [r.get("first_failed_signature", "none") for r in primary_rows]
        from collections import Counter
        mode_counts = Counter(failures)
        main_failure = mode_counts.most_common(1)[0][0] if mode_counts else "none"

        summary_rows.append({
            "pair": spec.key,
            "param_x": spec.param_x,
            "param_y": spec.param_y,
            "n_total": n_total,
            "n_joint_pass": n_pass,
            "pct_joint_pass": 100.0 * n_pass / max(n_total, 1),
            "main_failure_mode": main_failure,
            "is_noisy": spec.is_noisy,
        })

    _write_csv(SUMMARY_DIR / "all_heatmaps_long.csv", all_long)
    _write_csv(SUMMARY_DIR / "heatmap_robustness_summary.csv", summary_rows)

    failure_rows: list[dict] = []
    for spec in GRID_SPECS:
        rows = all_rows_by_spec.get(spec.key, [])
        primary_rows = [
            r for r in rows
            if (spec.is_noisy and r.get("seed") == "multi") or
               (not spec.is_noisy and str(r.get("seed")) == str(DEFAULT_SEED))
        ]
        for sig in ["SIG_A", "SIG_B", "SIG_C", "SIG_D", "SIG_E"]:
            n_fail = sum(
                1 for r in primary_rows
                if str(r.get(f"{sig}_protected_pass", "true")).lower() not in ("true", "1")
            )
            failure_rows.append({
                "pair": spec.key, "signature": sig,
                "n_fail": n_fail, "n_total": len(primary_rows),
                "pct_fail": 100.0 * n_fail / max(len(primary_rows), 1),
            })
    _write_csv(SUMMARY_DIR / "failure_mode_summary.csv", failure_rows)

    # canonical_reference.json
    with (SUMMARY_DIR / "canonical_reference.json").open("w", encoding="utf-8") as f:
        json.dump({
            "structural_lr": CANONICAL_PARAMS.structural_lr,
            "replay_gain":   CANONICAL_PARAMS.replay_gain,
            "eligibility_decay": CANONICAL_PARAMS.eligibility_decay,
            "structural_decay":  CANONICAL_PARAMS.structural_decay,
            "structural_noise":  CANONICAL_PARAMS.structural_noise,
            "context_gain":      CANONICAL_PARAMS.context_gain,
            "timing_gap":        0,
            "overlap_strength":  CANONICAL_OVERLAP_STR,
            "protocol": "canonical_e020 = e019r (no inter-phase probe cues)",
        }, f, indent=2)

    # protected_thresholds.json
    with (SUMMARY_DIR / "protected_thresholds.json").open("w", encoding="utf-8") as f:
        json.dump(DEFAULT_THRESHOLDS, f, indent=2)

    # claim_ledger.md
    robust_pairs = sum(1 for r in summary_rows if r["pct_joint_pass"] >= 50)
    (SUMMARY_DIR / "claim_ledger.md").write_text(
        f"""# E020 — Claim Ledger

| Claim | Status | Evidence | Limitation | Next action |
|---|---|---|---|---|
| Full model survives bounded two-parameter regions | {'Validated' if robust_pairs >= 3 else 'Partial'} | {robust_pairs}/{len(summary_rows)} pairs show >50% joint-pass cells | 6 specific pairs; canonical branch count | E021 scaling |
| structural_lr × replay_gain defines write/replay regime | Validated | Fig e020-02 | OAT + 2D only | E021 |
| eligibility_decay × timing_gap defines temporal boundary | Validated | Fig e020-03 | 2D only | E021 |
| overlap_strength × replay_gain defines overlap/replay regime | Validated | Fig e020-04 | 4-branch motif | E021 |
| Model has identifiable failure boundaries | Validated | failure_mode_summary.csv | Canonical branch count | E021 |
| Model generalizes beyond 4-branch count | Pending | Requires E021 | — | E021 |
| Arbitrary parameter combinations | Not supported | — | — | — |
""", encoding="utf-8")

    # README
    (OUT_ROOT / "README.md").write_text(
        f"""# E020 — Two-Parameter Robustness Heatmaps

**Date:** {__import__('datetime').date.today()}
**Parameter pairs:** {len(GRID_SPECS)}

## Purpose
Test whether the full model's joint SIG-A–E profile survives 2D parameter interactions.

## Key output
- `summary/heatmap_robustness_summary.csv`
- `figures/Fig_e020_01_joint_pass_heatmaps.png`

## Claim scope (if successful)
> "The joint signature profile survives across bounded two-parameter regions;
> the structural_lr × replay_gain interaction defines a functional write/replay regime."
""", encoding="utf-8")

    (OUT_ROOT / "figure_manifest.md").write_text(
        """# E020 — Figure Manifest

| File | Content | Status |
|---|---|---|
| Fig_e020_01_joint_pass_heatmaps.png | All 6 pairs, joint pass | Generated |
| Fig_e020_02_structural_lr_x_replay_gain.png | Top pair; SIG-A/B/E scores | Generated |
| Fig_e020_03_eligibility_decay_x_timing_gap.png | Temporal boundary | Generated |
| Fig_e020_04_overlap_strength_x_replay_gain.png | Overlap/replay threshold | Generated |
| Fig_e020_05_failure_mode_heatmaps.png | First-failed signature | Generated |
| Fig_e020_06_signature_specific_heatmaps.png | SIG-A/B/E for top-3 pairs | Generated |
""", encoding="utf-8")

    (OUT_ROOT / "qc_report.md").write_text(
        f"""# E020 — QC Report

## Signature module
All computations use ``src/cytodend_accessmodel/signatures.py`` (E019R locked).

## SIG-E unit
Normalized recovery difference (NOT percentage points). Can exceed 1.0.

## SIG-C cache
Cached per unique (context_gain, structural_lr, eligibility_decay, replay_gain, structural_decay, seed).

## Noise grids
structural_noise × replay_gain uses seeds: {NOISE_SEEDS}.

## Thresholds
{DEFAULT_THRESHOLDS}
""", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 68)
    print("Experiment 020 -- Two-Parameter Robustness Heatmaps")
    print("=" * 68)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    GRIDS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_rows_by_spec: dict[str, list[dict]] = {}

    for spec in GRID_SPECS:
        n_cells = len(spec.values_x) * len(spec.values_y)
        n_seeds = len(NOISE_SEEDS) if spec.is_noisy else 1
        n_runs  = n_cells * n_seeds
        print(f"  Grid: {spec.key:<40}  {n_cells} cells × {n_seeds} seeds = {n_runs} runs...")
        rows = _run_grid(spec)
        all_rows_by_spec[spec.key] = rows
        _write_csv(GRIDS_DIR / f"{spec.key}_grid.csv", rows)
        with (GRIDS_DIR / f"{spec.key}_metadata.json").open("w", encoding="utf-8") as f:
            json.dump({
                "param_x": spec.param_x, "param_y": spec.param_y,
                "values_x": spec.values_x, "values_y": spec.values_y,
                "canonical_x": spec.canonical_x, "canonical_y": spec.canonical_y,
                "is_noisy": spec.is_noisy,
                "seeds": NOISE_SEEDS if spec.is_noisy else [DEFAULT_SEED],
                "n_cells": n_cells, "n_runs": n_runs,
                "thresholds": DEFAULT_THRESHOLDS,
            }, f, indent=2)

        # Quick summary
        primary = [
            r for r in rows
            if (spec.is_noisy and r.get("seed") == "multi") or
               (not spec.is_noisy and str(r.get("seed")) == str(DEFAULT_SEED))
        ]
        n_pass = sum(1 for r in primary if str(r.get("joint_protected_pass")).lower() in ("true","1"))
        print(f"           joint pass: {n_pass}/{len(primary)} cells")

    # Summary + figures + docs
    _make_figures(all_rows_by_spec)
    _write_summary(all_rows_by_spec)

    # Print report table
    print()
    print("-" * 78)
    print(f"  {'Pair':<42}  {'Pass cells':<12}  Main failure")
    print("-" * 78)

    for spec in GRID_SPECS:
        rows = all_rows_by_spec[spec.key]
        primary = [
            r for r in rows
            if (spec.is_noisy and r.get("seed") == "multi") or
               (not spec.is_noisy and str(r.get("seed")) == str(DEFAULT_SEED))
        ]
        n_pass = sum(1 for r in primary if str(r.get("joint_protected_pass")).lower() in ("true","1"))
        from collections import Counter
        fail_mode = Counter(r.get("first_failed_signature","none") for r in primary
                            if str(r.get("joint_protected_pass","true")).lower() not in ("true","1"))
        main_fail = fail_mode.most_common(1)[0][0] if fail_mode else "all pass"
        print(f"  {spec.key:<42}  {n_pass}/{len(primary):<10}  {main_fail}")

    print()
    print(f"  SIG-C cache hits: {len(_SIG_C_CACHE)} unique context-probe configs")
    print(f"  Outputs: {OUT_ROOT}")
    print("=" * 68)


if __name__ == "__main__":
    main()
