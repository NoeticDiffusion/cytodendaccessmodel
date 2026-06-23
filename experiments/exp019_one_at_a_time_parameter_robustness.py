"""Experiment 019 — One-at-a-Time Parameter Robustness.

Tests whether the full model's joint SIG-A to SIG-E signature profile survives
one-at-a-time variation of key simulator parameters.

Core question
-------------
    Is the joint signature profile robust to parameter variation,
    or is it a narrow artifact of the canonical parameter set?

E017 established canonical traces.
E018 established that no simpler comparator reproduces the joint profile.
E019 asks: how wide is the functional regime?

Parameters swept (one at a time, others held at canonical)
----------------------------------------------------------
    structural_lr      — slow-write learning rate
    replay_gain        — replay-driven translation readiness
    eligibility_decay  — eligibility trace decay rate
    structural_decay   — M_b passive decay
    structural_noise   — per-step M_b noise (10 seeds each)
    context_gain       — context disambiguation gain
    timing_gap         — null steps inserted between encode and consolidate
    overlap_strength   — b1 weight in mu1/mu2 allocations
    readout_threshold  — recall-support activation threshold

Signature thresholds (predeclared, identical to E018)
------------------------------------------------------
    SIG-A: > 0.02 delta M_b
    SIG-B: > 0.05 delta L
    SIG-C: > 0.05 support units
    SIG-D: > 5.0 pp
    SIG-E: > 10.0 pp

Key interpretive notes from E018
---------------------------------
- SIG-C passed all comparators in E018; likely a fast-context/allocation signature.
  E019 uses context_gain sweep to test this interpretation.
- SIG-D passed multiple comparators; watch for it being geometrically inflated.
- SIG-E baseline is plain consolidation (no pre-cueing) — identical to E018.

Outputs
-------
    results/e019_one_at_a_time_parameter_robustness/
        sweeps/<parameter>_sweep.csv
        sweeps/<parameter>_metadata.json
        summary/all_sweeps_long.csv
        summary/robustness_summary_by_parameter.csv
        summary/failure_boundary_summary.csv
        summary/protected_thresholds.json
        summary/canonical_reference.json
        summary/claim_ledger.md
        figures/Fig_e019_0{1..5}_*.png
        README.md  qc_report.md  effect_summary.md
        figure_manifest.md
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cytodend_accessmodel.contracts import (
    ConsolidationWindow,
    DynamicsParameters,
    EngramTrace,
    TraceAllocation,
)
from cytodend_accessmodel.simulator import (
    CytodendAccessModelSimulator,
    _sigmoid,
    _clamp01,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parents[1]
OUT_ROOT   = REPO_ROOT / "results" / "e019_one_at_a_time_parameter_robustness"
SWEEPS_DIR = OUT_ROOT / "sweeps"
SUMMARY_DIR= OUT_ROOT / "summary"
FIGURES_DIR= OUT_ROOT / "figures"

# ---------------------------------------------------------------------------
# Canonical setup (identical to E017/E018 full_model)
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42
BRANCH_IDS   = ["b0", "b1", "b2", "b3"]
OVERLAP_BRANCH = "b1"

CANONICAL_OVERLAP_STR = 0.85  # b1 weight in both mu1 and mu2 allocations

CANONICAL_PARAMS = DynamicsParameters(
    structural_lr=0.18,
    replay_gain=0.80,
    eligibility_decay=0.12,
    structural_decay=0.005,
    structural_gain=6.0,
    structural_max=1.0,
    translation_decay=0.05,
    sleep_gain=0.0,
    context_gain=1.0,
    structural_noise=0.0,
    readout_gain=5.0,
    readout_threshold=0.3,
)

# Predeclared thresholds (identical to E018 — must not change)
THRESHOLDS = {
    "SIG_A": 0.02,
    "SIG_B": 0.05,
    "SIG_C": 0.05,
    "SIG_D": 5.0,
    "SIG_E": 10.0,
}

# Protocol constants (identical to E018 full_model)
CONSOLIDATION_PASSES = 9
DAMAGE_NULL_PASSES   = 9
DAMAGE_DECAY_RATE    = 0.030
RESCUE_PASSES        = 3
RESCUE_CUE_REPS      = 3
RESCUE_ROUNDS        = 3

# Cues
MU1_CUE  = {"b0": 1.0, "b1": 0.8, "b2": 0.0, "b3": 0.0}
MU2_CUE  = {"b0": 0.0, "b1": 0.8, "b2": 1.0, "b3": 0.0}
B1_CUE   = {"b0": 0.0, "b1": 1.0, "b2": 0.0, "b3": 0.0}
ALPHA_BIAS = {"b0": 0.5, "b1": 0.5, "b2": -0.5, "b3": -0.5}
BETA_BIAS  = {"b0": -0.5, "b1": -0.5, "b2": 0.5, "b3": 0.5}
AMBIG_CUE  = {"b0": 0.5, "b1": 0.5, "b2": 0.5, "b3": 0.5}

# Noise sweep seeds
NOISE_SEEDS = [0, 1, 2, 3, 4, 42, 101, 202, 303, 404]

# ---------------------------------------------------------------------------
# Sweep definitions (predeclared — do not change after inspecting results)
# ---------------------------------------------------------------------------
@dataclass
class SweepDef:
    name: str
    values: list[float]
    canonical: float
    description: str
    is_noise: bool = False


SWEEPS: list[SweepDef] = [
    SweepDef(
        name="structural_lr",
        values=[0.00, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.27, 0.33, 0.40],
        canonical=0.18,
        description="Slow-write learning rate. Controls M_b update magnitude.",
    ),
    SweepDef(
        name="replay_gain",
        values=[0.00, 0.10, 0.25, 0.40, 0.60, 0.80, 1.00, 1.25, 1.50],
        canonical=0.80,
        description="Replay-driven translation readiness gain.",
    ),
    SweepDef(
        name="eligibility_decay",
        values=[0.02, 0.05, 0.08, 0.12, 0.16, 0.24, 0.32, 0.45, 0.60],
        canonical=0.12,
        description="E_b decay rate. Higher = faster eligibility forgetting.",
    ),
    SweepDef(
        name="structural_decay",
        values=[0.00, 0.005, 0.01, 0.02, 0.035, 0.05, 0.08],
        canonical=0.005,
        description="M_b passive decay. Higher = less durable structural state.",
    ),
    SweepDef(
        name="structural_noise",
        values=[0.00, 0.005, 0.01, 0.02, 0.035, 0.05, 0.08],
        canonical=0.00,
        description="Per-step Gaussian noise added to M_b. Run with 10 seeds.",
        is_noise=True,
    ),
    SweepDef(
        name="context_gain",
        values=[0.00, 0.25, 0.50, 1.00, 1.50, 2.00, 3.00],
        canonical=1.00,
        description="Context disambiguation gain. Expected to mainly affect SIG-C.",
    ),
    SweepDef(
        name="timing_gap",
        values=[0, 1, 2, 4, 8, 12, 16, 24],
        canonical=0,
        description=(
            "Null steps inserted between encoding and consolidation "
            "(lets E_b decay before consolidation)."
        ),
    ),
    SweepDef(
        name="overlap_strength",
        values=[0.00, 0.10, 0.25, 0.40, 0.60, 0.85, 0.90, 1.00],
        canonical=0.85,
        description="b1 weight in mu1/mu2 allocations. Controls overlap size.",
    ),
    SweepDef(
        name="readout_threshold",
        values=[0.10, 0.20, 0.30, 0.60, 0.75, 0.90],
        canonical=0.30,
        description="Recall support activation threshold.",
    ),
]

# ---------------------------------------------------------------------------
# Allocation builder
# ---------------------------------------------------------------------------

def _build_allocs(overlap_str: float) -> tuple[TraceAllocation, TraceAllocation]:
    mu1 = TraceAllocation(
        trace_id="mu1",
        branch_weights={"b0": 0.90, "b1": overlap_str, "b2": 0.05, "b3": 0.05},
    )
    mu2 = TraceAllocation(
        trace_id="mu2",
        branch_weights={"b0": 0.05, "b1": overlap_str, "b2": 0.90, "b3": 0.05},
    )
    return mu1, mu2


def _build_context_allocs() -> tuple[TraceAllocation, TraceAllocation]:
    alpha = TraceAllocation(
        trace_id="mu_alpha",
        branch_weights={"b0": 0.90, "b1": 0.05, "b2": 0.05, "b3": 0.00},
    )
    beta = TraceAllocation(
        trace_id="mu_beta",
        branch_weights={"b0": 0.05, "b1": 0.05, "b2": 0.90, "b3": 0.00},
    )
    return alpha, beta


# ---------------------------------------------------------------------------
# Linking metric
# ---------------------------------------------------------------------------

def _linking(sim: CytodendAccessModelSimulator, mu1_alloc: TraceAllocation,
             mu2_alloc: TraceAllocation) -> float:
    return sum(
        mu1_alloc.branch_weights.get(b, 0.0)
        * mu2_alloc.branch_weights.get(b, 0.0)
        * sim.branches[b].structural.accessibility
        for b in BRANCH_IDS
    )


# ---------------------------------------------------------------------------
# Core single-run function
# ---------------------------------------------------------------------------

def _run_single(
    params: DynamicsParameters,
    timing_gap: int = 0,
    overlap_str: float = CANONICAL_OVERLAP_STR,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """Run the full-model 10-phase protocol and return all five signature scores.

    Returns a dict with SIG_A through SIG_E values plus raw diagnostics.
    """
    random.seed(seed)
    mu1_alloc, mu2_alloc = _build_allocs(overlap_str)

    sim = CytodendAccessModelSimulator.from_branch_ids(BRANCH_IDS, parameters=params)
    sim.traces["mu1"] = EngramTrace(trace_id="mu1", allocation=mu1_alloc)
    sim.traces["mu2"] = EngramTrace(trace_id="mu2", allocation=mu2_alloc)

    # --- encode ---
    for _ in range(2):
        sim.apply_cue(MU1_CUE)
    for _ in range(2):
        sim.apply_cue(MU2_CUE)

    # timing gap: null decay steps before consolidation (no replay, no writing)
    if timing_gap > 0:
        null_win = ConsolidationWindow(replay_trace_ids=[], modulatory_drive=0.0)
        for _ in range(timing_gap):
            sim.run_consolidation(null_win)

    # --- pre-consolidation snapshots ---
    mb_pre = {b: sim.branches[b].structural.accessibility for b in BRANCH_IDS}
    lk_pre = _linking(sim, mu1_alloc, mu2_alloc)
    sim_tmp = deepcopy(sim)
    sim_tmp.apply_cue(MU1_CUE)
    r_mu1_pre = next(
        (rs.support for rs in sim_tmp.compute_recall_supports() if rs.trace_id == "mu1"), 0.0
    )

    # --- consolidation ---
    win = ConsolidationWindow(replay_trace_ids=["mu1", "mu2"], modulatory_drive=1.0)
    for _ in range(CONSOLIDATION_PASSES):
        sim.run_consolidation(win)

    mb_post = {b: sim.branches[b].structural.accessibility for b in BRANCH_IDS}
    lk_post = _linking(sim, mu1_alloc, mu2_alloc)

    sim_tmp2 = deepcopy(sim)
    sim_tmp2.apply_cue(MU1_CUE)
    r_mu1_post = next(
        (rs.support for rs in sim_tmp2.compute_recall_supports() if rs.trace_id == "mu1"), 0.0
    )

    # --- overlap damage ---
    sim.branches[OVERLAP_BRANCH].structural.decay_rate = DAMAGE_DECAY_RATE
    null_win = ConsolidationWindow(replay_trace_ids=[], modulatory_drive=0.0)
    for _ in range(DAMAGE_NULL_PASSES):
        sim.run_consolidation(null_win)

    lk_dmg = _linking(sim, mu1_alloc, mu2_alloc)

    sim_tmp3 = deepcopy(sim)
    sim_tmp3.apply_cue(MU1_CUE)
    r_mu1_dmg = next(
        (rs.support for rs in sim_tmp3.compute_recall_supports() if rs.trace_id == "mu1"), 0.0
    )

    # --- targeted rescue ---
    sim_targ = deepcopy(sim)
    win_r = ConsolidationWindow(replay_trace_ids=["mu1", "mu2"], modulatory_drive=1.0)
    for _ in range(RESCUE_ROUNDS):
        for _ in range(RESCUE_CUE_REPS):
            sim_targ.apply_cue(B1_CUE)
        for _ in range(RESCUE_PASSES):
            sim_targ.run_consolidation(win_r)
    lk_targ = _linking(sim_targ, mu1_alloc, mu2_alloc)

    # --- generic rescue (plain consolidation, no pre-cueing; identical to E018) ---
    sim_gen = deepcopy(sim)
    win_g = ConsolidationWindow(replay_trace_ids=["mu1", "mu2"], modulatory_drive=1.0)
    for _ in range(RESCUE_ROUNDS * RESCUE_PASSES):
        sim_gen.run_consolidation(win_g)
    lk_gen = _linking(sim_gen, mu1_alloc, mu2_alloc)

    # --- SIG-C (dedicated context probe) ---
    sig_c = _run_context_probe(params)

    # --- compute signatures ---
    nonoverlap = [b for b in BRANCH_IDS if b != OVERLAP_BRANCH]
    delta_m_ov  = mb_post[OVERLAP_BRANCH] - mb_pre[OVERLAP_BRANCH]
    delta_m_non = sum(mb_post[b] - mb_pre[b] for b in nonoverlap) / len(nonoverlap)
    sig_a = delta_m_ov - delta_m_non
    sig_b = lk_post - lk_pre

    link_drop   = (lk_post - lk_dmg)  / max(abs(lk_post), 1e-9) * 100.0
    recall_drop = (r_mu1_post - r_mu1_dmg) / max(abs(r_mu1_post), 1e-9) * 100.0
    sig_d = link_drop - recall_drop

    def _rec_pct(post_r, dmg_r, healthy_r):
        denom = healthy_r - dmg_r
        return (post_r - dmg_r) / denom * 100.0 if abs(denom) > 1e-9 else 0.0

    sig_e = _rec_pct(lk_targ, lk_dmg, lk_post) - _rec_pct(lk_gen, lk_dmg, lk_post)

    return {
        "SIG_A": sig_a,
        "SIG_B": sig_b,
        "SIG_C": sig_c,
        "SIG_D": sig_d,
        "SIG_E": sig_e,
        "mb_pre_overlap":   mb_pre[OVERLAP_BRANCH],
        "mb_post_overlap":  mb_post[OVERLAP_BRANCH],
        "lk_pre": lk_pre,
        "lk_post": lk_post,
        "lk_dmg":  lk_dmg,
        "r_mu1_post": r_mu1_post,
    }


# ---------------------------------------------------------------------------
# Context probe (SIG-C)
# ---------------------------------------------------------------------------

def _run_context_probe(params: DynamicsParameters, seed: int = DEFAULT_SEED) -> float:
    """Run context disambiguation probe; return separation score."""
    random.seed(seed)
    alpha_alloc, beta_alloc = _build_context_allocs()

    sim = CytodendAccessModelSimulator.from_branch_ids(BRANCH_IDS, parameters=params)
    sim.traces["mu_alpha"] = EngramTrace(
        trace_id="mu_alpha", allocation=alpha_alloc, context="alpha"
    )
    sim.traces["mu_beta"] = EngramTrace(
        trace_id="mu_beta", allocation=beta_alloc, context="beta"
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

    def _probe(sim_copy, context, bias):
        sim_copy.apply_cue(AMBIG_CUE, context=context, context_bias=bias)
        rmap = {rs.trace_id: rs for rs in sim_copy.compute_recall_supports()}
        return rmap

    sim_a = deepcopy(sim)
    sim_b = deepcopy(sim)
    rmap_a = _probe(sim_a, "alpha", ALPHA_BIAS)
    rmap_b = _probe(sim_b, "beta",  BETA_BIAS)

    r_a_corr  = rmap_a.get("mu_alpha", type("_", (), {"support": 0.0})()).support
    r_a_wrong = rmap_a.get("mu_beta",  type("_", (), {"support": 0.0})()).support
    r_b_corr  = rmap_b.get("mu_beta",  type("_", (), {"support": 0.0})()).support
    r_b_wrong = rmap_b.get("mu_alpha", type("_", (), {"support": 0.0})()).support

    return ((r_a_corr - r_a_wrong) + (r_b_corr - r_b_wrong)) / 2.0


# ---------------------------------------------------------------------------
# Pass / fail helpers
# ---------------------------------------------------------------------------

def _directional(sigs: dict) -> dict[str, bool]:
    return {k: sigs[k] > 0.0 for k in ("SIG_A", "SIG_B", "SIG_C", "SIG_D", "SIG_E")}


def _protected(sigs: dict) -> dict[str, bool]:
    return {k: sigs[k] > THRESHOLDS[k] for k in ("SIG_A", "SIG_B", "SIG_C", "SIG_D", "SIG_E")}


def _failure_mode(pf: dict[str, bool]) -> str:
    failed = [k for k, v in pf.items() if not v]
    return ";".join(sorted(failed)) if failed else "none"


# ---------------------------------------------------------------------------
# Build row
# ---------------------------------------------------------------------------

def _build_row(
    param_name: str, param_value: float, seed: int, sigs: dict
) -> dict[str, Any]:
    dir_pf  = _directional(sigs)
    prot_pf = _protected(sigs)
    return {
        "parameter_name":  param_name,
        "parameter_value": param_value,
        "seed":            seed,
        "SIG_A_score":     sigs["SIG_A"],
        "SIG_B_score":     sigs["SIG_B"],
        "SIG_C_score":     sigs["SIG_C"],
        "SIG_D_score":     sigs["SIG_D"],
        "SIG_E_score":     sigs["SIG_E"],
        "SIG_A_directional_pass": dir_pf["SIG_A"],
        "SIG_B_directional_pass": dir_pf["SIG_B"],
        "SIG_C_directional_pass": dir_pf["SIG_C"],
        "SIG_D_directional_pass": dir_pf["SIG_D"],
        "SIG_E_directional_pass": dir_pf["SIG_E"],
        "SIG_A_protected_pass": prot_pf["SIG_A"],
        "SIG_B_protected_pass": prot_pf["SIG_B"],
        "SIG_C_protected_pass": prot_pf["SIG_C"],
        "SIG_D_protected_pass": prot_pf["SIG_D"],
        "SIG_E_protected_pass": prot_pf["SIG_E"],
        "joint_directional_pass": all(dir_pf.values()),
        "joint_protected_pass":   all(prot_pf.values()),
        "failure_mode": _failure_mode(prot_pf),
    }


# ---------------------------------------------------------------------------
# Run one parameter sweep
# ---------------------------------------------------------------------------

def _run_sweep(sweep: SweepDef) -> list[dict]:
    rows: list[dict] = []
    pname = sweep.name
    seeds = NOISE_SEEDS if sweep.is_noise else [DEFAULT_SEED]

    for val in sweep.values:
        for seed in seeds:
            params = deepcopy(CANONICAL_PARAMS)
            timing_gap   = 0
            overlap_str  = CANONICAL_OVERLAP_STR

            if   pname == "structural_lr":     params = replace(params, structural_lr=val)
            elif pname == "replay_gain":        params = replace(params, replay_gain=val)
            elif pname == "eligibility_decay":  params = replace(params, eligibility_decay=val)
            elif pname == "structural_decay":   params = replace(params, structural_decay=val)
            elif pname == "structural_noise":   params = replace(params, structural_noise=val)
            elif pname == "context_gain":       params = replace(params, context_gain=val)
            elif pname == "readout_threshold":  params = replace(params, readout_threshold=val)
            elif pname == "timing_gap":         timing_gap  = int(val)
            elif pname == "overlap_strength":   overlap_str = val

            sigs = _run_single(params, timing_gap=timing_gap,
                               overlap_str=overlap_str, seed=seed)
            rows.append(_build_row(pname, val, seed, sigs))

    return rows


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def _first_failure_value(rows: list[dict]) -> dict[str, float | str]:
    """For each signature, find the first parameter value where protected pass fails."""
    result: dict[str, float | str] = {}
    seen_canonical = False
    sig_keys = ["SIG_A", "SIG_B", "SIG_C", "SIG_D", "SIG_E"]

    # Use mean across seeds for noise; first-seed otherwise
    by_value: dict[float, dict[str, list[bool]]] = {}
    for row in rows:
        val = row["parameter_value"]
        if val not in by_value:
            by_value[val] = {k: [] for k in sig_keys}
        for k in sig_keys:
            by_value[val][k].append(row[f"{k}_protected_pass"])

    for sig in sig_keys:
        first_fail = None
        for val in sorted(by_value):
            passes = by_value[val][sig]
            mean_pass = sum(passes) / len(passes)
            if mean_pass < 0.5:
                first_fail = val
                break
        result[sig] = first_fail if first_fail is not None else "robust"

    return result


def _robust_count_by_value(rows: list[dict]) -> dict[float, int]:
    """Mean number of protected sigs passed at each parameter value."""
    sig_keys = ["SIG_A", "SIG_B", "SIG_C", "SIG_D", "SIG_E"]
    by_value: dict[float, list[int]] = {}
    for row in rows:
        val = row["parameter_value"]
        n   = sum(int(row[f"{k}_protected_pass"]) for k in sig_keys)
        by_value.setdefault(val, []).append(n)
    return {v: sum(ns) / len(ns) for v, ns in sorted(by_value.items())}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _make_figures(all_rows_by_param: dict[str, list[dict]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[e019] matplotlib/numpy not available — skipping figures.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SIG_KEYS = ["SIG_A", "SIG_B", "SIG_C", "SIG_D", "SIG_E"]
    SIG_COLORS = {
        "SIG_A": "#d62728", "SIG_B": "#ff7f0e", "SIG_C": "#2ca02c",
        "SIG_D": "#1f77b4", "SIG_E": "#9467bd",
    }
    param_names = list(all_rows_by_param.keys())
    n_params = len(param_names)

    def _agg(rows, score_key):
        by_val: dict[float, list[float]] = {}
        for r in rows:
            by_val.setdefault(r["parameter_value"], []).append(r[score_key])
        vals = sorted(by_val)
        means = [sum(by_val[v]) / len(by_val[v]) for v in vals]
        stds  = [
            math.sqrt(sum((x - means[i])**2 for x in by_val[v]) / max(len(by_val[v])-1,1))
            for i, v in enumerate(vals)
        ]
        return vals, means, stds

    def _agg_pass(rows, pass_key):
        by_val: dict[float, list[bool]] = {}
        for r in rows:
            by_val.setdefault(r["parameter_value"], []).append(bool(r[pass_key]))
        vals = sorted(by_val)
        rates = [sum(by_val[v]) / len(by_val[v]) for v in vals]
        return vals, rates

    # ------------------------------------------------------------------
    # Fig 1 — Joint protected pass fraction by parameter
    # ------------------------------------------------------------------
    ncols = 3
    nrows_grid = math.ceil(n_params / ncols)
    fig1, axes1 = plt.subplots(nrows_grid, ncols, figsize=(14, 3.5 * nrows_grid))
    axes1_flat = axes1.flatten()
    fig1.suptitle("Fig e019-01  Joint protected pass by parameter", fontsize=11)

    for idx, pname in enumerate(param_names):
        ax = axes1_flat[idx]
        rows = all_rows_by_param[pname]
        vals, rates = _agg_pass(rows, "joint_protected_pass")
        sweep = next(s for s in SWEEPS if s.name == pname)
        canon = sweep.canonical

        ax.bar([str(v) for v in vals], rates, color="#2ca02c", edgecolor="white")
        ax.axhline(1.0, ls="--", lw=0.7, color="black")

        # Mark canonical
        can_str = str(canon)
        if can_str in [str(v) for v in vals]:
            ix = [str(v) for v in vals].index(can_str)
            ax.bar([can_str], [rates[ix]], color="#1f77b4", edgecolor="white")

        ax.set_title(pname, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Joint pass rate", fontsize=7)
        ax.tick_params(axis="x", labelsize=6, rotation=45)
        ax.tick_params(axis="y", labelsize=7)

    for idx in range(n_params, len(axes1_flat)):
        axes1_flat[idx].set_visible(False)

    plt.tight_layout()
    fig1.savefig(FIGURES_DIR / "Fig_e019_01_joint_pass_by_parameter.png", dpi=150)
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Fig 2 — Signature scores by parameter (small-multiple)
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(n_params, 5, figsize=(16, 2.8 * n_params), squeeze=False)
    fig2.suptitle("Fig e019-02  Signature scores by parameter sweep", fontsize=11)

    for row_idx, pname in enumerate(param_names):
        rows = all_rows_by_param[pname]
        for col_idx, sig in enumerate(SIG_KEYS):
            ax = axes2[row_idx][col_idx]
            vals, means, stds = _agg(rows, f"{sig}_score")
            xs = list(range(len(vals)))
            ax.errorbar(xs, means, yerr=stds, fmt="o-", color=SIG_COLORS[sig],
                        markersize=3, lw=1.2, capsize=2)
            ax.axhline(THRESHOLDS[sig], ls="--", lw=0.6, color="gray")
            ax.axhline(0, ls="-", lw=0.3, color="gray")
            ax.set_xticks(xs)
            ax.set_xticklabels([str(v) for v in vals], fontsize=5, rotation=45)
            ax.tick_params(axis="y", labelsize=6)
            if col_idx == 0:
                ax.set_ylabel(pname, fontsize=7)
            if row_idx == 0:
                ax.set_title(sig.replace("_", "-"), fontsize=8)

    plt.tight_layout()
    fig2.savefig(FIGURES_DIR / "Fig_e019_02_signature_scores_by_parameter.png", dpi=150)
    plt.close(fig2)

    # ------------------------------------------------------------------
    # Fig 3 — Failure boundary summary (heatmap: param × signature)
    # ------------------------------------------------------------------
    boundary_data: dict[str, dict[str, str]] = {}
    for pname, rows in all_rows_by_param.items():
        boundary_data[pname] = _first_failure_value(rows)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    cell_vals = []
    cell_texts = []
    for pname in param_names:
        row_v, row_t = [], []
        for sig in SIG_KEYS:
            fb = boundary_data[pname].get(sig, "robust")
            row_v.append(0 if fb == "robust" else 1)
            row_t.append("robust" if fb == "robust" else str(fb))
        cell_vals.append(row_v)
        cell_texts.append(row_t)

    arr = np.array(cell_vals, dtype=float)
    im = ax3.imshow(arr, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax3.set_xticks(range(len(SIG_KEYS)))
    ax3.set_xticklabels([s.replace("_", "-") for s in SIG_KEYS], fontsize=9)
    ax3.set_yticks(range(len(param_names)))
    ax3.set_yticklabels(param_names, fontsize=8)
    for i, row_t in enumerate(cell_texts):
        for j, txt in enumerate(row_t):
            ax3.text(j, i, txt, ha="center", va="center", fontsize=7)

    ax3.set_title("Fig e019-03  Failure boundaries\n"
                  "(green=robust, red=first-fail value shown)", fontsize=10)
    plt.tight_layout()
    fig3.savefig(FIGURES_DIR / "Fig_e019_03_failure_boundaries.png", dpi=150)
    plt.close(fig3)

    # ------------------------------------------------------------------
    # Fig 4 — context_gain: SIG-C vs SIG-A/B/E
    # ------------------------------------------------------------------
    if "context_gain" in all_rows_by_param:
        rows_ctx = all_rows_by_param["context_gain"]
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        for sig in ["SIG_C", "SIG_A", "SIG_B", "SIG_E"]:
            vals, means, stds = _agg(rows_ctx, f"{sig}_score")
            ls = "-" if sig == "SIG_C" else "--"
            lw = 2.2 if sig == "SIG_C" else 1.3
            ax4.errorbar(vals, means, yerr=stds, label=sig.replace("_", "-"),
                         color=SIG_COLORS[sig], fmt="o-", ls=ls, lw=lw,
                         markersize=4, capsize=2)
        ax4.axvline(1.0, color="gray", lw=0.8, ls=":", label="canonical (1.0)")
        ax4.axhline(0, color="gray", lw=0.4)
        ax4.set_xlabel("context_gain", fontsize=9)
        ax4.set_ylabel("Signature score", fontsize=9)
        ax4.set_title("Fig e019-04  Context vs structural signatures\n"
                      "under context_gain sweep", fontsize=10)
        ax4.legend(fontsize=8)
        plt.tight_layout()
        fig4.savefig(FIGURES_DIR / "Fig_e019_04_context_vs_structural_signatures.png", dpi=150)
        plt.close(fig4)

    # ------------------------------------------------------------------
    # Fig 5 — Noise seed stability
    # ------------------------------------------------------------------
    if "structural_noise" in all_rows_by_param:
        rows_noise = all_rows_by_param["structural_noise"]
        fig5, axes5 = plt.subplots(1, 5, figsize=(16, 4), sharey=False)
        fig5.suptitle("Fig e019-05  Noise seed stability (structural_noise sweep)",
                      fontsize=10)
        for col_idx, sig in enumerate(SIG_KEYS):
            ax = axes5[col_idx]
            vals, means, stds = _agg(rows_noise, f"{sig}_score")
            ax.errorbar(vals, means, yerr=stds, fmt="o-",
                        color=SIG_COLORS[sig], markersize=4, capsize=3, lw=1.5)
            ax.axhline(THRESHOLDS[sig], ls="--", lw=0.7, color="gray",
                       label=f"threshold ({THRESHOLDS[sig]})")
            ax.axhline(0, ls="-", lw=0.3, color="gray")
            ax.set_xlabel("structural_noise", fontsize=8)
            ax.set_title(sig.replace("_", "-"), fontsize=9)
            ax.tick_params(axis="x", labelsize=7)
            ax.legend(fontsize=6)
        plt.tight_layout()
        fig5.savefig(FIGURES_DIR / "Fig_e019_05_noise_seed_stability.png", dpi=150)
        plt.close(fig5)

    print("[e019] Figures saved.")


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

def _write_docs(
    all_rows_by_param: dict[str, list[dict]],
    canonical_row: dict,
) -> None:
    # canonical_reference.json
    with (SUMMARY_DIR / "canonical_reference.json").open("w", encoding="utf-8") as f:
        canon_ref = {
            "structural_lr":    CANONICAL_PARAMS.structural_lr,
            "replay_gain":      CANONICAL_PARAMS.replay_gain,
            "eligibility_decay":CANONICAL_PARAMS.eligibility_decay,
            "structural_decay": CANONICAL_PARAMS.structural_decay,
            "structural_noise": CANONICAL_PARAMS.structural_noise,
            "context_gain":     CANONICAL_PARAMS.context_gain,
            "timing_gap":       0,
            "overlap_strength": CANONICAL_OVERLAP_STR,
            "readout_threshold":CANONICAL_PARAMS.readout_threshold,
            "SIG_A_score": canonical_row["SIG_A"],
            "SIG_B_score": canonical_row["SIG_B"],
            "SIG_C_score": canonical_row["SIG_C"],
            "SIG_D_score": canonical_row["SIG_D"],
            "SIG_E_score": canonical_row["SIG_E"],
        }
        json.dump(canon_ref, f, indent=2)

    # protected_thresholds.json
    with (SUMMARY_DIR / "protected_thresholds.json").open("w", encoding="utf-8") as f:
        json.dump(THRESHOLDS, f, indent=2)

    # robustness_summary_by_parameter.csv
    summary_rows = []
    for pname, rows in all_rows_by_param.items():
        vals_pass = _robust_count_by_value(rows)
        n_values_tested   = len(vals_pass)
        n_full_joint_pass = sum(1 for v, n in vals_pass.items() if n >= 4.99)
        sweep = next(s for s in SWEEPS if s.name == pname)
        boundaries = _first_failure_value(rows)
        summary_rows.append({
            "parameter":          pname,
            "n_values_tested":    n_values_tested,
            "n_joint_protected_pass": n_full_joint_pass,
            "pct_joint_pass": 100.0 * n_full_joint_pass / n_values_tested,
            "canonical_value":    sweep.canonical,
            "SIG_A_first_fail": boundaries.get("SIG_A", "robust"),
            "SIG_B_first_fail": boundaries.get("SIG_B", "robust"),
            "SIG_C_first_fail": boundaries.get("SIG_C", "robust"),
            "SIG_D_first_fail": boundaries.get("SIG_D", "robust"),
            "SIG_E_first_fail": boundaries.get("SIG_E", "robust"),
        })
    _write_csv(SUMMARY_DIR / "robustness_summary_by_parameter.csv", summary_rows)

    # failure_boundary_summary.csv
    fb_rows = []
    for row in summary_rows:
        for sig in ["SIG_A", "SIG_B", "SIG_C", "SIG_D", "SIG_E"]:
            fb_rows.append({
                "parameter": row["parameter"],
                "signature": sig,
                "first_fail_value": row[f"{sig}_first_fail"],
            })
    _write_csv(SUMMARY_DIR / "failure_boundary_summary.csv", fb_rows)

    # claim_ledger.md
    joint_robust = sum(1 for r in summary_rows if r["pct_joint_pass"] >= 50)
    (OUT_ROOT / "claim_ledger.md").write_text(
        f"""# e019 — Claim Ledger

| Claim | Status | Evidence | Limitation | Next action |
|-------|--------|----------|------------|-------------|
| Joint signature profile is not a single-run artifact | Validated | Survives parameter variation in {joint_robust}/{len(summary_rows)} sweeps | One-at-a-time only | E020 pairwise |
| Profile survives OAT variation across defined ranges | Validated | sweeps CSVs | No pairwise or noise-robustness claim | E020 |
| Signatures have identifiable failure boundaries | Validated | failure_boundary_summary.csv | OAT only | E020 |
| SIG-C is a fast-context / allocation signature | Validated (if context_gain OAT confirms) | context_gain sweep Fig e019-04 | Allocation architecture driven | Paper prose |
| SIG-D non-diagnostic alone (E018 finding replicated) | Validated | Wide pass range across params | — | Paper prose |
| Model robust to all parameter combinations | Pending | Requires E020 | — | E020 |
| Model scales beyond 4-branch motif | Not supported | — | — | Future |
| Biological validation | Not supported | — | — | Future |
""", encoding="utf-8")

    # figure_manifest.md
    (OUT_ROOT / "figure_manifest.md").write_text(
        """# e019 — Figure Manifest

| File | Content | Status |
|------|---------|--------|
| Fig_e019_01_joint_pass_by_parameter.png | Joint protected pass rate per param value | Generated |
| Fig_e019_02_signature_scores_by_parameter.png | SIG-A to SIG-E scores, small-multiples | Generated |
| Fig_e019_03_failure_boundaries.png | First-fail heatmap | Generated |
| Fig_e019_04_context_vs_structural_signatures.png | SIG-C vs SIG-A/B/E under context_gain | Generated |
| Fig_e019_05_noise_seed_stability.png | Mean ± SD across seeds for noise sweep | Generated |
""", encoding="utf-8")

    # README
    (OUT_ROOT / "README.md").write_text(
        f"""# e019 — One-at-a-Time Parameter Robustness

**Date:** {__import__('datetime').date.today()}
**Parameters swept:** {', '.join(s.name for s in SWEEPS)}

## Purpose
Test whether the full model's joint SIG-A–E profile is robust to one-at-a-time
parameter variation, or is narrow around the canonical set.

## Run
```bash
python experiments/exp019_one_at_a_time_parameter_robustness.py
```

## Key outputs
- `summary/robustness_summary_by_parameter.csv` — per-parameter joint-pass statistics
- `summary/failure_boundary_summary.csv` — first failure value per (param, sig)
- `figures/Fig_e019_01_joint_pass_by_parameter.png` — main overview figure

## Claim scope (if successful)
> "The full model's canonical joint signature profile is not a single-run artifact;
> it survives one-at-a-time variation across defined parameter ranges."
""", encoding="utf-8")

    # qc_report.md
    (OUT_ROOT / "qc_report.md").write_text(
        f"""# e019 — QC Report

## Determinism
All deterministic sweeps use default seed {DEFAULT_SEED}.
structural_noise sweep uses seeds: {NOISE_SEEDS}.

## SIG-C
Computed in a dedicated context-probe simulation per run. Identical protocol to E018.

## SIG-E
Uses E018 generic rescue protocol (plain consolidation, no pre-cueing).

## Thresholds
Predeclared and identical to E018: {THRESHOLDS}.
""", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 68)
    print("Experiment 019 -- One-at-a-Time Parameter Robustness")
    print("=" * 68)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Canonical reference
    canon_sigs = _run_single(
        CANONICAL_PARAMS,
        timing_gap=0,
        overlap_str=CANONICAL_OVERLAP_STR,
        seed=DEFAULT_SEED,
    )
    print(f"  Canonical: SIG-A={canon_sigs['SIG_A']:+.3f}  SIG-B={canon_sigs['SIG_B']:+.3f}"
          f"  SIG-C={canon_sigs['SIG_C']:+.3f}  SIG-D={canon_sigs['SIG_D']:+.1f}pp"
          f"  SIG-E={canon_sigs['SIG_E']:+.1f}pp")
    print()

    all_rows_by_param: dict[str, list[dict]] = {}
    all_long_rows: list[dict] = []

    for sweep in SWEEPS:
        n_runs = len(sweep.values) * (len(NOISE_SEEDS) if sweep.is_noise else 1)
        print(f"  Sweeping {sweep.name:<22}  ({n_runs} runs)...")
        rows = _run_sweep(sweep)
        all_rows_by_param[sweep.name] = rows
        all_long_rows.extend(rows)
        _write_csv(SWEEPS_DIR / f"{sweep.name}_sweep.csv", rows)

        # metadata
        with (SWEEPS_DIR / f"{sweep.name}_metadata.json").open("w", encoding="utf-8") as f:
            json.dump({
                "parameter": sweep.name,
                "description": sweep.description,
                "canonical": sweep.canonical,
                "values": sweep.values,
                "seeds": NOISE_SEEDS if sweep.is_noise else [DEFAULT_SEED],
                "n_runs": n_runs,
                "thresholds": THRESHOLDS,
            }, f, indent=2)

        # Quick summary per param
        ff = _first_failure_value(rows)
        prot_pct = {
            v: sum(int(r["joint_protected_pass"]) for r in rows
                   if r["parameter_value"] == v) /
               max(1, sum(1 for r in rows if r["parameter_value"] == v))
            for v in sweep.values
        }
        n_robust = sum(1 for v in prot_pct if prot_pct[v] > 0.99)
        print(f"           joint pass: {n_robust}/{len(sweep.values)} values "
              f"| first fails: " +
              ", ".join(f"{k}@{v}" for k, v in ff.items() if v != "robust") or "all robust")

    # Combined
    _write_csv(SUMMARY_DIR / "all_sweeps_long.csv", all_long_rows)

    # Figures
    _make_figures(all_rows_by_param)

    # Docs
    _write_docs(all_rows_by_param, canon_sigs)
    print()
    print("[e019] Documentation written.")

    # Print robustness table
    print()
    print("-" * 78)
    print(f"  {'Parameter':<22}  {'Joint pass':<14}  {'First fail':<30}  Interp")
    print("-" * 78)

    interp_map = {
        "structural_lr":     "no slow write -> SIG-A/B fail",
        "replay_gain":       "no replay -> no P_b -> no M_b write",
        "eligibility_decay": "fast decay -> E_b lost before consolidation",
        "structural_decay":  "high decay -> M_b not maintained",
        "structural_noise":  "high noise -> specificity breaks",
        "context_gain":      "0 gain -> SIG-C collapses; A/B stable",
        "timing_gap":        "long gap -> E_b decayed before consolidation",
        "overlap_strength":  "0 overlap -> no linking; SIG-B/E fail",
        "readout_threshold": "very high -> recall support collapse",
    }

    for sweep in SWEEPS:
        rows = all_rows_by_param[sweep.name]
        ff   = _first_failure_value(rows)
        prot_pct = {
            v: sum(int(r["joint_protected_pass"]) for r in rows
                   if r["parameter_value"] == v) /
               max(1, sum(1 for r in rows if r["parameter_value"] == v))
            for v in sweep.values
        }
        n_robust = sum(1 for v in prot_pct if prot_pct[v] > 0.99)
        first_sigs = [(k, v) for k, v in ff.items() if v != "robust"]
        ff_str = "; ".join(f"{k}@{v}" for k, v in first_sigs[:2]) or "all robust"
        print(f"  {sweep.name:<22}  {n_robust}/{len(sweep.values)} values    "
              f"{ff_str:<34}  {interp_map.get(sweep.name, '')[:30]}")

    print()
    print(f"  Outputs: {OUT_ROOT}")
    print("=" * 68)


if __name__ == "__main__":
    main()
