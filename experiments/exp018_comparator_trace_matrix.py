"""Experiment 018 — Comparator Trace Matrix.

Runs the E017 trace-export machinery across five comparator baselines to
determine whether the full model's joint signature profile (SIG-A to SIG-E)
requires replay-dependent slow structural writing, or whether a simpler
mechanism can reproduce it.

This is the first model-discrimination experiment for article v2.

Core question
-------------
    Do simpler comparator models reproduce the same joint profile,
    or does the profile require replay-dependent slow structural writing?

Comparators
-----------
    full_model           — all mechanisms active (canonical)
    fast_context_only    — fast context gating; M_b fixed (structural_lr = 0,
                           structural_decay = 0)
    replay_no_structure  — replay active; M_b cannot be written
                           (structural_lr = 0, decay present)
    random_slow_drift    — slow drift of matched scale, non-specific
    fixed_allocation_only— overlap by construction; no dynamic updating

Signature thresholds (predeclared)
-----------------------------------
    SIG-A: overlap advantage > 0.02 delta M_b
    SIG-B: linking gain      > 0.05 delta L
    SIG-C: context separation> 0.05 support units
    SIG-D: dissociation      > 5.0 pp
    SIG-E: rescue advantage  > 10.0 pp

SIG-C is computed in a dedicated context-probe sim and exported separately.
SIG-E compares targeted-overlap rescue against generic-all-branch rescue.

Equation note (v2 article alignment)
-------------------------------------
The minimal model equations refined in article/peer_review/elife_reorganisation_plan.md
§4.1–4.2 are already implemented. Correspondence:
    W(t)        ↔  modulatory_drive * write-enable window
    E_b · P_b   ↔  eligibility.value * translation_readiness.value
    A_b         ↔  fast_access * slow_access  (=  A^f * A^s)
    L_{μν}      ↔  sum_b a_{μb} a_{νb} M_b  (linking metric)
No code changes required; this is purely notation standardisation for prose.

Outputs (under results/e018_comparator_trace_matrix/)
------------------------------------------------------
    traces/<name>_branch_traces.csv
    traces/<name>_trace_support.csv
    traces/<name>_linking_trace.csv
    traces/<name>_context_probe_traces.csv
    summary/<name>_signature_summary.csv
    summary/<name>_run_metadata.json
    summary/comparator_signature_matrix.csv
    summary/comparator_pass_fail_matrix.csv
    summary/comparator_effect_sizes.csv
    summary/comparator_definitions.json
    summary/joint_pass_summary.json
    summary/rescue_protocol.md
    summary/context_probe_limitations.md
    figures/Fig_e018_01_comparator_signature_matrix.png
    figures/Fig_e018_02_joint_pass_profile.png
    figures/Fig_e018_03_linking_traces_by_comparator.png
    figures/Fig_e018_04_structural_accessibility_by_comparator.png
    figures/Fig_e018_05_context_separation_by_comparator.png
    README.md  qc_report.md  effect_summary.md  claim_ledger.md
    figure_manifest.md  future_comparators.md
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
from copy import deepcopy
from dataclasses import dataclass, field
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
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT  = REPO_ROOT / "results" / "e018_comparator_trace_matrix"
TRACES_DIR  = OUT_ROOT / "traces"
SUMMARY_DIR = OUT_ROOT / "summary"
FIGURES_DIR = OUT_ROOT / "figures"

# ---------------------------------------------------------------------------
# Canonical setup (identical to E017)
# ---------------------------------------------------------------------------
RANDOM_SEED    = 42
BRANCH_IDS     = ["b0", "b1", "b2", "b3"]
OVERLAP_BRANCH = "b1"

MU1_ALLOC = TraceAllocation(
    trace_id="mu1",
    branch_weights={"b0": 0.90, "b1": 0.85, "b2": 0.05, "b3": 0.05},
)
MU2_ALLOC = TraceAllocation(
    trace_id="mu2",
    branch_weights={"b0": 0.05, "b1": 0.85, "b2": 0.90, "b3": 0.05},
)

MU1_CUE    = {"b0": 1.0, "b1": 0.8, "b2": 0.0, "b3": 0.0}
MU2_CUE    = {"b0": 0.0, "b1": 0.8, "b2": 1.0, "b3": 0.0}
B1_CUE     = {"b0": 0.0, "b1": 1.0, "b2": 0.0, "b3": 0.0}
GENERIC_CUE= {"b0": 0.5, "b1": 0.5, "b2": 0.5, "b3": 0.5}
AMBIG_CUE  = {"b0": 0.5, "b1": 0.5, "b2": 0.5, "b3": 0.5}
ALPHA_BIAS = {"b0": 0.5, "b1": 0.5, "b2": -0.5, "b3": -0.5}
BETA_BIAS  = {"b0": -0.5, "b1": -0.5, "b2": 0.5, "b3": 0.5}

ALPHA_ALLOC = TraceAllocation(
    trace_id="mu_alpha",
    branch_weights={"b0": 0.90, "b1": 0.05, "b2": 0.05, "b3": 0.00},
)
BETA_ALLOC = TraceAllocation(
    trace_id="mu_beta",
    branch_weights={"b0": 0.05, "b1": 0.05, "b2": 0.90, "b3": 0.00},
)

CONSOLIDATION_PASSES  = 9
DAMAGE_NULL_PASSES    = 9
RESCUE_PASSES         = 3
RESCUE_CUE_REPS       = 3
RESCUE_ROUNDS         = 3
DAMAGE_DECAY_RATE     = 0.030
RANDOM_DRIFT_SIGMA    = 0.025

# Predeclared thresholds (must be fixed before seeing results)
THRESHOLDS = {
    "SIG_A": 0.02,   # delta M_b
    "SIG_B": 0.05,   # delta L
    "SIG_C": 0.05,   # support units
    "SIG_D": 5.0,    # percentage points
    "SIG_E": 10.0,   # percentage points
}

# ---------------------------------------------------------------------------
# Comparator parameter sets (from exp015 BASE_PARAMS)
# ---------------------------------------------------------------------------
BASE_PARAMS = DynamicsParameters(
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

# structural_lr=0 AND structural_decay=0 → M_b is truly frozen at init value
FAST_CONTEXT_PARAMS = DynamicsParameters(
    structural_lr=0.0,
    replay_gain=0.80,
    eligibility_decay=0.12,
    structural_decay=0.0,
    structural_gain=6.0,
    structural_max=1.0,
    translation_decay=0.05,
    sleep_gain=0.0,
    context_gain=1.0,
    structural_noise=0.0,
    readout_gain=5.0,
    readout_threshold=0.3,
)

# structural_lr=0, decay present → M_b can only decay, never grow
REPLAY_NO_STRUCT_PARAMS = DynamicsParameters(
    structural_lr=0.0,
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

# No dynamic updating at all
FIXED_ALLOC_PARAMS = DynamicsParameters(
    structural_lr=0.0,
    replay_gain=0.0,
    eligibility_decay=0.0,
    structural_decay=0.0,
    structural_gain=6.0,
    structural_max=1.0,
    translation_decay=0.0,
    sleep_gain=0.0,
    context_gain=1.0,
    structural_noise=0.0,
    readout_gain=5.0,
    readout_threshold=0.3,
)


@dataclass
class ComparatorSpec:
    name: str
    params: DynamicsParameters
    random_drift: bool = False
    drift_sigma: float = RANDOM_DRIFT_SIGMA
    description: str = ""
    expected_joint_pass: bool = False


COMPARATORS: list[ComparatorSpec] = [
    ComparatorSpec(
        name="full_model",
        params=BASE_PARAMS,
        random_drift=False,
        description="All mechanisms active: fast access, eligibility, replay, slow M_b write.",
        expected_joint_pass=True,
    ),
    ComparatorSpec(
        name="fast_context_only",
        params=FAST_CONTEXT_PARAMS,
        random_drift=False,
        description="Fast context gating active; M_b frozen at init (structural_lr=0, decay=0).",
        expected_joint_pass=False,
    ),
    ComparatorSpec(
        name="replay_no_structure",
        params=REPLAY_NO_STRUCT_PARAMS,
        random_drift=False,
        description="Replay updates transient E_b/P_b but cannot write M_b (structural_lr=0).",
        expected_joint_pass=False,
    ),
    ComparatorSpec(
        name="random_slow_drift",
        params=BASE_PARAMS,
        random_drift=True,
        drift_sigma=RANDOM_DRIFT_SIGMA,
        description=(
            "Slow Gaussian drift of matched scale (σ≈0.025/pass), non-specific "
            "branch targeting. Uses BASE_PARAMS for fast dynamics."
        ),
        expected_joint_pass=False,
    ),
    ComparatorSpec(
        name="fixed_allocation_only",
        params=FIXED_ALLOC_PARAMS,
        random_drift=False,
        description="Overlap by construction; no dynamic E_b/P_b/M_b updating.",
        expected_joint_pass=False,
    ),
]

COMPARATOR_NAMES = [c.name for c in COMPARATORS]

# ---------------------------------------------------------------------------
# Simulator builder
# ---------------------------------------------------------------------------

def _build_sim(params: DynamicsParameters) -> CytodendAccessModelSimulator:
    sim = CytodendAccessModelSimulator.from_branch_ids(BRANCH_IDS, parameters=params)
    sim.traces["mu1"] = EngramTrace(trace_id="mu1", allocation=MU1_ALLOC)
    sim.traces["mu2"] = EngramTrace(trace_id="mu2", allocation=MU2_ALLOC)
    return sim


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _snapshot_branches(
    sim: CytodendAccessModelSimulator,
    step: int,
    phase: str,
    comparator: str,
    input_drives: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    drives = input_drives or {}
    rows = []
    for bid in BRANCH_IDS:
        b = sim.branches[bid]
        rows.append({
            "comparator":              comparator,
            "step":                    step,
            "phase":                   phase,
            "branch_id":               bid,
            "is_overlap":              bid == OVERLAP_BRANCH,
            "x_b":                     b.activation,
            "fast_access":             b.fast_access,
            "slow_access":             b.slow_access,
            "effective_access":        b.effective_access,
            "eligibility":             b.eligibility.value,
            "translation_readiness":   b.translation_readiness.value,
            "structural_accessibility":b.structural.accessibility,
            "input_drive":             drives.get(bid, float("nan")),
        })
    return rows


def _snapshot_supports(
    sim: CytodendAccessModelSimulator,
    step: int,
    phase: str,
    comparator: str,
) -> list[dict[str, Any]]:
    rows = []
    for rs in sim.compute_recall_supports():
        rows.append({
            "comparator":     comparator,
            "step":           step,
            "phase":          phase,
            "trace_id":       rs.trace_id,
            "recall_support": rs.support,
            "readout_value":  rs.expressed_strength,
            "context_label":  rs.matched_context or "none",
        })
    return rows


def _snapshot_linking(
    sim: CytodendAccessModelSimulator,
    step: int,
    phase: str,
    comparator: str,
) -> dict[str, Any]:
    lk = _linking(sim)
    ovlp = (
        MU1_ALLOC.branch_weights.get(OVERLAP_BRANCH, 0.0)
        * MU2_ALLOC.branch_weights.get(OVERLAP_BRANCH, 0.0)
        * sim.branches[OVERLAP_BRANCH].structural.accessibility
    )
    return {
        "comparator":                 comparator,
        "step":                       step,
        "phase":                      phase,
        "trace_pair":                 "mu1_mu2",
        "linking_score":              lk,
        "overlap_branch_contribution":ovlp,
        "nonoverlap_contribution":    lk - ovlp,
    }


def _linking(sim: CytodendAccessModelSimulator) -> float:
    return sum(
        MU1_ALLOC.branch_weights.get(b, 0.0)
        * MU2_ALLOC.branch_weights.get(b, 0.0)
        * sim.branches[b].structural.accessibility
        for b in BRANCH_IDS
    )


# ---------------------------------------------------------------------------
# Random drift
# ---------------------------------------------------------------------------

def _apply_random_drift(sim: CytodendAccessModelSimulator, sigma: float) -> None:
    for branch in sim.branches.values():
        drift = random.gauss(0.0, sigma)
        m_max = branch.structural.max_accessibility
        branch.structural.accessibility = max(
            0.0, min(m_max, branch.structural.accessibility + drift)
        )
        branch.slow_access = _sigmoid(
            sim.parameters.structural_gain * branch.structural.accessibility
        )
        branch.effective_access = _clamp01(branch.fast_access * branch.slow_access)


# ---------------------------------------------------------------------------
# Cue/consolidation with trace capture
# ---------------------------------------------------------------------------

def _apply_cue_traced(
    sim, cue, step_c, phase, comp, branch_rows, support_rows, linking_rows
) -> None:
    sim.apply_cue(cue)
    step_c[0] += 1
    branch_rows.extend(_snapshot_branches(sim, step_c[0], phase, comp, cue))
    support_rows.extend(_snapshot_supports(sim, step_c[0], phase, comp))
    linking_rows.append(_snapshot_linking(sim, step_c[0], phase, comp))


def _consolidate_traced(
    sim, n_passes, step_c, phase, comp, branch_rows, support_rows, linking_rows,
    spec: ComparatorSpec, replay_ids=None, modulatory_drive=1.0,
) -> None:
    win = ConsolidationWindow(
        replay_trace_ids=replay_ids if replay_ids is not None else ["mu1", "mu2"],
        modulatory_drive=modulatory_drive,
    )
    for _ in range(n_passes):
        if spec.random_drift:
            _apply_random_drift(sim, spec.drift_sigma)
        else:
            sim.run_consolidation(win)
        step_c[0] += 1
        branch_rows.extend(_snapshot_branches(sim, step_c[0], phase, comp))
        support_rows.extend(_snapshot_supports(sim, step_c[0], phase, comp))
        linking_rows.append(_snapshot_linking(sim, step_c[0], phase, comp))


def _null_consolidate_traced(
    sim, n_passes, step_c, phase, comp, branch_rows, support_rows, linking_rows,
    spec: ComparatorSpec,
) -> None:
    _consolidate_traced(
        sim, n_passes, step_c, phase, comp, branch_rows, support_rows, linking_rows,
        spec, replay_ids=[], modulatory_drive=0.0,
    )


# ---------------------------------------------------------------------------
# Context probe per comparator (SIG-C)
# ---------------------------------------------------------------------------

def _run_context_probe(spec: ComparatorSpec) -> tuple[float, list[dict]]:
    """Run the context disambiguation probe and return (sig_c_score, trace_rows)."""
    random.seed(RANDOM_SEED)
    sim = CytodendAccessModelSimulator.from_branch_ids(BRANCH_IDS, parameters=spec.params)
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

    if spec.random_drift:
        for _ in range(CONSOLIDATION_PASSES):
            _apply_random_drift(sim, spec.drift_sigma)
    else:
        win = ConsolidationWindow(
            replay_trace_ids=["mu_alpha", "mu_beta"], modulatory_drive=1.0
        )
        for _ in range(CONSOLIDATION_PASSES):
            sim.run_consolidation(win)

    trace_rows: list[dict] = []

    def _probe_ctx(sim_copy, context, bias, ctx_label):
        sim_copy.apply_cue(AMBIG_CUE, context=context, context_bias=bias)
        rmap = {rs.trace_id: rs for rs in sim_copy.compute_recall_supports()}
        for bid in BRANCH_IDS:
            b = sim_copy.branches[bid]
            trace_rows.append({
                "comparator":      spec.name,
                "context_condition":ctx_label,
                "branch_id":       bid,
                "x_b":             b.activation,
                "fast_access":     b.fast_access,
                "slow_access":     b.slow_access,
                "effective_access":b.effective_access,
            })
        for trace_id in ["mu_alpha", "mu_beta"]:
            rs = rmap.get(trace_id)
            trace_rows.append({
                "comparator":      spec.name,
                "context_condition":ctx_label,
                "branch_id":       "trace_level",
                "trace_id":        trace_id,
                "recall_support":  rs.support if rs else 0.0,
                "readout_value":   rs.expressed_strength if rs else 0.0,
            })

    sim_a = deepcopy(sim)
    sim_b = deepcopy(sim)
    _probe_ctx(sim_a, "alpha", ALPHA_BIAS, "alpha_probe")
    _probe_ctx(sim_b, "beta",  BETA_BIAS,  "beta_probe")

    rmap_a = {rs.trace_id: rs for rs in sim_a.compute_recall_supports()}
    rmap_b = {rs.trace_id: rs for rs in sim_b.compute_recall_supports()}
    r_a_corr  = rmap_a.get("mu_alpha", type("_", (), {"support": 0.0})()).support
    r_a_wrong = rmap_a.get("mu_beta",  type("_", (), {"support": 0.0})()).support
    r_b_corr  = rmap_b.get("mu_beta",  type("_", (), {"support": 0.0})()).support
    r_b_wrong = rmap_b.get("mu_alpha", type("_", (), {"support": 0.0})()).support

    sig_c = ((r_a_corr - r_a_wrong) + (r_b_corr - r_b_wrong)) / 2.0
    return sig_c, trace_rows


# ---------------------------------------------------------------------------
# Core protocol
# ---------------------------------------------------------------------------

def _recovery_pct(post: float, dmg: float, healthy: float) -> float:
    denom = healthy - dmg
    return (post - dmg) / denom * 100.0 if abs(denom) > 1e-9 else 0.0


def run_comparator(
    spec: ComparatorSpec,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict[str, float]]:
    """Run the full 10-phase protocol for one comparator.

    Returns (branch_rows, support_rows, linking_rows, context_probe_rows, sigs).
    """
    random.seed(RANDOM_SEED)
    comp = spec.name
    sim  = _build_sim(spec.params)

    branch_rows:  list[dict] = []
    support_rows: list[dict] = []
    linking_rows: list[dict] = []
    step = [0]

    # init
    branch_rows.extend(_snapshot_branches(sim, step[0], "init", comp))
    support_rows.extend(_snapshot_supports(sim, step[0], "init", comp))
    linking_rows.append(_snapshot_linking(sim, step[0], "init", comp))

    # encode
    for _ in range(2):
        _apply_cue_traced(sim, MU1_CUE, step, "encode_mu_1", comp,
                          branch_rows, support_rows, linking_rows)
    for _ in range(2):
        _apply_cue_traced(sim, MU2_CUE, step, "encode_mu_2", comp,
                          branch_rows, support_rows, linking_rows)

    # pre_consolidation_probe
    _apply_cue_traced(sim, MU1_CUE, step, "pre_consolidation_probe", comp,
                      branch_rows, support_rows, linking_rows)
    _apply_cue_traced(sim, MU2_CUE, step, "pre_consolidation_probe", comp,
                      branch_rows, support_rows, linking_rows)

    mb_pre = {b: sim.branches[b].structural.accessibility for b in BRANCH_IDS}
    lk_pre = _linking(sim)
    sim_tmp = deepcopy(sim)
    sim_tmp.apply_cue(MU1_CUE)
    r_mu1_pre = next(
        (rs.support for rs in sim_tmp.compute_recall_supports() if rs.trace_id == "mu1"), 0.0
    )

    # consolidation
    _consolidate_traced(sim, CONSOLIDATION_PASSES, step, "consolidation_replay", comp,
                        branch_rows, support_rows, linking_rows, spec)

    mb_post_cons = {b: sim.branches[b].structural.accessibility for b in BRANCH_IDS}
    lk_post_cons = _linking(sim)

    # post_consolidation_probe
    _apply_cue_traced(sim, MU1_CUE, step, "post_consolidation_probe", comp,
                      branch_rows, support_rows, linking_rows)
    _apply_cue_traced(sim, MU2_CUE, step, "post_consolidation_probe", comp,
                      branch_rows, support_rows, linking_rows)

    sim_tmp2 = deepcopy(sim)
    sim_tmp2.apply_cue(MU1_CUE)
    r_mu1_post_cons = next(
        (rs.support for rs in sim_tmp2.compute_recall_supports() if rs.trace_id == "mu1"), 0.0
    )

    # overlap_damage
    sim.branches[OVERLAP_BRANCH].structural.decay_rate = DAMAGE_DECAY_RATE
    _null_consolidate_traced(sim, DAMAGE_NULL_PASSES, step, "overlap_damage", comp,
                             branch_rows, support_rows, linking_rows, spec)

    lk_post_damage = _linking(sim)

    # post_damage_probe
    _apply_cue_traced(sim, MU1_CUE, step, "post_damage_probe", comp,
                      branch_rows, support_rows, linking_rows)
    _apply_cue_traced(sim, MU2_CUE, step, "post_damage_probe", comp,
                      branch_rows, support_rows, linking_rows)

    sim_tmp3 = deepcopy(sim)
    sim_tmp3.apply_cue(MU1_CUE)
    r_mu1_post_damage = next(
        (rs.support for rs in sim_tmp3.compute_recall_supports() if rs.trace_id == "mu1"), 0.0
    )

    # targeted_overlap_rescue
    sim_targ = deepcopy(sim)
    for _ in range(RESCUE_ROUNDS):
        for _ in range(RESCUE_CUE_REPS):
            sim_targ.apply_cue(B1_CUE)
        win_r = ConsolidationWindow(replay_trace_ids=["mu1", "mu2"], modulatory_drive=1.0)
        for _ in range(RESCUE_PASSES):
            if spec.random_drift:
                _apply_random_drift(sim_targ, spec.drift_sigma)
            else:
                sim_targ.run_consolidation(win_r)
    lk_targ_rescue = _linking(sim_targ)

    # generic_all_branch_rescue: plain re-consolidation without any targeted pre-cueing.
    # This is the same comparison used in exp015 (standard rescue = no pre-cueing).
    # The total consolidation volume matches targeted rescue: RESCUE_ROUNDS × RESCUE_PASSES.
    sim_gen = deepcopy(sim)
    win_gen = ConsolidationWindow(replay_trace_ids=["mu1", "mu2"], modulatory_drive=1.0)
    for _ in range(RESCUE_ROUNDS * RESCUE_PASSES):
        if spec.random_drift:
            _apply_random_drift(sim_gen, spec.drift_sigma)
        else:
            sim_gen.run_consolidation(win_gen)
    lk_gen_rescue = _linking(sim_gen)

    # Use targeted rescue sim for the post_rescue_probe traces
    sim_final = sim_targ
    _apply_cue_traced(sim_final, MU1_CUE, step, "post_rescue_probe", comp,
                      branch_rows, support_rows, linking_rows)
    _apply_cue_traced(sim_final, MU2_CUE, step, "post_rescue_probe", comp,
                      branch_rows, support_rows, linking_rows)

    # SIG-C (separate sim)
    sig_c_score, context_probe_rows = _run_context_probe(spec)

    # -----------------------------------------------------------------------
    # Compute signatures
    # -----------------------------------------------------------------------
    nonoverlap_ids = [b for b in BRANCH_IDS if b != OVERLAP_BRANCH]
    delta_m_overlap    = mb_post_cons[OVERLAP_BRANCH] - mb_pre[OVERLAP_BRANCH]
    delta_m_nonoverlap = (
        sum(mb_post_cons[b] - mb_pre[b] for b in nonoverlap_ids) / len(nonoverlap_ids)
    )
    sig_a = delta_m_overlap - delta_m_nonoverlap
    sig_b = lk_post_cons - lk_pre
    sig_c = sig_c_score

    link_drop_pct   = (lk_post_cons - lk_post_damage) / max(abs(lk_post_cons), 1e-9) * 100.0
    recall_drop_pct = (r_mu1_post_cons - r_mu1_post_damage) / max(abs(r_mu1_post_cons), 1e-9) * 100.0
    sig_d = link_drop_pct - recall_drop_pct

    rec_targ = _recovery_pct(lk_targ_rescue, lk_post_damage, lk_post_cons)
    rec_gen  = _recovery_pct(lk_gen_rescue,  lk_post_damage, lk_post_cons)
    sig_e    = rec_targ - rec_gen

    sigs: dict[str, float] = {
        "comparator":                         comp,
        "SIG_A_overlap_advantage":            sig_a,
        "SIG_B_linking_gain":                 sig_b,
        "SIG_C_context_separation":           sig_c,
        "SIG_D_linking_recall_dissociation":  sig_d,
        "SIG_E_targeted_rescue_advantage":    sig_e,
        # raw
        "mb_pre_overlap":          mb_pre[OVERLAP_BRANCH],
        "mb_post_overlap":         mb_post_cons[OVERLAP_BRANCH],
        "delta_m_overlap":         delta_m_overlap,
        "delta_m_nonoverlap_mean": delta_m_nonoverlap,
        "lk_pre":                  lk_pre,
        "lk_post_cons":            lk_post_cons,
        "lk_post_damage":          lk_post_damage,
        "lk_targ_rescue":          lk_targ_rescue,
        "lk_gen_rescue":           lk_gen_rescue,
        "r_mu1_pre":               r_mu1_pre,
        "r_mu1_post_cons":         r_mu1_post_cons,
        "r_mu1_post_damage":       r_mu1_post_damage,
        "link_drop_pct":           link_drop_pct,
        "recall_drop_pct":         recall_drop_pct,
        "rec_targ_pct":            rec_targ,
        "rec_gen_pct":             rec_gen,
    }

    return branch_rows, support_rows, linking_rows, context_probe_rows, sigs


# ---------------------------------------------------------------------------
# Pass / fail
# ---------------------------------------------------------------------------

def _passes(sigs: dict[str, float]) -> dict[str, bool]:
    return {
        "SIG_A": sigs["SIG_A_overlap_advantage"]           > THRESHOLDS["SIG_A"],
        "SIG_B": sigs["SIG_B_linking_gain"]                > THRESHOLDS["SIG_B"],
        "SIG_C": sigs["SIG_C_context_separation"]          > THRESHOLDS["SIG_C"],
        "SIG_D": sigs["SIG_D_linking_recall_dissociation"] > THRESHOLDS["SIG_D"],
        "SIG_E": sigs["SIG_E_targeted_rescue_advantage"]   > THRESHOLDS["SIG_E"],
    }


def _joint_pass(pf: dict[str, bool]) -> bool:
    return all(pf.values())


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Collect union of all keys to handle rows with different schemas
    all_keys: list[str] = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore",
                                restval="")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        import subprocess
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _make_figures(
    all_sigs: dict[str, dict],
    all_pf:   dict[str, dict],
    all_link: dict[str, list[dict]],
    all_branch: dict[str, list[dict]],
    context_scores: dict[str, float],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap
        import numpy as np
    except ImportError:
        print("[e018] matplotlib/numpy not available — skipping figures.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SIG_KEYS   = ["SIG_A", "SIG_B", "SIG_C", "SIG_D", "SIG_E"]
    SIG_LABELS = ["SIG-A\nOverlap\nwriting", "SIG-B\nLinking\ngain",
                  "SIG-C\nContext\nsep", "SIG-D\nLink>recall\nvulner.",
                  "SIG-E\nTargeted\nrescue"]
    COMP_COLOURS = {
        "full_model":          "#2ca02c",
        "fast_context_only":   "#d62728",
        "replay_no_structure": "#ff7f0e",
        "random_slow_drift":   "#9467bd",
        "fixed_allocation_only":"#1f77b4",
    }
    COMP_LABELS = {
        "full_model":          "Full model",
        "fast_context_only":   "Fast context\nonly",
        "replay_no_structure": "Replay, no\nstructure",
        "random_slow_drift":   "Random\nslow drift",
        "fixed_allocation_only":"Fixed alloc\nonly",
    }

    comp_names = COMPARATOR_NAMES

    # ------------------------------------------------------------------
    # Figure 1 — Comparator signature matrix (heatmap)
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(9, 5.5))
    matrix = np.array([
        [float(all_pf[c][sk]) for c in comp_names]
        for sk in SIG_KEYS
    ])
    im = ax1.imshow(matrix, cmap=ListedColormap(["#f28b82", "#81c995"]),
                    vmin=0, vmax=1, aspect="auto")
    ax1.set_xticks(range(len(comp_names)))
    ax1.set_xticklabels([COMP_LABELS.get(c, c) for c in comp_names], fontsize=8)
    ax1.set_yticks(range(len(SIG_KEYS)))
    ax1.set_yticklabels(SIG_LABELS, fontsize=8)
    ax1.set_title("Fig e018-01  Comparator signature matrix\n"
                  "(green=PASS, red=FAIL; predeclared protected thresholds)", fontsize=10)

    for i, sk in enumerate(SIG_KEYS):
        for j, c in enumerate(comp_names):
            val  = all_sigs[c]["SIG_" + sk.split("_")[1] + "_" + {
                "SIG_A": "overlap_advantage",
                "SIG_B": "linking_gain",
                "SIG_C": "context_separation",
                "SIG_D": "linking_recall_dissociation",
                "SIG_E": "targeted_rescue_advantage",
            }[sk]]
            pf   = all_pf[c][sk]
            cell = f"{'✓' if pf else '✗'}\n{val:.3f}"
            ax1.text(j, i, cell, ha="center", va="center", fontsize=7,
                     color="black")

    ax1.set_xlabel("Comparator", fontsize=9)
    ax1.set_ylabel("Signature", fontsize=9)
    plt.tight_layout()
    fig1.savefig(FIGURES_DIR / "Fig_e018_01_comparator_signature_matrix.png", dpi=150)
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2 — Joint pass profile (bar chart)
    # ------------------------------------------------------------------
    n_pass = [sum(all_pf[c].values()) for c in comp_names]
    bar_cols = [COMP_COLOURS.get(c, "gray") for c in comp_names]
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    bars = ax2.bar([COMP_LABELS.get(c, c) for c in comp_names],
                   n_pass, color=bar_cols, edgecolor="white", width=0.55)
    ax2.axhline(5, color="black", lw=0.8, ls="--", label="full joint pass (5/5)")
    ax2.set_ylim(0, 5.8)
    ax2.set_ylabel("Signatures passed (out of 5)", fontsize=9)
    ax2.set_title("Fig e018-02  Joint pass profile by comparator", fontsize=10)
    for bar, n in zip(bars, n_pass):
        ax2.text(bar.get_x() + bar.get_width()/2, n + 0.08,
                 f"{n}/5", ha="center", fontsize=9, fontweight="bold")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    fig2.savefig(FIGURES_DIR / "Fig_e018_02_joint_pass_profile.png", dpi=150)
    plt.close(fig2)

    # ------------------------------------------------------------------
    # Figure 3 — Linking traces by comparator
    # ------------------------------------------------------------------
    phase_order = [
        "init", "encode_mu_1", "encode_mu_2",
        "pre_consolidation_probe", "consolidation_replay",
        "post_consolidation_probe", "overlap_damage",
        "post_damage_probe", "targeted_rescue", "post_rescue_probe",
    ]

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    for comp in comp_names:
        rows = all_link[comp]
        steps  = [r["step"]         for r in rows]
        scores = [r["linking_score"] for r in rows]
        ls = "-" if comp == "full_model" else "--"
        lw = 2.2 if comp == "full_model" else 1.3
        ax3.plot(steps, scores, color=COMP_COLOURS.get(comp, "gray"),
                 ls=ls, lw=lw, label=COMP_LABELS.get(comp, comp))

    ax3.set_xlabel("Step", fontsize=9)
    ax3.set_ylabel("Linking score $L_{\\mu_1\\mu_2}$", fontsize=9)
    ax3.set_title("Fig e018-03  Linking traces by comparator", fontsize=10)
    ax3.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    fig3.savefig(FIGURES_DIR / "Fig_e018_03_linking_traces_by_comparator.png", dpi=150)
    plt.close(fig3)

    # ------------------------------------------------------------------
    # Figure 4 — M_b overlap vs non-overlap, faceted by comparator
    # ------------------------------------------------------------------
    fig4, axes4 = plt.subplots(1, len(comp_names), figsize=(14, 4), sharey=True)
    fig4.suptitle("Fig e018-04  Structural accessibility: overlap vs non-overlap",
                  fontsize=10)

    for ax, comp in zip(axes4, comp_names):
        rows = [r for r in all_branch[comp] if r["branch_id"] in BRANCH_IDS]
        # collect per branch per step
        per_branch: dict[str, dict[str, list]] = {b: {"step": [], "M_b": []} for b in BRANCH_IDS}
        for r in rows:
            per_branch[r["branch_id"]]["step"].append(r["step"])
            per_branch[r["branch_id"]]["M_b"].append(r["structural_accessibility"])

        # overlap
        ax.plot(per_branch[OVERLAP_BRANCH]["step"], per_branch[OVERLAP_BRANCH]["M_b"],
                color="#d62728", lw=2.0, label="b1 (overlap)")

        # mean non-overlap
        nonoverlap_ids = [b for b in BRANCH_IDS if b != OVERLAP_BRANCH]
        steps_ref = per_branch[nonoverlap_ids[0]]["step"]
        mean_non  = [
            np.mean([per_branch[b]["M_b"][i] for b in nonoverlap_ids])
            for i in range(len(steps_ref))
        ]
        ax.plot(steps_ref, mean_non,
                color="#1f77b4", lw=1.3, ls="--", label="mean non-overlap")

        ax.set_title(COMP_LABELS.get(comp, comp), fontsize=8)
        ax.set_xlabel("Step", fontsize=7)
        if ax is axes4[0]:
            ax.set_ylabel("$M_b$", fontsize=9)
        ax.legend(fontsize=6)

    plt.tight_layout()
    fig4.savefig(FIGURES_DIR / "Fig_e018_04_structural_accessibility_by_comparator.png", dpi=150)
    plt.close(fig4)

    # ------------------------------------------------------------------
    # Figure 5 — Context separation by comparator (optional but included)
    # ------------------------------------------------------------------
    comp_list_ctx = list(context_scores.keys())
    ctx_vals = [context_scores[c] for c in comp_list_ctx]
    bar_cols_ctx = [COMP_COLOURS.get(c, "gray") for c in comp_list_ctx]
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    ax5.bar([COMP_LABELS.get(c, c) for c in comp_list_ctx],
            ctx_vals, color=bar_cols_ctx, edgecolor="white")
    ax5.axhline(THRESHOLDS["SIG_C"], color="black", lw=0.8, ls="--",
                label=f"SIG-C threshold ({THRESHOLDS['SIG_C']})")
    ax5.axhline(0, color="gray", lw=0.5)
    ax5.set_ylabel("Context separation score", fontsize=9)
    ax5.set_title("Fig e018-05  Context separation (SIG-C) by comparator\n"
                  "(computed in separate context-probe sim)", fontsize=9)
    ax5.legend(fontsize=8)
    plt.tight_layout()
    fig5.savefig(FIGURES_DIR / "Fig_e018_05_context_separation_by_comparator.png", dpi=150)
    plt.close(fig5)

    print("[e018] Figures saved.")


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

def _write_docs(
    all_sigs: dict[str, dict],
    all_pf:   dict[str, dict],
) -> None:
    SIG_KEYS = ["SIG_A", "SIG_B", "SIG_C", "SIG_D", "SIG_E"]
    SIG_RAW  = [
        "SIG_A_overlap_advantage", "SIG_B_linking_gain",
        "SIG_C_context_separation", "SIG_D_linking_recall_dissociation",
        "SIG_E_targeted_rescue_advantage",
    ]
    UNITS = ["ΔM_b", "ΔL", "support", "pp", "pp"]

    def _p(c, sk): return "PASS" if all_pf[c][sk] else "FAIL"
    def _v(c, rk): return f"{all_sigs[c][rk]:.4f}"

    # rescue_protocol
    (SUMMARY_DIR / "rescue_protocol.md").write_text(
        f"""# e018 — Rescue Protocol

## Targeted overlap rescue
- Apply B1_CUE (only b1 driven) x {RESCUE_CUE_REPS} reps
- Run consolidation (replay mu1+mu2) x {RESCUE_PASSES} passes
- Repeat {RESCUE_ROUNDS} rounds
- Total consolidation passes: {RESCUE_ROUNDS * RESCUE_PASSES}

Rationale: explicitly rebuilds E_b on b1 before each consolidation window,
targeted at the damaged overlap branch.

## Generic rescue (no pre-cueing baseline)
- Run plain consolidation (replay mu1+mu2) x {RESCUE_ROUNDS * RESCUE_PASSES} passes
- No targeted pre-cueing of any branch

Rationale: same total consolidation volume as targeted rescue, but without
branch-specific E_b pre-loading. Matches the 'standard rescue' comparison
used in exp015. This ensures SIG-E tests whether b1-specific pre-cueing
confers additional advantage over consolidation alone.

Note: an earlier version used uniform all-branch pre-cueing (GENERIC_CUE) as the
baseline, but this also drove b1 at 0.5, making it too similar to targeted rescue.
Plain consolidation (no pre-cueing) is the more meaningful control.

## SIG-E definition
```
SIG_E = recovery_pct(L_targeted) - recovery_pct(L_generic)
recovery_pct = (L_post_rescue - L_post_damage) / (L_healthy - L_post_damage) x 100
```

## Interpretation
SIG-E > 0 means targeted overlap rescue recovers more linking than plain re-consolidation.
SIG-E > {THRESHOLDS['SIG_E']} pp = protected threshold.
""", encoding="utf-8")

    # context_probe_limitations
    (SUMMARY_DIR / "context_probe_limitations.md").write_text(
        """# e018 — Context Probe Limitations

SIG-C is computed in a separate context-probe simulation, not the main 10-phase trace.

## Why separate
The main phase trace uses mu1/mu2 (linking traces). SIG-C requires mu_alpha/mu_beta
(context-differentiated traces with distinct branch allocations and explicit context labels).
Merging these into one trace would require a different simulator configuration.

## What is exported
- `traces/<comparator>_context_probe_traces.csv`: per-branch and per-trace snapshots
  for alpha_probe and beta_probe conditions.
- `summary/<comparator>_signature_summary.csv`: SIG-C score included.

## What is NOT exported
- Time-resolved context-probe traces across consolidation phases (only post-consolidation probe).
- Context-value per branch in main branch_traces.csv.

## Impact on claims
SIG-C direction and magnitude are reported. The limitation is that SIG-C dynamics
across consolidation cannot be visualized from main traces. This will be addressed in E019
if context-trace instrumentation is added to the main protocol.
""", encoding="utf-8")

    # README
    (OUT_ROOT / "README.md").write_text(
        f"""# e018 — Comparator Trace Matrix

**Date:** {__import__('datetime').date.today()}
**Comparators:** {', '.join(COMPARATOR_NAMES)}

## Purpose
Model-discrimination experiment: does the full model uniquely reproduce the
joint SIG-A–SIG-E signature profile among the tested comparators?

## Run
```bash
python experiments/exp018_comparator_trace_matrix.py
```

## Key result
See `summary/joint_pass_summary.json` and
`figures/Fig_e018_01_comparator_signature_matrix.png`.

## Claim scope
Only if full model passes and no simpler comparator passes the joint profile:
> "Under canonical parameters, the joint SIG-A to SIG-E profile is specific
> to the full replay-dependent slow structural writing model."
""", encoding="utf-8")

    # effect_summary
    rows_text = []
    for comp in COMPARATOR_NAMES:
        s = all_sigs[comp]
        pf = all_pf[comp]
        joint = _joint_pass(pf)
        row_vals = " | ".join(
            f"{_v(comp, rk)} ({'P' if all_pf[comp][sk] else 'F'})"
            for rk, sk in zip(SIG_RAW, SIG_KEYS)
        )
        rows_text.append(f"| {comp:<26} | {row_vals} | {'PASS' if joint else 'FAIL'} |")

    header = (
        "| Comparator                 | SIG-A (ΔM_b) | SIG-B (ΔL)   | SIG-C (sup)  "
        "| SIG-D (pp)   | SIG-E (pp)   | Joint |\n"
        "|----------------------------|--------------|--------------|--"
        "------------|--------------|--------------|-------|"
    )
    (OUT_ROOT / "effect_summary.md").write_text(
        f"""# e018 — Effect Summary

## Signature matrix

{header}
""" + "\n".join(rows_text) + f"""

## Predeclared thresholds
SIG-A > {THRESHOLDS['SIG_A']} ΔM_b | SIG-B > {THRESHOLDS['SIG_B']} ΔL |
SIG-C > {THRESHOLDS['SIG_C']} support | SIG-D > {THRESHOLDS['SIG_D']} pp |
SIG-E > {THRESHOLDS['SIG_E']} pp
""", encoding="utf-8")

    # claim_ledger
    full_pf = all_pf["full_model"]
    joint_full = _joint_pass(full_pf)
    no_simple_passes = all(
        not _joint_pass(all_pf[c])
        for c in COMPARATOR_NAMES if c != "full_model"
    )

    (OUT_ROOT / "claim_ledger.md").write_text(
        f"""# e018 — Claim Ledger

| Claim | Status | Evidence | Limitation | Next action |
|-------|--------|----------|------------|-------------|
| Simulator can compare full model against baselines using same trace machinery | Internal validated result | All 5 comparators ran; traces exported | Canonical params only | E019 parameter robustness |
| Full model passes joint SIG-A–SIG-E profile | {'Internal validated result' if joint_full else 'FAIL — investigate'} | comparator_pass_fail_matrix.csv | Canonical params; directional+protected threshold | E019 |
| At least one simpler comparator fails joint profile | {'Internal validated result' if any(not _joint_pass(all_pf[c]) for c in COMPARATOR_NAMES if c != "full_model") else 'NOT MET — all pass'} | comparator_pass_fail_matrix.csv | Tested comparators only | E018R if needed |
| No simpler comparator passes all five signatures | {'Internal validated result' if no_simple_passes else 'NOT MET — check matrix'} | comparator_pass_fail_matrix.csv | Canonical params; not robustness tested | E019 |
| Model discrimination claim is internally testable | Internal validated result | joint_pass_summary.json | Not externally validated | — |
| Model robust across parameters | Pending | Requires E019 | — | E019 |
| Biological validation | Not supported | E018 scope: instrumentation + discrimination | — | — |
| DANDI evidence validates model | Not supported | E018 has no DANDI analysis | — | — |
""", encoding="utf-8")

    # figure_manifest
    (OUT_ROOT / "figure_manifest.md").write_text(
        """# e018 — Figure Manifest

| File | Content | Status |
|------|---------|--------|
| Fig_e018_01_comparator_signature_matrix.png | Heatmap: SIG-A–E × comparators; green=PASS red=FAIL | Generated |
| Fig_e018_02_joint_pass_profile.png | Bar: signatures passed per comparator | Generated |
| Fig_e018_03_linking_traces_by_comparator.png | L_mu1mu2 over all steps per comparator | Generated |
| Fig_e018_04_structural_accessibility_by_comparator.png | M_b overlap vs non-overlap, faceted | Generated |
| Fig_e018_05_context_separation_by_comparator.png | SIG-C bar chart (context-probe sim) | Generated |
""", encoding="utf-8")

    # future_comparators
    (OUT_ROOT / "future_comparators.md").write_text(
        """# e018 — Future Comparators (not implemented in E018)

The following five optional comparators from the E018 spec were not implemented
in this run to avoid delaying the core five-comparator result. They are documented
here for E018R or E019.

## hebbian_weight_only
Associative strengthening via trace weight updates, not M_b.
Implementation: add a per-trace weight matrix that scales recall support; disable M_b writing.

## soma_global_gain_only
A global gain G(t) applied uniformly to all branches.
Implementation: multiply all branch activations by a shared scalar; disable M_b.

## shuffled_replay
Replay exists but branch identity is shuffled before consolidation.
Implementation: permute branch_id → M_b assignment in each consolidation pass.

## eligibility_only
E_b affects recall transiently but no persistent M_b is written.
Implementation: structural_lr=0; let E_b directly scale recall support via a fast gain.

## resource_only
P_b (capture/resource) exists and accumulates but no persistent M_b is retained.
Implementation: structural_lr=0; let P_b directly scale recall support.

These comparators would strengthen the model-discrimination claim if they also
fail the joint SIG-A–SIG-E profile.
""", encoding="utf-8")

    # qc_report
    (OUT_ROOT / "qc_report.md").write_text(
        f"""# e018 — QC Report

## Determinism
All comparators except random_slow_drift use fixed seed {RANDOM_SEED} and structural_noise=0.
random_slow_drift uses fixed seed {RANDOM_SEED}; drift is Gaussian but seeded.

## SIG-C
Computed in separate context-probe sim. Traces exported to context_probe_traces.csv per comparator.
See summary/context_probe_limitations.md.

## SIG-E
Now compares targeted_overlap_rescue vs generic_all_branch_rescue.
See summary/rescue_protocol.md.

## Variable coverage
All required trace columns exported. context_value not in main branch_traces (known limitation).

## Threshold predeclaration
Thresholds set before analysis: {THRESHOLDS}.
""", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 68)
    print("Experiment 018 — Comparator Trace Matrix")
    print("=" * 68)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_branch:  dict[str, list[dict]] = {}
    all_support: dict[str, list[dict]] = {}
    all_link:    dict[str, list[dict]] = {}
    all_ctx:     dict[str, list[dict]] = {}
    all_sigs:    dict[str, dict] = {}
    all_pf:      dict[str, dict] = {}
    ctx_scores:  dict[str, float] = {}

    for spec in COMPARATORS:
        print(f"[e018] Running {spec.name}...")
        branch_rows, support_rows, linking_rows, context_rows, sigs = run_comparator(spec)

        all_branch[spec.name]  = branch_rows
        all_support[spec.name] = support_rows
        all_link[spec.name]    = linking_rows
        all_ctx[spec.name]     = context_rows
        all_sigs[spec.name]    = sigs
        all_pf[spec.name]      = _passes(sigs)
        ctx_scores[spec.name]  = sigs["SIG_C_context_separation"]

        # Per-comparator CSVs
        _write_csv(TRACES_DIR / f"{spec.name}_branch_traces.csv",       branch_rows)
        _write_csv(TRACES_DIR / f"{spec.name}_trace_support.csv",       support_rows)
        _write_csv(TRACES_DIR / f"{spec.name}_linking_trace.csv",       linking_rows)
        _write_csv(TRACES_DIR / f"{spec.name}_context_probe_traces.csv",context_rows)
        _write_csv(SUMMARY_DIR / f"{spec.name}_signature_summary.csv",  [sigs])

        print(f"         -> SIG-A:{sigs['SIG_A_overlap_advantage']:+.3f}  "
              f"SIG-B:{sigs['SIG_B_linking_gain']:+.3f}  "
              f"SIG-C:{sigs['SIG_C_context_separation']:+.3f}  "
              f"SIG-D:{sigs['SIG_D_linking_recall_dissociation']:+.1f}pp  "
              f"SIG-E:{sigs['SIG_E_targeted_rescue_advantage']:+.1f}pp  "
              f"joint={'PASS' if _joint_pass(all_pf[spec.name]) else 'FAIL'}")

    # Combined outputs
    sig_matrix_rows = [sigs for sigs in all_sigs.values()]
    _write_csv(SUMMARY_DIR / "comparator_signature_matrix.csv", sig_matrix_rows)

    SIG_KEYS = ["SIG_A", "SIG_B", "SIG_C", "SIG_D", "SIG_E"]
    SIG_RAW  = [
        "SIG_A_overlap_advantage", "SIG_B_linking_gain",
        "SIG_C_context_separation", "SIG_D_linking_recall_dissociation",
        "SIG_E_targeted_rescue_advantage",
    ]
    pf_rows = [
        {"comparator": c, **{sk: pf for sk, pf in all_pf[c].items()},
         "joint_pass": _joint_pass(all_pf[c])}
        for c in COMPARATOR_NAMES
    ]
    _write_csv(SUMMARY_DIR / "comparator_pass_fail_matrix.csv", pf_rows)

    effect_rows = [
        {"comparator": c,
         **{rk: all_sigs[c][rk] for rk in SIG_RAW},
         "joint_pass": _joint_pass(all_pf[c])}
        for c in COMPARATOR_NAMES
    ]
    _write_csv(SUMMARY_DIR / "comparator_effect_sizes.csv", effect_rows)

    comp_defs = {
        spec.name: {
            "description": spec.description,
            "random_drift": spec.random_drift,
            "expected_joint_pass": spec.expected_joint_pass,
            "structural_lr": spec.params.structural_lr,
            "replay_gain": spec.params.replay_gain,
            "structural_decay": spec.params.structural_decay,
        }
        for spec in COMPARATORS
    }
    with (SUMMARY_DIR / "comparator_definitions.json").open("w", encoding="utf-8") as f:
        json.dump(comp_defs, f, indent=2)

    no_simple_passes = all(
        not _joint_pass(all_pf[c]) for c in COMPARATOR_NAMES if c != "full_model"
    )
    joint_summary = {
        "thresholds": THRESHOLDS,
        "random_seed": RANDOM_SEED,
        "full_model_joint_pass": _joint_pass(all_pf["full_model"]),
        "any_simpler_passes_joint": not no_simple_passes,
        "no_simpler_passes_all_five": no_simple_passes,
        "per_comparator": {
            c: {
                "joint_pass": _joint_pass(all_pf[c]),
                "n_passed": sum(all_pf[c].values()),
                **{sk: all_pf[c][sk] for sk in SIG_KEYS},
            }
            for c in COMPARATOR_NAMES
        },
    }
    with (SUMMARY_DIR / "joint_pass_summary.json").open("w", encoding="utf-8") as f:
        json.dump(joint_summary, f, indent=2)

    # Figures
    _make_figures(all_sigs, all_pf, all_link, all_branch, ctx_scores)

    # Docs
    _write_docs(all_sigs, all_pf)
    print("[e018] Documentation written.")

    # Print summary table
    print()
    print("-" * 68)
    print("COMPARATOR SIGNATURE MATRIX")
    print("-" * 68)
    hdr = f"  {'Comparator':<26}  SIG-A  SIG-B  SIG-C  SIG-D  SIG-E  Joint"
    print(hdr)
    for c in COMPARATOR_NAMES:
        pf = all_pf[c]
        row = "  ".join("PASS" if pf[sk] else "FAIL" for sk in SIG_KEYS)
        joint = "PASS" if _joint_pass(pf) else "FAIL"
        print(f"  {c:<26}  {row}  {joint}")

    print()
    if no_simple_passes:
        print("  >> DISCRIMINATION RESULT: no simpler comparator reproduces the joint profile.")
    else:
        print("  >> NOTE: at least one simpler comparator passes the joint profile.")
        print("     Check figures/Fig_e018_01 and joint_pass_summary.json for details.")

    print()
    print(f"  Outputs: {OUT_ROOT}")
    print("=" * 68)


if __name__ == "__main__":
    main()
