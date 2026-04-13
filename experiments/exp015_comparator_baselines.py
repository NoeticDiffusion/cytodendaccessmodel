"""
Experiment 015 - Comparator Baselines
======================================
Directly answers the peer reviewer's strongest objection:

  "The LR-ablation is good, but I still want to see competing models that
   can explain at least some of the effects without slow structural writing."

Four baselines are compared against the full model on the paper's five
most exposed signatures:

  SIG-A  Overlap-branch structural strengthening (ΔM_b1)
  SIG-B  Linking gain after consolidation (% change in L_mu1_mu2)
  SIG-C  Context disambiguation (correct - wrong context support)
  SIG-D  Linking vs recall vulnerability dissociation (link_drop - recall_drop %)
  SIG-E  Targeted rescue selectivity (overlap-rescue advantage %)

Baselines
---------
  full_model              : standard simulator, structural_lr = 0.18
  fast_context_only       : structural state fixed (no learning), context bias present
  replay_no_structure     : replay updates transient P_b/E_b but M_b cannot be written
                            (equivalent to structural_lr = 0, matching exp014 ablation,
                            included here for comparator completeness)
  random_slow_drift       : a slow term of matched scale drives M_b randomly
                            (Gaussian drift with sigma matching full-model mean shift)
  fixed_allocation_only   : hand-designed overlap allocation preserved, but
                            structural updating entirely removed

A strong result is NOT that the full model wins every metric by a large
margin. A strong result is that NO simpler baseline reproduces the JOINT
signature profile of the full model.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import random
from copy import deepcopy
from math import sqrt

from cytodend_accessmodel.simulator import CytodendAccessModelSimulator, _sigmoid, _clamp01
from cytodend_accessmodel.contracts import (
    BranchState,
    ConsolidationWindow,
    DynamicsParameters,
    EngramTrace,
    StructuralState,
    TraceAllocation,
)

# ---------------------------------------------------------------------------
# Canonical network (same as exp001 / exp014)
# ---------------------------------------------------------------------------
BRANCH_IDS = ["b0", "b1", "b2", "b3"]

MU1_ALLOC = TraceAllocation(
    trace_id="mu1",
    branch_weights={"b0": 0.90, "b1": 0.85, "b2": 0.05, "b3": 0.05},
)
MU2_ALLOC = TraceAllocation(
    trace_id="mu2",
    branch_weights={"b0": 0.05, "b1": 0.85, "b2": 0.90, "b3": 0.05},
)

# Cue inputs
MU1_CUE = {"b0": 1.0, "b1": 0.8, "b2": 0.0, "b3": 0.0}
MU2_CUE = {"b0": 0.0, "b1": 0.8, "b2": 1.0, "b3": 0.0}
AMBIG_CUE = {"b0": 0.5, "b1": 0.5, "b2": 0.5, "b3": 0.5}
B1_CUE = {"b0": 0.0, "b1": 1.0, "b2": 0.0, "b3": 0.0}

# Context biases (correct context for mu_alpha = b0/b1, mu_beta = b2/b3)
ALPHA_BIAS = {"b0": 0.5, "b1": 0.5, "b2": -0.5, "b3": -0.5}
BETA_BIAS = {"b0": -0.5, "b1": -0.5, "b2": 0.5, "b3": 0.5}

MU_ALPHA_ALLOC = TraceAllocation(
    trace_id="mu_alpha",
    branch_weights={"b0": 0.90, "b1": 0.05, "b2": 0.05, "b3": 0.00},
)
MU_BETA_ALLOC = TraceAllocation(
    trace_id="mu_beta",
    branch_weights={"b0": 0.05, "b1": 0.05, "b2": 0.90, "b3": 0.00},
)

# ---------------------------------------------------------------------------
# Parameters
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
)

# fast_context_only: no structural learning, same context gain
FAST_CONTEXT_PARAMS = DynamicsParameters(
    structural_lr=0.0,
    replay_gain=0.80,
    eligibility_decay=0.12,
    structural_decay=0.0,    # no decay either → truly fixed structural state
    structural_gain=6.0,
    structural_max=1.0,
    translation_decay=0.05,
    sleep_gain=0.0,
    context_gain=1.0,
    structural_noise=0.0,
)

# replay_no_structure: identical to the LR=0 ablation from exp014
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
)

# fixed_allocation_only: hand-designed overlap allocation, no structural updating
FIXED_ALLOC_PARAMS = DynamicsParameters(
    structural_lr=0.0,
    replay_gain=0.0,         # replay doesn't write P_b either
    eligibility_decay=0.0,
    structural_decay=0.0,
    structural_gain=6.0,
    structural_max=1.0,
    translation_decay=0.0,
    sleep_gain=0.0,
    context_gain=1.0,
    structural_noise=0.0,
)


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def _build(params: DynamicsParameters) -> CytodendAccessModelSimulator:
    sim = CytodendAccessModelSimulator.from_branch_ids(BRANCH_IDS, parameters=params)
    sim.traces["mu1"] = EngramTrace(trace_id="mu1", allocation=MU1_ALLOC)
    sim.traces["mu2"] = EngramTrace(trace_id="mu2", allocation=MU2_ALLOC)
    return sim


def _build_context_sim(params: DynamicsParameters) -> CytodendAccessModelSimulator:
    sim = CytodendAccessModelSimulator.from_branch_ids(BRANCH_IDS, parameters=params)
    sim.traces["mu_alpha"] = EngramTrace(
        trace_id="mu_alpha", allocation=MU_ALPHA_ALLOC, context="alpha"
    )
    sim.traces["mu_beta"] = EngramTrace(
        trace_id="mu_beta", allocation=MU_BETA_ALLOC, context="beta"
    )
    return sim


def _mb(sim: CytodendAccessModelSimulator) -> dict[str, float]:
    return {b: sim.branches[b].structural.accessibility for b in BRANCH_IDS}


def _linking(sim: CytodendAccessModelSimulator) -> float:
    return sum(
        MU1_ALLOC.branch_weights.get(b, 0.0)
        * MU2_ALLOC.branch_weights.get(b, 0.0)
        * sim.branches[b].structural.accessibility
        for b in BRANCH_IDS
    )


def _encode(sim: CytodendAccessModelSimulator) -> None:
    for _ in range(2):
        sim.apply_cue(MU1_CUE)
    for _ in range(2):
        sim.apply_cue(MU2_CUE)


def _consolidate(
    sim: CytodendAccessModelSimulator,
    n_passes: int = 9,
    drive: float = 1.0,
    replay: list[str] | None = None,
) -> None:
    win = ConsolidationWindow(
        replay_trace_ids=replay if replay is not None else ["mu1", "mu2"],
        modulatory_drive=drive,
    )
    for _ in range(n_passes):
        sim.run_consolidation(win)


def _null_consolidate(sim: CytodendAccessModelSimulator, n: int = 9) -> None:
    win = ConsolidationWindow(replay_trace_ids=[], modulatory_drive=0.0)
    for _ in range(n):
        sim.run_consolidation(win)


def _recall(sim: CytodendAccessModelSimulator, cue: dict, tid: str) -> float:
    sim.apply_cue(cue)
    rmap = {rs.trace_id: rs for rs in sim.compute_recall_supports()}
    return rmap[tid].support if tid in rmap else 0.0


def _recovery_pct(post: float, dmg: float, healthy: float) -> float:
    denom = healthy - dmg
    return (post - dmg) / denom * 100.0 if abs(denom) > 1e-9 else 0.0


# ---------------------------------------------------------------------------
# Random-drift baseline: injects Gaussian M_b drift of matched scale
# ---------------------------------------------------------------------------
def _apply_random_drift(sim: CytodendAccessModelSimulator, sigma: float) -> None:
    """Apply one step of non-specific Gaussian drift to all M_b values."""
    import random as _random
    for branch in sim.branches.values():
        drift = _random.gauss(0.0, sigma)
        m_max = branch.structural.max_accessibility
        branch.structural.accessibility = max(
            0.0, min(m_max, branch.structural.accessibility + drift)
        )
        branch.slow_access = _sigmoid(
            sim.parameters.structural_gain * branch.structural.accessibility
        )
        branch.effective_access = _clamp01(branch.fast_access * branch.slow_access)


def _consolidate_random_drift(
    sim: CytodendAccessModelSimulator,
    n_passes: int = 9,
    drift_sigma: float = 0.025,
) -> None:
    """Consolidation pass with random (non-specific) slow drift."""
    for _ in range(n_passes):
        _apply_random_drift(sim, drift_sigma)


# ---------------------------------------------------------------------------
# SIG-A + SIG-B: Structural writing and linking
# ---------------------------------------------------------------------------

def sig_ab(label: str, params: DynamicsParameters, *, random_drift: bool = False) -> dict:
    sim = _build(params)
    mb_pre = _mb(sim)
    lk_pre = _linking(sim)
    _encode(sim)

    if random_drift:
        # Use a drift sigma matched to the mean per-step shift of the full model
        # (empirically ~ 0.025 per pass over 9 passes gives ~0.22 rise on active branches)
        _consolidate_random_drift(sim, n_passes=9, drift_sigma=0.025)
    else:
        _consolidate(sim, n_passes=9)

    mb_post = _mb(sim)
    lk_post = _linking(sim)

    delta_b1 = mb_post["b1"] - mb_pre["b1"]
    delta_b0 = mb_post["b0"] - mb_pre["b0"]
    lk_change_pct = (lk_post - lk_pre) / max(abs(lk_pre), 1e-9) * 100.0
    overlap_advantage = delta_b1 - delta_b0   # b1 is overlap; b0 is single-trace

    return {
        "label": label,
        "mb_pre_b1": mb_pre["b1"],
        "mb_post_b1": mb_post["b1"],
        "delta_b1": delta_b1,
        "delta_b0": delta_b0,
        "overlap_advantage": overlap_advantage,
        "lk_pre": lk_pre,
        "lk_post": lk_post,
        "lk_change_pct": lk_change_pct,
    }


# ---------------------------------------------------------------------------
# SIG-C: Context disambiguation
# ---------------------------------------------------------------------------

def sig_c(label: str, params: DynamicsParameters, *, random_drift: bool = False) -> dict:
    sim = _build_context_sim(params)

    # Encode both traces
    for _ in range(2):
        sim.apply_cue({"b0": 1.0, "b1": 0.0, "b2": 0.0, "b3": 0.0}, context="alpha")
    for _ in range(2):
        sim.apply_cue({"b0": 0.0, "b1": 0.0, "b2": 1.0, "b3": 0.0}, context="beta")

    # Consolidate
    if random_drift:
        _consolidate_random_drift(sim, n_passes=9, drift_sigma=0.025)
    else:
        win = ConsolidationWindow(
            replay_trace_ids=["mu_alpha", "mu_beta"], modulatory_drive=1.0
        )
        for _ in range(9):
            sim.run_consolidation(win)

    # Probe with ambiguous cue under correct / wrong context
    sim_alpha = deepcopy(sim)
    sim_beta = deepcopy(sim)

    sim_alpha.apply_cue(AMBIG_CUE, context="alpha", context_bias=ALPHA_BIAS)
    rmap_alpha = {rs.trace_id: rs for rs in sim_alpha.compute_recall_supports()}
    r_alpha_correct = rmap_alpha.get("mu_alpha", type("X", (), {"support": 0.0})()).support
    r_alpha_wrong   = rmap_alpha.get("mu_beta",  type("X", (), {"support": 0.0})()).support

    sim_beta.apply_cue(AMBIG_CUE, context="beta", context_bias=BETA_BIAS)
    rmap_beta = {rs.trace_id: rs for rs in sim_beta.compute_recall_supports()}
    r_beta_correct = rmap_beta.get("mu_beta",  type("X", (), {"support": 0.0})()).support
    r_beta_wrong   = rmap_beta.get("mu_alpha", type("X", (), {"support": 0.0})()).support

    context_sep = ((r_alpha_correct - r_alpha_wrong) + (r_beta_correct - r_beta_wrong)) / 2.0

    return {
        "label": label,
        "r_alpha_correct": r_alpha_correct,
        "r_alpha_wrong": r_alpha_wrong,
        "r_beta_correct": r_beta_correct,
        "r_beta_wrong": r_beta_wrong,
        "context_sep": context_sep,
    }


# ---------------------------------------------------------------------------
# SIG-D: Linking vs recall vulnerability dissociation
# ---------------------------------------------------------------------------

def sig_d(label: str, params: DynamicsParameters, *, random_drift: bool = False) -> dict:
    sim = _build(params)
    _encode(sim)

    if random_drift:
        _consolidate_random_drift(sim, n_passes=9, drift_sigma=0.025)
    else:
        _consolidate(sim, n_passes=9)

    # Healthy state
    h_link = _linking(sim)
    h_recall = _recall(deepcopy(sim), MU1_CUE, "mu1")

    # Apply focal pathology: raise decay rate on overlap branch and run null
    sim_path = deepcopy(sim)
    sim_path.branches["b1"].structural.decay_rate = 0.030
    _null_consolidate(sim_path, n=9)

    d_link = _linking(sim_path)
    d_recall = _recall(deepcopy(sim_path), MU1_CUE, "mu1")

    link_drop_pct   = (h_link - d_link) / max(abs(h_link), 1e-9) * 100.0
    recall_drop_pct = (h_recall - d_recall) / max(abs(h_recall), 1e-9) * 100.0
    dissociation = link_drop_pct - recall_drop_pct

    return {
        "label": label,
        "h_link": h_link,
        "d_link": d_link,
        "h_recall": h_recall,
        "d_recall": d_recall,
        "link_drop_pct": link_drop_pct,
        "recall_drop_pct": recall_drop_pct,
        "dissociation": dissociation,
    }


# ---------------------------------------------------------------------------
# SIG-E: Targeted rescue selectivity
# ---------------------------------------------------------------------------

def sig_e(label: str, params: DynamicsParameters, *, random_drift: bool = False) -> dict:
    sim = _build(params)
    _encode(sim)

    if random_drift:
        _consolidate_random_drift(sim, n_passes=9, drift_sigma=0.025)
    else:
        _consolidate(sim, n_passes=9)

    h_link = _linking(sim)

    # Apply focal damage
    sim.branches["b1"].structural.decay_rate = 0.030
    _null_consolidate(sim, n=9)
    d_link = _linking(sim)

    # Standard rescue
    sim_std = deepcopy(sim)
    if random_drift:
        _consolidate_random_drift(sim_std, n_passes=9, drift_sigma=0.025)
    else:
        _consolidate(sim_std, n_passes=9)
    link_std = _linking(sim_std)

    # Overlap-targeted rescue (pre-cue b1 to build eligibility then consolidate)
    sim_ovlp = deepcopy(sim)
    for _ in range(3):
        for _ in range(3):
            sim_ovlp.apply_cue(B1_CUE)
        if random_drift:
            _consolidate_random_drift(sim_ovlp, n_passes=3, drift_sigma=0.025)
        else:
            _consolidate(sim_ovlp, n_passes=3)
    link_ovlp = _linking(sim_ovlp)

    rec_std  = _recovery_pct(link_std,  d_link, h_link)
    rec_ovlp = _recovery_pct(link_ovlp, d_link, h_link)
    rescue_advantage = rec_ovlp - rec_std

    return {
        "label": label,
        "h_link": h_link,
        "d_link": d_link,
        "link_std": link_std,
        "link_ovlp": link_ovlp,
        "rec_std": rec_std,
        "rec_ovlp": rec_ovlp,
        "rescue_advantage": rescue_advantage,
    }


# ---------------------------------------------------------------------------
# Main: run all baselines on all signatures
# ---------------------------------------------------------------------------

BASELINES = [
    ("full_model",            BASE_PARAMS,             False),
    ("fast_context_only",     FAST_CONTEXT_PARAMS,     False),
    ("replay_no_structure",   REPLAY_NO_STRUCT_PARAMS, False),
    ("random_slow_drift",     BASE_PARAMS,             True),   # uses BASE_PARAMS but replaces consolidation
    ("fixed_allocation_only", FIXED_ALLOC_PARAMS,      False),
]


def main() -> None:
    random.seed(42)

    results_ab: list[dict] = []
    results_c:  list[dict] = []
    results_d:  list[dict] = []
    results_e:  list[dict] = []

    for label, params, rand_drift in BASELINES:
        results_ab.append(sig_ab(label, params, random_drift=rand_drift))
        results_c.append(sig_c(label, params,  random_drift=rand_drift))
        results_d.append(sig_d(label, params,  random_drift=rand_drift))
        results_e.append(sig_e(label, params,  random_drift=rand_drift))

    print("\n" + "=" * 78)
    print("Experiment 015 - Comparator Baselines")
    print("=" * 78)
    print()
    print("Full model = cytodend_accessmodel with structural_lr = 0.18")
    print("Baselines  = four simpler alternatives (see module docstring)")
    print()

    # ---- SIG-A + SIG-B ----
    print("-" * 78)
    print("SIG-A  Overlap-branch structural strengthening  &  SIG-B  Linking gain")
    print("-" * 78)
    hdr = f"{'Baseline':<26}  {'dM_b1':>8}  {'dM_b0':>8}  {'OvlpAdv':>8}  {'Lk_pre':>7}  {'Lk_post':>7}  {'Lk%chg':>8}"
    print(hdr)
    for r in results_ab:
        print(
            f"{r['label']:<26}  {r['delta_b1']:>+8.4f}  {r['delta_b0']:>+8.4f}  "
            f"{r['overlap_advantage']:>+8.4f}  {r['lk_pre']:>7.4f}  {r['lk_post']:>7.4f}  "
            f"{r['lk_change_pct']:>+8.1f}%"
        )

    # ---- SIG-C ----
    print()
    print("-" * 78)
    print("SIG-C  Context disambiguation under ambiguous cue")
    print("-" * 78)
    hdr = f"{'Baseline':<26}  {'Corr_a':>7}  {'Wrong_a':>8}  {'Corr_b':>7}  {'Wrong_b':>8}  {'CtxSep':>7}"
    print(hdr)
    for r in results_c:
        print(
            f"{r['label']:<26}  {r['r_alpha_correct']:>7.4f}  {r['r_alpha_wrong']:>8.4f}  "
            f"{r['r_beta_correct']:>7.4f}  {r['r_beta_wrong']:>8.4f}  {r['context_sep']:>+7.4f}"
        )

    # ---- SIG-D ----
    print()
    print("-" * 78)
    print("SIG-D  Linking vs recall vulnerability dissociation (focal pathology)")
    print("-" * 78)
    hdr = f"{'Baseline':<26}  {'Lk_h':>6}  {'Lk_d':>6}  {'Lk_drop%':>9}  {'Rc_h':>6}  {'Rc_d':>6}  {'Rc_drop%':>9}  {'Dissoc':>7}"
    print(hdr)
    for r in results_d:
        print(
            f"{r['label']:<26}  {r['h_link']:>6.4f}  {r['d_link']:>6.4f}  "
            f"{r['link_drop_pct']:>+9.1f}%  {r['h_recall']:>6.4f}  {r['d_recall']:>6.4f}  "
            f"{r['recall_drop_pct']:>+9.1f}%  {r['dissociation']:>+7.1f}pp"
        )

    # ---- SIG-E ----
    print()
    print("-" * 78)
    print("SIG-E  Targeted rescue selectivity (overlap-targeted vs standard rescue)")
    print("-" * 78)
    hdr = f"{'Baseline':<26}  {'L_h':>6}  {'L_d':>6}  {'L_std':>6}  {'L_ovlp':>6}  {'Rec_std%':>9}  {'Rec_ovlp%':>10}  {'RscAdv%':>8}"
    print(hdr)
    for r in results_e:
        print(
            f"{r['label']:<26}  {r['h_link']:>6.4f}  {r['d_link']:>6.4f}  "
            f"{r['link_std']:>6.4f}  {r['link_ovlp']:>6.4f}  "
            f"{r['rec_std']:>+9.1f}  {r['rec_ovlp']:>+10.1f}  {r['rescue_advantage']:>+8.1f}"
        )

    # ---- Signature profile summary ----
    print()
    print("-" * 78)
    print("JOINT SIGNATURE PROFILE  (pass = directionally correct for full-model claim)")
    print("-" * 78)

    full_ab = results_ab[0]
    full_c  = results_c[0]
    full_d  = results_d[0]
    full_e  = results_e[0]

    # Thresholds derived from full model
    thr_ab_ovlp = full_ab["overlap_advantage"] * 0.5
    thr_ab_lk   = 5.0
    thr_c_sep   = full_c["context_sep"] * 0.5
    thr_d_dis   = 5.0       # pp: linking drops ≥5pp more than recall
    thr_e_adv   = 10.0      # pp: targeted rescue > standard by ≥10pp

    print(f"  Thresholds: OvlpAdv>{thr_ab_ovlp:.3f} | Lk%>{thr_ab_lk:.0f}% | "
          f"CtxSep>{thr_c_sep:.3f} | Dissoc>{thr_d_dis:.0f}pp | RscAdv>{thr_e_adv:.0f}pp")
    print()

    hdr = f"{'Baseline':<26}  {'SIG-A':>6}  {'SIG-B':>6}  {'SIG-C':>6}  {'SIG-D':>6}  {'SIG-E':>6}  {'JOINT':>6}"
    print(hdr)

    for i, (label, _, _) in enumerate(BASELINES):
        rab = results_ab[i]
        rc  = results_c[i]
        rd  = results_d[i]
        re  = results_e[i]

        sa = "PASS" if rab["overlap_advantage"] >= thr_ab_ovlp else "FAIL"
        sb = "PASS" if rab["lk_change_pct"] >= thr_ab_lk else "FAIL"
        sc = "PASS" if rc["context_sep"] >= thr_c_sep else "FAIL"
        sd = "PASS" if rd["dissociation"] >= thr_d_dis else "FAIL"
        se = "PASS" if re["rescue_advantage"] >= thr_e_adv else "FAIL"
        joint = "PASS" if all(x == "PASS" for x in [sa, sb, sc, sd, se]) else "FAIL"

        print(f"  {label:<24}  {sa:>6}  {sb:>6}  {sc:>6}  {sd:>6}  {se:>6}  {joint:>6}")

    print()
    print("Key: SIG-A=overlap branch strengthening  SIG-B=linking gain  SIG-C=context sep")
    print("     SIG-D=linking>recall vulnerability  SIG-E=targeted rescue selectivity")
    print()
    print("A baseline that fails the JOINT profile cannot substitute for the full model.")


if __name__ == "__main__":
    main()
