"""
Experiment 007 – Branch Heterogeneity
======================================
Question: Does the theory still work when branches are not interchangeable?

Heterogeneous network
---------------------
  b0  – "plastic-fast"   : M_b_init = 0.70, decay_rate = 0.010
          High baseline, fast-forgetting. Allocated to mu1.
  b1  – "balanced"       : M_b_init = 0.50, decay_rate = 0.005
          Overlap branch.  Shared by both traces.
  b2  – "stable-slow"    : M_b_init = 0.30, decay_rate = 0.003
          Low baseline, slow decay.  Allocated to mu2.
          Builds slowly but persists.
  b3  – "dormant"        : M_b_init = 0.15, decay_rate = 0.005
          Very low baseline, rarely recruited.

Compared against:
  Homogeneous baseline (all branches M_b_init = 0.50, decay_rate = 0.005)
  using the same trace allocations and encoding protocol.

Metrics
-------
  1. Linking L(mu1, mu2) – does it emerge despite heterogeneity?
  2. Context separation  – does context-sensitive recall survive?
  3. Preferred-branch test – does b0's head-start create early dominance?
  4. Forgetting test – after consolidation, 20 null steps;
                       does b0 (high decay) forget faster than b2 (low decay)?

Expected outcomes
-----------------
  - Linking emerges in both conditions (overlap at b1 is the driver).
  - Context separation survives: correct context still wins.
  - b0 shows highest M_b early; b2 catches up slowly over repeated nights.
  - After consolidation stops, b0 decays fastest.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cytodend_accessmodel.simulator import CytodendAccessModelSimulator
from cytodend_accessmodel.contracts import (
    ConsolidationWindow,
    DynamicsParameters,
    EngramTrace,
    TraceAllocation,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BRANCH_IDS = ["b0", "b1", "b2", "b3"]
N_NIGHTS   = 6
PASSES_PER_NIGHT = 3

BASE_PARAMS = DynamicsParameters(
    structural_lr=0.18,
    replay_gain=0.80,
    eligibility_decay=0.12,
    structural_decay=0.006,
    structural_gain=6.0,
    structural_max=1.0,
    translation_decay=0.05,
    sleep_gain=0.0,
)

MU1_ALLOC = TraceAllocation(trace_id="mu1", branch_weights={"b0": 0.90, "b1": 0.85, "b2": 0.05, "b3": 0.05})
MU2_ALLOC = TraceAllocation(trace_id="mu2", branch_weights={"b0": 0.05, "b1": 0.85, "b2": 0.90, "b3": 0.05})

CUE_MU1_CORRECT   = {"b0": 1.0, "b1": 0.8, "b2": 0.0, "b3": 0.0}
CUE_MU1_WRONG_CTX = {"b0": 0.0, "b1": 0.2, "b2": 1.0, "b3": 0.0}  # context mismatch
CUE_MU2_CORRECT   = {"b0": 0.0, "b1": 0.8, "b2": 1.0, "b3": 0.0}
CUE_MU2_WRONG_CTX = {"b0": 1.0, "b1": 0.2, "b2": 0.0, "b3": 0.0}

# Heterogeneous branch properties
HETERO_INIT: dict[str, float] = {"b0": 0.70, "b1": 0.50, "b2": 0.30, "b3": 0.15}
HETERO_DECAY: dict[str, float] = {"b0": 0.010, "b1": 0.005, "b2": 0.003, "b3": 0.005}

# Homogeneous branch properties (matching exp001 baseline)
HOMO_INIT:  dict[str, float] = {"b0": 0.50, "b1": 0.50, "b2": 0.50, "b3": 0.50}
HOMO_DECAY: dict[str, float] = {"b0": 0.005, "b1": 0.005, "b2": 0.005, "b3": 0.005}

N_NULL_STEPS = 20  # null consolidation passes for forgetting test

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(init: dict[str, float], decay: dict[str, float]) -> CytodendAccessModelSimulator:
    sim = CytodendAccessModelSimulator.from_branch_ids(BRANCH_IDS, parameters=BASE_PARAMS)
    for bid in BRANCH_IDS:
        sim.branches[bid].structural.accessibility = init[bid]
        sim.branches[bid].structural.decay_rate = decay[bid]
        # Recompute slow/effective access to match new M_b
        from cytodend_accessmodel.simulator import _sigmoid, _clamp01
        sim.branches[bid].slow_access = _sigmoid(
            BASE_PARAMS.structural_gain * init[bid]
        )
        sim.branches[bid].effective_access = _clamp01(
            sim.branches[bid].fast_access * sim.branches[bid].slow_access
        )
    sim.traces["mu1"] = EngramTrace(trace_id="mu1", allocation=MU1_ALLOC)
    sim.traces["mu2"] = EngramTrace(trace_id="mu2", allocation=MU2_ALLOC)
    return sim


def _encode(sim: CytodendAccessModelSimulator, n: int = 2) -> None:
    for _ in range(n):
        sim.apply_cue(CUE_MU1_CORRECT)
    for _ in range(n):
        sim.apply_cue(CUE_MU2_CORRECT)


def _consolidate_night(sim: CytodendAccessModelSimulator) -> None:
    win = ConsolidationWindow(
        replay_trace_ids=["mu1", "mu2"],
        modulatory_drive=1.0,
    )
    for _ in range(PASSES_PER_NIGHT):
        sim.run_consolidation(win)


def _mb(sim: CytodendAccessModelSimulator) -> dict[str, float]:
    return {bid: sim.branches[bid].structural.accessibility for bid in BRANCH_IDS}


def _linking(sim: CytodendAccessModelSimulator) -> float:
    return sum(
        MU1_ALLOC.branch_weights.get(bid, 0.0)
        * MU2_ALLOC.branch_weights.get(bid, 0.0)
        * sim.branches[bid].structural.accessibility
        for bid in BRANCH_IDS
    )


def _recall_pair(
    sim: CytodendAccessModelSimulator,
    cue_correct: dict[str, float],
    cue_wrong: dict[str, float],
    trace_id: str,
) -> tuple[float, float]:
    """Return (support_with_correct_cue, support_with_wrong_context_cue) for trace_id."""
    sim.apply_cue(cue_correct)
    r_correct = {rs.trace_id: rs for rs in sim.compute_recall_supports()}
    sim.apply_cue(cue_wrong)
    r_wrong = {rs.trace_id: rs for rs in sim.compute_recall_supports()}
    rc = r_correct.get(trace_id)
    rw = r_wrong.get(trace_id)
    return (rc.support if rc else 0.0), (rw.support if rw else 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # -----------------------------------------------------------------------
    # Build both simulators
    # -----------------------------------------------------------------------
    sim_homo  = _build(HOMO_INIT,  HOMO_DECAY)
    sim_hetero = _build(HETERO_INIT, HETERO_DECAY)

    _encode(sim_homo)
    _encode(sim_hetero)

    # Track M_b per night
    nights_homo:   list[dict[str, float]] = [_mb(sim_homo)]
    nights_hetero: list[dict[str, float]] = [_mb(sim_hetero)]

    for _ in range(N_NIGHTS):
        _consolidate_night(sim_homo)
        _consolidate_night(sim_hetero)
        nights_homo.append(_mb(sim_homo))
        nights_hetero.append(_mb(sim_hetero))

    # -----------------------------------------------------------------------
    # Forgetting test: N_NULL_STEPS null consolidation passes (no replay)
    # -----------------------------------------------------------------------
    null_win = ConsolidationWindow(replay_trace_ids=[], modulatory_drive=0.0)
    mb_post_consolidation_homo   = _mb(sim_homo)
    mb_post_consolidation_hetero = _mb(sim_hetero)

    for _ in range(N_NULL_STEPS):
        sim_homo.run_consolidation(null_win)
        sim_hetero.run_consolidation(null_win)

    mb_after_forget_homo   = _mb(sim_homo)
    mb_after_forget_hetero = _mb(sim_hetero)

    # -----------------------------------------------------------------------
    # Context separation
    # -----------------------------------------------------------------------
    r_mu1_correct_homo,  r_mu1_wrong_homo  = _recall_pair(sim_homo,  CUE_MU1_CORRECT, CUE_MU1_WRONG_CTX, "mu1")
    r_mu2_correct_homo,  r_mu2_wrong_homo  = _recall_pair(sim_homo,  CUE_MU2_CORRECT, CUE_MU2_WRONG_CTX, "mu2")
    r_mu1_correct_hetero, r_mu1_wrong_hetero = _recall_pair(sim_hetero, CUE_MU1_CORRECT, CUE_MU1_WRONG_CTX, "mu1")
    r_mu2_correct_hetero, r_mu2_wrong_hetero = _recall_pair(sim_hetero, CUE_MU2_CORRECT, CUE_MU2_WRONG_CTX, "mu2")

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print("\n=== Experiment 007: Branch Heterogeneity ===\n")

    for label, nights in [("Homogeneous", nights_homo), ("Heterogeneous", nights_hetero)]:
        print(f"--- {label}: M_b per branch per night ---")
        print(f"{'Night':>6}  {'b0(plastic)':>12}  {'b1(overlap)':>12}  {'b2(stable)':>12}  {'b3(dormant)':>12}  {'Link':>7}")
        for n, mb in enumerate(nights):
            sim_ref = sim_homo if label == "Homogeneous" else sim_hetero
            lk = sum(
                MU1_ALLOC.branch_weights.get(bid, 0.0)
                * MU2_ALLOC.branch_weights.get(bid, 0.0)
                * mb[bid]
                for bid in BRANCH_IDS
            )
            print(f"{n:>6}  {mb['b0']:>12.4f}  {mb['b1']:>12.4f}  {mb['b2']:>12.4f}  {mb['b3']:>12.4f}  {lk:>7.4f}")
        print()

    print("--- Linking L(mu1, mu2) at night 6 ---")
    lk_homo   = _linking(sim_homo)
    lk_hetero = _linking(sim_hetero)
    print(f"  Homogeneous:  {lk_homo:.4f}")
    print(f"  Heterogeneous:{lk_hetero:.4f}")

    print("\n--- Context separation after forgetting test ---")
    print(f"{'Condition':<16}  {'Homo R_correct':>14}  {'Homo R_wrong':>12}  {'Hetero R_correct':>16}  {'Hetero R_wrong':>14}")
    print(f"  mu1 recall:   {r_mu1_correct_homo:>14.4f}  {r_mu1_wrong_homo:>12.4f}  {r_mu1_correct_hetero:>16.4f}  {r_mu1_wrong_hetero:>14.4f}")
    print(f"  mu2 recall:   {r_mu2_correct_homo:>14.4f}  {r_mu2_wrong_homo:>12.4f}  {r_mu2_correct_hetero:>16.4f}  {r_mu2_wrong_hetero:>14.4f}")

    print("\n--- Forgetting test: M_b decay after null consolidation ---")
    print(f"{'Branch':<8}  {'Homo: post-consol':>18}  {'Homo: post-forget':>18}  {'Homo loss':>10}  "
          f"{'Hetero: post-consol':>20}  {'Hetero: post-forget':>20}  {'Hetero loss':>12}")
    for bid in BRANCH_IDS:
        hl = mb_post_consolidation_homo[bid]   - mb_after_forget_homo[bid]
        el = mb_post_consolidation_hetero[bid] - mb_after_forget_hetero[bid]
        print(
            f"{bid:<8}  {mb_post_consolidation_homo[bid]:>18.4f}  {mb_after_forget_homo[bid]:>18.4f}  {hl:>10.4f}"
            f"  {mb_post_consolidation_hetero[bid]:>20.4f}  {mb_after_forget_hetero[bid]:>20.4f}  {el:>12.4f}"
        )

    # -----------------------------------------------------------------------
    # Acceptance criteria
    # -----------------------------------------------------------------------
    print("\n--- Acceptance criteria ---")

    # C1: Linking emerges in both conditions
    c1 = lk_homo > 0.25 and lk_hetero > 0.20
    print(f"  C1  Linking > threshold (homo>0.25, hetero>0.20):  "
          f"homo={lk_homo:.4f}  hetero={lk_hetero:.4f}  {'PASS' if c1 else 'FAIL'}")

    # C2: Context separation survives heterogeneity
    ctx_sep_hetero = (r_mu1_correct_hetero > r_mu1_wrong_hetero and
                      r_mu2_correct_hetero > r_mu2_wrong_hetero)
    c2 = ctx_sep_hetero
    print(f"  C2  Context separation in heterogeneous condition:  {'PASS' if c2 else 'FAIL'}")
    print(f"        mu1 correct={r_mu1_correct_hetero:.4f} > wrong={r_mu1_wrong_hetero:.4f}")
    print(f"        mu2 correct={r_mu2_correct_hetero:.4f} > wrong={r_mu2_wrong_hetero:.4f}")

    # C3: b0 (high decay) forgets faster than b2 (low decay) in heterogeneous condition
    b0_loss_hetero = mb_post_consolidation_hetero["b0"] - mb_after_forget_hetero["b0"]
    b2_loss_hetero = mb_post_consolidation_hetero["b2"] - mb_after_forget_hetero["b2"]
    c3 = b0_loss_hetero > b2_loss_hetero
    print(f"  C3  b0 (decay=0.010) forgets faster than b2 (decay=0.003):  "
          f"b0_loss={b0_loss_hetero:.4f}  b2_loss={b2_loss_hetero:.4f}  {'PASS' if c3 else 'FAIL'}")

    # C4: b0 early advantage — at night 1, b0 > b2 in heterogeneous condition
    b0_n1 = nights_hetero[1]["b0"]
    b2_n1 = nights_hetero[1]["b2"]
    c4 = b0_n1 > b2_n1
    print(f"  C4  b0 has early advantage over b2 (night 1):  "
          f"b0={b0_n1:.4f}  b2={b2_n1:.4f}  {'PASS' if c4 else 'FAIL'}")

    all_pass = c1 and c2 and c3 and c4
    print(f"\n  Result: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    if all_pass:
        print(
            "  Core mechanisms survive branch heterogeneity:\n"
            "  linking emerges, context separation persists,\n"
            "  and branch-specific decay rates create differential forgetting."
        )


if __name__ == "__main__":
    main()
