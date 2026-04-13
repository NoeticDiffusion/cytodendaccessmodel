"""
Experiment 011 – Branch Topology / Neighbourhood Effects
=========================================================
Question: Does local neighbourhood structure change which branches are
co-written and how focal damage spreads functionally?

Spillover mechanism (added to simulator)
-----------------------------------------
When `spillover_rate > 0` and `branch_adjacency` is set, each branch's
replay overlap is augmented by the MEAN of its neighbours' direct overlaps,
scaled by `spillover_rate`.  This models local dendritic calcium propagation:
a strongly-activated branch can partially drive consolidation in its
cytoskeletal neighbourhood.

Network topology: 6-branch line
---------------------------------
  b0 - b1 - b2 - b3 - b4 - b5

Trace allocations:
  mu1: primarily b1  (encoding focus near left of line)
  mu2: primarily b3  (encoding focus near right of line)
  Overlap: none by allocation, but with spillover b2 (between b1 and b3)
           receives contribution from both neighbours.

Three spillover conditions (× healthy vs damaged):
  no_spillover    spillover_rate = 0.00 (baseline)
  mild_spillover  spillover_rate = 0.25
  strong_spillover spillover_rate = 0.50

Metrics
-------
  M_b per branch (does neighbourhood co-strengthen?)
  L(mu1, mu2)    (does spillover create indirect linking?)
  Damage spread: focal b2 damage -> how much does b1 and b3 lose?
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
BRANCH_IDS = ["b0", "b1", "b2", "b3", "b4", "b5"]

# Line adjacency: b_i adjacent to b_{i-1} and b_{i+1}
LINE_ADJACENCY: dict[str, tuple[str, ...]] = {
    "b0": ("b1",),
    "b1": ("b0", "b2"),
    "b2": ("b1", "b3"),
    "b3": ("b2", "b4"),
    "b4": ("b3", "b5"),
    "b5": ("b4",),
}

MU1_ALLOC = TraceAllocation(
    trace_id="mu1",
    branch_weights={"b1": 0.90, "b0": 0.50, "b2": 0.10},
)
MU2_ALLOC = TraceAllocation(
    trace_id="mu2",
    branch_weights={"b3": 0.90, "b4": 0.50, "b2": 0.10},
)

MU1_CUE = {"b0": 0.6, "b1": 1.0, "b2": 0.1}
MU2_CUE = {"b2": 0.1, "b3": 1.0, "b4": 0.6}

SPILLOVER_CONDITIONS = [
    ("no_spillover",     0.00),
    ("mild_spillover",   0.25),
    ("strong_spillover", 0.50),
]

BASE_PARAMS_TEMPLATE = dict(
    structural_lr=0.18,
    replay_gain=0.80,
    eligibility_decay=0.12,
    structural_decay=0.005,
    structural_gain=6.0,
    structural_max=1.0,
    translation_decay=0.05,
    sleep_gain=0.0,
)

N_NIGHTS = 5
PASSES_PER_NIGHT = 3
FOCAL_DECAY = 0.025

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(spillover: float, with_adjacency: bool = True) -> CytodendAccessModelSimulator:
    params = DynamicsParameters(**BASE_PARAMS_TEMPLATE, spillover_rate=spillover)
    sim = CytodendAccessModelSimulator.from_branch_ids(BRANCH_IDS, parameters=params)
    if with_adjacency:
        sim.branch_adjacency = LINE_ADJACENCY
    sim.traces["mu1"] = EngramTrace(trace_id="mu1", allocation=MU1_ALLOC)
    sim.traces["mu2"] = EngramTrace(trace_id="mu2", allocation=MU2_ALLOC)
    return sim


def _encode(sim: CytodendAccessModelSimulator) -> None:
    for _ in range(2):
        sim.apply_cue(MU1_CUE)
    for _ in range(2):
        sim.apply_cue(MU2_CUE)


def _consolidate(sim: CytodendAccessModelSimulator, n: int = N_NIGHTS) -> None:
    win = ConsolidationWindow(
        replay_trace_ids=["mu1", "mu2"],
        modulatory_drive=1.0,
    )
    for _ in range(n * PASSES_PER_NIGHT):
        sim.run_consolidation(win)


def _null(sim: CytodendAccessModelSimulator, n: int) -> None:
    win = ConsolidationWindow(replay_trace_ids=[], modulatory_drive=0.0)
    for _ in range(n * PASSES_PER_NIGHT):
        sim.run_consolidation(win)


def _mb(sim: CytodendAccessModelSimulator) -> dict[str, float]:
    return {b: sim.branches[b].structural.accessibility for b in BRANCH_IDS}


def _linking(sim: CytodendAccessModelSimulator) -> float:
    return sum(
        MU1_ALLOC.branch_weights.get(b, 0.0)
        * MU2_ALLOC.branch_weights.get(b, 0.0)
        * sim.branches[b].structural.accessibility
        for b in BRANCH_IDS
    )


def _recall(sim: CytodendAccessModelSimulator, tid: str) -> float:
    cue = MU1_CUE if tid == "mu1" else MU2_CUE
    sim.apply_cue(cue)
    rmap = {rs.trace_id: rs for rs in sim.compute_recall_supports()}
    return rmap[tid].support if tid in rmap else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== Experiment 011: Branch Topology / Neighbourhood Effects ===\n")
    print("Network: line topology  b0-b1-b2-b3-b4-b5")
    print("  mu1: primarily b1 (left focus), mu2: primarily b3 (right focus)\n")

    # ------------------------------------------------------------------
    # Part A: Healthy consolidation across spillover conditions
    # ------------------------------------------------------------------
    print("--- Part A: M_b per branch after consolidation (healthy) ---")
    print(f"{'Condition':<22}  {'b0':>7}  {'b1':>7}  {'b2(mid)':>8}  {'b3':>7}  {'b4':>7}  {'b5':>7}  {'Link':>7}")

    results_healthy: dict[str, dict] = {}
    for label, spill in SPILLOVER_CONDITIONS:
        sim = _build(spill)
        _encode(sim)
        _consolidate(sim)
        mb = _mb(sim)
        lk = _linking(sim)
        results_healthy[label] = {"mb": mb, "linking": lk, "spillover": spill}
        print(
            f"{label:<22}  {mb['b0']:>7.4f}  {mb['b1']:>7.4f}  {mb['b2']:>8.4f}  "
            f"{mb['b3']:>7.4f}  {mb['b4']:>7.4f}  {mb['b5']:>7.4f}  {lk:>7.4f}"
        )

    # ------------------------------------------------------------------
    # Part B: Spillover spreads functional damage through neighbourhood
    # ------------------------------------------------------------------
    print("\n--- Part B: Focal damage at b2 — impact on b1 and b3 ---")
    print(f"{'Condition':<22}  {'b1_loss':>10}  {'b3_loss':>10}  {'b2_loss':>10}  {'link_loss':>10}")

    for label, spill in SPILLOVER_CONDITIONS:
        sim_pre  = _build(spill)
        _encode(sim_pre)
        _consolidate(sim_pre)
        mb_pre = _mb(sim_pre)
        lk_pre = _linking(sim_pre)

        # Apply damage to b2
        sim_pre.branches["b2"].structural.decay_rate = FOCAL_DECAY
        _null(sim_pre, 3)

        mb_post = _mb(sim_pre)
        lk_post = _linking(sim_pre)

        b1_loss = mb_pre["b1"] - mb_post["b1"]
        b3_loss = mb_pre["b3"] - mb_post["b3"]
        b2_loss = mb_pre["b2"] - mb_post["b2"]
        lk_loss = lk_pre - lk_post
        print(
            f"{label:<22}  {b1_loss:>10.4f}  {b3_loss:>10.4f}  "
            f"{b2_loss:>10.4f}  {lk_loss:>10.4f}"
        )

    # ------------------------------------------------------------------
    # Acceptance criteria
    # ------------------------------------------------------------------
    print("\n--- Acceptance criteria ---")

    r_no   = results_healthy["no_spillover"]
    r_mild = results_healthy["mild_spillover"]
    r_str  = results_healthy["strong_spillover"]

    # C1: Spillover increases M_b at the middle (non-allocated) branch b2
    #     b2 has small allocation (0.10 for mu1, 0.10 for mu2) but is adjacent
    #     to both b1 and b3 which are strongly driven.  Spillover should boost b2.
    c1 = r_str["mb"]["b2"] > r_no["mb"]["b2"]
    print(f"  C1  Spillover increases b2 (mid) M_b:  "
          f"no_spillover={r_no['mb']['b2']:.4f}  strong={r_str['mb']['b2']:.4f}  "
          f"{'PASS' if c1 else 'FAIL'}")

    # C2: Spillover increases linking L(mu1, mu2)
    #     Since b2 gets strengthened via spillover from both sides, and b2 is
    #     in both allocations (weight 0.10), linking should increase.
    c2 = r_str["linking"] > r_no["linking"]
    print(f"  C2  Spillover increases linking L(mu1,mu2):  "
          f"no_spillover={r_no['linking']:.4f}  strong={r_str['linking']:.4f}  "
          f"{'PASS' if c2 else 'FAIL'}")

    # C3: Spillover is monotone with spillover_rate
    c3 = r_str["mb"]["b2"] >= r_mild["mb"]["b2"] >= r_no["mb"]["b2"]
    print(f"  C3  b2 M_b monotone with spillover_rate:  "
          f"no={r_no['mb']['b2']:.4f} <= mild={r_mild['mb']['b2']:.4f} <= strong={r_str['mb']['b2']:.4f}  "
          f"{'PASS' if c3 else 'FAIL'}")

    # C4: Strong spillover increases b1 and b4 M_b (neighborhood co-strengthening)
    #     b0 (b1's outer neighbour) should also receive mild spillover from b1
    c4 = (r_str["mb"]["b0"] > r_no["mb"]["b0"] and
          r_str["mb"]["b4"] > r_no["mb"]["b4"])
    print(f"  C4  Spillover co-strengthens outer neighbours (b0, b4):  "
          f"b0: no={r_no['mb']['b0']:.4f} -> strong={r_str['mb']['b0']:.4f}  "
          f"b4: no={r_no['mb']['b4']:.4f} -> strong={r_str['mb']['b4']:.4f}  "
          f"{'PASS' if c4 else 'FAIL'}")

    all_pass = c1 and c2 and c3 and c4
    print(f"\n  Result: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    if all_pass:
        print(
            "  Neighbourhood spillover co-strengthens adjacent branches,\n"
            "  including branches with minimal direct trace allocation.\n"
            "  Spillover creates indirect linking between traces that have\n"
            "  no shared allocation but share a topological neighbourhood.\n"
            "  This is interpretable as cytoskeletal propagation from active\n"
            "  dendritic subunits to their immediate neighbours."
        )


if __name__ == "__main__":
    main()
