"""
Experiment 014 – Structural Gate Ablation
==========================================
Negative control / ablation for the core paper claim:

  "The slow structural accessibility layer is necessary for
   branch-specific consolidation, overlap-driven linking, and
   selective rescue."

Ablation: structural_lr = 0
-----------------------------
Setting the structural learning rate to zero removes the ability of
replay-like consolidation to write structural accessibility.
M_b can no longer increase; it only decays slowly from its initial value.
A_b^s therefore stays near its initial value and never differentially
strengthens the overlap branch.

This is the minimal and cleanest ablation: it does not touch the fast
dynamics (cue-driven activation), context mechanisms, or readout. It
removes only the capacity to write the slow structural layer.

Three key predictions for the ablation:
  1. Overlap branch shows NO preferential strengthening (delta_M_b1 ≈ 0)
  2. Linking metric does NOT rise above pre-consolidation baseline
  3. Targeted rescue shows NO advantage over standard rescue
     (because rescue operates via the structural layer)

A failure of all three predictions under ablation, combined with their
presence under the full model, constitutes a clean mechanistic control.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from copy import deepcopy

from cytodend_accessmodel.simulator import CytodendAccessModelSimulator, _sigmoid, _clamp01
from cytodend_accessmodel.contracts import (
    ConsolidationWindow,
    DynamicsParameters,
    EngramTrace,
    TraceAllocation,
)

# ---------------------------------------------------------------------------
# Canonical network (same as exp001 / exp009)
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
MU1_CUE = {"b0": 1.0, "b1": 0.8, "b2": 0.0, "b3": 0.0}
MU2_CUE = {"b0": 0.0, "b1": 0.8, "b2": 1.0, "b3": 0.0}
B1_CUE  = {"b0": 0.0, "b1": 1.0, "b2": 0.0, "b3": 0.0}

FULL_PARAMS = DynamicsParameters(
    structural_lr=0.18,
    replay_gain=0.80,
    eligibility_decay=0.12,
    structural_decay=0.005,
    structural_gain=6.0,
    structural_max=1.0,
    translation_decay=0.05,
    sleep_gain=0.0,
)

ABLATED_PARAMS = DynamicsParameters(
    structural_lr=0.00,    # <-- ablation: no structural writing
    replay_gain=0.80,
    eligibility_decay=0.12,
    structural_decay=0.005,
    structural_gain=6.0,
    structural_max=1.0,
    translation_decay=0.05,
    sleep_gain=0.0,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(params: DynamicsParameters) -> CytodendAccessModelSimulator:
    sim = CytodendAccessModelSimulator.from_branch_ids(BRANCH_IDS, parameters=params)
    sim.traces["mu1"] = EngramTrace(trace_id="mu1", allocation=MU1_ALLOC)
    sim.traces["mu2"] = EngramTrace(trace_id="mu2", allocation=MU2_ALLOC)
    return sim


def _encode(sim: CytodendAccessModelSimulator) -> None:
    for _ in range(2):
        sim.apply_cue(MU1_CUE)
    for _ in range(2):
        sim.apply_cue(MU2_CUE)


def _consolidate(sim: CytodendAccessModelSimulator, n_nights: int = 3, drive: float = 1.0,
                 replay: list[str] | None = None) -> None:
    win = ConsolidationWindow(
        replay_trace_ids=replay if replay is not None else ["mu1", "mu2"],
        modulatory_drive=drive,
    )
    for _ in range(n_nights * 3):
        sim.run_consolidation(win)


def _null(sim: CytodendAccessModelSimulator, n: int) -> None:
    win = ConsolidationWindow(replay_trace_ids=[], modulatory_drive=0.0)
    for _ in range(n * 3):
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


def _recall(sim: CytodendAccessModelSimulator, cue: dict[str, float], tid: str) -> float:
    sim.apply_cue(cue)
    rmap = {rs.trace_id: rs for rs in sim.compute_recall_supports()}
    return rmap[tid].support if tid in rmap else 0.0


def _recovery_pct(post: float, dmg: float, healthy: float) -> float:
    denom = healthy - dmg
    return (post - dmg) / denom * 100.0 if abs(denom) > 1e-9 else 0.0


# ---------------------------------------------------------------------------
# Part A: Structural consolidation ablation
# ---------------------------------------------------------------------------

def part_a() -> dict:
    results = {}
    for label, params in [("full_model", FULL_PARAMS), ("lr_ablation", ABLATED_PARAMS)]:
        sim = _build(params)
        mb_pre = _mb(sim)
        lk_pre = _linking(sim)
        _encode(sim)
        _consolidate(sim, 3)
        mb_post = _mb(sim)
        lk_post = _linking(sim)
        r_mu1 = _recall(sim, MU1_CUE, "mu1")
        results[label] = {
            "mb_pre": mb_pre,
            "mb_post": mb_post,
            "lk_pre": lk_pre,
            "lk_post": lk_post,
            "r_mu1": r_mu1,
            "delta_b1": mb_post["b1"] - mb_pre["b1"],
            "delta_b0": mb_post["b0"] - mb_pre["b0"],
            "lk_change_pct": (lk_post - lk_pre) / max(lk_pre, 1e-9) * 100,
        }
    return results


# ---------------------------------------------------------------------------
# Part B: Rescue ablation
# ---------------------------------------------------------------------------

def part_b() -> dict:
    results = {}
    for label, params in [("full_model", FULL_PARAMS), ("lr_ablation", ABLATED_PARAMS)]:
        # Phase A: healthy
        sim = _build(params)
        _encode(sim)
        _consolidate(sim, 3)
        h_link = _linking(sim)

        # Phase B: focal damage
        sim.branches["b1"].structural.decay_rate = 0.025
        _null(sim, 3)
        d_link = _linking(sim)

        # Phase C rescue - standard
        sim_std = deepcopy(sim)
        _consolidate(sim_std, 3, drive=1.0)
        link_std = _linking(sim_std)

        # Phase C rescue - overlap targeted
        sim_ovlp = deepcopy(sim)
        for _ in range(3):
            for _ in range(3):
                sim_ovlp.apply_cue(B1_CUE)
            _consolidate(sim_ovlp, 1, drive=1.0)
        link_ovlp = _linking(sim_ovlp)

        rec_std  = _recovery_pct(link_std,  d_link, h_link)
        rec_ovlp = _recovery_pct(link_ovlp, d_link, h_link)

        results[label] = {
            "h_link": h_link,
            "d_link": d_link,
            "link_std": link_std,
            "link_ovlp": link_ovlp,
            "rec_std": rec_std,
            "rec_ovlp": rec_ovlp,
            "rescue_advantage": rec_ovlp - rec_std,
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    a = part_a()
    b = part_b()

    print("\n=== Experiment 014: Structural Gate Ablation ===\n")
    print("Ablation: structural_lr = 0.0  (slow structural layer cannot write)")
    print("Full model: structural_lr = 0.18\n")

    # --- Part A: Consolidation ---
    print("--- Part A: Branch-specific structural consolidation ---")
    print(f"{'Condition':<16}  {'delta_M_b1':>12}  {'delta_M_b0':>12}  {'Link_pre':>9}  {'Link_post':>10}  {'Link_chg%':>10}")
    for label in ["full_model", "lr_ablation"]:
        r = a[label]
        print(
            f"{label:<16}  {r['delta_b1']:>+12.5f}  {r['delta_b0']:>+12.5f}  "
            f"{r['lk_pre']:>9.5f}  {r['lk_post']:>10.5f}  {r['lk_change_pct']:>+10.1f}%"
        )

    # --- Part B: Rescue ---
    print("\n--- Part B: Rescue selectivity (targeted vs standard) ---")
    print(f"{'Condition':<16}  {'L_healthy':>10}  {'L_damaged':>10}  {'L_std_rsc':>10}  "
          f"{'L_ovlp_rsc':>11}  {'Rec_std%':>9}  {'Rec_ovlp%':>10}  {'Adv%':>7}")
    for label in ["full_model", "lr_ablation"]:
        r = b[label]
        print(
            f"{label:<16}  {r['h_link']:>10.4f}  {r['d_link']:>10.4f}  "
            f"{r['link_std']:>10.4f}  {r['link_ovlp']:>11.4f}  "
            f"{r['rec_std']:>9.1f}  {r['rec_ovlp']:>10.1f}  {r['rescue_advantage']:>+7.1f}"
        )

    # --- Acceptance criteria ---
    print("\n--- Acceptance criteria ---")

    # C1: Full model shows overlap branch preferential strengthening
    delta_b1_full = a["full_model"]["delta_b1"]
    delta_b0_full = a["full_model"]["delta_b0"]
    delta_b1_abl  = a["lr_ablation"]["delta_b1"]
    c1 = delta_b1_full > 0.10 and delta_b1_abl < 0.01
    print(f"  C1  Overlap branch strengthening requires structural lr:  "
          f"full={delta_b1_full:+.5f}  ablated={delta_b1_abl:+.5f}  "
          f"{'PASS' if c1 else 'FAIL'}")

    # C2: Full model shows linking growth; ablation shows near-zero or negative
    lk_chg_full = a["full_model"]["lk_change_pct"]
    lk_chg_abl  = a["lr_ablation"]["lk_change_pct"]
    c2 = lk_chg_full > 5.0 and lk_chg_abl < 2.0
    print(f"  C2  Linking growth requires structural lr:  "
          f"full={lk_chg_full:+.1f}%  ablated={lk_chg_abl:+.1f}%  "
          f"{'PASS' if c2 else 'FAIL'}")

    # C3: Full model shows rescue advantage (overlap_targeted > standard);
    #     ablation shows near-zero rescue advantage
    adv_full = b["full_model"]["rescue_advantage"]
    adv_abl  = b["lr_ablation"]["rescue_advantage"]
    c3 = adv_full > 10.0 and abs(adv_abl) < adv_full / 3.0
    print(f"  C3  Rescue selectivity requires structural lr:  "
          f"full_advantage={adv_full:+.1f}%  ablated_advantage={adv_abl:+.1f}%  "
          f"{'PASS' if c3 else 'FAIL'}")

    all_pass = c1 and c2 and c3
    print(f"\n  Result: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    if all_pass:
        print(
            "  The slow structural layer is mechanistically necessary.\n"
            "  Without it (structural_lr=0):\n"
            "    - overlap branch does not preferentially strengthen\n"
            "    - linking does not rise above baseline\n"
            "    - targeted rescue has no selective advantage over standard rescue\n"
            "  All three effects are eliminated by a single-parameter ablation."
        )


if __name__ == "__main__":
    main()
