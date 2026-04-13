"""
Experiment 009 – Rescue of Linking Under Structural Damage
===========================================================
Question: Can targeted intervention restore structural accessibility and
linking after focal degradation?

Protocol
--------
Three phases per condition:

  Phase A (healthy baseline):
    Standard encoding (2 passes each trace) + 3 consolidation nights.

  Phase B (damage):
    Focal overlap-branch vulnerability: raise b1's decay_rate to 0.025
    and run 3 null nights (no replay).  b1 loses M_b; linking degrades.

  Phase C (rescue):
    3 more nights with one of four rescue protocols:

      no_rescue          – null windows continue (damage control)
      standard           – joint replay, modulatory_drive = 1.0
      high_drive         – joint replay, modulatory_drive = 2.0
      overlap_targeted   – pre-cue b1 (drive = 1.0) to rebuild eligibility,
                           then joint replay, drive = 1.0.
                           Mimics targeted reactivation of the overlap branch.

Key metrics
-----------
  M_b[b1]   – overlap-branch structural accessibility
  L(mu1,mu2) – linking
  R_mu1, R_mu2 – individual recall support
  recovery_L  – (post_rescue_L - post_damage_L) / (healthy_L - post_damage_L)

Expected outcome
----------------
  overlap_targeted > high_drive > standard >> no_rescue
  Linking recovers more slowly than recall under standard rescue.
  overlap_targeted partially closes the linking gap selectively.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cytodend_accessmodel.simulator import CytodendAccessModelSimulator, _sigmoid, _clamp01
from cytodend_accessmodel.contracts import (
    ConsolidationWindow,
    DynamicsParameters,
    EngramTrace,
    TraceAllocation,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BRANCH_IDS  = ["b0", "b1", "b2", "b3"]
OVERLAP_BID = "b1"
N_HEALTHY_NIGHTS  = 3
N_DAMAGE_NIGHTS   = 3
N_RESCUE_NIGHTS   = 3
PASSES_PER_NIGHT  = 3
FOCAL_DECAY_RATE  = 0.025   # 5x normal during damage phase
PRE_CUE_REPS      = 3       # b1-targeted pre-cues per rescue night

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
B1_CUE  = {"b0": 0.0, "b1": 1.0, "b2": 0.0, "b3": 0.0}  # overlap-targeted cue

BASE_PARAMS = DynamicsParameters(
    structural_lr=0.18,
    replay_gain=0.80,
    eligibility_decay=0.12,
    structural_decay=0.005,
    structural_gain=6.0,
    structural_max=1.0,
    translation_decay=0.05,
    sleep_gain=0.0,
)

RESCUE_PROTOCOLS = ["no_rescue", "standard", "high_drive", "overlap_targeted"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build() -> CytodendAccessModelSimulator:
    sim = CytodendAccessModelSimulator.from_branch_ids(BRANCH_IDS, parameters=BASE_PARAMS)
    sim.traces["mu1"] = EngramTrace(trace_id="mu1", allocation=MU1_ALLOC)
    sim.traces["mu2"] = EngramTrace(trace_id="mu2", allocation=MU2_ALLOC)
    return sim


def _encode(sim: CytodendAccessModelSimulator) -> None:
    for _ in range(2):
        sim.apply_cue(MU1_CUE)
    for _ in range(2):
        sim.apply_cue(MU2_CUE)


def _consolidate_nights(
    sim: CytodendAccessModelSimulator,
    n_nights: int,
    drive: float = 1.0,
    replay_ids: list[str] | None = None,
) -> None:
    win = ConsolidationWindow(
        replay_trace_ids=replay_ids if replay_ids is not None else ["mu1", "mu2"],
        modulatory_drive=drive,
    )
    for _ in range(n_nights * PASSES_PER_NIGHT):
        sim.run_consolidation(win)


def _null_nights(sim: CytodendAccessModelSimulator, n_nights: int) -> None:
    win = ConsolidationWindow(replay_trace_ids=[], modulatory_drive=0.0)
    for _ in range(n_nights * PASSES_PER_NIGHT):
        sim.run_consolidation(win)


def _linking(sim: CytodendAccessModelSimulator) -> float:
    return sum(
        MU1_ALLOC.branch_weights.get(b, 0.0)
        * MU2_ALLOC.branch_weights.get(b, 0.0)
        * sim.branches[b].structural.accessibility
        for b in BRANCH_IDS
    )


def _recall(sim: CytodendAccessModelSimulator) -> tuple[float, float]:
    sim.apply_cue(MU1_CUE)
    r1m = {rs.trace_id: rs for rs in sim.compute_recall_supports()}
    sim.apply_cue(MU2_CUE)
    r2m = {rs.trace_id: rs for rs in sim.compute_recall_supports()}
    return (
        r1m["mu1"].support if "mu1" in r1m else 0.0,
        r2m["mu2"].support if "mu2" in r2m else 0.0,
    )


def _mb(sim: CytodendAccessModelSimulator) -> dict[str, float]:
    return {b: sim.branches[b].structural.accessibility for b in BRANCH_IDS}


def _snap(sim: CytodendAccessModelSimulator) -> dict:
    r1, r2 = _recall(sim)
    return {
        "mb": _mb(sim),
        "linking": _linking(sim),
        "r_mu1": r1,
        "r_mu2": r2,
    }


def _recompute_access(sim: CytodendAccessModelSimulator) -> None:
    """Recompute slow_access and effective_access after direct M_b change."""
    for branch in sim.branches.values():
        branch.slow_access = _sigmoid(
            BASE_PARAMS.structural_gain * branch.structural.accessibility
        )
        branch.effective_access = _clamp01(branch.fast_access * branch.slow_access)


def _run_rescue_night(
    sim: CytodendAccessModelSimulator,
    protocol: str,
) -> None:
    """Run one rescue night (PASSES_PER_NIGHT passes)."""
    if protocol == "no_rescue":
        _null_nights(sim, 1)
    elif protocol == "standard":
        _consolidate_nights(sim, 1, drive=1.0)
    elif protocol == "high_drive":
        _consolidate_nights(sim, 1, drive=2.0)
    elif protocol == "overlap_targeted":
        # Rebuild eligibility at the overlap branch with targeted pre-cues
        for _ in range(PRE_CUE_REPS):
            sim.apply_cue(B1_CUE)
        # Then consolidate with standard drive
        _consolidate_nights(sim, 1, drive=1.0)


# ---------------------------------------------------------------------------
# Run one protocol
# ---------------------------------------------------------------------------

def run_protocol(protocol: str) -> dict:
    from copy import deepcopy

    # Phase A: healthy baseline
    sim = _build()
    _encode(sim)
    _consolidate_nights(sim, N_HEALTHY_NIGHTS)
    snap_healthy = _snap(sim)

    # Apply focal damage to b1 decay rate and run null nights
    sim.branches[OVERLAP_BID].structural.decay_rate = FOCAL_DECAY_RATE
    _null_nights(sim, N_DAMAGE_NIGHTS)
    snap_post_damage = _snap(sim)

    # Phase C: rescue
    for _ in range(N_RESCUE_NIGHTS):
        _run_rescue_night(sim, protocol)
    snap_post_rescue = _snap(sim)

    return {
        "protocol": protocol,
        "healthy":      snap_healthy,
        "post_damage":  snap_post_damage,
        "post_rescue":  snap_post_rescue,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _recovery_pct(post_rescue: float, post_damage: float, healthy: float) -> float:
    denom = healthy - post_damage
    if abs(denom) < 1e-9:
        return 0.0
    return (post_rescue - post_damage) / denom * 100.0


def main() -> None:
    results = [run_protocol(p) for p in RESCUE_PROTOCOLS]

    print("\n=== Experiment 009: Rescue of Linking Under Structural Damage ===\n")
    print("Damage: focal b1 decay_rate raised to 0.025 + 3 null nights.\n")

    # Phase snapshots
    print("--- M_b per phase ---")
    print(f"{'Protocol':<22}  Phase  {'b0':>7}  {'b1(OL)':>8}  {'b2':>7}  {'b3':>7}  {'Link':>8}  {'R_mu1':>7}  {'R_mu2':>7}")
    for r in results:
        for phase_key, label in [("healthy", "A-hlth"), ("post_damage", "B-dmg"), ("post_rescue", "C-rsc")]:
            s = r[phase_key]
            print(
                f"{r['protocol'] if label == 'A-hlth' else '':<22}  {label}  "
                f"{s['mb']['b0']:>7.4f}  {s['mb']['b1']:>8.4f}  "
                f"{s['mb']['b2']:>7.4f}  {s['mb']['b3']:>7.4f}  "
                f"{s['linking']:>8.4f}  {s['r_mu1']:>7.4f}  {s['r_mu2']:>7.4f}"
            )
        print()

    # Recovery table
    print("--- Recovery percentages (post_rescue vs healthy) ---")
    print(f"{'Protocol':<22}  {'RecL%':>8}  {'RecR_mu1%':>10}  {'RecR_mu2%':>10}  {'b1_RecM%':>10}")
    for r in results:
        h = r["healthy"]
        d = r["post_damage"]
        e = r["post_rescue"]
        rec_link  = _recovery_pct(e["linking"], d["linking"], h["linking"])
        rec_r1    = _recovery_pct(e["r_mu1"],   d["r_mu1"],   h["r_mu1"])
        rec_r2    = _recovery_pct(e["r_mu2"],   d["r_mu2"],   h["r_mu2"])
        rec_b1    = _recovery_pct(e["mb"]["b1"], d["mb"]["b1"], h["mb"]["b1"])
        print(f"{r['protocol']:<22}  {rec_link:>8.1f}  {rec_r1:>10.1f}  {rec_r2:>10.1f}  {rec_b1:>10.1f}")

    # Acceptance criteria
    ref   = next(r for r in results if r["protocol"] == "no_rescue")
    std   = next(r for r in results if r["protocol"] == "standard")
    hdri  = next(r for r in results if r["protocol"] == "high_drive")
    ovlp  = next(r for r in results if r["protocol"] == "overlap_targeted")

    h_link = ref["healthy"]["linking"]
    d_link = ref["post_damage"]["linking"]

    print("\n--- Acceptance criteria ---")

    # C1: Damage actually degrades linking
    c1 = d_link < h_link - 0.02
    print(f"  C1  Damage degrades linking (>0.02 drop):  "
          f"healthy={h_link:.4f}  post_damage={d_link:.4f}  "
          f"drop={h_link - d_link:.4f}  {'PASS' if c1 else 'FAIL'}")

    # C2: Any rescue beats no_rescue
    rec_std   = _recovery_pct(std["post_rescue"]["linking"], d_link, h_link)
    rec_nr    = _recovery_pct(ref["post_rescue"]["linking"], d_link, h_link)
    c2 = rec_std > rec_nr + 5.0
    print(f"  C2  standard rescue beats no_rescue by >5% recovery:  "
          f"standard={rec_std:.1f}%  no_rescue={rec_nr:.1f}%  {'PASS' if c2 else 'FAIL'}")

    # C3: high_drive >= standard
    rec_hdri = _recovery_pct(hdri["post_rescue"]["linking"], d_link, h_link)
    c3 = rec_hdri >= rec_std - 2.0
    print(f"  C3  high_drive linking recovery >= standard:  "
          f"high_drive={rec_hdri:.1f}%  standard={rec_std:.1f}%  {'PASS' if c3 else 'FAIL'}")

    # C4: overlap_targeted achieves higher b1 recovery than standard
    rec_b1_ovlp = _recovery_pct(
        ovlp["post_rescue"]["mb"]["b1"], ref["post_damage"]["mb"]["b1"], h_link
    )
    rec_b1_std  = _recovery_pct(
        std["post_rescue"]["mb"]["b1"],  ref["post_damage"]["mb"]["b1"], h_link
    )
    c4 = ovlp["post_rescue"]["mb"]["b1"] >= std["post_rescue"]["mb"]["b1"]
    print(f"  C4  overlap_targeted achieves >= b1 M_b recovery vs standard:  "
          f"OL_targeted={ovlp['post_rescue']['mb']['b1']:.4f}  "
          f"standard={std['post_rescue']['mb']['b1']:.4f}  {'PASS' if c4 else 'FAIL'}")

    # C5: recall recovers faster than linking under standard rescue
    rec_recall_std = (_recovery_pct(std["post_rescue"]["r_mu1"], ref["post_damage"]["r_mu1"], ref["healthy"]["r_mu1"])
                    + _recovery_pct(std["post_rescue"]["r_mu2"], ref["post_damage"]["r_mu2"], ref["healthy"]["r_mu2"])) / 2
    c5 = rec_recall_std > rec_std
    print(f"  C5  Recall recovers faster than linking under standard rescue:  "
          f"recall_recovery={rec_recall_std:.1f}%  link_recovery={rec_std:.1f}%  "
          f"{'PASS' if c5 else 'FAIL'}")
    print(f"      [Asymmetric recovery = linking harder to restore than recall]")

    all_pass = c1 and c2 and c3 and c4 and c5
    print(f"\n  Result: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    if all_pass:
        print(
            "  Structural linking is recoverable but partial.\n"
            "  overlap_targeted rescue selectively restores b1 accessibility.\n"
            "  Recall recovers faster than linking — confirming asymmetric\n"
            "  vulnerability also means asymmetric recoverability."
        )


if __name__ == "__main__":
    main()
