"""
Experiment 006 – Asymmetric Consolidation
==========================================
Question: Can the model generate stable long-term asymmetry between traces through
differences in encoding or replay history?

Network
-------
  b0  – mu1-specific branch    (allocation weight 0.90 for mu1, 0.05 for mu2)
  b1  – overlap branch         (0.85 for mu1, 0.85 for mu2)
  b2  – mu2-specific branch    (0.05 for mu1, 0.90 for mu2)
  b3  – unrelated branch       (0.05 for both)

Five conditions (8 consolidation nights, 3 passes per night)
-------------------------------------------------------------
  symmetric          – equal encoding (2 passes each), equal priority, joint replay
  encoding_adv       – mu1 gets 5 encoding passes, mu2 gets 2
  priority_adv       – mu1.replay_priority = 3.0, mu2 = 1.0; equal encoding
  replay_freq_adv    – mu1 replayed every night, mu2 only nights 5-8
  modulator_adv      – per-night: window A (mu1 only, drive=1.5) then window B
                       (mu2 only, drive=0.5); equal encoding

Divergence metric: M_b[b0] − M_b[b2]  (mu1-branch advantage over mu2-branch)

Expected outcome: symmetric condition stays near zero divergence; all four
asymmetric conditions show growing divergence where mu1's branches dominate.
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
N_NIGHTS = 8
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
MU1_CUE   = {"b0": 1.0, "b1": 0.8, "b2": 0.0, "b3": 0.0}
MU2_CUE   = {"b0": 0.0, "b1": 0.8, "b2": 1.0, "b3": 0.0}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build() -> CytodendAccessModelSimulator:
    return CytodendAccessModelSimulator.from_branch_ids(BRANCH_IDS, parameters=BASE_PARAMS)


def _encode(sim: CytodendAccessModelSimulator, trace_id: str, n_passes: int) -> None:
    """Encode trace by applying its cue N times to build eligibility."""
    cue = MU1_CUE if trace_id == "mu1" else MU2_CUE
    for _ in range(n_passes):
        sim.apply_cue(cue)


def _add_traces(
    sim: CytodendAccessModelSimulator,
    mu1_priority: float = 1.0,
    mu2_priority: float = 1.0,
) -> None:
    sim.traces["mu1"] = EngramTrace(
        trace_id="mu1",
        allocation=MU1_ALLOC,
        replay_priority=mu1_priority,
    )
    sim.traces["mu2"] = EngramTrace(
        trace_id="mu2",
        allocation=MU2_ALLOC,
        replay_priority=mu2_priority,
    )


def _snapshot_mb(sim: CytodendAccessModelSimulator) -> dict[str, float]:
    return {bid: sim.branches[bid].structural.accessibility for bid in BRANCH_IDS}


def _recall(sim: CytodendAccessModelSimulator) -> tuple[float, float]:
    sim.apply_cue(MU1_CUE)
    r1_map = {rs.trace_id: rs for rs in sim.compute_recall_supports()}
    r1 = r1_map["mu1"].support if "mu1" in r1_map else 0.0
    sim.apply_cue(MU2_CUE)
    r2_map = {rs.trace_id: rs for rs in sim.compute_recall_supports()}
    r2 = r2_map["mu2"].support if "mu2" in r2_map else 0.0
    return r1, r2


def _joint_window() -> ConsolidationWindow:
    return ConsolidationWindow(
        replay_trace_ids=["mu1", "mu2"],
        modulatory_drive=1.0,
        sleep_drive=0.0,
    )


# ---------------------------------------------------------------------------
# Run condition
# ---------------------------------------------------------------------------

def run_condition(
    n_enc_mu1: int = 2,
    n_enc_mu2: int = 2,
    mu1_priority: float = 1.0,
    mu2_priority: float = 1.0,
    condition: str = "symmetric",
) -> dict:
    """
    Returns per-night M_b snapshots, recall, linking, and divergence values.
    """
    sim = _build()
    _encode(sim, "mu1", n_enc_mu1)
    _encode(sim, "mu2", n_enc_mu2)
    _add_traces(sim, mu1_priority=mu1_priority, mu2_priority=mu2_priority)

    nights: list[dict[str, float]] = [_snapshot_mb(sim)]  # night 0 = post-encoding

    for night_idx in range(1, N_NIGHTS + 1):
        if condition == "replay_freq_adv":
            # mu2 only starts getting replay from night 5 onwards
            if night_idx < 5:
                windows = [ConsolidationWindow(
                    replay_trace_ids=["mu1"],
                    modulatory_drive=1.0,
                )]
            else:
                windows = [_joint_window()]
        elif condition == "modulator_adv":
            # Two separate windows per night: mu1 strong, mu2 weak
            windows = [
                ConsolidationWindow(
                    replay_trace_ids=["mu1"],
                    modulatory_drive=1.5,
                ),
                ConsolidationWindow(
                    replay_trace_ids=["mu2"],
                    modulatory_drive=0.5,
                ),
            ]
        else:
            windows = [_joint_window()]

        for _ in range(PASSES_PER_NIGHT):
            for win in windows:
                sim.run_consolidation(win)

        nights.append(_snapshot_mb(sim))

    # Final recall and linking
    r_mu1, r_mu2 = _recall(sim)

    linking = sum(
        MU1_ALLOC.branch_weights.get(bid, 0.0)
        * MU2_ALLOC.branch_weights.get(bid, 0.0)
        * sim.branches[bid].structural.accessibility
        for bid in BRANCH_IDS
    )

    return {
        "condition": condition,
        "nights": nights,
        "r_mu1": r_mu1,
        "r_mu2": r_mu2,
        "linking": linking,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    conditions = [
        dict(condition="symmetric",         n_enc_mu1=2, n_enc_mu2=2,
             mu1_priority=1.0, mu2_priority=1.0),
        dict(condition="encoding_adv",       n_enc_mu1=5, n_enc_mu2=2,
             mu1_priority=1.0, mu2_priority=1.0),
        dict(condition="priority_adv",       n_enc_mu1=2, n_enc_mu2=2,
             mu1_priority=3.0, mu2_priority=1.0),
        dict(condition="replay_freq_adv",    n_enc_mu1=2, n_enc_mu2=2,
             mu1_priority=1.0, mu2_priority=1.0),
        dict(condition="modulator_adv",      n_enc_mu1=2, n_enc_mu2=2,
             mu1_priority=1.0, mu2_priority=1.0),
    ]

    results = [run_condition(**c) for c in conditions]

    # ------------------------------------------------------------------
    # Print divergence table: M_b[b0] - M_b[b2] per night
    # ------------------------------------------------------------------
    print("\n=== Experiment 006: Asymmetric Consolidation ===\n")
    print("Divergence = M_b[b0] - M_b[b2]  (positive = mu1 advantage)\n")

    header = f"{'Condition':<20}" + "".join(f"  Nt{n:d}" for n in range(N_NIGHTS + 1))
    print(header)
    print("-" * len(header))

    for res in results:
        divs = [res["nights"][n]["b0"] - res["nights"][n]["b2"] for n in range(N_NIGHTS + 1)]
        row = f"{res['condition']:<20}" + "".join(f"  {d:+.3f}" for d in divs)
        print(row)

    # ------------------------------------------------------------------
    # Per-branch M_b at night 8
    # ------------------------------------------------------------------
    print("\n--- M_b at night 8 ---")
    print(f"{'Condition':<20}  {'b0':>7}  {'b1':>7}  {'b2':>7}  {'b3':>7}")
    for res in results:
        mb = res["nights"][N_NIGHTS]
        print(
            f"{res['condition']:<20}  {mb['b0']:>7.4f}  {mb['b1']:>7.4f}"
            f"  {mb['b2']:>7.4f}  {mb['b3']:>7.4f}"
        )

    # ------------------------------------------------------------------
    # Final recall and linking
    # ------------------------------------------------------------------
    print("\n--- Final recall and linking (night 8) ---")
    print(f"{'Condition':<20}  {'R_mu1':>7}  {'R_mu2':>7}  {'Link':>7}  {'R_mu1-R_mu2':>12}")
    for res in results:
        diff = res["r_mu1"] - res["r_mu2"]
        print(
            f"{res['condition']:<20}  {res['r_mu1']:>7.4f}  {res['r_mu2']:>7.4f}"
            f"  {res['linking']:>7.4f}  {diff:>+12.4f}"
        )

    # ------------------------------------------------------------------
    # Acceptance criteria
    # ------------------------------------------------------------------
    print("\n--- Acceptance criteria ---")
    sym  = next(r for r in results if r["condition"] == "symmetric")
    rfa  = next(r for r in results if r["condition"] == "replay_freq_adv")
    enc  = next(r for r in results if r["condition"] == "encoding_adv")

    # C1: Symmetric condition stays near-zero divergence at night 8
    sym_div = abs(sym["nights"][N_NIGHTS]["b0"] - sym["nights"][N_NIGHTS]["b2"])
    c1 = sym_div < 0.06
    print(f"  C1  Symmetric stays near-zero divergence (<0.06):  "
          f"div={sym_div:.4f}  {'PASS' if c1 else 'FAIL'}")

    # C2: replay_freq_adv shows clear positive divergence (strongest effect)
    rfa_div = rfa["nights"][N_NIGHTS]["b0"] - rfa["nights"][N_NIGHTS]["b2"]
    c2 = rfa_div > 0.05
    print(f"  C2  replay_freq_adv shows positive structural divergence (>0.05):  "
          f"div={rfa_div:+.4f}  {'PASS' if c2 else 'FAIL'}")

    # C3: replay_freq_adv shows clear recall asymmetry (R_mu1 > R_mu2)
    c3 = rfa["r_mu1"] > rfa["r_mu2"]
    print(f"  C3  replay_freq_adv: R(mu1) > R(mu2):  "
          f"R_mu1={rfa['r_mu1']:.4f}  R_mu2={rfa['r_mu2']:.4f}  {'PASS' if c3 else 'FAIL'}")

    # C4: encoding advantage is transient — at night 8 encoding_adv shows smaller divergence
    #     than replay_freq_adv (structural asymmetry requires sustained replay, not just
    #     initial encoding differences)
    enc_div = enc["nights"][N_NIGHTS]["b0"] - enc["nights"][N_NIGHTS]["b2"]
    c4 = abs(rfa_div) > abs(enc_div)
    print(f"  C4  Replay-frequency effect > encoding effect at night 8 (asymmetry requires")
    print(f"      sustained replay, not just initial encoding):  "
          f"rfa={rfa_div:+.4f}  enc={enc_div:+.4f}  {'PASS' if c4 else 'FAIL'}")
    print(f"      [Note: encoding advantages are transient — they wash out as eligibility")
    print(f"       decays and both traces receive equal replay thereafter.]")

    all_pass = c1 and c2 and c3 and c4
    print(f"\n  Result: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    if all_pass:
        print(
            "  Stable long-term structural asymmetry requires SUSTAINED differential\n"
            "  replay, not just initial encoding differences. Encoding advantages are\n"
            "  transient and wash out as eligibility decays over consolidation nights.\n"
            "  Replay frequency is the dominant driver of long-term M_b divergence."
        )


if __name__ == "__main__":
    main()
