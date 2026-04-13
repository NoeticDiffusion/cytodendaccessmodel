"""Experiment 001: Minimal branch linking test.

Validates the core hypothesis:
  slow structural state gates which dendritic subunits can participate
  in encoding, stabilization, and retrieval.

Two traces with partial overlap are encoded. A consolidation window is
then run. We inspect whether M_b changes branch-specifically and whether
recall support shifts in the predicted direction.

Deliverable format specified in:
  Project cytodend_accessmodel/Diary/001_first_test_minimal_branch_linking.md
"""

from __future__ import annotations

from copy import deepcopy

from cytodend_accessmodel import (
    ConsolidationWindow,
    CytodendAccessModelSimulator,
    DynamicsParameters,
    EngramTrace,
    TraceAllocation,
)


BRANCH_IDS = ["b0", "b1", "b2", "b3"]

# mu1 is strong on b0, b1 — b3 gets a small drive as background noise
MU1_ALLOCATION = TraceAllocation(
    trace_id="mu1",
    branch_weights={"b0": 0.9, "b1": 0.85, "b2": 0.05, "b3": 0.05},
)

# mu2 is strong on b1, b2 — b3 gets a small drive as background noise
MU2_ALLOCATION = TraceAllocation(
    trace_id="mu2",
    branch_weights={"b0": 0.05, "b1": 0.85, "b2": 0.9, "b3": 0.05},
)

MU1_CUE = {"b0": 1.0, "b1": 0.9, "b2": 0.05, "b3": 0.0}
MU2_CUE = {"b0": 0.05, "b1": 0.9, "b2": 1.0, "b3": 0.0}


def build_simulator() -> CytodendAccessModelSimulator:
    params = DynamicsParameters(
        fast_gain=2.0,
        structural_gain=2.0,
        eligibility_decay=0.1,
        translation_decay=0.05,
        structural_lr=0.20,
        structural_decay=0.005,
        structural_max=1.0,
        replay_gain=1.2,
        sleep_gain=0.8,
        readout_gain=5.0,
        readout_threshold=0.3,
    )
    sim = CytodendAccessModelSimulator.from_branch_ids(
        BRANCH_IDS, spines_per_branch=3, parameters=params
    )
    sim.add_trace(EngramTrace(trace_id="mu1", allocation=MU1_ALLOCATION, label="trace-mu1"))
    sim.add_trace(EngramTrace(trace_id="mu2", allocation=MU2_ALLOCATION, label="trace-mu2"))
    return sim


def log_branch_state(sim: CytodendAccessModelSimulator, *, label: str) -> None:
    print(f"\n--- {label} ---")
    print(
        f"  {'branch':<8}  {'M_b':>8}  {'E_b':>8}  {'P_b':>8}  "
        f"{'A_b^f':>8}  {'A_b^s':>8}  {'A_b':>8}  {'x_b':>8}"
    )
    for bid in BRANCH_IDS:
        b = sim.branches[bid]
        print(
            f"  {bid:<8}  "
            f"{b.structural.accessibility:>8.4f}  "
            f"{b.eligibility.value:>8.4f}  "
            f"{b.translation_readiness.value:>8.4f}  "
            f"{b.fast_access:>8.4f}  "
            f"{b.slow_access:>8.4f}  "
            f"{b.effective_access:>8.4f}  "
            f"{b.activation:>8.4f}"
        )


def log_recall_supports(sim: CytodendAccessModelSimulator, *, label: str) -> None:
    supports = sim.compute_recall_supports()
    print(f"\n  Recall supports [{label}]:")
    for rs in supports:
        print(
            f"    {rs.trace_id}: support={rs.support:.4f}  "
            f"expressed={rs.expressed_strength:.4f}  "
            f"active_branches={rs.active_branches}"
        )


def log_linking(sim: CytodendAccessModelSimulator, *, label: str) -> None:
    """L(mu1, mu2) = sum_b a_mu1b * a_mu2b * M_b"""
    linking = 0.0
    for bid in BRANCH_IDS:
        w1 = MU1_ALLOCATION.branch_weights.get(bid, 0.0)
        w2 = MU2_ALLOCATION.branch_weights.get(bid, 0.0)
        m_b = sim.branches[bid].structural.accessibility
        linking += w1 * w2 * m_b
    print(f"\n  L(mu1, mu2) [{label}] = {linking:.5f}")


def capture_m_b(sim: CytodendAccessModelSimulator) -> dict[str, float]:
    return {bid: sim.branches[bid].structural.accessibility for bid in BRANCH_IDS}


def run() -> None:
    print("=" * 64)
    print("Experiment 001: Minimal Branch Linking")
    print("=" * 64)

    sim = build_simulator()

    # ----------------------------------------------------------------
    # Phase 1: encode mu1 (two cue passes to build eligibility)
    # ----------------------------------------------------------------
    print("\n[Phase 1] Encoding mu1 (2 cue passes)")
    sim.apply_cue(MU1_CUE)
    sim.apply_cue(MU1_CUE)
    log_branch_state(sim, label="after mu1 encoding")

    # ----------------------------------------------------------------
    # Phase 2: encode mu2 (two cue passes)
    # ----------------------------------------------------------------
    print("\n[Phase 2] Encoding mu2 (2 cue passes)")
    sim.apply_cue(MU2_CUE)
    sim.apply_cue(MU2_CUE)
    log_branch_state(sim, label="after mu2 encoding")

    # ----------------------------------------------------------------
    # Pre-consolidation recall
    # ----------------------------------------------------------------
    pre_m_b = capture_m_b(sim)

    print("\n[Pre-consolidation recall — cue mu1]")
    sim.apply_cue(MU1_CUE)
    log_recall_supports(sim, label="pre-consolidation, cue=mu1")

    print("\n[Pre-consolidation recall — cue mu2]")
    sim.apply_cue(MU2_CUE)
    log_recall_supports(sim, label="pre-consolidation, cue=mu2")

    log_linking(sim, label="pre-consolidation")

    # ----------------------------------------------------------------
    # Phase 3: sleep-like consolidation replaying both traces
    # ----------------------------------------------------------------
    print("\n[Phase 3] Consolidation window (replay both traces, 3 passes)")
    for pass_idx in range(3):
        window = ConsolidationWindow(
            window_id=f"sleep-pass-{pass_idx}",
            modulatory_drive=1.0,
            sleep_drive=1.0,
            replay_trace_ids=("mu1", "mu2"),
        )
        report = sim.run_consolidation(window)
        print(
            f"  pass {pass_idx}: branches_updated={report.branches_updated}  "
            f"mean_shift={report.mean_structural_shift:.5f}  "
            f"mean_P_b={report.mean_translation_readiness:.5f}"
        )

    log_branch_state(sim, label="after consolidation")

    # ----------------------------------------------------------------
    # Post-consolidation recall
    # ----------------------------------------------------------------
    post_m_b = capture_m_b(sim)

    print("\n[Post-consolidation recall — cue mu1]")
    sim.apply_cue(MU1_CUE)
    log_recall_supports(sim, label="post-consolidation, cue=mu1")

    print("\n[Post-consolidation recall — cue mu2]")
    sim.apply_cue(MU2_CUE)
    log_recall_supports(sim, label="post-consolidation, cue=mu2")

    log_linking(sim, label="post-consolidation")

    # ----------------------------------------------------------------
    # Summary table: pre/post M_b per branch
    # ----------------------------------------------------------------
    print("\n" + "=" * 64)
    print("SUMMARY: Branch-wise M_b change")
    print("=" * 64)
    print(f"  {'branch':<8}  {'M_b pre':>10}  {'M_b post':>10}  {'delta':>10}  note")
    for bid in BRANCH_IDS:
        pre = pre_m_b[bid]
        post = post_m_b[bid]
        delta = post - pre
        note = ""
        if bid == "b1":
            note = "<-- overlap branch"
        elif bid in ("b0", "b2"):
            note = "<-- single-trace branch"
        else:
            note = "<-- unrelated"
        print(f"  {bid:<8}  {pre:>10.5f}  {post:>10.5f}  {delta:>+10.5f}  {note}")

    # ----------------------------------------------------------------
    # Hypothesis verdict
    # ----------------------------------------------------------------
    print("\n" + "=" * 64)
    print("VERDICT")
    print("=" * 64)
    b1_delta = post_m_b["b1"] - pre_m_b["b1"]
    b3_delta = post_m_b["b3"] - pre_m_b["b3"]
    b0_delta = post_m_b["b0"] - pre_m_b["b0"]
    b2_delta = post_m_b["b2"] - pre_m_b["b2"]

    overlap_stronger_than_unrelated = b1_delta > b3_delta
    replay_changed_m_b = any(
        abs(post_m_b[bid] - pre_m_b[bid]) > 1e-6 for bid in BRANCH_IDS
    )
    overlap_exceeds_single_trace = b1_delta >= max(b0_delta, b2_delta)

    print(f"  Replay changed M_b:                    {'YES' if replay_changed_m_b else 'NO  <FAILURE>'}")
    print(f"  Overlap branch stronger than unrelated:{'YES' if overlap_stronger_than_unrelated else 'NO  <FAILURE>'}")
    print(f"  Overlap branch >= single-trace branches:{'YES' if overlap_exceeds_single_trace else 'NO  (borderline)'}")

    if replay_changed_m_b and overlap_stronger_than_unrelated:
        print("\n  Result: SUPPORTS the architectural hypothesis.")
        print("  Structural accessibility was written in a branch-specific way,")
        print("  with the overlap branch (b1) showing the strongest consolidation pressure.")
    else:
        print("\n  Result: CHALLENGES the architectural hypothesis.")
        print("  See failure mode notes in Diary 001.")


if __name__ == "__main__":
    run()
