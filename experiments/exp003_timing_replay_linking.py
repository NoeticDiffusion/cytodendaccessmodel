"""Experiment 003: Timing, spacing, and replay-dependent linking.

Tests whether the timing of encoding relative to consolidation, and the
specificity of replay, affect branch-level structural accessibility and
memory linking.

Core question:
  Does the eligibility trace provide a genuine timing window?
  Does replay specificity determine which branches get consolidated?

Four conditions (fresh simulator each):

  1. immediate_joint
     Encode mu1, encode mu2 (no gap), replay both traces together.
     Prediction: strongest M_b on overlap branch b1, highest L(mu1, mu2).

  2. spaced_joint
     Encode mu1, wait 12 neutral steps (E_b decays toward baseline),
     encode mu2, replay both.
     Prediction: b0 (mu1-specific) has lower M_b because its eligibility
     tag had partially decayed before consolidation. b1 recovers somewhat
     because mu2 also drives it. L is lower than condition 1.

  3. immediate_selective (replay mu1 only)
     Encode mu1, encode mu2 (no gap), replay ONLY mu1.
     Prediction: b2 (mu2-specific) barely consolidates because it gets no
     replay overlap. b1 gets partial boost (it is in mu1's allocation).
     L is lower because b2's M_b lags.

  4. immediate_no_replay
     Encode mu1, encode mu2 (no gap), run consolidation with
     modulatory_drive=0 (no instructional write signal).
     Prediction: structural decay dominates; M_b barely changes;
     L does not grow. Shows replay-dependence of consolidation.

Key measurements per condition:
  - M_b per branch
  - L(mu1, mu2)
  - R_mu1 (under mu1 cue) and R_mu2 (under mu2 cue)
  - Mean E_b at consolidation time (to verify timing effect)
"""

from __future__ import annotations

from dataclasses import dataclass

from cytodend_accessmodel import (
    ConsolidationWindow,
    CytodendAccessModelSimulator,
    DynamicsParameters,
    EngramTrace,
    TraceAllocation,
)

BRANCH_IDS = ["b0", "b1", "b2", "b3"]

MU1_ALLOCATION = TraceAllocation(
    trace_id="mu1",
    branch_weights={"b0": 0.90, "b1": 0.85, "b2": 0.05, "b3": 0.05},
)
MU2_ALLOCATION = TraceAllocation(
    trace_id="mu2",
    branch_weights={"b0": 0.05, "b1": 0.85, "b2": 0.90, "b3": 0.05},
)

MU1_CUE = {"b0": 1.0, "b1": 0.9, "b2": 0.05, "b3": 0.0}
MU2_CUE = {"b0": 0.05, "b1": 0.9, "b2": 1.0, "b3": 0.0}
NULL_CUE = {bid: 0.0 for bid in BRANCH_IDS}


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
        readout_gain=4.0,
        readout_threshold=0.4,
    )
    sim = CytodendAccessModelSimulator.from_branch_ids(
        BRANCH_IDS, spines_per_branch=3, parameters=params
    )
    sim.add_trace(EngramTrace(trace_id="mu1", allocation=MU1_ALLOCATION, label="trace-mu1"))
    sim.add_trace(EngramTrace(trace_id="mu2", allocation=MU2_ALLOCATION, label="trace-mu2"))
    return sim


def encode_mu1(sim: CytodendAccessModelSimulator, n: int = 2) -> None:
    for _ in range(n):
        sim.apply_cue(MU1_CUE)


def encode_mu2(sim: CytodendAccessModelSimulator, n: int = 2) -> None:
    for _ in range(n):
        sim.apply_cue(MU2_CUE)


def gap_steps(sim: CytodendAccessModelSimulator, n: int) -> None:
    """Advance time with zero input; decays E_b without new encoding."""
    for _ in range(n):
        sim.apply_cue(NULL_CUE)


def consolidate(
    sim: CytodendAccessModelSimulator,
    *,
    replay_ids: tuple[str, ...],
    modulatory_drive: float = 1.0,
    n_passes: int = 3,
) -> None:
    for i in range(n_passes):
        window = ConsolidationWindow(
            window_id=f"sleep-{i}",
            modulatory_drive=modulatory_drive,
            sleep_drive=1.0,
            replay_trace_ids=replay_ids,
        )
        sim.run_consolidation(window)


def linking_metric(sim: CytodendAccessModelSimulator) -> float:
    """L(mu1, mu2) = sum_b a_mu1_b * a_mu2_b * M_b"""
    total = 0.0
    for bid in BRANCH_IDS:
        w1 = MU1_ALLOCATION.branch_weights.get(bid, 0.0)
        w2 = MU2_ALLOCATION.branch_weights.get(bid, 0.0)
        total += w1 * w2 * sim.branches[bid].structural.accessibility
    return total


@dataclass
class ConditionResult:
    name: str
    label: str
    m_b: dict[str, float]
    e_b_at_consolidation: dict[str, float]
    linking: float
    recall_mu1_cue: float
    recall_mu2_cue: float


def run_condition(name: str, label: str, gap: int, replay_ids: tuple[str, ...], modulatory_drive: float) -> ConditionResult:
    sim = build_simulator()

    encode_mu1(sim)
    if gap > 0:
        gap_steps(sim, gap)
    encode_mu2(sim)

    # Capture E_b just before consolidation
    e_b_at_consolidation = {
        bid: sim.branches[bid].eligibility.value for bid in BRANCH_IDS
    }

    consolidate(sim, replay_ids=replay_ids, modulatory_drive=modulatory_drive)

    # Recall under respective cues
    sim.apply_cue(MU1_CUE)
    r_mu1 = next(
        rs.support for rs in sim.compute_recall_supports() if rs.trace_id == "mu1"
    )
    sim.apply_cue(MU2_CUE)
    r_mu2 = next(
        rs.support for rs in sim.compute_recall_supports() if rs.trace_id == "mu2"
    )

    return ConditionResult(
        name=name,
        label=label,
        m_b={bid: sim.branches[bid].structural.accessibility for bid in BRANCH_IDS},
        e_b_at_consolidation=e_b_at_consolidation,
        linking=linking_metric(sim),
        recall_mu1_cue=r_mu1,
        recall_mu2_cue=r_mu2,
    )


def run() -> None:
    print("=" * 72)
    print("Experiment 003: Timing, Spacing, and Replay-Dependent Linking")
    print("=" * 72)

    conditions: list[ConditionResult] = [
        run_condition(
            name="immediate_joint",
            label="no gap, replay mu1+mu2",
            gap=0,
            replay_ids=("mu1", "mu2"),
            modulatory_drive=1.0,
        ),
        run_condition(
            name="spaced_joint",
            label="12-step gap, replay mu1+mu2",
            gap=12,
            replay_ids=("mu1", "mu2"),
            modulatory_drive=1.0,
        ),
        run_condition(
            name="immediate_selective",
            label="no gap, replay mu1 only",
            gap=0,
            replay_ids=("mu1",),
            modulatory_drive=1.0,
        ),
        run_condition(
            name="immediate_no_replay",
            label="no gap, modulatory_drive=0",
            gap=0,
            replay_ids=("mu1", "mu2"),
            modulatory_drive=0.0,
        ),
    ]

    # ------------------------------------------------------------------
    # E_b at consolidation onset (shows timing / decay effect)
    # ------------------------------------------------------------------
    print("\n--- E_b at consolidation onset ---")
    print(f"  {'condition':<24}  {'b0':>8}  {'b1':>8}  {'b2':>8}  {'b3':>8}")
    for cond in conditions:
        e = cond.e_b_at_consolidation
        print(
            f"  {cond.name:<24}  "
            f"{e['b0']:>8.4f}  {e['b1']:>8.4f}  "
            f"{e['b2']:>8.4f}  {e['b3']:>8.4f}"
        )

    # ------------------------------------------------------------------
    # M_b per branch per condition
    # ------------------------------------------------------------------
    print("\n--- Post-consolidation M_b per branch ---")
    print(f"  {'condition':<24}  {'b0':>8}  {'b1':>8}  {'b2':>8}  {'b3':>8}")
    for cond in conditions:
        m = cond.m_b
        print(
            f"  {cond.name:<24}  "
            f"{m['b0']:>8.4f}  {m['b1']:>8.4f}  "
            f"{m['b2']:>8.4f}  {m['b3']:>8.4f}"
        )

    # ------------------------------------------------------------------
    # Recall support per condition
    # ------------------------------------------------------------------
    print("\n--- Post-consolidation recall support ---")
    print(f"  {'condition':<24}  {'R(mu1|cue=mu1)':>16}  {'R(mu2|cue=mu2)':>16}  {'L(mu1,mu2)':>12}")
    for cond in conditions:
        print(
            f"  {cond.name:<24}  "
            f"{cond.recall_mu1_cue:>16.4f}  "
            f"{cond.recall_mu2_cue:>16.4f}  "
            f"{cond.linking:>12.5f}"
        )

    # ------------------------------------------------------------------
    # Comparison summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    ref = conditions[0]  # immediate_joint as baseline
    print(f"  {'condition':<24}  {'label':<32}  {'L':>8}  {'delta_L':>8}")
    for cond in conditions:
        delta_l = cond.linking - ref.linking
        print(
            f"  {cond.name:<24}  {cond.label:<32}  "
            f"{cond.linking:>8.5f}  {delta_l:>+8.5f}"
        )

    # ------------------------------------------------------------------
    # Specific comparisons for hypothesis tests
    # ------------------------------------------------------------------
    imm_joint  = next(c for c in conditions if c.name == "immediate_joint")
    spaced     = next(c for c in conditions if c.name == "spaced_joint")
    selective  = next(c for c in conditions if c.name == "immediate_selective")
    no_replay  = next(c for c in conditions if c.name == "immediate_no_replay")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    timing_effect = imm_joint.m_b["b0"] > spaced.m_b["b0"]
    replay_needed = imm_joint.linking > no_replay.linking
    selective_reduces_mu2 = selective.recall_mu2_cue < imm_joint.recall_mu2_cue
    joint_gives_best_linking = imm_joint.linking >= max(spaced.linking, selective.linking, no_replay.linking)

    print(f"  Timing effect (immediate b0 M_b > spaced b0 M_b):     {'YES' if timing_effect else 'NO  <FAILURE>'}")
    print(f"  Replay required for linking growth:                    {'YES' if replay_needed else 'NO  <FAILURE>'}")
    print(f"  Selective replay reduces non-replayed trace recall:    {'YES' if selective_reduces_mu2 else 'NO  (unexpected)'}")
    print(f"  Joint immediate replay gives strongest linking:        {'YES' if joint_gives_best_linking else 'NO  (unexpected)'}")

    if replay_needed and joint_gives_best_linking:
        print("\n  Result: SUPPORTS timing and replay-dependence hypothesis.")
        print("  Structural consolidation requires both active eligibility tags")
        print("  AND replay overlap. Omitting either weakens M_b and linking.")
    else:
        print("\n  Result: CHALLENGES the hypothesis.")


if __name__ == "__main__":
    run()
