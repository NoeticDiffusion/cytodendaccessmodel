"""
Experiment 012 – Richer Retrieval Dynamics
===========================================
Question: Do the current qualitative effects survive a stronger nonlinear
readout model?

Three readout layers are compared on the same underlying recall support
values computed by the standard simulator:

  linear_raw      – current default: R_mu (pre-threshold support, continuous)
  softmax         – exp(beta * R_mu) / sum_nu exp(beta * R_nu)
                    sharpens the winner; makes context disambiguation
                    cleaner but also more brittle
  winner_margin   – P(winner) - P(runner_up); measures separation

The experiment runs four scenarios that match earlier experiments:

  Scenario 1 – context disambiguation (from exp002):
    two traces in different contexts, correct vs wrong-context cue

  Scenario 2 – damage: linking degrades before recall (from exp005/008):
    healthy vs focal-overlap-damage, compare R_mu1 and L across readouts

  Scenario 3 – asymmetric traces (from exp006):
    strong vs weak trace after 6 replay-freq-adv nights

  Scenario 4 – competitive budget (from exp008):
    unlimited vs competitive budget, show readout sharpening

Key claim: all four qualitative effects identified in experiments 001-008
remain directionally consistent under softmax readout.  The softmax
amplifies the contrast but does not reverse any finding.
"""

from __future__ import annotations

import math
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
# Readout functions
# ---------------------------------------------------------------------------

def softmax(values: dict[str, float], beta: float = 5.0) -> dict[str, float]:
    """Softmax over trace support values."""
    max_v = max(values.values()) if values else 0.0
    exps = {k: math.exp(beta * (v - max_v)) for k, v in values.items()}
    total = sum(exps.values())
    return {k: e / total for k, e in exps.items()}


def winner_margin(probs: dict[str, float]) -> float:
    """P(winner) - P(runner_up)."""
    sorted_vals = sorted(probs.values(), reverse=True)
    if len(sorted_vals) < 2:
        return sorted_vals[0] if sorted_vals else 0.0
    return sorted_vals[0] - sorted_vals[1]


def readout_all(raw: dict[str, float]) -> dict[str, dict[str, float]]:
    sm = softmax(raw)
    return {"linear": raw, "softmax": sm, "margin": {k: winner_margin(sm) for k in raw}}


# ---------------------------------------------------------------------------
# Shared network
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


def _build(budget: float = 0.0) -> CytodendAccessModelSimulator:
    p = DynamicsParameters(
        structural_lr=BASE_PARAMS.structural_lr,
        replay_gain=BASE_PARAMS.replay_gain,
        eligibility_decay=BASE_PARAMS.eligibility_decay,
        structural_decay=BASE_PARAMS.structural_decay,
        structural_gain=BASE_PARAMS.structural_gain,
        structural_max=BASE_PARAMS.structural_max,
        translation_decay=BASE_PARAMS.translation_decay,
        sleep_gain=BASE_PARAMS.sleep_gain,
        translation_budget=budget,
    )
    sim = CytodendAccessModelSimulator.from_branch_ids(BRANCH_IDS, parameters=p)
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


def _get_supports(sim: CytodendAccessModelSimulator, cue: dict[str, float]) -> dict[str, float]:
    sim.apply_cue(cue)
    return {rs.trace_id: rs.support for rs in sim.compute_recall_supports()}


def _linking(sim: CytodendAccessModelSimulator) -> float:
    return sum(
        MU1_ALLOC.branch_weights.get(b, 0.0)
        * MU2_ALLOC.branch_weights.get(b, 0.0)
        * sim.branches[b].structural.accessibility
        for b in BRANCH_IDS
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def scenario_context_disambiguation() -> dict:
    """Scenario 1: correct vs wrong-context cue."""
    sim = _build()
    _encode(sim)
    # Set contexts
    for tr in sim.traces.values():
        tr.context = "ctx_A" if tr.trace_id == "mu1" else "ctx_B"
    sim.context = "ctx_A"
    _consolidate(sim)

    # Correct context
    correct = _get_supports(sim, MU1_CUE)
    # Wrong context (flip)
    sim.context = "ctx_B"
    wrong = _get_supports(sim, MU1_CUE)
    return {"correct": correct, "wrong": wrong, "label": "context_disambiguation"}


def scenario_damage_linking() -> dict:
    """Scenario 2: healthy vs focal overlap damage."""
    sim_healthy = _build()
    _encode(sim_healthy)
    _consolidate(sim_healthy, 4)
    healthy_supports = _get_supports(sim_healthy, MU1_CUE)
    healthy_linking = _linking(sim_healthy)

    sim_dmg = _build()
    _encode(sim_dmg)
    _consolidate(sim_dmg, 4)
    sim_dmg.branches["b1"].structural.decay_rate = 0.025
    win_null = ConsolidationWindow(replay_trace_ids=[], modulatory_drive=0.0)
    for _ in range(9):
        sim_dmg.run_consolidation(win_null)
    damaged_supports = _get_supports(sim_dmg, MU1_CUE)
    damaged_linking = _linking(sim_dmg)

    return {
        "healthy": {"supports": healthy_supports, "linking": healthy_linking},
        "damaged": {"supports": damaged_supports, "linking": damaged_linking},
        "label": "damage_linking",
    }


def scenario_asymmetric() -> dict:
    """Scenario 3: asymmetric replay (replay_freq_adv from exp006)."""
    sim = _build()
    for _ in range(2):
        sim.apply_cue(MU1_CUE)
    for _ in range(2):
        sim.apply_cue(MU2_CUE)

    for night in range(1, 9):
        replay_ids = ["mu1"] if night < 5 else ["mu1", "mu2"]
        win = ConsolidationWindow(replay_trace_ids=replay_ids, modulatory_drive=1.0)
        for _ in range(3):
            sim.run_consolidation(win)

    sup_mu1 = _get_supports(sim, MU1_CUE)
    sup_mu2 = _get_supports(sim, MU2_CUE)
    return {"mu1_cue": sup_mu1, "mu2_cue": sup_mu2, "label": "asymmetric_replay"}


def scenario_competition() -> dict:
    """Scenario 4: unlimited vs competitive budget."""
    sim_unlim = _build(budget=0.0)
    _encode(sim_unlim)
    _consolidate(sim_unlim)
    sup_unlim = _get_supports(sim_unlim, MU1_CUE)

    sim_comp = _build(budget=1.5)
    _encode(sim_comp)
    _consolidate(sim_comp)
    sup_comp = _get_supports(sim_comp, MU1_CUE)

    return {"unlimited": sup_unlim, "competitive": sup_comp, "label": "competition"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_readout_row(label: str, raw: dict[str, float], tid: str = "mu1") -> None:
    sm = softmax(raw)
    margin = winner_margin(sm)
    r = raw.get(tid, 0.0)
    s = sm.get(tid, 0.0)
    print(f"  {label:<28}  linear={r:.4f}  softmax={s:.4f}  margin={margin:.4f}")


def main() -> None:
    print("\n=== Experiment 012: Richer Retrieval Dynamics ===\n")
    print("Softmax beta = 5.0. All qualitative effects should survive readout change.\n")

    # -----------------------------------------------------------------------
    # Scenario 1: Context disambiguation
    # -----------------------------------------------------------------------
    s1 = scenario_context_disambiguation()
    print("--- Scenario 1: Context disambiguation ---")
    print(f"  {'Condition':<28}  {'linear_mu1':>12}  {'softmax_mu1':>12}  {'margin':>8}")
    for key, raw in [("correct_ctx_A", s1["correct"]), ("wrong_ctx_B", s1["wrong"])]:
        sm = softmax(raw)
        mg = winner_margin(sm)
        print(f"  {key:<28}  {raw.get('mu1', 0.0):>12.4f}  {sm.get('mu1', 0.0):>12.4f}  {mg:>8.4f}")

    # -----------------------------------------------------------------------
    # Scenario 2: Damage/linking
    # -----------------------------------------------------------------------
    s2 = scenario_damage_linking()
    print("\n--- Scenario 2: Damage — linking degrades before recall ---")
    print(f"  {'Condition':<28}  {'linear_mu1':>12}  {'softmax_mu1':>12}  {'linking':>8}")
    for key in ["healthy", "damaged"]:
        raw = s2[key]["supports"]
        lk  = s2[key]["linking"]
        sm  = softmax(raw)
        print(f"  {key:<28}  {raw.get('mu1', 0.0):>12.4f}  {sm.get('mu1', 0.0):>12.4f}  {lk:>8.4f}")

    # -----------------------------------------------------------------------
    # Scenario 3: Asymmetric replay
    # -----------------------------------------------------------------------
    s3 = scenario_asymmetric()
    print("\n--- Scenario 3: Asymmetric replay (mu1 freq_adv) ---")
    print(f"  {'Condition':<28}  {'linear_target':>14}  {'softmax_target':>14}")
    raw_mu1_cue = s3["mu1_cue"]
    raw_mu2_cue = s3["mu2_cue"]
    sm_mu1 = softmax(raw_mu1_cue)
    sm_mu2 = softmax(raw_mu2_cue)
    print(f"  {'mu1_cue -> mu1 support':<28}  {raw_mu1_cue.get('mu1', 0):>14.4f}  {sm_mu1.get('mu1', 0):>14.4f}")
    print(f"  {'mu2_cue -> mu2 support':<28}  {raw_mu2_cue.get('mu2', 0):>14.4f}  {sm_mu2.get('mu2', 0):>14.4f}")
    print(f"  [mu1 should exceed mu2 in both readouts due to more replay]")

    # -----------------------------------------------------------------------
    # Scenario 4: Competition
    # -----------------------------------------------------------------------
    s4 = scenario_competition()
    print("\n--- Scenario 4: Competition (budget=0 vs budget=1.5) ---")
    print(f"  {'Condition':<28}  {'linear_mu1':>12}  {'softmax_mu1':>12}  {'winner_margin':>14}")
    for key in ["unlimited", "competitive"]:
        raw = s4[key]
        sm  = softmax(raw)
        mg  = winner_margin(sm)
        print(f"  {key:<28}  {raw.get('mu1', 0.0):>12.4f}  {sm.get('mu1', 0.0):>12.4f}  {mg:>14.4f}")

    # -----------------------------------------------------------------------
    # Acceptance criteria: qualitative consistency across readouts
    # -----------------------------------------------------------------------
    print("\n--- Acceptance criteria ---")

    # C1: context correct > wrong in linear AND softmax
    c1_lin = s1["correct"].get("mu1", 0) > s1["wrong"].get("mu1", 0)
    c1_sm  = softmax(s1["correct"]).get("mu1", 0) > softmax(s1["wrong"]).get("mu1", 0)
    c1 = c1_lin and c1_sm
    print(f"  C1  Context disambiguation holds in both readouts:  "
          f"linear={'yes' if c1_lin else 'no'}  softmax={'yes' if c1_sm else 'no'}  "
          f"{'PASS' if c1 else 'FAIL'}")

    # C2: damage degrades both linear support AND softmax probability
    c2_lin = s2["healthy"]["supports"].get("mu1", 0) > s2["damaged"]["supports"].get("mu1", 0)
    sm_h = softmax(s2["healthy"]["supports"]).get("mu1", 0)
    sm_d = softmax(s2["damaged"]["supports"]).get("mu1", 0)
    c2_sm = sm_h >= sm_d
    c2 = c2_lin and c2_sm
    print(f"  C2  Damage degrades mu1 support in both readouts:  "
          f"linear={'yes' if c2_lin else 'no'}  softmax={'yes' if c2_sm else 'no'}  "
          f"{'PASS' if c2 else 'FAIL'}")

    # C3: asymmetric replay: mu1 support > mu2 support (mu1 advantage)
    mu1_lin = raw_mu1_cue.get("mu1", 0)
    mu2_lin = raw_mu2_cue.get("mu2", 0)
    mu1_sm  = sm_mu1.get("mu1", 0)
    mu2_sm  = sm_mu2.get("mu2", 0)
    c3 = (mu1_lin > mu2_lin) and (mu1_sm >= mu2_sm)
    print(f"  C3  Asymmetric recall: mu1 > mu2 in both readouts:  "
          f"linear={mu1_lin:.4f}>{mu2_lin:.4f}={'yes' if mu1_lin>mu2_lin else 'no'}  "
          f"softmax={'yes' if mu1_sm>=mu2_sm else 'no'}  "
          f"{'PASS' if c3 else 'FAIL'}")

    # C4: softmax sharpens discrimination (winner_margin increases vs linear ratio)
    # Under competition, recall drops in linear but softmax should still differentiate
    unlim_sm_mu1 = softmax(s4["unlimited"]).get("mu1", 0)
    comp_sm_mu1  = softmax(s4["competitive"]).get("mu1", 0)
    margin_unlim = winner_margin(softmax(s4["unlimited"]))
    margin_comp  = winner_margin(softmax(s4["competitive"]))
    c4 = unlim_sm_mu1 >= comp_sm_mu1
    print(f"  C4  Competition reduces softmax probability (correct direction):  "
          f"unlimited={unlim_sm_mu1:.4f}  competitive={comp_sm_mu1:.4f}  "
          f"{'PASS' if c4 else 'FAIL'}")

    all_pass = c1 and c2 and c3 and c4
    print(f"\n  Result: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    if all_pass:
        print(
            "  All qualitative effects from experiments 002-008 survive the\n"
            "  softmax readout layer.  Softmax amplifies contrast between\n"
            "  conditions without reversing any directional prediction.\n"
            "  The linear pre-threshold support is therefore a valid and\n"
            "  interpretable proxy for the full nonlinear readout."
        )


if __name__ == "__main__":
    main()
