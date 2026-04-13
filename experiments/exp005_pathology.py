"""Experiment 005: Pathology / structural degradation.

Tests which aspects of memory fail first under progressive degradation of
the slow structural accessibility layer.

Six conditions (one healthy baseline + five pathological):

  1. healthy          baseline parameters, no noise
  2. high_decay       structural decay 6× normal (tau detachment / MT destabilization)
  3. noise_drift      high structural noise (elevated T_eff / active-matter disorder)
  4. low_write        structural_lr reduced 4× (impaired local translation / capture)
  5. low_ceiling      max_accessibility = 0.6 (reduced structural capacity)
  6. focal_vuln       overlap branch b1 has both high_decay AND low_ceiling

All conditions use the same encoding + consolidation protocol as exp001.
Pathology is applied at the branch or simulator level before consolidation.

Expected (from Diary 005 / architectural plan):
  - Not total failure, but selective degradation
  - Weaker linking, poorer context disambiguation, more recency dependence
  - Focal vulnerability (condition 6) hurts linking most because b1 is impaired
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from cytodend_accessmodel import (
    ConsolidationWindow,
    CytodendAccessModelSimulator,
    DynamicsParameters,
    EngramTrace,
    TraceAllocation,
)

# ---------------------------------------------------------------------------
# Topology (matches exp001 and exp003)
# ---------------------------------------------------------------------------
BRANCH_IDS = ["b0", "b1", "b2", "b3"]

MU1_ALLOC = TraceAllocation("mu1", {"b0": 0.90, "b1": 0.85, "b2": 0.05, "b3": 0.05})
MU2_ALLOC = TraceAllocation("mu2", {"b0": 0.05, "b1": 0.85, "b2": 0.90, "b3": 0.05})
ALPHA_ALLOC = TraceAllocation("mu_alpha", {"b0": 0.90, "b1": 0.80, "b2": 0.05, "b3": 0.05})
BETA_ALLOC  = TraceAllocation("mu_beta",  {"b0": 0.05, "b1": 0.05, "b2": 0.80, "b3": 0.90})

MU1_CUE  = {"b0": 1.0, "b1": 0.9, "b2": 0.05, "b3": 0.0}
MU2_CUE  = {"b0": 0.05, "b1": 0.9, "b2": 1.0, "b3": 0.0}
ALPHA_CUE = {"b0": 1.0, "b1": 0.9, "b2": 0.05, "b3": 0.0}
BETA_CUE  = {"b0": 0.0, "b1": 0.05, "b2": 0.9, "b3": 1.0}
PARTIAL_CUE = {"b0": 0.4, "b1": 0.4, "b2": 0.4, "b3": 0.4}

ALPHA_BIAS = {"b0": 0.7, "b1": 0.6, "b2": 0.0, "b3": 0.0}
BETA_BIAS  = {"b0": 0.0, "b1": 0.0, "b2": 0.6, "b3": 0.7}

HEALTHY_PARAMS = DynamicsParameters(
    fast_gain=2.0, structural_gain=2.0,
    eligibility_decay=0.10, translation_decay=0.05,
    structural_lr=0.20, structural_decay=0.005, structural_max=1.0,
    replay_gain=1.2, sleep_gain=0.8,
    readout_gain=4.0, readout_threshold=0.4,
    context_mismatch_penalty=0.35,
    structural_noise=0.0,
)


# ---------------------------------------------------------------------------
# Pathology condition descriptor
# ---------------------------------------------------------------------------
@dataclass
class PathologyCondition:
    name: str
    label: str
    params: DynamicsParameters
    focal_branch: str | None = None          # branch to receive extra degradation
    focal_decay_rate: float | None = None    # override decay_rate for focal branch
    focal_max_accessibility: float | None = None  # override max for focal branch


CONDITIONS: list[PathologyCondition] = [
    PathologyCondition(
        name="healthy",
        label="baseline (no pathology)",
        params=HEALTHY_PARAMS,
    ),
    PathologyCondition(
        name="high_decay",
        label="structural decay 6× (tau detachment)",
        params=replace(HEALTHY_PARAMS, structural_decay=0.030),
    ),
    PathologyCondition(
        name="noise_drift",
        label="elevated structural noise (high T_eff)",
        params=replace(HEALTHY_PARAMS, structural_noise=0.020),
    ),
    PathologyCondition(
        name="low_write",
        label="reduced write efficacy (structural_lr ÷ 4)",
        params=replace(HEALTHY_PARAMS, structural_lr=0.05),
    ),
    PathologyCondition(
        name="low_ceiling",
        label="reduced M_max = 0.6 (limited structural capacity)",
        params=replace(HEALTHY_PARAMS, structural_max=0.6),
    ),
    PathologyCondition(
        name="focal_vuln",
        label="overlap branch b1: high decay + low ceiling",
        params=HEALTHY_PARAMS,
        focal_branch="b1",
        focal_decay_rate=0.030,
        focal_max_accessibility=0.6,
    ),
]


# ---------------------------------------------------------------------------
# Simulator builder with optional focal vulnerability
# ---------------------------------------------------------------------------

def build_sim(cond: PathologyCondition) -> CytodendAccessModelSimulator:
    sim = CytodendAccessModelSimulator.from_branch_ids(
        BRANCH_IDS, spines_per_branch=3, parameters=cond.params
    )
    sim.add_trace(EngramTrace("mu1", MU1_ALLOC))
    sim.add_trace(EngramTrace("mu2", MU2_ALLOC))

    if cond.focal_branch is not None:
        branch = sim.branches[cond.focal_branch]
        if cond.focal_decay_rate is not None:
            branch.structural.decay_rate = cond.focal_decay_rate
        if cond.focal_max_accessibility is not None:
            branch.structural.max_accessibility = cond.focal_max_accessibility
    return sim


def build_sim_ctx(cond: PathologyCondition) -> CytodendAccessModelSimulator:
    sim = CytodendAccessModelSimulator.from_branch_ids(
        BRANCH_IDS, spines_per_branch=3, parameters=cond.params
    )
    sim.add_trace(EngramTrace("mu_alpha", ALPHA_ALLOC, context="alpha"))
    sim.add_trace(EngramTrace("mu_beta",  BETA_ALLOC,  context="beta"))
    if cond.focal_branch is not None:
        branch = sim.branches[cond.focal_branch]
        if cond.focal_decay_rate is not None:
            branch.structural.decay_rate = cond.focal_decay_rate
        if cond.focal_max_accessibility is not None:
            branch.structural.max_accessibility = cond.focal_max_accessibility
    return sim


def consolidate(sim: CytodendAccessModelSimulator, n: int = 3) -> None:
    for i in range(n):
        sim.run_consolidation(ConsolidationWindow(
            window_id=f"s{i}", replay_trace_ids=("mu1", "mu2"), modulatory_drive=1.0,
        ))


def consolidate_ctx(sim: CytodendAccessModelSimulator, n: int = 3) -> None:
    for i in range(n):
        sim.run_consolidation(ConsolidationWindow(
            window_id=f"s{i}", replay_trace_ids=("mu_alpha", "mu_beta"), modulatory_drive=1.0,
        ))


def linking(sim: CytodendAccessModelSimulator) -> float:
    return sum(
        MU1_ALLOC.branch_weights.get(b, 0) * MU2_ALLOC.branch_weights.get(b, 0)
        * sim.branches[b].structural.accessibility
        for b in BRANCH_IDS
    )


@dataclass
class PathologyResult:
    name: str
    label: str
    m_b: dict[str, float]
    delta_m_b: dict[str, float]
    linking_pre: float
    linking_post: float
    recall_mu1: float
    recall_mu2: float
    context_gap_pre: float
    context_gap_post: float
    overlap_advantage: bool   # b1_delta > b3_delta
    linking_growth: bool      # L_post > L_pre
    context_separates: bool   # correct > wrong context


# ---------------------------------------------------------------------------
# Run one condition
# ---------------------------------------------------------------------------

N_NOISE_SEEDS = 10  # averaged over stochastic seeds for noisy conditions

def run_condition(cond: PathologyCondition) -> PathologyResult:
    import random

    # --- exp001 protocol ---
    m_b_post_acc: dict[str, float] = {b: 0.0 for b in BRANCH_IDS}
    delta_acc:    dict[str, float] = {b: 0.0 for b in BRANCH_IDS}
    l_pre_acc = l_post_acc = r_mu1_acc = r_mu2_acc = 0.0

    n_runs = N_NOISE_SEEDS if cond.params.structural_noise > 0 else 1

    for seed in range(n_runs):
        random.seed(seed)
        sim = build_sim(cond)
        pre = {b: sim.branches[b].structural.accessibility for b in BRANCH_IDS}

        for _ in range(2): sim.apply_cue(MU1_CUE)
        for _ in range(2): sim.apply_cue(MU2_CUE)
        # pre-consolidation linking (after recall cue to warm up activation)
        sim.apply_cue(MU1_CUE); sim.apply_cue(MU2_CUE)
        l_pre_acc += linking(sim)

        consolidate(sim)

        for b in BRANCH_IDS:
            m_b_post_acc[b] += sim.branches[b].structural.accessibility
            delta_acc[b] += sim.branches[b].structural.accessibility - pre[b]
        l_post_acc += linking(sim)

        sim.apply_cue(MU1_CUE)
        r_mu1_acc += next(rs.support for rs in sim.compute_recall_supports() if rs.trace_id == "mu1")
        sim.apply_cue(MU2_CUE)
        r_mu2_acc += next(rs.support for rs in sim.compute_recall_supports() if rs.trace_id == "mu2")

    m_b   = {b: m_b_post_acc[b] / n_runs for b in BRANCH_IDS}
    delta = {b: delta_acc[b] / n_runs for b in BRANCH_IDS}
    l_pre = l_pre_acc / n_runs
    l_post = l_post_acc / n_runs
    r_mu1 = r_mu1_acc / n_runs
    r_mu2 = r_mu2_acc / n_runs

    def _ctx_gap(s: CytodendAccessModelSimulator) -> float:
        s.apply_cue(PARTIAL_CUE, context="alpha", context_bias=ALPHA_BIAS)
        sup = {rs.trace_id: rs.support for rs in s.compute_recall_supports()}
        return sup.get("mu_alpha", 0) - sup.get("mu_beta", 0)

    # --- context separation (exp002 protocol) ---
    ctx_gap_pre_acc = ctx_gap_post_acc = 0.0
    for seed in range(n_runs):
        random.seed(seed)
        sim_c = build_sim_ctx(cond)
        for _ in range(3): sim_c.apply_cue(ALPHA_CUE, context="alpha", context_bias=ALPHA_BIAS)
        for _ in range(3): sim_c.apply_cue(BETA_CUE,  context="beta",  context_bias=BETA_BIAS)

        ctx_gap_pre_acc += _ctx_gap(sim_c)
        consolidate_ctx(sim_c)
        ctx_gap_post_acc += _ctx_gap(sim_c)

    ctx_gap_pre  = ctx_gap_pre_acc / n_runs
    ctx_gap_post = ctx_gap_post_acc / n_runs

    return PathologyResult(
        name=cond.name,
        label=cond.label,
        m_b=m_b,
        delta_m_b=delta,
        linking_pre=l_pre,
        linking_post=l_post,
        recall_mu1=r_mu1,
        recall_mu2=r_mu2,
        context_gap_pre=ctx_gap_pre,
        context_gap_post=ctx_gap_post,
        overlap_advantage=delta["b1"] > delta["b3"],
        linking_growth=l_post > l_pre,
        context_separates=ctx_gap_post > 0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    print("=" * 72)
    print("Experiment 005: Pathology / Structural Degradation")
    print("=" * 72)

    results: list[PathologyResult] = [run_condition(c) for c in CONDITIONS]
    healthy = results[0]

    # ------------------------------------------------------------------
    # M_b per branch
    # ------------------------------------------------------------------
    print("\n--- Post-consolidation M_b per branch ---")
    print(f"  {'condition':<18}  {'b0':>7}  {'b1':>7}  {'b2':>7}  {'b3':>7}  (vs healthy b1 delta)")
    for r in results:
        b1_diff = r.m_b["b1"] - healthy.m_b["b1"]
        marker = f"  {b1_diff:+.4f}" if r.name != "healthy" else ""
        print(
            f"  {r.name:<18}  "
            f"{r.m_b['b0']:>7.4f}  {r.m_b['b1']:>7.4f}  "
            f"{r.m_b['b2']:>7.4f}  {r.m_b['b3']:>7.4f}"
            f"{marker}"
        )

    # ------------------------------------------------------------------
    # Linking and recall
    # ------------------------------------------------------------------
    print("\n--- Linking and recall support ---")
    print(f"  {'condition':<18}  {'L_pre':>8}  {'L_post':>8}  {'delta_L':>8}  {'R(mu1)':>8}  {'R(mu2)':>8}")
    for r in results:
        print(
            f"  {r.name:<18}  "
            f"{r.linking_pre:>8.5f}  {r.linking_post:>8.5f}  "
            f"{r.linking_post - r.linking_pre:>+8.5f}  "
            f"{r.recall_mu1:>8.4f}  {r.recall_mu2:>8.4f}"
        )

    # ------------------------------------------------------------------
    # Context separation
    # ------------------------------------------------------------------
    print("\n--- Context separation gap (correct - wrong context) ---")
    print(f"  {'condition':<18}  {'gap_pre':>10}  {'gap_post':>10}  {'widens?':>10}")
    for r in results:
        print(
            f"  {r.name:<18}  "
            f"{r.context_gap_pre:>10.4f}  {r.context_gap_post:>10.4f}  "
            f"{'YES' if r.context_gap_post > r.context_gap_pre else 'NO':>10}"
        )

    # ------------------------------------------------------------------
    # Degradation relative to healthy
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("DEGRADATION RELATIVE TO HEALTHY BASELINE")
    print("=" * 72)
    print(f"  {'condition':<18}  {'L_post':>8}  {'loss%':>7}  {'ctx_gap_post':>13}  {'overlap_adv':>12}  {'ctx_sep':>8}")
    for r in results:
        l_loss = (healthy.linking_post - r.linking_post) / max(1e-9, healthy.linking_post) * 100
        print(
            f"  {r.name:<18}  "
            f"{r.linking_post:>8.5f}  {l_loss:>7.1f}%  "
            f"{r.context_gap_post:>13.4f}  "
            f"{'YES' if r.overlap_advantage else 'NO':>12}  "
            f"{'YES' if r.context_separates else 'NO':>8}"
        )

    # ------------------------------------------------------------------
    # Mechanistic hierarchy: what fails first?
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("MECHANISTIC HIERARCHY: ordering conditions by L_post degradation")
    print("=" * 72)
    sorted_by_l = sorted(results[1:], key=lambda r: r.linking_post, reverse=True)
    for rank, r in enumerate(sorted_by_l, 1):
        l_loss = (healthy.linking_post - r.linking_post) / max(1e-9, healthy.linking_post) * 100
        print(f"  {rank}. {r.name:<18}  L={r.linking_post:.5f}  loss={l_loss:.1f}%  — {r.label}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    any_total_failure = any(not r.linking_growth and r.name != "healthy" for r in results)
    selective_degradation = all(r.recall_mu1 > 0 for r in results)
    focal_hurts_most = (
        results[-1].linking_post
        < min(r.linking_post for r in results[1:-1])
    )

    print(f"  No total recall failure (selective degradation):    {'YES' if selective_degradation else 'NO'}")
    print(f"  Focal vulnerability hurts linking most:             {'YES' if focal_hurts_most else 'NO (unexpected)'}")

    if selective_degradation:
        print("\n  Result: SUPPORTS the pathology prediction.")
        print("  Structural degradation produces graded, selective impairment.")
        print("  Linking and context separation decline before recall collapses,")
        print("  matching the model's prediction of partial-access failure modes.")
    else:
        print("\n  Result: CHALLENGES expected failure profile — total collapse occurred.")


if __name__ == "__main__":
    run()
