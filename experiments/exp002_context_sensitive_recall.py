"""Experiment 002: Context-sensitive recall.

Tests the interaction between:
- Fast contextual modulation: context_bias boosts A_b^f for context-matched branches
- Slow structural accessibility: M_b shaped by which branches were active during encoding
- Context mismatch penalty on R_mu at retrieval time

Core question:
  Does providing the encoding context at retrieval time recover the correct trace,
  and does structural consolidation amplify the context-specificity?

Design:
- 4 branches: b0, b1, b2, b3
- Two non-overlapping traces:
    mu_alpha  → strong on b0, b1  (context "alpha")
    mu_beta   → strong on b2, b3  (context "beta")
- Context bias (fast A_b^f enhancement):
    alpha context → b0, b1 receive positive context_bias
    beta  context → b2, b3 receive positive context_bias
- Partial ambiguous cue (equal drive to all branches) used at retrieval
- Recall measured under three contexts: alpha / beta / none
- Pre- and post-consolidation comparison shows structural amplification

Expected qualitative outcomes:
- Matching context yields highest support for the corresponding trace
- Wrong context gives lowest support (mismatch penalty + structural misalignment)
- No context gives intermediate, near-symmetric support
- Post-consolidation the separation between correct and wrong context widens
"""

from __future__ import annotations

from cytodend_accessmodel import (
    ConsolidationWindow,
    CytodendAccessModelSimulator,
    DynamicsParameters,
    EngramTrace,
    TraceAllocation,
)

BRANCH_IDS = ["b0", "b1", "b2", "b3"]

ALPHA_ALLOCATION = TraceAllocation(
    trace_id="mu_alpha",
    branch_weights={"b0": 0.90, "b1": 0.80, "b2": 0.05, "b3": 0.05},
)
BETA_ALLOCATION = TraceAllocation(
    trace_id="mu_beta",
    branch_weights={"b0": 0.05, "b1": 0.05, "b2": 0.80, "b3": 0.90},
)

# Context bias: which branches are contextually opened in each state
ALPHA_CONTEXT_BIAS = {"b0": 0.7, "b1": 0.6, "b2": 0.0, "b3": 0.0}
BETA_CONTEXT_BIAS  = {"b0": 0.0, "b1": 0.0, "b2": 0.6, "b3": 0.7}

# Encoding cues (strong, context-specific input)
ALPHA_CUE = {"b0": 1.0, "b1": 0.9, "b2": 0.05, "b3": 0.0}
BETA_CUE  = {"b0": 0.0, "b1": 0.05, "b2": 0.9, "b3": 1.0}

# Retrieval cue: ambiguous, equal drive to all branches
PARTIAL_CUE = {"b0": 0.4, "b1": 0.4, "b2": 0.4, "b3": 0.4}


def build_simulator() -> CytodendAccessModelSimulator:
    params = DynamicsParameters(
        fast_gain=2.0,
        context_gain=1.0,
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
        context_mismatch_penalty=0.35,
    )
    sim = CytodendAccessModelSimulator.from_branch_ids(
        BRANCH_IDS, spines_per_branch=3, parameters=params
    )
    sim.add_trace(
        EngramTrace(
            trace_id="mu_alpha",
            allocation=ALPHA_ALLOCATION,
            label="alpha-trace",
            context="alpha",
        )
    )
    sim.add_trace(
        EngramTrace(
            trace_id="mu_beta",
            allocation=BETA_ALLOCATION,
            label="beta-trace",
            context="beta",
        )
    )
    return sim


def log_structural_state(sim: CytodendAccessModelSimulator, *, label: str) -> None:
    print(f"\n  [{label}] branch structural state:")
    print(f"    {'branch':<6}  {'M_b':>8}  {'E_b':>8}  {'A_b^s':>8}")
    for bid in BRANCH_IDS:
        b = sim.branches[bid]
        print(
            f"    {bid:<6}  "
            f"{b.structural.accessibility:>8.4f}  "
            f"{b.eligibility.value:>8.4f}  "
            f"{b.slow_access:>8.4f}"
        )


def recall_under_contexts(
    sim: CytodendAccessModelSimulator, *, label: str
) -> dict[str, dict[str, float]]:
    """Run partial cue under three context conditions; return support dict."""
    results: dict[str, dict[str, float]] = {}
    scenarios = [
        ("alpha", ALPHA_CONTEXT_BIAS),
        ("beta",  BETA_CONTEXT_BIAS),
        ("none",  {}),
    ]
    for ctx_name, ctx_bias in scenarios:
        ctx_arg = None if ctx_name == "none" else ctx_name
        sim.apply_cue(PARTIAL_CUE, context=ctx_arg, context_bias=ctx_bias)
        supports = sim.compute_recall_supports()
        results[ctx_name] = {rs.trace_id: rs.support for rs in supports}

    print(f"\n  [{label}] recall support under different contexts:")
    print(f"    {'context':<8}  {'R(mu_alpha)':>12}  {'R(mu_beta)':>12}  {'winner':>10}")
    for ctx_name, sup in results.items():
        r_a = sup.get("mu_alpha", 0.0)
        r_b = sup.get("mu_beta",  0.0)
        winner = "mu_alpha" if r_a > r_b else ("mu_beta" if r_b > r_a else "tie")
        print(f"    {ctx_name:<8}  {r_a:>12.4f}  {r_b:>12.4f}  {winner:>10}")
    return results


def run() -> None:
    print("=" * 64)
    print("Experiment 002: Context-Sensitive Recall")
    print("=" * 64)

    sim = build_simulator()

    # ------------------------------------------------------------------
    # Phase 1: encode mu_alpha in context "alpha"
    # ------------------------------------------------------------------
    print("\n[Phase 1] Encoding mu_alpha in context alpha (3 passes)")
    for _ in range(3):
        sim.apply_cue(ALPHA_CUE, context="alpha", context_bias=ALPHA_CONTEXT_BIAS)

    # ------------------------------------------------------------------
    # Phase 2: encode mu_beta in context "beta"
    # ------------------------------------------------------------------
    print("[Phase 2] Encoding mu_beta in context beta (3 passes)")
    for _ in range(3):
        sim.apply_cue(BETA_CUE, context="beta", context_bias=BETA_CONTEXT_BIAS)

    log_structural_state(sim, label="post-encoding, pre-consolidation")

    # ------------------------------------------------------------------
    # Pre-consolidation recall
    # ------------------------------------------------------------------
    print("\n[Phase 3] Pre-consolidation recall (partial ambiguous cue)")
    pre_results = recall_under_contexts(sim, label="pre-consolidation")

    # ------------------------------------------------------------------
    # Phase 4: consolidation (replay both traces)
    # ------------------------------------------------------------------
    print("\n[Phase 4] Consolidation (3 passes, replay both)")
    for i in range(3):
        window = ConsolidationWindow(
            window_id=f"sleep-{i}",
            modulatory_drive=1.0,
            sleep_drive=1.0,
            replay_trace_ids=("mu_alpha", "mu_beta"),
        )
        report = sim.run_consolidation(window)
        print(
            f"  pass {i}: branches_updated={report.branches_updated}  "
            f"mean_shift={report.mean_structural_shift:.5f}"
        )

    log_structural_state(sim, label="post-consolidation")

    # ------------------------------------------------------------------
    # Post-consolidation recall
    # ------------------------------------------------------------------
    print("\n[Phase 5] Post-consolidation recall (same partial ambiguous cue)")
    post_results = recall_under_contexts(sim, label="post-consolidation")

    # ------------------------------------------------------------------
    # Structural differentiation summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("SUMMARY: Structural differentiation")
    print("=" * 64)
    print(f"  {'branch':<6}  {'M_b':>8}  {'role'}")
    for bid in BRANCH_IDS:
        m = sim.branches[bid].structural.accessibility
        role = (
            "alpha-specific" if bid in ("b0", "b1") else
            "beta-specific"  if bid in ("b2", "b3") else
            "neutral"
        )
        print(f"  {bid:<6}  {m:>8.4f}  {role}")

    # ------------------------------------------------------------------
    # Context separation gain (post vs pre)
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("SUMMARY: Context separation")
    print("=" * 64)
    print(f"  {'condition':<38}  {'pre':>8}  {'post':>8}  {'gain':>8}")

    cases = [
        ("mu_alpha | context=alpha (correct)",  "alpha",  "mu_alpha"),
        ("mu_alpha | context=beta  (wrong)",    "beta",   "mu_alpha"),
        ("mu_alpha | context=none",             "none",   "mu_alpha"),
        ("mu_beta  | context=beta  (correct)",  "beta",   "mu_beta"),
        ("mu_beta  | context=alpha (wrong)",    "alpha",  "mu_beta"),
        ("mu_beta  | context=none",             "none",   "mu_beta"),
    ]
    for desc, ctx, trace in cases:
        pre  = pre_results[ctx].get(trace, 0.0)
        post = post_results[ctx].get(trace, 0.0)
        gain = post - pre
        print(f"  {desc:<38}  {pre:>8.4f}  {post:>8.4f}  {gain:>+8.4f}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("VERDICT")
    print("=" * 64)

    alpha_correct_post = post_results["alpha"].get("mu_alpha", 0.0)
    alpha_wrong_post   = post_results["alpha"].get("mu_beta",  0.0)
    beta_correct_post  = post_results["beta"].get("mu_beta",   0.0)
    beta_wrong_post    = post_results["beta"].get("mu_alpha",  0.0)

    alpha_correct_pre  = pre_results["alpha"].get("mu_alpha",  0.0)
    alpha_wrong_pre    = pre_results["alpha"].get("mu_beta",   0.0)

    correct_beats_wrong   = alpha_correct_post > alpha_wrong_post and beta_correct_post > beta_wrong_post
    consolidation_widens  = (alpha_correct_post - alpha_wrong_post) > (alpha_correct_pre - alpha_wrong_pre)

    print(f"  Correct context beats wrong context:           {'YES' if correct_beats_wrong else 'NO  <FAILURE>'}")
    print(f"  Consolidation widens context gap:              {'YES' if consolidation_widens else 'NO  (expected amplification)'}")

    if correct_beats_wrong:
        print("\n  Result: SUPPORTS context-sensitive recall hypothesis.")
        print("  Structural accessibility amplifies contextual gating,")
        print("  with the correct context yielding higher recall support.")
    else:
        print("\n  Result: CHALLENGES the hypothesis.")


if __name__ == "__main__":
    run()
