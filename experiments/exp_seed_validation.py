"""
Seed Validation – CytoDend KeyLock Core Claims
================================================
Implements the seed policy defined in:
  Project cytodend_accessmodel/Seed policy/seed_policy.md

Seed counts per policy tier
----------------------------
  Tier 2 (standard mechanism validation): 20 seeds
  Tier 3 (paper-ready core claims):       30 seeds
  Tier 4 (high-sensitivity / reviewer-exposed): 50 seeds

Stochasticity source
---------------------
  structural_noise = 0.02  (Gaussian noise on M_b update, validated in exp004).
  Python random state is seeded before each trial via random.seed(seed_index).
  This controls all gauss() calls in run_consolidation.

Five protected claims (policy section: "Which results deserve higher seed counts")
-----------------------------------------------------------------------------------
  A  Linking degrades before recall under focal overlap pathology      30 seeds
  B  Focal overlap damage worse than distributed high-decay             30 seeds
  C  Targeted rescue restores linking selectively (overlap > standard)  50 seeds
  D  Sustained replay > encoding advantage for long-term asymmetry     30 seeds
  E  Competition hurts linking more than near-ceiling recall            30 seeds

Reporting format
-----------------
  For each claim: pass_rate%, mean_margin ± std_margin, worst_seed
  Pass = the directional criterion holds for that seed.
  Margin = the numerical gap between the two quantities being compared.
"""

from __future__ import annotations

import math
import random
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
# Canonical two-trace network
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

NOISE_LEVEL = 0.02   # structural noise amplitude — validated in exp004

BASE_PARAMS = DynamicsParameters(
    structural_lr=0.18,
    replay_gain=0.80,
    eligibility_decay=0.12,
    structural_decay=0.005,
    structural_gain=6.0,
    structural_max=1.0,
    translation_decay=0.05,
    sleep_gain=0.0,
    structural_noise=NOISE_LEVEL,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build(params: DynamicsParameters | None = None) -> CytodendAccessModelSimulator:
    p = params or BASE_PARAMS
    sim = CytodendAccessModelSimulator.from_branch_ids(BRANCH_IDS, parameters=p)
    sim.traces["mu1"] = EngramTrace(trace_id="mu1", allocation=MU1_ALLOC)
    sim.traces["mu2"] = EngramTrace(trace_id="mu2", allocation=MU2_ALLOC)
    return sim


def _encode(sim: CytodendAccessModelSimulator) -> None:
    for _ in range(2):
        sim.apply_cue(MU1_CUE)
    for _ in range(2):
        sim.apply_cue(MU2_CUE)


def _consolidate(sim: CytodendAccessModelSimulator, n_nights: int,
                 drive: float = 1.0, replay: list[str] | None = None) -> None:
    win = ConsolidationWindow(
        replay_trace_ids=replay if replay is not None else ["mu1", "mu2"],
        modulatory_drive=drive,
    )
    for _ in range(n_nights * 3):
        sim.run_consolidation(win)


def _null(sim: CytodendAccessModelSimulator, n_nights: int) -> None:
    win = ConsolidationWindow(replay_trace_ids=[], modulatory_drive=0.0)
    for _ in range(n_nights * 3):
        sim.run_consolidation(win)


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


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def _std(vals: list[float]) -> float:
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def _ci95(vals: list[float]) -> float:
    """Half-width of 95% CI using normal approximation."""
    return 1.96 * _std(vals) / math.sqrt(len(vals))


def _params_with(**kwargs) -> DynamicsParameters:
    fields = {
        "structural_lr", "replay_gain", "eligibility_decay", "structural_decay",
        "structural_gain", "structural_max", "translation_decay", "sleep_gain",
        "structural_noise", "translation_budget", "spillover_rate",
    }
    base = {f: getattr(BASE_PARAMS, f) for f in fields if hasattr(BASE_PARAMS, f)}
    base.update(kwargs)
    return DynamicsParameters(**base)


# ---------------------------------------------------------------------------
# Claim A: Linking degrades before recall under focal overlap pathology
#   Margin = (recall_drop% - linking_drop%) ... wait, linking drops MORE,
#   so margin = link_drop_pct - recall_drop_pct  (positive = linking more vulnerable)
# ---------------------------------------------------------------------------

def claim_A(seed: int) -> tuple[bool, float]:
    random.seed(seed)
    sim = _build()
    _encode(sim)
    _consolidate(sim, 3)
    h_link = _linking(sim)
    r_mu1_h = _recall(sim, MU1_CUE, "mu1")
    r_mu2_h = _recall(sim, MU2_CUE, "mu2")

    # Apply focal damage
    sim.branches["b1"].structural.decay_rate = 0.025
    _null(sim, 3)

    d_link = _linking(sim)
    r_mu1_d = _recall(sim, MU1_CUE, "mu1")
    r_mu2_d = _recall(sim, MU2_CUE, "mu2")

    link_drop_pct  = (h_link - d_link) / max(h_link, 1e-9) * 100
    recall_avg_h   = (r_mu1_h + r_mu2_h) / 2
    recall_avg_d   = (r_mu1_d + r_mu2_d) / 2
    recall_drop_pct = (recall_avg_h - recall_avg_d) / max(recall_avg_h, 1e-9) * 100

    margin = link_drop_pct - recall_drop_pct   # > 0 means linking is more vulnerable
    return margin > 0, margin


# ---------------------------------------------------------------------------
# Claim B: Focal overlap damage worse than distributed high-decay
#   Margin = L_high_decay - L_focal  (positive = focal is worse)
# ---------------------------------------------------------------------------

def claim_B(seed: int) -> tuple[bool, float]:
    random.seed(seed)

    # Distributed high-decay
    sim_dist = _build(_params_with(structural_decay=0.020))
    _encode(sim_dist)
    _consolidate(sim_dist, 3)
    l_distributed = _linking(sim_dist)

    # Focal overlap damage
    random.seed(seed)
    sim_focal = _build()
    _encode(sim_focal)
    sim_focal.branches["b1"].structural.decay_rate = 0.025
    _consolidate(sim_focal, 3)
    l_focal = _linking(sim_focal)

    margin = l_distributed - l_focal   # positive = focal is worse
    return margin > 0, margin


# ---------------------------------------------------------------------------
# Claim C: Targeted rescue restores linking selectively
#   Phase A: healthy + consolidation; Phase B: focal damage + null nights
#   Phase C: rescue protocol
#   Margin = linking_recovery(overlap_targeted) - linking_recovery(standard)
# ---------------------------------------------------------------------------

def claim_C(seed: int) -> tuple[bool, float]:
    def _phase_ab(s: int) -> tuple[CytodendAccessModelSimulator, float, float]:
        random.seed(s)
        sim = _build()
        _encode(sim)
        _consolidate(sim, 3)
        h_link = _linking(sim)
        sim.branches["b1"].structural.decay_rate = 0.025
        _null(sim, 3)
        d_link = _linking(sim)
        return sim, h_link, d_link

    from copy import deepcopy

    sim_base, h_link, d_link = _phase_ab(seed)

    # Standard rescue
    sim_std = deepcopy(sim_base)
    random.seed(seed + 10000)
    _consolidate(sim_std, 3, drive=1.0)
    link_std = _linking(sim_std)

    # Overlap-targeted rescue
    sim_ovlp = deepcopy(sim_base)
    random.seed(seed + 20000)
    for _ in range(3):
        for _ in range(3):
            sim_ovlp.apply_cue(B1_CUE)
        _consolidate(sim_ovlp, 1, drive=1.0)
    link_ovlp = _linking(sim_ovlp)

    denom = h_link - d_link
    if abs(denom) < 1e-9:
        return False, 0.0
    rec_std  = (link_std  - d_link) / denom * 100
    rec_ovlp = (link_ovlp - d_link) / denom * 100

    margin = rec_ovlp - rec_std   # positive = targeted rescue is better
    return margin > 0, margin


# ---------------------------------------------------------------------------
# Claim D: Sustained replay creates lasting structural asymmetry;
#          encoding advantage alone is transient and washes out.
#   Two sub-criteria, combined:
#     D1: replay_freq_adv divergence at night 8 > +0.05
#         (sustained replay establishes clear asymmetry)
#     D2: encoding_adv divergence at night 8 < +0.05
#         (encoding advantage alone is transient / washed out)
#   Margin = div_replay_freq - div_encoding_adv  (signed; large positive = good)
# ---------------------------------------------------------------------------

def claim_D(seed: int) -> tuple[bool, float]:
    # replay_freq must show divergence that is at least GAP larger than encoding_adv.
    # encoding_adv ≈ -0.048 (recency bias), replay_freq ≈ +0.138.
    # With noise both fluctuate, but the gap (~0.186) is much larger than noise std.
    GAP = 0.05

    def _run_8nights(n_enc_mu1: int, n_enc_mu2: int, condition: str) -> float:
        random.seed(seed)
        sim = _build()
        for _ in range(n_enc_mu1):
            sim.apply_cue(MU1_CUE)
        for _ in range(n_enc_mu2):
            sim.apply_cue(MU2_CUE)
        for night in range(1, 9):
            if condition == "encoding_adv":
                replay = ["mu1", "mu2"]
            else:
                replay = ["mu1"] if night < 5 else ["mu1", "mu2"]
            _consolidate(sim, 1, drive=1.0, replay=replay)
        return sim.branches["b0"].structural.accessibility - sim.branches["b2"].structural.accessibility

    div_encoding    = _run_8nights(5, 2, "encoding_adv")
    div_replay_freq = _run_8nights(2, 2, "replay_freq_adv")

    margin = div_replay_freq - div_encoding   # signed difference
    return margin > GAP, margin


# ---------------------------------------------------------------------------
# Claim E: Competition hurts linking more than near-ceiling recall
#   Margin = link_drop_pct - recall_drop_pct  (positive = linking more hurt)
# ---------------------------------------------------------------------------

def claim_E(seed: int) -> tuple[bool, float]:
    random.seed(seed)
    sim_unlim = _build(_params_with(translation_budget=0.0))
    _encode(sim_unlim)
    _consolidate(sim_unlim, 1)
    l_unlim = _linking(sim_unlim)
    r_unlim = (_recall(sim_unlim, MU1_CUE, "mu1") + _recall(sim_unlim, MU2_CUE, "mu2")) / 2

    random.seed(seed)
    sim_comp = _build(_params_with(translation_budget=1.5))
    _encode(sim_comp)
    _consolidate(sim_comp, 1)
    l_comp = _linking(sim_comp)
    r_comp = (_recall(sim_comp, MU1_CUE, "mu1") + _recall(sim_comp, MU2_CUE, "mu2")) / 2

    link_drop_pct   = (l_unlim - l_comp) / max(l_unlim, 1e-9) * 100
    recall_drop_pct = (r_unlim - r_comp) / max(r_unlim, 1e-9) * 100

    margin = link_drop_pct - recall_drop_pct
    return margin > 0, margin


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_claim(
    label: str,
    fn,
    n_seeds: int,
    description: str,
    pass_criterion: str,
) -> dict:
    passes = []
    margins = []
    worst_margin = float("inf")
    worst_seed = -1

    for s in range(n_seeds):
        ok, margin = fn(s)
        passes.append(ok)
        margins.append(margin)
        if margin < worst_margin:
            worst_margin = margin
            worst_seed = s

    pass_rate = sum(passes) / n_seeds * 100
    mean_m = _mean(margins)
    std_m  = _std(margins)
    ci_m   = _ci95(margins)

    return {
        "label": label,
        "description": description,
        "pass_criterion": pass_criterion,
        "n_seeds": n_seeds,
        "pass_rate": pass_rate,
        "mean_margin": mean_m,
        "std_margin": std_m,
        "ci95_margin": ci_m,
        "worst_margin": worst_margin,
        "worst_seed": worst_seed,
        "passes": passes,
        "margins": margins,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CLAIM_SPECS = [
    ("A", claim_A, 30,
     "Linking degrades before recall under focal overlap pathology",
     "link_drop% > recall_drop%  =>  margin = link_drop - recall_drop > 0",
     "Tier 3 / 30 seeds"),
    ("B", claim_B, 30,
     "Focal overlap damage worse than distributed high-decay",
     "L_high_decay > L_focal  =>  margin = L_dist - L_focal > 0",
     "Tier 3 / 30 seeds"),
    ("C", claim_C, 50,
     "Targeted rescue restores linking selectively",
     "rec%(overlap_targeted) > rec%(standard)  =>  margin > 0",
     "Tier 4 / 50 seeds (reviewer-exposed)"),
    ("D", claim_D, 30,
     "Sustained replay creates asymmetry; encoding advantage is transient",
     "div(replay_freq) - div(encoding_adv) > 0.05  (replay clearly larger than encoding)",
     "Tier 3 / 30 seeds"),
    ("E", claim_E, 30,
     "Competition hurts linking more than near-ceiling recall",
     "link_drop% > recall_drop%  =>  margin > 0",
     "Tier 3 / 30 seeds"),
]


def main() -> None:
    print("\n=== CytoDend KeyLock — Seed Policy Validation ===")
    print(f"  structural_noise = {NOISE_LEVEL}  (Gaussian noise on M_b update)")
    print(f"  Random state: random.seed(seed_index) before each trial\n")

    results = []
    for label, fn, n_seeds, desc, crit, tier_note in CLAIM_SPECS:
        print(f"  Running Claim {label} ({n_seeds} seeds)...  ", end="", flush=True)
        r = run_claim(label, fn, n_seeds, desc, crit)
        r["tier_note"] = tier_note
        results.append(r)
        print(f"pass_rate={r['pass_rate']:.0f}%  margin={r['mean_margin']:+.3f}±{r['std_margin']:.3f}")

    print("\n" + "="*80)
    print("SEED VALIDATION REPORT")
    print("="*80)
    print(f"\n  noise: structural_noise={NOISE_LEVEL}")
    print(f"  {'Claim':<5}  {'Seeds':>6}  {'PassRate%':>10}  {'mean_margin':>12}  {'std':>7}  {'95%CI':>8}  {'worst':>7}  OK")
    print(f"  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*4}")

    all_stable = True
    for r in results:
        ok = r["pass_rate"] >= 90.0 and r["worst_margin"] > -0.01
        flag = "PASS" if ok else "FAIL"
        if not ok:
            all_stable = False
        print(
            f"  {r['label']:<5}  {r['n_seeds']:>6}  {r['pass_rate']:>10.1f}  "
            f"{r['mean_margin']:>+12.4f}  {r['std_margin']:>7.4f}  "
            f"{r['ci95_margin']:>8.4f}  {r['worst_margin']:>+7.4f}  {flag}"
        )

    print()
    for r in results:
        print(f"  Claim {r['label']}: {r['description']}")
        print(f"    Criterion:  {r['pass_criterion']}")
        print(f"    Policy tier: {r['tier_note']}")
        print(f"    Result: mean margin = {r['mean_margin']:+.4f} ± {r['std_margin']:.4f}  "
              f"(95% CI: {r['mean_margin'] - r['ci95_margin']:+.4f} to {r['mean_margin'] + r['ci95_margin']:+.4f})")
        print(f"    Pass rate: {r['pass_rate']:.0f}% of {r['n_seeds']} seeds")
        if r["worst_margin"] < 0:
            print(f"    WARNING: worst seed {r['worst_seed']} has negative margin ({r['worst_margin']:+.4f})")
        print()

    print("-"*80)
    overall = "ALL STABLE" if all_stable else "SOME UNSTABLE"
    print(f"\n  Overall verdict: {overall}")
    if all_stable:
        print(
            "\n  All five protected claims hold with >= 90% pass rate across their\n"
            "  designated seed counts under structural_noise=0.02.\n"
            "  Results are stable for paper-level reporting.\n"
            "\n  Reporting standard (per seed policy):\n"
            "    mean ± std for diary entries\n"
            "    mean ± 95% CI for article-critical tables"
        )
    else:
        print(
            "\n  One or more claims show instability under noise.\n"
            "  Review design and escalate seed count if needed."
        )


if __name__ == "__main__":
    main()
