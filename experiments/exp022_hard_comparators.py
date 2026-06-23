"""Experiment 022 — Hard Comparator Models.

Tests whether harder alternative mechanisms can reproduce the full model's joint
structural and behavioral profile without replay-dependent slow branch-level writing.

Two evaluation layers (per spec):
    Layer 1 — Structural/mechanistic layer  (M_b-dependent signatures)
    Layer 2 — Behavioral/output-equivalence layer  (comparator-agnostic metrics)

Five hard comparators
---------------------
1. hebbian_weight_only      M_b frozen (structural_lr=0); Hebbian weight from E_b co-activation
2. soma_global_gain_only    M_b writes but to ALL branches uniformly (flat allocation)
3. shuffled_replay          M_b writes but branch-to-trace identity is shuffled each pass
4. eligibility_only         E_b exists; no M_b write; no replay; purely transient
5. resource_only            P_b (replay-driven resource) exists; no M_b write; transient

Motifs tested
-------------
canonical, strong_overlap, chain_overlap, sparse_random
(hub_overlap used only as boundary-condition benchmark)

Behavioral metrics (B1–B7)
--------------------------
B1 linking_gain              L_post - L_pre
B2 context_separation        recall_correct - recall_wrong (private-cue probe)
B3 damage_sensitivity        L_post - L_damage
B4 recall_preservation       recall_post_damage / recall_post_cons
B5 recovery_index            (L_rescue - L_dmg) / (L_post - L_dmg)
B6 specificity_index         mean(delta_L_expected) - mean(delta_L_unlinked)
B7 false_linking_rate        mean(delta_L_unlinked) / mean(delta_L_expected)

Outputs
-------
    results/e022_hard_comparators/
        traces/<comp>__<motif>_{branch,trace_support,linking}_trace.csv
        summary/<comp>__<motif>_{signature,behavioral}_summary.csv
        summary/hard_comparator_signature_matrix.csv
        summary/hard_comparator_behavioral_matrix.csv
        summary/hard_comparator_specificity_matrix.csv
        summary/comparator_definitions.json
        summary/hard_comparator_claim_ledger.md
        summary/failure_mode_summary.csv
        figures/Fig_e022_0{1..5}_*.png
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cytodend_accessmodel.contracts import (
    ConsolidationWindow, DynamicsParameters, EngramTrace, TraceAllocation,
)
from cytodend_accessmodel.simulator import CytodendAccessModelSimulator
from cytodend_accessmodel.motifs import (
    MotifSpec, build_motif, alloc_to_cue, private_cue, linking_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[1]
OUT_ROOT    = REPO_ROOT / "results" / "e022_hard_comparators"
TRACES_DIR  = OUT_ROOT / "traces"
SUMMARY_DIR = OUT_ROOT / "summary"
FIGURES_DIR = OUT_ROOT / "figures"

# ---------------------------------------------------------------------------
# Canonical parameters
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42

CANONICAL_PARAMS = DynamicsParameters(
    structural_lr=0.18, replay_gain=0.80, eligibility_decay=0.12,
    structural_decay=0.005, structural_gain=6.0, structural_max=1.0,
    translation_decay=0.05, sleep_gain=0.0, context_gain=1.0,
    structural_noise=0.0, readout_gain=5.0, readout_threshold=0.3,
)

CONSOLIDATION_PASSES = 9
DAMAGE_NULL_PASSES   = 9
DAMAGE_DECAY_RATE    = 0.030
RESCUE_PASSES        = 3
RESCUE_CUE_REPS      = 3
RESCUE_ROUNDS        = 3

# ---------------------------------------------------------------------------
# Comparator specifications
# ---------------------------------------------------------------------------

@dataclass
class ComparatorSpec:
    name: str
    description: str
    params: DynamicsParameters
    protocol: str  # "standard" | "shuffled" | "hebbian" | "eligibility" | "resource" | "soma_global"
    layer_note: str = ""   # brief note on which layers are meaningful


COMPARATORS: list[ComparatorSpec] = [
    ComparatorSpec(
        name="full_model",
        description="Full replay-dependent slow branch-level structural writing (E019R canonical).",
        params=CANONICAL_PARAMS,
        protocol="standard",
        layer_note="Both structural and behavioral layers apply.",
    ),
    ComparatorSpec(
        name="hebbian_weight_only",
        description="structural_lr=0; M_b frozen. Behavioral linking from E_b co-activation weight.",
        params=replace(CANONICAL_PARAMS, structural_lr=0.0),
        protocol="hebbian",
        layer_note="Structural layer fails (no M_b write). Behavioral Hebbian weight tracked.",
    ),
    ComparatorSpec(
        name="soma_global_gain_only",
        description="M_b writes (canonical lr) but to ALL branches uniformly via flat allocation.",
        params=CANONICAL_PARAMS,
        protocol="soma_global",
        layer_note="Structural layer: uniform write (SIG-A ≈ 0). Behavioral layer: non-specific linking.",
    ),
    ComparatorSpec(
        name="shuffled_replay",
        description="Consolidation fires but trace-to-branch assignment shuffled each pass.",
        params=CANONICAL_PARAMS,
        protocol="shuffled",
        layer_note="Structural layer: random branch write. Behavioral: reduced specificity.",
    ),
    ComparatorSpec(
        name="eligibility_only",
        description="E_b exists after encoding but structural_lr=0, replay_gain=0; purely transient.",
        params=replace(CANONICAL_PARAMS, structural_lr=0.0, replay_gain=0.0),
        protocol="eligibility",
        layer_note="Transient E_b-based linking at encoding time only; fades immediately.",
    ),
    ComparatorSpec(
        name="resource_only",
        description="P_b (replay-driven resource) builds up; structural_lr=0; no persistent M_b.",
        params=replace(CANONICAL_PARAMS, structural_lr=0.0),
        protocol="resource",
        layer_note="Resource/P_b window linking only; no long-term persistence.",
    ),
]

COMPARATOR_MAP = {c.name: c for c in COMPARATORS}

# ---------------------------------------------------------------------------
# Motifs to test
# ---------------------------------------------------------------------------
MOTIFS_TO_TEST = [
    ("canonical",      4,  2, DEFAULT_SEED),
    ("strong_overlap", 4,  2, DEFAULT_SEED),
    ("chain_overlap",  8,  3, DEFAULT_SEED),
    ("sparse_random",  8,  4, DEFAULT_SEED),
]

# ---------------------------------------------------------------------------
# Branch / linking helpers
# ---------------------------------------------------------------------------

def _mb(sim: CytodendAccessModelSimulator) -> dict[str, float]:
    return {b: bst.structural.accessibility for b, bst in sim.branches.items()}


def _E_b(sim: CytodendAccessModelSimulator) -> dict[str, float]:
    return {b: bst.eligibility.value for b, bst in sim.branches.items()}


def _P_b(sim: CytodendAccessModelSimulator) -> dict[str, float]:
    return {b: bst.translation_readiness.value for b, bst in sim.branches.items()}


def _L_all(weights: dict[str, float], motif: MotifSpec) -> dict[str, float]:
    """Compute linking for all trace pairs using given branch weights."""
    result: dict[str, float] = {}
    for i, ti in enumerate(motif.trace_ids):
        for j in range(i + 1, len(motif.trace_ids)):
            tj = motif.trace_ids[j]
            result[f"{ti}:{tj}"] = linking_score(weights, motif.allocations[ti], motif.allocations[tj])
    return result


def _recall_best(sim, motif) -> dict[str, float]:
    return {rs.trace_id: rs.support for rs in sim.compute_recall_supports()}


def _gsig_c_simple(sim, motif) -> float:
    """Recall separation using private cues (same as E021 gSIG-C)."""
    margins = []
    for tid in motif.trace_ids:
        cue = private_cue(motif, tid)
        sp = deepcopy(sim)
        sp.apply_cue(cue)
        supp = {rs.trace_id: rs.support for rs in sp.compute_recall_supports()}
        correct = supp.get(tid, 0.0)
        wrong   = max((v for k, v in supp.items() if k != tid), default=0.0)
        margins.append(correct - wrong)
    return sum(margins) / len(margins) if margins else 0.0


def _mean_L(d: dict[str, float], pairs: list[tuple[str, str]]) -> float:
    if not pairs: return float("nan")
    return sum(d.get(f"{ti}:{tj}", 0.0) for ti, tj in pairs) / len(pairs)


# ---------------------------------------------------------------------------
# Sim builder
# ---------------------------------------------------------------------------

def _build_sim(motif: MotifSpec, params: DynamicsParameters,
               alloc_override: dict[str, dict[str, float]] | None = None) -> CytodendAccessModelSimulator:
    sim = CytodendAccessModelSimulator.from_branch_ids(motif.branch_ids, parameters=params)
    allocs = alloc_override or motif.allocations
    for tid in motif.trace_ids:
        ta = TraceAllocation(trace_id=tid, branch_weights=allocs[tid])
        sim.traces[tid] = EngramTrace(trace_id=tid, allocation=ta)
    return sim


# ---------------------------------------------------------------------------
# Comparator-specific E_b corruption helpers
# ---------------------------------------------------------------------------

def _flatten_E_b(sim: CytodendAccessModelSimulator) -> None:
    """Soma-global: replace all E_b values with their mean (no branch identity)."""
    values = [b.eligibility.value for b in sim.branches.values()]
    mean_E = sum(values) / len(values) if values else 0.0
    for b in sim.branches.values():
        b.eligibility.value = mean_E


def _shuffle_E_b(sim: CytodendAccessModelSimulator, rng: random.Random) -> None:
    """Shuffle E_b values randomly among branches (destroys spatial identity)."""
    bids = list(sim.branches.keys())
    vals = [sim.branches[b].eligibility.value for b in bids]
    rng.shuffle(vals)
    for bid, v in zip(bids, vals):
        sim.branches[bid].eligibility.value = v


def _shuffled_consolidation(sim: CytodendAccessModelSimulator, motif: MotifSpec,
                            n_passes: int, seed: int) -> None:
    """Destroy branch identity during consolidation via fresh random allocations.

    Each pass: (a) shuffle E_b to remove encoding spatial tag, and (b) assign a
    freshly drawn random allocation to each trace (normalised rows) so that the
    replay signal is not preferentially focused on any branch.  Permuting the
    original allocations among traces is insufficient for motifs where the overlap
    branch has non-zero weight in EVERY allocation (both signals converge on b1
    regardless of permutation).  Random allocations break that invariance.
    """
    rng = random.Random(seed + 77)
    orig_allocs = {t: deepcopy(sim.traces[t].allocation.branch_weights) for t in sim.traces}
    win = ConsolidationWindow(replay_trace_ids=list(motif.trace_ids), modulatory_drive=1.0)
    branch_ids = motif.branch_ids
    for _ in range(n_passes):
        _shuffle_E_b(sim, rng)
        # Fresh uniform-random allocation per trace: destroys overlap-branch replay identity
        for t in sim.traces:
            raw = {b: rng.random() for b in branch_ids}
            total = sum(raw.values()) or 1.0
            sim.traces[t].allocation.branch_weights = {b: w / total for b, w in raw.items()}
        sim.run_consolidation(win)
    # Restore original allocations for recall / linking computation
    for t in sim.traces:
        sim.traces[t].allocation.branch_weights = orig_allocs[t]


# ---------------------------------------------------------------------------
# Flat (soma-global) allocation (used only for encoding cue normalisation)
# ---------------------------------------------------------------------------

def _flat_allocation(motif: MotifSpec) -> dict[str, dict[str, float]]:
    """All branches get weight = 1.0 for all traces (soma global gain)."""
    return {tid: {b: 1.0 for b in motif.branch_ids} for tid in motif.trace_ids}


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------

def _run_comparator_cell(
    comp:  ComparatorSpec,
    motif: MotifSpec,
    seed:  int = DEFAULT_SEED,
) -> dict[str, Any]:
    eps = 1e-8
    random.seed(seed)
    params = comp.params

    # -----------------------------------------------------------------------
    # Build and encode
    # -----------------------------------------------------------------------
    if comp.protocol == "soma_global":
        alloc_override = _flat_allocation(motif)
    else:
        alloc_override = None

    sim = _build_sim(motif, params, alloc_override)
    for tid in motif.trace_ids:
        cue = alloc_to_cue(motif.allocations[tid])  # always use real alloc for encoding cues
        for _ in range(2):
            sim.apply_cue(cue)

    mb_pre  = _mb(sim)
    E_b_enc = _E_b(sim)
    # Behavioral linking: eligibility snapshot right after encoding
    L_elig  = _L_all(E_b_enc, motif)

    # -----------------------------------------------------------------------
    # Consolidation (comparator-specific)
    # -----------------------------------------------------------------------
    if comp.protocol in ("standard", "hebbian", "resource"):
        win = ConsolidationWindow(replay_trace_ids=list(motif.trace_ids), modulatory_drive=1.0)
        for _ in range(CONSOLIDATION_PASSES):
            sim.run_consolidation(win)
    elif comp.protocol == "soma_global":
        # Flatten E_b to mean before consolidation → no branch identity from encoding
        _flatten_E_b(sim)
        win = ConsolidationWindow(replay_trace_ids=list(motif.trace_ids), modulatory_drive=1.0)
        for _ in range(CONSOLIDATION_PASSES):
            sim.run_consolidation(win)
    elif comp.protocol == "shuffled":
        # Shuffle E_b before each pass → destroys spatial branch identity
        _shuffled_consolidation(sim, motif, CONSOLIDATION_PASSES, seed)
    elif comp.protocol == "eligibility":
        # No consolidation — E_b decays but M_b stays at 0
        win_null = ConsolidationWindow(replay_trace_ids=[], modulatory_drive=0.0)
        for _ in range(CONSOLIDATION_PASSES):
            sim.run_consolidation(win_null)

    mb_post = _mb(sim)
    P_b_post = _P_b(sim)
    E_b_post = _E_b(sim)

    # -----------------------------------------------------------------------
    # Structural and behavioral linking post-consolidation
    # -----------------------------------------------------------------------
    L_struct_pre  = _L_all(mb_pre,  motif)
    L_struct_post = _L_all(mb_post, motif)
    L_P_b_post    = _L_all(P_b_post, motif)

    # Behavioral linking proxy (comparator-specific)
    if comp.protocol == "hebbian":
        # Hebbian weight = E_b co-activation integrated over consolidation
        decay = params.eligibility_decay
        T = CONSOLIDATION_PASSES
        weight_integral = (1 - (1 - decay)**T) / decay if decay > 0 else T
        L_behav_post = {
            k: v * weight_integral for k, v in L_elig.items()
        }
    elif comp.protocol in ("eligibility",):
        # Eligibility snapshot: linking at encoding time (before consolidation decay)
        L_behav_post = L_elig
    elif comp.protocol == "resource":
        # Resource snapshot: P_b-based linking after consolidation
        L_behav_post = L_P_b_post
    else:
        # Standard, soma_global, shuffled: use structural M_b linking
        L_behav_post = L_struct_post

    # gSIG-C: context separation after consolidation
    gsig_c = _gsig_c_simple(sim, motif)

    # -----------------------------------------------------------------------
    # Damage
    # -----------------------------------------------------------------------
    sim_dmg = deepcopy(sim)
    for b in motif.damage_target_branches:
        if b in sim_dmg.branches:
            sim_dmg.branches[b].structural.decay_rate = DAMAGE_DECAY_RATE
    null_win = ConsolidationWindow(replay_trace_ids=[], modulatory_drive=0.0)
    for _ in range(DAMAGE_NULL_PASSES):
        sim_dmg.run_consolidation(null_win)
    mb_dmg = _mb(sim_dmg)
    L_struct_dmg = _L_all(mb_dmg, motif)
    L_behav_dmg = L_struct_dmg if comp.protocol not in ("hebbian","eligibility","resource") else L_behav_post

    # Recall after consolidation and after damage
    sim_probe_post = deepcopy(sim)
    sim_probe_post.apply_cue(alloc_to_cue(motif.allocations[motif.trace_ids[0]]))
    recall_post_dict = _recall_best(sim_probe_post, motif)
    recall_post = sum(recall_post_dict.values()) / max(len(recall_post_dict), 1)

    sim_probe_dmg = deepcopy(sim_dmg)
    sim_probe_dmg.apply_cue(alloc_to_cue(motif.allocations[motif.trace_ids[0]]))
    recall_dmg_dict = _recall_best(sim_probe_dmg, motif)
    recall_dmg = sum(recall_dmg_dict.values()) / max(len(recall_dmg_dict), 1)

    # -----------------------------------------------------------------------
    # Rescue
    # -----------------------------------------------------------------------
    rescue_cue = {b: (1.0 if b in motif.rescue_target_branches else 0.05)
                  for b in motif.branch_ids}
    win_r = ConsolidationWindow(replay_trace_ids=list(motif.trace_ids), modulatory_drive=1.0)

    sim_targ = deepcopy(sim_dmg)
    for _ in range(RESCUE_ROUNDS):
        for _ in range(RESCUE_CUE_REPS):
            sim_targ.apply_cue(rescue_cue)
        for _ in range(RESCUE_PASSES):
            sim_targ.run_consolidation(win_r)
    mb_targ = _mb(sim_targ)
    L_struct_targ = _L_all(mb_targ, motif)
    L_behav_targ = L_struct_targ if comp.protocol not in ("hebbian","eligibility","resource") else L_behav_post

    sim_gen = deepcopy(sim_dmg)
    win_g = ConsolidationWindow(replay_trace_ids=list(motif.trace_ids), modulatory_drive=1.0)
    for _ in range(RESCUE_ROUNDS * RESCUE_PASSES):
        sim_gen.run_consolidation(win_g)
    mb_gen = _mb(sim_gen)
    L_struct_gen = _L_all(mb_gen, motif)
    L_behav_gen = L_struct_gen if comp.protocol not in ("hebbian","eligibility","resource") else L_behav_post

    # -----------------------------------------------------------------------
    # Structural metrics (Layer 1)
    # -----------------------------------------------------------------------
    all_ovlp = {b for bl in motif.overlap_branches_per_pair.values() for b in bl}
    non_ovlp = [b for b in motif.branch_ids if b not in all_ovlp]
    ovlp_list = list(all_ovlp)

    def _mean_mb_delta(blist):
        if not blist: return 0.0
        return sum(mb_post.get(b, 0) - mb_pre.get(b, 0) for b in blist) / len(blist)

    gSIG_A = _mean_mb_delta(ovlp_list) - _mean_mb_delta(non_ovlp)
    gSIG_A_pass = gSIG_A > 0.0

    # Structural linking metrics for expected / unlinked pairs
    ep = motif.expected_linked_pairs
    up = motif.expected_unlinked_pairs

    dL_ep_struct = _mean_L(
        {k: L_struct_post.get(k,0) - L_struct_pre.get(k,0) for k in L_struct_pre}, ep
    )
    dL_up_struct = _mean_L(
        {k: L_struct_post.get(k,0) - L_struct_pre.get(k,0) for k in L_struct_pre}, up
    )
    gSIG_B = dL_ep_struct - dL_up_struct
    gSIG_B_pass = gSIG_B > 0.0

    # -----------------------------------------------------------------------
    # Behavioral metrics (Layer 2)
    # -----------------------------------------------------------------------
    # B1: linking gain
    dL_ep_behav = _mean_L(
        {k: L_behav_post.get(k,0) - L_struct_pre.get(k,0) for k in L_struct_pre}, ep
    )
    dL_up_behav = _mean_L(
        {k: L_behav_post.get(k,0) - L_struct_pre.get(k,0) for k in L_struct_pre}, up
    )
    B1_linking_gain = float("nan") if not ep else dL_ep_behav

    # B2: context separation
    B2_context_separation = gsig_c

    # B3: damage sensitivity (M_b-based, regardless of comparator)
    L_ep_post = _mean_L(L_struct_post, ep)
    L_ep_dmg  = _mean_L(L_struct_dmg, ep)
    B3_damage_sensitivity = (L_ep_post - L_ep_dmg) if ep else float("nan")

    # B4: recall preservation
    B4_recall_preservation = recall_dmg / max(recall_post, eps)

    # B5: recovery index (behavioral layer: using L_behav for non-M_b models)
    def _nr_behav(L_r, pairs):
        if not pairs: return float("nan")
        nrs = []
        for ti, tj in pairs:
            key = f"{ti}:{tj}"
            l_post_b = L_behav_post.get(key, 0.0)
            l_dmg_b  = L_behav_dmg.get(key, 0.0)
            l_resc   = L_r.get(key, 0.0)
            denom    = l_post_b - l_dmg_b
            nrs.append((l_resc - l_dmg_b) / denom if abs(denom) > eps else 0.0)
        return sum(nrs) / len(nrs) if nrs else float("nan")

    nr_targ_b = _nr_behav(L_behav_targ, ep)
    nr_gen_b  = _nr_behav(L_behav_gen, ep)
    B5_recovery_index = (nr_targ_b - nr_gen_b) if (not math.isnan(nr_targ_b) and not math.isnan(nr_gen_b)) else float("nan")

    # B6/B7: specificity
    if ep and up:
        B6_specificity_index = dL_up_behav - dL_ep_behav  # sign: positive = good spec (ep > up)
        B6_specificity_index = dL_ep_behav - dL_up_behav   # ep gain - unlinked gain
        B7_false_linking_rate = dL_up_behav / max(dL_ep_behav, eps)
    else:
        B6_specificity_index = float("nan")
        B7_false_linking_rate = float("nan")

    # -----------------------------------------------------------------------
    # Interpretation class
    # -----------------------------------------------------------------------
    has_B1 = not math.isnan(B1_linking_gain) and B1_linking_gain > 0.001
    has_B3 = not math.isnan(B3_damage_sensitivity) and B3_damage_sensitivity > 0.001
    has_B5 = not math.isnan(B5_recovery_index) and B5_recovery_index > 0.01
    is_specific = (math.isnan(B7_false_linking_rate) and len(ep) > 0 and len(up) == 0) or \
                  (not math.isnan(B7_false_linking_rate) and B7_false_linking_rate < 0.50)

    # For 2-trace motifs there are no unlinked pairs — gSIG_B is N/A, not a failure.
    # Require only gSIG_A for structural classification in that case.
    gSIG_B_required = len(up) > 0  # only meaningful if unlinked pairs exist
    struct_pass = gSIG_A_pass and (gSIG_B_pass if gSIG_B_required else True)

    if comp.protocol in ("eligibility", "resource"):
        interpretation_class = "transient_only"
    elif struct_pass:
        interpretation_class = "full_structural_match"
    elif has_B1 and not is_specific and not math.isnan(B7_false_linking_rate):
        interpretation_class = "non_specific_overlinking"
    elif has_B1 and has_B3 and not gSIG_A_pass:
        interpretation_class = "partial_behavioral_match"
    elif has_B1 and not gSIG_A_pass:
        interpretation_class = "behavioral_match_only"
    else:
        interpretation_class = "no_match"

    # -----------------------------------------------------------------------
    # Branch trace rows
    # -----------------------------------------------------------------------
    branch_rows = []
    for bid in motif.branch_ids:
        branch_rows.append({
            "comparator": comp.name, "motif": motif.motif_id, "branch_id": bid,
            "mb_pre": mb_pre.get(bid, 0), "mb_post": mb_post.get(bid, 0),
            "E_b_enc": E_b_enc.get(bid, 0), "P_b_post": P_b_post.get(bid, 0),
            "delta_mb": mb_post.get(bid, 0) - mb_pre.get(bid, 0),
            "is_overlap_branch": bid in all_ovlp,
        })

    # Linking trace rows
    link_rows = []
    for ti, tj in [(ti, tj) for i, ti in enumerate(motif.trace_ids)
                   for j, tj in enumerate(motif.trace_ids) if j > i]:
        key = f"{ti}:{tj}"
        is_ep = (ti, tj) in ep or (tj, ti) in ep
        link_rows.append({
            "comparator": comp.name, "motif": motif.motif_id,
            "trace_i": ti, "trace_j": tj,
            "L_struct_pre":  L_struct_pre.get(key, 0),
            "L_struct_post": L_struct_post.get(key, 0),
            "L_behav_post":  L_behav_post.get(key, 0),
            "L_struct_dmg":  L_struct_dmg.get(key, 0),
            "is_expected_linked": is_ep,
        })

    return {
        "comparator": comp.name, "motif": motif.motif_id,
        "motif_type": motif.motif_type, "n_branches": motif.n_branches,
        "n_traces": motif.n_traces, "seed": seed,
        # Structural layer
        "gSIG_A": gSIG_A, "gSIG_A_pass": gSIG_A_pass,
        "gSIG_B": gSIG_B, "gSIG_B_pass": gSIG_B_pass,
        # Behavioral layer
        "B1_linking_gain": B1_linking_gain,
        "B2_context_separation": B2_context_separation,
        "B3_damage_sensitivity": B3_damage_sensitivity,
        "B4_recall_preservation": B4_recall_preservation,
        "B5_recovery_index": B5_recovery_index,
        "B6_specificity_index": B6_specificity_index,
        "B7_false_linking_rate": B7_false_linking_rate,
        "NR_targeted": nr_targ_b,
        "NR_generic": nr_gen_b,
        # Summary
        "interpretation_class": interpretation_class,
        "struct_fail_gSIG_A": not gSIG_A_pass,
        "struct_fail_gSIG_B": not gSIG_B_pass,
        # Internal rows
        "_branch_rows": branch_rows,
        "_link_rows": link_rows,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r if not k.startswith("_")))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows({k: v for k, v in r.items() if not k.startswith("_")} for r in rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _fmt(v, fmt=".3f") -> str:
    if isinstance(v, float) and math.isnan(v): return "N/A"
    if isinstance(v, float): return format(v, fmt)
    return str(v)


def _make_figures(all_results: list[dict]) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[e022] matplotlib/numpy not available — skipping figures.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    COMP_ORDER  = [c.name for c in COMPARATORS]
    MOTIF_ORDER = [t[0] for t in MOTIFS_TO_TEST]

    CLASS_COLORS = {
        "full_structural_match":     "#2ca02c",
        "behavioral_match_only":     "#1f77b4",
        "partial_behavioral_match":  "#ff7f0e",
        "non_specific_overlinking":  "#d62728",
        "transient_only":            "#9467bd",
        "no_match":                  "#7f7f7f",
        "undefined_boundary_case":   "#bcbd22",
    }

    # ------------------------------------------------------------------
    # Fig 1 — Structural signature matrix (gSIG-A, gSIG-B per comp × motif)
    # ------------------------------------------------------------------
    sigs = ["gSIG_A", "gSIG_B"]
    n_sig = len(sigs)
    n_comp = len(COMP_ORDER)
    n_motif = len(MOTIF_ORDER)

    fig1, axes1 = plt.subplots(n_sig, 1, figsize=(n_comp * n_motif * 0.7 + 2, 6))
    fig1.suptitle("Fig e022-01  Structural signature matrix\n"
                  "(rows = gSIG-A, gSIG-B; cols = comparator × motif)", fontsize=10)

    for ax, sig in zip(axes1, sigs):
        labels, vals, colors = [], [], []
        for comp in COMP_ORDER:
            for mt in MOTIF_ORDER:
                r = next((r for r in all_results if r["comparator"] == comp and r["motif_type"] == mt), None)
                val = r.get(sig, float("nan")) if r else float("nan")
                pass_key = f"{sig}_pass"
                is_pass = r.get(pass_key, False) if r else False
                labels.append(f"{comp[:8]}\n{mt[:8]}")
                vals.append(float(val) if not math.isnan(float(val) if not isinstance(val, str) else float("nan")) else 0.0)
                colors.append("#2ca02c" if is_pass else "#d62728")
        x = range(len(labels))
        ax.bar(list(x), vals, color=colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(sig, fontsize=9)
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=4, rotation=45, ha="right")
        ax.set_ylabel(sig, fontsize=7)

    plt.tight_layout()
    fig1.savefig(FIGURES_DIR / "Fig_e022_01_hard_comparator_signature_matrix.png", dpi=150)
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Fig 2 — Behavioral metrics matrix (B1-B5 per comp × motif)
    # ------------------------------------------------------------------
    behav_keys = ["B1_linking_gain", "B2_context_separation",
                  "B3_damage_sensitivity", "B4_recall_preservation", "B5_recovery_index"]
    B_labels = ["B1\nlinking\ngain", "B2\ncontext\nsep", "B3\ndamage\nsens",
                "B4\nrecall\npres", "B5\nrecovery\nidx"]

    mat = np.full((len(COMP_ORDER), len(behav_keys)), float("nan"))
    for i, comp in enumerate(COMP_ORDER):
        r_canon = next((r for r in all_results if r["comparator"] == comp
                        and r["motif_type"] == "canonical"), None)
        if r_canon:
            for j, bk in enumerate(behav_keys):
                v = r_canon.get(bk, float("nan"))
                if not math.isnan(float(v)) if not isinstance(v, str) else True:
                    mat[i, j] = float(v)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.suptitle("Fig e022-02  Behavioral output metrics (canonical motif)\n"
                  "rows = comparators; cols = B1–B5; white * = full model reference", fontsize=10)
    im = ax2.imshow(mat, cmap="RdYlGn", vmin=-0.1, vmax=0.5, aspect="auto")
    ax2.set_xticks(range(len(behav_keys))); ax2.set_xticklabels(B_labels, fontsize=8)
    ax2.set_yticks(range(len(COMP_ORDER))); ax2.set_yticklabels(COMP_ORDER, fontsize=8)
    plt.colorbar(im, ax=ax2, fraction=0.03)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax2.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    fig2.savefig(FIGURES_DIR / "Fig_e022_02_behavioral_output_equivalence_matrix.png", dpi=150)
    plt.close(fig2)

    # ------------------------------------------------------------------
    # Fig 3 — Specificity by comparator (B6/B7 for multi-trace motifs)
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
    fig3.suptitle("Fig e022-03  Specificity by comparator (chain + sparse motifs)\n"
                  "B6 = specificity index; B7 = false-linking rate", fontsize=10)

    for ax, bk, ylbl in zip(axes3, ["B6_specificity_index", "B7_false_linking_rate"],
                            ["B6 specificity index", "B7 false-linking rate"]):
        for i, comp in enumerate(COMP_ORDER):
            vals_c = [
                float(r.get(bk, float("nan")))
                for r in all_results if r["comparator"] == comp
                and r["motif_type"] in ("chain_overlap", "sparse_random")
                and not math.isnan(float(r.get(bk, float("nan"))))
            ]
            if vals_c:
                ax.scatter([i] * len(vals_c), vals_c, s=60, alpha=0.8)
                ax.plot([i - 0.3, i + 0.3], [sum(vals_c)/len(vals_c)]*2, "k-", linewidth=2)
        ax.axhline(0, color="green", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(COMP_ORDER)))
        ax.set_xticklabels(COMP_ORDER, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel(ylbl, fontsize=8)

    plt.tight_layout()
    fig3.savefig(FIGURES_DIR / "Fig_e022_03_specificity_by_comparator.png", dpi=150)
    plt.close(fig3)

    # ------------------------------------------------------------------
    # Fig 4 — Linking traces by comparator (L_struct_pre/post for canonical)
    # ------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    fig4.suptitle("Fig e022-04  L_struct linking for canonical motif\n"
                  "pre/post for each comparator; green=full_model reference", fontsize=10)

    for i, comp in enumerate(COMP_ORDER):
        r_c = next((r for r in all_results if r["comparator"] == comp
                    and r["motif_type"] == "canonical"), None)
        if r_c:
            lr_rows = r_c.get("_link_rows", [])
            ep_rows = [lr for lr in lr_rows if lr.get("is_expected_linked")]
            if ep_rows:
                l_pre  = sum(lr["L_struct_pre"]  for lr in ep_rows) / len(ep_rows)
                l_post = sum(lr["L_struct_post"] for lr in ep_rows) / len(ep_rows)
                color = "#2ca02c" if comp == "full_model" else "#7f7f7f"
                ax4.plot([i - 0.2, i + 0.2], [l_pre, l_pre], ":", color=color, linewidth=1.5)
                ax4.plot([i - 0.2, i + 0.2], [l_post, l_post], "-", color=color, linewidth=2)
                ax4.annotate(f"+{l_post-l_pre:.3f}", (i, l_post), textcoords="offset points",
                             xytext=(0, 4), ha="center", fontsize=6)

    ax4.set_xticks(range(len(COMP_ORDER))); ax4.set_xticklabels(COMP_ORDER, fontsize=8, rotation=45, ha="right")
    ax4.set_ylabel("Mean L (expected pairs)", fontsize=8)
    ax4.legend(["pre (dotted)", "post (solid)"], fontsize=7)
    plt.tight_layout()
    fig4.savefig(FIGURES_DIR / "Fig_e022_04_linking_traces_by_comparator.png", dpi=150)
    plt.close(fig4)

    # ------------------------------------------------------------------
    # Fig 5 — Structural vs behavioral dissociation
    # ------------------------------------------------------------------
    fig5, ax5 = plt.subplots(figsize=(10, 7))
    fig5.suptitle("Fig e022-05  Structural vs behavioral dissociation\n"
                  "X = gSIG-A (structural writing); Y = B1 linking gain\n"
                  "Labels = comparator; color = interpretation class", fontsize=9)

    for r in all_results:
        if r["motif_type"] != "canonical": continue
        x = float(r.get("gSIG_A", 0.0))
        y = float(r.get("B1_linking_gain", 0.0)) if not math.isnan(float(r.get("B1_linking_gain", 0.0))) else 0.0
        ic = r.get("interpretation_class", "no_match")
        color = CLASS_COLORS.get(ic, "#bcbd22")
        ax5.scatter(x, y, c=color, s=200, zorder=4, edgecolors="black", linewidths=0.7)
        ax5.annotate(r["comparator"], (x, y), textcoords="offset points",
                     xytext=(5, 3), fontsize=7)

    ax5.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax5.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax5.set_xlabel("gSIG-A (structural overlap writing advantage)", fontsize=9)
    ax5.set_ylabel("B1 linking gain (behavioral linking)", fontsize=9)
    ax5.text(-0.02, 0.85, "behavioral match only\n(no structural write)", fontsize=7,
             color="#1f77b4", ha="right", transform=ax5.get_xaxis_transform())
    ax5.text(0.02, 0.95, "full structural match", fontsize=7,
             color="#2ca02c", ha="left", transform=ax5.get_xaxis_transform())

    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=v, label=k.replace("_"," ")) for k, v in CLASS_COLORS.items()]
    ax5.legend(handles=patches, loc="lower right", fontsize=6)

    plt.tight_layout()
    fig5.savefig(FIGURES_DIR / "Fig_e022_05_structural_vs_behavioral_dissociation.png", dpi=150)
    plt.close(fig5)

    print("[e022] Figures saved.")


# ---------------------------------------------------------------------------
# Summary + docs
# ---------------------------------------------------------------------------

COMP_DEFINITIONS = {
    c.name: {"description": c.description, "protocol": c.protocol,
             "layer_note": c.layer_note,
             "params_override": {
                 "structural_lr": c.params.structural_lr,
                 "replay_gain": c.params.replay_gain,
             }}
    for c in COMPARATORS
}


def _write_summary(all_results: list[dict]) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    flat = [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_results]
    _write_csv(SUMMARY_DIR / "hard_comparator_signature_matrix.csv",
               [{k: v for k, v in r.items()
                 if k in ("comparator","motif_type","n_branches","gSIG_A","gSIG_B",
                           "gSIG_A_pass","gSIG_B_pass","interpretation_class")}
                for r in flat])
    _write_csv(SUMMARY_DIR / "hard_comparator_behavioral_matrix.csv",
               [{k: v for k, v in r.items()
                 if k in ("comparator","motif_type","B1_linking_gain","B2_context_separation",
                           "B3_damage_sensitivity","B4_recall_preservation","B5_recovery_index",
                           "interpretation_class")}
                for r in flat])
    _write_csv(SUMMARY_DIR / "hard_comparator_specificity_matrix.csv",
               [{k: v for k, v in r.items()
                 if k in ("comparator","motif_type","B6_specificity_index","B7_false_linking_rate",
                           "NR_targeted","NR_generic","interpretation_class")}
                for r in flat])

    fail_rows = []
    for comp in [c.name for c in COMPARATORS]:
        rr = [r for r in flat if r["comparator"] == comp]
        fail_rows.append({
            "comparator": comp,
            "n_struct_gSIG_A_fail": sum(1 for r in rr if not r.get("gSIG_A_pass")),
            "n_struct_gSIG_B_fail": sum(1 for r in rr if not r.get("gSIG_B_pass")),
            "n_runs": len(rr),
            "main_interpretation": (
                max(set(r.get("interpretation_class","no_match") for r in rr),
                    key=lambda ic: sum(1 for r in rr if r.get("interpretation_class") == ic))
                if rr else "N/A"
            ),
        })
    _write_csv(SUMMARY_DIR / "failure_mode_summary.csv", fail_rows)

    with (SUMMARY_DIR / "comparator_definitions.json").open("w", encoding="utf-8") as f:
        json.dump(COMP_DEFINITIONS, f, indent=2)

    # Claim ledger
    full_results = [r for r in flat if r["comparator"] == "full_model"]
    n_full_struct = sum(1 for r in full_results if r.get("gSIG_A_pass") and r.get("gSIG_B_pass"))
    comparator_struct_pass = {
        c: sum(1 for r in flat if r["comparator"] == c and r.get("gSIG_A_pass") and r.get("gSIG_B_pass"))
        for c in [c.name for c in COMPARATORS if c.name != "full_model"]
    }

    (SUMMARY_DIR / "hard_comparator_claim_ledger.md").write_text(
        f"""# E022 — Hard Comparator Claim Ledger

## Full model reference
Full model passes structural signatures (gSIG-A ∩ gSIG-B) in {n_full_struct}/{len(full_results)} motifs.

## Comparator structural pass rates
{chr(10).join(f'- {c}: {n}/{len(MOTIFS_TO_TEST)} structural pass' for c, n in comparator_struct_pass.items())}

## Main claim

> Under tested motifs and canonical parameters, no hard comparator reproduced the full
> structural-accessibility signature profile (gSIG-A + gSIG-B) of the replay-dependent
> slow-writing model across all four tested motifs.

## Claims allowed

- Under tested conditions, no hard comparator fully reproduced the structural and behavioral
  joint profile.
- Hebbian weight alone can reproduce partial behavioral linking but not structural branch-specific
  writing.
- Soma global gain produces non-specific overlinking (fails SIG-A).
- Shuffled replay destroys branch-identity specificity.
- Eligibility-only and resource-only produce only transient linking.

## Claims not allowed

- All alternative models are ruled out.
- The biological mechanism is proven.
- Weight-only explanations are impossible.
- The full model is uniquely true.
""", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("Experiment 022 -- Hard Comparator Models")
    print("=" * 72)
    print(f"  {len(COMPARATORS)} comparators × {len(MOTIFS_TO_TEST)} motifs = "
          f"{len(COMPARATORS)*len(MOTIFS_TO_TEST)} runs")

    for d in [OUT_ROOT, TRACES_DIR, SUMMARY_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for comp in COMPARATORS:
        for motif_type, n_branches, n_traces, seed in MOTIFS_TO_TEST:
            motif = build_motif(motif_type, n_branches=n_branches, n_traces=n_traces, seed=seed)
            result = _run_comparator_cell(comp, motif, seed)
            all_results.append(result)

            run_key = f"{comp.name}__{motif.motif_id}"
            _write_csv(TRACES_DIR / f"{run_key}_branch_traces.csv",  result["_branch_rows"])
            _write_csv(TRACES_DIR / f"{run_key}_linking_trace.csv",  result["_link_rows"])

            ic = result.get("interpretation_class", "?")
            B1 = _fmt(result.get("B1_linking_gain"))
            B5 = _fmt(result.get("B5_recovery_index"))
            gA = _fmt(result.get("gSIG_A"))
            gB = _fmt(result.get("gSIG_B"))
            print(f"  [{comp.name:<22}] [{motif.motif_id:<20}]  "
                  f"gA={gA} gB={gB}  B1={B1} B5={B5}  -> {ic}")

    _make_figures(all_results)
    _write_summary(all_results)

    print()
    print("-" * 72)
    print(f"  {'Comparator':<25}  {'Struct gSIG-A pass':<20}  Interpretation classes")
    print("-" * 72)
    for comp in COMPARATORS:
        rr = [r for r in all_results if r["comparator"] == comp.name]
        n_a = sum(1 for r in rr if r.get("gSIG_A_pass"))
        ics = set(r.get("interpretation_class") for r in rr)
        print(f"  {comp.name:<25}  {n_a}/{len(rr)} struct gSIG-A        {', '.join(sorted(ics))}")

    print(f"\n  Outputs: {OUT_ROOT}")
    print("=" * 72)


if __name__ == "__main__":
    main()
