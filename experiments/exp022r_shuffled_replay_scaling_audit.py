"""Experiment 022R — Shuffled Replay Scaling Audit.

Resolves the E022 open question: does shuffled_replay's apparent partial
structural-specificity match (gSIG-A ≈ 0.091 for 4-branch canonical) decay as
branch count increases?

Design
------
- Comparator : shuffled_replay  (random allocation per pass + E_b shuffle)
- Reference  : full_model       (canonical E019R protocol, deterministic)
- Motif types: canonical, strong_overlap
- Branch counts: 4, 8, 16, 32  (optional 64)
- Seeds      : 20 shuffled-replay seeds per (motif_type, n_branches) condition

Primary endpoint
----------------
ratio_to_full_model = mean(gSIG_A_shuffled) / gSIG_A_full
Does this ratio decrease monotonically with n_branches?

Secondary endpoint
------------------
Does shuffled_replay ever pass the full structural-accessibility profile at n >= 8?

Outputs
-------
results/e022r_shuffled_replay_scaling_audit/
    summary/shuffled_replay_scaling.csv
    summary/shuffled_replay_vs_full_model.csv
    summary/claim_ledger.md
    figures/Fig_e022r_01_gsig_a_vs_branch_count.png
    figures/Fig_e022r_02_ratio_to_full_model.png
    figures/Fig_e022r_03_seed_spread.png
"""
from __future__ import annotations

import csv
import math
import random
import statistics
import sys
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cytodend_accessmodel.contracts import (
    ConsolidationWindow, DynamicsParameters, EngramTrace, TraceAllocation,
)
from cytodend_accessmodel.simulator import CytodendAccessModelSimulator
from cytodend_accessmodel.motifs import (
    MotifSpec, build_motif, alloc_to_cue, linking_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[1]
OUT_ROOT    = REPO_ROOT / "results" / "e022r_shuffled_replay_scaling_audit"
SUMMARY_DIR = OUT_ROOT / "summary"
FIGURES_DIR = OUT_ROOT / "figures"

# ---------------------------------------------------------------------------
# Canonical parameters (E019R lock)
# ---------------------------------------------------------------------------
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
# Sweep configuration
# ---------------------------------------------------------------------------
MOTIF_TYPES   = ["canonical", "strong_overlap"]
BRANCH_COUNTS = [4, 8, 16, 32]
N_TRACES      = 2          # 2-trace motifs throughout
N_SEEDS       = 20         # shuffled-replay seeds
BASE_SEED     = 42         # full-model reference seed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mb(sim: CytodendAccessModelSimulator) -> dict[str, float]:
    return {b: st.structural.accessibility for b, st in sim.branches.items()}


def _L_all(weights: dict[str, float], motif: MotifSpec) -> dict[str, float]:
    result: dict[str, float] = {}
    for i, ti in enumerate(motif.trace_ids):
        for j in range(i + 1, len(motif.trace_ids)):
            tj = motif.trace_ids[j]
            result[f"{ti}:{tj}"] = linking_score(weights, motif.allocations[ti], motif.allocations[tj])
    return result


def _mean_mb_delta(blist: list[str], mb_pre: dict, mb_post: dict) -> float:
    if not blist:
        return 0.0
    return sum(mb_post.get(b, 0) - mb_pre.get(b, 0) for b in blist) / len(blist)


def _shuffle_E_b(sim: CytodendAccessModelSimulator, rng: random.Random) -> None:
    bids = list(sim.branches.keys())
    vals = [sim.branches[b].eligibility.value for b in bids]
    rng.shuffle(vals)
    for bid, v in zip(bids, vals):
        sim.branches[bid].eligibility.value = v


# ---------------------------------------------------------------------------
# Build simulator with motif allocations
# ---------------------------------------------------------------------------

def _build_sim(motif: MotifSpec, params: DynamicsParameters) -> CytodendAccessModelSimulator:
    sim = CytodendAccessModelSimulator.from_branch_ids(motif.branch_ids, parameters=params)
    for tid in motif.trace_ids:
        ta = TraceAllocation(trace_id=tid, branch_weights=motif.allocations[tid])
        sim.traces[tid] = EngramTrace(trace_id=tid, allocation=ta)
    return sim


# ---------------------------------------------------------------------------
# Standard (full-model) consolidation
# ---------------------------------------------------------------------------

def _standard_consolidation(sim: CytodendAccessModelSimulator,
                             motif: MotifSpec, n_passes: int) -> None:
    win = ConsolidationWindow(replay_trace_ids=list(motif.trace_ids), modulatory_drive=1.0)
    for _ in range(n_passes):
        sim.run_consolidation(win)


# ---------------------------------------------------------------------------
# Shuffled consolidation (random alloc per pass + E_b shuffle)
# ---------------------------------------------------------------------------

def _shuffled_consolidation(sim: CytodendAccessModelSimulator,
                             motif: MotifSpec, n_passes: int, rng: random.Random) -> None:
    orig_allocs = {t: deepcopy(sim.traces[t].allocation.branch_weights) for t in sim.traces}
    win = ConsolidationWindow(replay_trace_ids=list(motif.trace_ids), modulatory_drive=1.0)
    branch_ids = motif.branch_ids
    for _ in range(n_passes):
        _shuffle_E_b(sim, rng)
        for t in sim.traces:
            raw = {b: rng.random() for b in branch_ids}
            total = sum(raw.values()) or 1.0
            sim.traces[t].allocation.branch_weights = {b: w / total for b, w in raw.items()}
        sim.run_consolidation(win)
    for t in sim.traces:
        sim.traces[t].allocation.branch_weights = orig_allocs[t]


# ---------------------------------------------------------------------------
# Damage + rescue protocol (shared; used for gSIG_E / B5)
# ---------------------------------------------------------------------------

def _damage_and_rescue(sim: CytodendAccessModelSimulator,
                       motif: MotifSpec) -> tuple[dict, dict, dict]:
    """Returns (mb_dmg, mb_targ, mb_gen)."""
    sim_dmg = deepcopy(sim)
    for b in motif.damage_target_branches:
        if b in sim_dmg.branches:
            sim_dmg.branches[b].structural.decay_rate = DAMAGE_DECAY_RATE
    null_win = ConsolidationWindow(replay_trace_ids=[], modulatory_drive=0.0)
    for _ in range(DAMAGE_NULL_PASSES):
        sim_dmg.run_consolidation(null_win)
    mb_dmg = _mb(sim_dmg)

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

    sim_gen = deepcopy(sim_dmg)
    win_g = ConsolidationWindow(replay_trace_ids=list(motif.trace_ids), modulatory_drive=1.0)
    for _ in range(RESCUE_ROUNDS * RESCUE_PASSES):
        sim_gen.run_consolidation(win_g)
    mb_gen = _mb(sim_gen)

    return mb_dmg, mb_targ, mb_gen


# ---------------------------------------------------------------------------
# Single-cell run
# ---------------------------------------------------------------------------

def _run_cell(motif: MotifSpec, params: DynamicsParameters,
              seed: int, is_shuffled: bool) -> dict[str, Any]:
    rng = random.Random(seed)
    sim = _build_sim(motif, params)

    # Encode (2 cue reps per trace)
    for tid in motif.trace_ids:
        cue = alloc_to_cue(motif.allocations[tid])
        for _ in range(2):
            sim.apply_cue(cue)

    mb_pre = _mb(sim)

    # Consolidation
    if is_shuffled:
        _shuffled_consolidation(sim, motif, CONSOLIDATION_PASSES, rng)
    else:
        _standard_consolidation(sim, motif, CONSOLIDATION_PASSES)

    mb_post = _mb(sim)

    # gSIG-A: overlap branch write advantage
    all_ovlp = {b for bl in motif.overlap_branches_per_pair.values() for b in bl}
    non_ovlp = [b for b in motif.branch_ids if b not in all_ovlp]
    ovlp_list = list(all_ovlp)
    gSIG_A = _mean_mb_delta(ovlp_list, mb_pre, mb_post) - _mean_mb_delta(non_ovlp, mb_pre, mb_post)

    # gSIG-B: expected-pair linking gain vs unlinked pairs (N/A for 2-trace)
    ep = motif.expected_linked_pairs
    up = motif.expected_unlinked_pairs
    L_pre  = _L_all(mb_pre, motif)
    L_post = _L_all(mb_post, motif)

    def _mean_L_delta(pairs):
        if not pairs: return float("nan")
        return sum(L_post.get(f"{ti}:{tj}", 0) - L_pre.get(f"{ti}:{tj}", 0)
                   for ti, tj in pairs) / len(pairs)

    dL_ep = _mean_L_delta(ep)
    dL_up = _mean_L_delta(up)
    gSIG_B = (dL_ep - dL_up) if (ep and up) else float("nan")

    # gSIG-E (B5): damage + rescue recovery index
    eps = 1e-8
    mb_dmg, mb_targ, mb_gen = _damage_and_rescue(sim, motif)
    L_dmg  = _L_all(mb_dmg,  motif)
    L_targ = _L_all(mb_targ, motif)
    L_gen  = _L_all(mb_gen,  motif)

    def _nr(L_r, pairs):
        if not pairs: return float("nan")
        nrs = []
        for ti, tj in pairs:
            key = f"{ti}:{tj}"
            l_p = L_post.get(key, 0)
            l_d = L_dmg.get(key, 0)
            l_r = L_r.get(key, 0)
            denom = l_p - l_d
            nrs.append((l_r - l_d) / denom if abs(denom) > eps else 0.0)
        return sum(nrs) / len(nrs)

    NR_targ = _nr(L_targ, ep)
    NR_gen  = _nr(L_gen, ep)
    gSIG_E  = (NR_targ - NR_gen) if (not math.isnan(NR_targ) and not math.isnan(NR_gen)) else float("nan")

    # Behavioral linking gain (B1)
    B1 = float("nan") if not ep else dL_ep

    return {
        "is_shuffled": is_shuffled, "seed": seed,
        "motif_type": motif.motif_type, "n_branches": motif.n_branches,
        "n_traces": motif.n_traces, "n_overlap_branches": len(ovlp_list),
        "gSIG_A": gSIG_A, "gSIG_B": gSIG_B, "gSIG_E": gSIG_E,
        "B1_linking_gain": B1, "NR_targeted": NR_targ, "NR_generic": NR_gen,
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
        w.writerows(rows)


def _fmt(v) -> str:
    if isinstance(v, float) and math.isnan(v): return "N/A"
    if isinstance(v, float): return f"{v:.4f}"
    return str(v)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _make_figures(comparison_rows: list[dict], raw_rows: list[dict]) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[e022r] matplotlib not available — skipping figures.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    motif_types = MOTIF_TYPES
    branch_counts = sorted({int(r["n_branches"]) for r in comparison_rows})
    colors = {"canonical": "#1f77b4", "strong_overlap": "#ff7f0e"}
    markers_full = {"canonical": "s", "strong_overlap": "^"}
    markers_shfl = {"canonical": "o", "strong_overlap": "D"}

    # ------------------------------------------------------------------
    # Fig 01 — gSIG-A vs branch count
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    fig1.suptitle("Fig e022r-01  gSIG-A vs branch count\n"
                  "Full model (lines) and shuffled replay mean ± SD (bands)", fontsize=10)

    for mt in motif_types:
        rows_c = [r for r in comparison_rows if r["motif_type"] == mt]
        bc    = [int(r["n_branches"]) for r in rows_c]
        full  = [float(r["full_gSIG_A"]) for r in rows_c]
        s_mn  = [float(r["shuffled_mean"]) for r in rows_c]
        s_sd  = [float(r["shuffled_sd"])  for r in rows_c]

        ax1.plot(bc, full, "--", marker=markers_full[mt], label=f"full_model {mt}",
                 color=colors[mt], linewidth=1.5, alpha=0.9)
        ax1.plot(bc, s_mn, "-",   marker=markers_shfl[mt], label=f"shuffled {mt}",
                 color=colors[mt], linewidth=2, alpha=0.9)
        ax1.fill_between(bc,
                         [m - s for m, s in zip(s_mn, s_sd)],
                         [m + s for m, s in zip(s_mn, s_sd)],
                         alpha=0.20, color=colors[mt])

    ax1.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax1.set_xlabel("Number of branches", fontsize=10)
    ax1.set_ylabel("gSIG-A", fontsize=10)
    ax1.set_xticks(branch_counts)
    ax1.legend(fontsize=7)
    plt.tight_layout()
    fig1.savefig(FIGURES_DIR / "Fig_e022r_01_gsig_a_vs_branch_count.png", dpi=150)
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Fig 02 — Ratio to full model
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    fig2.suptitle("Fig e022r-02  ratio_to_full_model vs branch count\n"
                  "ratio = mean(shuffled gSIG-A) / full-model gSIG-A", fontsize=10)

    for mt in motif_types:
        rows_c = [r for r in comparison_rows if r["motif_type"] == mt]
        bc     = [int(r["n_branches"]) for r in rows_c]
        ratio  = []
        for r in rows_c:
            full_gA = float(r["full_gSIG_A"])
            s_mean  = float(r["shuffled_mean"])
            ratio.append(s_mean / full_gA if abs(full_gA) > 1e-8 else float("nan"))
        ax2.plot(bc, ratio, "-o", color=colors[mt], label=mt, linewidth=2)
        for x, y in zip(bc, ratio):
            if not math.isnan(y):
                ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                             xytext=(0, 6), ha="center", fontsize=7)

    ax2.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax2.axhline(1, color="green", linestyle="--", linewidth=0.8, label="ratio = 1.0 (match)")
    ax2.axhline(0.3, color="orange", linestyle="--", linewidth=0.8, label="ratio = 0.3")
    ax2.set_xlabel("Number of branches", fontsize=10)
    ax2.set_ylabel("gSIG-A ratio (shuffled / full)", fontsize=10)
    ax2.set_xticks(branch_counts)
    ax2.set_ylim(-0.5, 1.5)
    ax2.legend(fontsize=7)
    plt.tight_layout()
    fig2.savefig(FIGURES_DIR / "Fig_e022r_02_ratio_to_full_model.png", dpi=150)
    plt.close(fig2)

    # ------------------------------------------------------------------
    # Fig 03 — Seed spread (boxplot per condition)
    # ------------------------------------------------------------------
    n_conds = len(motif_types) * len(branch_counts)
    fig3, ax3 = plt.subplots(figsize=(max(10, n_conds * 1.1), 5))
    fig3.suptitle("Fig e022r-03  gSIG-A seed spread (shuffled replay)\n"
                  "Boxes per motif_type × branch_count; green dashes = full model value", fontsize=9)

    positions, box_data, labels, ref_vals = [], [], [], []
    pos = 0
    for mt in motif_types:
        for nb in branch_counts:
            seeds_vals = [float(r["gSIG_A"]) for r in raw_rows
                          if r["motif_type"] == mt and int(r["n_branches"]) == nb
                          and r["is_shuffled"] == "True" and not math.isnan(float(r["gSIG_A"]))]
            full_ref   = next((float(r["gSIG_A"]) for r in raw_rows
                               if r["motif_type"] == mt and int(r["n_branches"]) == nb
                               and r["is_shuffled"] == "False"), None)
            if seeds_vals:
                positions.append(pos)
                box_data.append(seeds_vals)
                labels.append(f"{mt[:4]}\nn={nb}")
                ref_vals.append((pos, full_ref))
                pos += 1

    if box_data:
        bp = ax3.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                         medianprops={"color": "black", "linewidth": 2})
        for patch, mt in zip(bp["boxes"], [m for m in motif_types for _ in branch_counts]):
            patch.set_facecolor(colors.get(mt, "#aec7e8"))
            patch.set_alpha(0.7)
        for pos, ref in ref_vals:
            if ref is not None:
                ax3.plot([pos - 0.4, pos + 0.4], [ref, ref], "--", color="green",
                         linewidth=2, zorder=5)

    ax3.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax3.set_xticks(list(range(len(labels)))); ax3.set_xticklabels(labels, fontsize=7)
    ax3.set_ylabel("gSIG-A", fontsize=9)
    plt.tight_layout()
    fig3.savefig(FIGURES_DIR / "Fig_e022r_03_seed_spread.png", dpi=150)
    plt.close(fig3)

    print("[e022r] Figures saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("Experiment 022R -- Shuffled Replay Scaling Audit")
    print("=" * 72)
    n_full     = len(MOTIF_TYPES) * len(BRANCH_COUNTS)
    n_shuffled = len(MOTIF_TYPES) * len(BRANCH_COUNTS) * N_SEEDS
    print(f"  {n_full} full-model + {n_shuffled} shuffled-replay runs "
          f"({N_SEEDS} seeds x {len(BRANCH_COUNTS)} branch counts x {len(MOTIF_TYPES)} motifs)")

    for d in [OUT_ROOT, SUMMARY_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict]      = []
    comparison_rows: list[dict] = []

    for mt in MOTIF_TYPES:
        for nb in BRANCH_COUNTS:
            motif = build_motif(mt, n_branches=nb, n_traces=N_TRACES, seed=BASE_SEED)

            # Full model reference (single run)
            full_result = _run_cell(motif, CANONICAL_PARAMS, BASE_SEED, is_shuffled=False)
            raw_rows.append(full_result)

            # Shuffled replay (N_SEEDS seeds)
            shfl_results = []
            for seed in range(N_SEEDS):
                r = _run_cell(motif, CANONICAL_PARAMS, seed=seed + 100, is_shuffled=True)
                raw_rows.append(r)
                shfl_results.append(r)

            gA_vals  = [r["gSIG_A"] for r in shfl_results if not math.isnan(r["gSIG_A"])]
            gE_vals  = [r["gSIG_E"] for r in shfl_results if not math.isnan(r["gSIG_E"])]
            full_gA  = full_result["gSIG_A"]
            ratio    = (statistics.mean(gA_vals) / full_gA
                        if gA_vals and abs(full_gA) > 1e-8 else float("nan"))

            comp_row = {
                "motif_type": mt, "n_branches": nb, "n_traces": N_TRACES,
                "full_gSIG_A": full_gA,
                "full_gSIG_E": full_result["gSIG_E"],
                "shuffled_mean": statistics.mean(gA_vals) if gA_vals else float("nan"),
                "shuffled_sd":   statistics.stdev(gA_vals) if len(gA_vals) > 1 else 0.0,
                "shuffled_min":  min(gA_vals) if gA_vals else float("nan"),
                "shuffled_max":  max(gA_vals) if gA_vals else float("nan"),
                "shuffled_gSIG_E_mean": statistics.mean(gE_vals) if gE_vals else float("nan"),
                "ratio_to_full_model": ratio,
                "n_seeds": len(gA_vals),
            }
            comparison_rows.append(comp_row)

            print(f"  [{mt:<16}] n={nb:2d}  full_gA={_fmt(full_gA)}  "
                  f"shfl_mean={_fmt(comp_row['shuffled_mean'])} "
                  f"shfl_sd={_fmt(comp_row['shuffled_sd'])}  "
                  f"ratio={_fmt(ratio)}")

    _write_csv(SUMMARY_DIR / "shuffled_replay_scaling.csv",   raw_rows)
    _write_csv(SUMMARY_DIR / "shuffled_replay_vs_full_model.csv", comparison_rows)

    _make_figures(comparison_rows, [
        {k: str(v) for k, v in r.items()} for r in raw_rows
    ])

    # ------------------------------------------------------------------
    # Claim ledger
    # ------------------------------------------------------------------
    # Check if ratio decays monotonically
    decay_confirmed: dict[str, bool] = {}
    for mt in MOTIF_TYPES:
        rows_mt = [r for r in comparison_rows if r["motif_type"] == mt]
        ratios  = [float(r["ratio_to_full_model"]) for r in rows_mt if not math.isnan(float(r["ratio_to_full_model"]))]
        decay_confirmed[mt] = all(ratios[i] >= ratios[i+1] for i in range(len(ratios)-1)) if len(ratios) > 1 else False

    any_full_pass_large = any(
        float(r["shuffled_mean"]) >= float(r["full_gSIG_A"]) * 0.9
        for r in comparison_rows
        if int(r["n_branches"]) >= 8 and not math.isnan(float(r["shuffled_mean"]))
    )

    ledger_lines = ["# E022R — Shuffled Replay Scaling Audit: Claim Ledger\n",
                    f"## Branch counts tested: {BRANCH_COUNTS}\n",
                    f"## Seeds per condition: {N_SEEDS}\n\n",
                    "## Monotonic decay of ratio_to_full_model?\n"]
    for mt, ok in decay_confirmed.items():
        ledger_lines.append(f"- {mt}: {'YES (confirmed)' if ok else 'NO (not monotonic)'}\n")

    ledger_lines += [
        "\n## Does shuffled_replay ever pass full joint profile at n >= 8?\n",
        f"- {'YES — treat as serious alternative' if any_full_pass_large else 'NO — sampling artifact confirmed'}\n\n",
    ]

    if all(decay_confirmed.values()) and not any_full_pass_large:
        ledger_lines += [
            "## Allowed claim\n\n",
            "> Shuffled replay can partially mimic overlap-branch writing in the smallest\n",
            "> four-branch motif because random reassignment has a high chance of revisiting\n",
            "> the same branch (probability 1/n_branches per allocation weight).  This\n",
            "> apparent match decays as branch allocation space increases, supporting the\n",
            "> interpretation that identity-preserving replay, not replay alone, is required\n",
            "> for scalable structural specificity.\n",
        ]
    else:
        ledger_lines += [
            "## Required update\n\n",
            "> shuffled_replay shows a persistent structural match at n >= 8 or non-monotonic\n",
            "> decay. Treat shuffled_replay as a serious alternative mechanism and update the\n",
            "> E022 comparator matrix accordingly.\n",
        ]

    (SUMMARY_DIR / "claim_ledger.md").write_text("".join(ledger_lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print(f"  {'Motif':<18}  {'n':<4}  {'full_gA':<10}  "
          f"{'shfl_mean':<11}  {'shfl_sd':<10}  {'ratio':<8}")
    print("-" * 72)
    for r in comparison_rows:
        print(f"  {r['motif_type']:<18}  {r['n_branches']:<4}  "
              f"{_fmt(r['full_gSIG_A']):<10}  "
              f"{_fmt(r['shuffled_mean']):<11}  "
              f"{_fmt(r['shuffled_sd']):<10}  "
              f"{_fmt(r['ratio_to_full_model']):<8}")

    print()
    for mt, ok in decay_confirmed.items():
        print(f"  Monotonic decay ({mt}): {'YES' if ok else 'NO'}")
    print(f"  Full pass at n>=8: {'YES (serious alternative)' if any_full_pass_large else 'NO (artifact confirmed)'}")
    print(f"\n  Outputs: {OUT_ROOT}")
    print("=" * 72)


if __name__ == "__main__":
    main()
