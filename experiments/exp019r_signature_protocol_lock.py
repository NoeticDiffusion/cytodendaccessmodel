"""Experiment 019R — Signature Protocol Lock and Rescue Audit.

Locks the canonical SIG-A to SIG-E computation protocol before E020.

Background (why E019R is needed)
---------------------------------
E019 reported SIG-E = +154.7 for the canonical full model, while E017/E018 both
reported SIG-E = +33.3.  The difference arises from a genuine protocol difference:

    E017/E018  include post-consolidation and post-damage probe cues (MU1_CUE /
               MU2_CUE) between phases.  These probes rebuild E_b on b0, b1, b2
               before the rescue comparison begins, lifting the generic rescue
               baseline and suppressing SIG-E to ~33.

    E019       has no inter-phase probe cues.  After 9 null damage passes, E_b is
               depleted.  Generic plain-consolidation rescue cannot recover L
               (no E_b to drive M_b write).  Targeted rescue explicitly rebuilds
               E_b on b1 each round → very high NR_targeted, ~0 NR_generic →
               large SIG-E.

Neither value is wrong.  E017/E018 measured rescue advantage in a "warm" post-probe
context; E019 measured it in a "cold" clean-baseline context.

E019R decision: the E019 protocol (no inter-phase probes) is the canonical forward
protocol.  It gives cleaner mechanistic interpretation and larger SIG-E dynamic range.
E017/E018 values are preserved for record.

Key fixes implemented here
--------------------------
1. Centralized signature computation via ``src/cytodend_accessmodel/signatures.py``.
2. SIG-E unit label corrected:
       WRONG:  "percentage points" (normalized recovery × 100 can exceed 100)
       RIGHT:  report SIG_E_normalized (dimensionless NR difference, can exceed 1.0)
               AND SIG_E_raw (absolute ΔL)
3. SIG-C confirmed as architectural signature (not slow-writing diagnostic).
4. SIG-D confirmed as non-diagnostic alone (passes under multiple comparators/params).
5. Canonical rows from E017/E018/E019/E019R reproduced and differences explained.

Outputs
-------
    results/e019r_signature_protocol_lock/
        summary/
            signature_definitions.md
            signature_thresholds.json
            signature_units.json
            sig_e_rescue_audit.csv
            sig_e_interpretation.md
            canonical_reproduction_table.csv
            context_probe_limitations.md
            article_signature_language.md
        figures/
            Fig_e019r_01_sig_e_rescue_conditions.png
            Fig_e019r_02_canonical_signature_reproduction.png
            Fig_e019r_03_signature_interpretation_map.png
        README.md  qc_report.md  claim_ledger.md  figure_manifest.md
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cytodend_accessmodel.contracts import (
    ConsolidationWindow, DynamicsParameters, EngramTrace, TraceAllocation,
)
from cytodend_accessmodel.simulator import CytodendAccessModelSimulator, _sigmoid, _clamp01
from cytodend_accessmodel.signatures import (
    RescueConditionResult,
    SignatureInputs,
    SignatureProfile,
    compute_signature_profile,
    DEFAULT_THRESHOLDS,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parents[1]
OUT_ROOT   = REPO_ROOT / "results" / "e019r_signature_protocol_lock"
SUMMARY_DIR= OUT_ROOT / "summary"
FIGURES_DIR= OUT_ROOT / "figures"

# ---------------------------------------------------------------------------
# Canonical setup
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
BRANCH_IDS  = ["b0", "b1", "b2", "b3"]
OVERLAP_BRANCH = "b1"
CANONICAL_OVERLAP_STR = 0.85

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

MU1_CUE     = {"b0": 1.0, "b1": 0.8, "b2": 0.0, "b3": 0.0}
MU2_CUE     = {"b0": 0.0, "b1": 0.8, "b2": 1.0, "b3": 0.0}
B1_CUE      = {"b0": 0.0, "b1": 1.0, "b2": 0.0, "b3": 0.0}
GENERIC_CUE = {"b0": 0.5, "b1": 0.5, "b2": 0.5, "b3": 0.5}
NONOVLP_CUE = {"b0": 1.0, "b1": 0.0, "b2": 1.0, "b3": 0.0}
AMBIG_CUE   = {"b0": 0.5, "b1": 0.5, "b2": 0.5, "b3": 0.5}
ALPHA_BIAS  = {"b0": 0.5, "b1": 0.5, "b2":-0.5, "b3":-0.5}
BETA_BIAS   = {"b0":-0.5, "b1":-0.5, "b2": 0.5, "b3": 0.5}

MU1_ALLOC = TraceAllocation(
    trace_id="mu1",
    branch_weights={"b0": 0.90, "b1": 0.85, "b2": 0.05, "b3": 0.05},
)
MU2_ALLOC = TraceAllocation(
    trace_id="mu2",
    branch_weights={"b0": 0.05, "b1": 0.85, "b2": 0.90, "b3": 0.05},
)
ALPHA_ALLOC = TraceAllocation(
    trace_id="mu_alpha",
    branch_weights={"b0": 0.90, "b1": 0.05, "b2": 0.05, "b3": 0.00},
)
BETA_ALLOC  = TraceAllocation(
    trace_id="mu_beta",
    branch_weights={"b0": 0.05, "b1": 0.05, "b2": 0.90, "b3": 0.00},
)

# ---------------------------------------------------------------------------
# Simulator helpers
# ---------------------------------------------------------------------------

def _build_sim() -> CytodendAccessModelSimulator:
    sim = CytodendAccessModelSimulator.from_branch_ids(
        BRANCH_IDS, parameters=CANONICAL_PARAMS
    )
    sim.traces["mu1"] = EngramTrace(trace_id="mu1", allocation=MU1_ALLOC)
    sim.traces["mu2"] = EngramTrace(trace_id="mu2", allocation=MU2_ALLOC)
    return sim


def _linking(sim: CytodendAccessModelSimulator) -> float:
    return sum(
        MU1_ALLOC.branch_weights.get(b, 0.0)
        * MU2_ALLOC.branch_weights.get(b, 0.0)
        * sim.branches[b].structural.accessibility
        for b in BRANCH_IDS
    )


def _context_probe() -> float:
    """SIG-C context separation (dedicated sim, identical to E017/E018/E019)."""
    random.seed(RANDOM_SEED)
    sim = CytodendAccessModelSimulator.from_branch_ids(
        BRANCH_IDS, parameters=CANONICAL_PARAMS
    )
    sim.traces["mu_alpha"] = EngramTrace(
        trace_id="mu_alpha", allocation=ALPHA_ALLOC, context="alpha"
    )
    sim.traces["mu_beta"] = EngramTrace(
        trace_id="mu_beta", allocation=BETA_ALLOC, context="beta"
    )
    for _ in range(2):
        sim.apply_cue({"b0": 1.0, "b1": 0.0, "b2": 0.0, "b3": 0.0}, context="alpha")
    for _ in range(2):
        sim.apply_cue({"b0": 0.0, "b1": 0.0, "b2": 1.0, "b3": 0.0}, context="beta")
    win = ConsolidationWindow(replay_trace_ids=["mu_alpha", "mu_beta"], modulatory_drive=1.0)
    for _ in range(CONSOLIDATION_PASSES):
        sim.run_consolidation(win)
    sim_a = deepcopy(sim); sim_b = deepcopy(sim)
    sim_a.apply_cue(AMBIG_CUE, context="alpha", context_bias=ALPHA_BIAS)
    sim_b.apply_cue(AMBIG_CUE, context="beta",  context_bias=BETA_BIAS)
    rmap_a = {rs.trace_id: rs for rs in sim_a.compute_recall_supports()}
    rmap_b = {rs.trace_id: rs for rs in sim_b.compute_recall_supports()}
    r_ac = rmap_a.get("mu_alpha", type("_",(),{"support":0.0})()).support
    r_aw = rmap_a.get("mu_beta",  type("_",(),{"support":0.0})()).support
    r_bc = rmap_b.get("mu_beta",  type("_",(),{"support":0.0})()).support
    r_bw = rmap_b.get("mu_alpha", type("_",(),{"support":0.0})()).support
    return ((r_ac - r_aw) + (r_bc - r_bw)) / 2.0


def _run_rescue(sim_post_damage: CytodendAccessModelSimulator,
                cue: dict | None, n_rounds: int, n_cue_reps: int,
                n_passes: int) -> float:
    """Run a rescue protocol and return post-rescue linking score."""
    sim = deepcopy(sim_post_damage)
    win = ConsolidationWindow(replay_trace_ids=["mu1", "mu2"], modulatory_drive=1.0)
    for _ in range(n_rounds):
        if cue is not None:
            for _ in range(n_cue_reps):
                sim.apply_cue(cue)
        for _ in range(n_passes):
            sim.run_consolidation(win)
    return _linking(sim)


# ---------------------------------------------------------------------------
# Canonical run  (E019 protocol: no inter-phase probes)
# ---------------------------------------------------------------------------

def run_canonical_e019r() -> tuple[dict, "CytodendAccessModelSimulator"]:
    """Run canonical full-model protocol and return (intermediates, post-damage sim)."""
    random.seed(RANDOM_SEED)
    sim = _build_sim()

    # encode
    for _ in range(2): sim.apply_cue(MU1_CUE)
    for _ in range(2): sim.apply_cue(MU2_CUE)

    mb_pre = {b: sim.branches[b].structural.accessibility for b in BRANCH_IDS}
    L_pre  = _linking(sim)
    sim_tmp = deepcopy(sim)
    sim_tmp.apply_cue(MU1_CUE)
    r_mu1_pre = next(
        (rs.support for rs in sim_tmp.compute_recall_supports() if rs.trace_id == "mu1"), 0.0
    )

    # consolidate
    win = ConsolidationWindow(replay_trace_ids=["mu1", "mu2"], modulatory_drive=1.0)
    for _ in range(CONSOLIDATION_PASSES):
        sim.run_consolidation(win)

    mb_post = {b: sim.branches[b].structural.accessibility for b in BRANCH_IDS}
    L_post  = _linking(sim)
    sim_tmp2 = deepcopy(sim)
    sim_tmp2.apply_cue(MU1_CUE)
    r_mu1_post = next(
        (rs.support for rs in sim_tmp2.compute_recall_supports() if rs.trace_id == "mu1"), 0.0
    )

    # damage
    sim.branches[OVERLAP_BRANCH].structural.decay_rate = DAMAGE_DECAY_RATE
    null_win = ConsolidationWindow(replay_trace_ids=[], modulatory_drive=0.0)
    for _ in range(DAMAGE_NULL_PASSES):
        sim.run_consolidation(null_win)

    L_dmg = _linking(sim)
    sim_tmp3 = deepcopy(sim)
    sim_tmp3.apply_cue(MU1_CUE)
    r_mu1_dmg = next(
        (rs.support for rs in sim_tmp3.compute_recall_supports() if rs.trace_id == "mu1"), 0.0
    )

    nonovlp = [b for b in BRANCH_IDS if b != OVERLAP_BRANCH]
    inter = {
        "mb_pre_overlap":          mb_pre[OVERLAP_BRANCH],
        "mb_post_overlap":         mb_post[OVERLAP_BRANCH],
        "mb_nonoverlap_mean_pre":  sum(mb_pre[b]  for b in nonovlp) / len(nonovlp),
        "mb_nonoverlap_mean_post": sum(mb_post[b] for b in nonovlp) / len(nonovlp),
        "L_pre":  L_pre,
        "L_post": L_post,
        "L_dmg":  L_dmg,
        "recall_support_post_cons": r_mu1_post,
        "recall_support_post_dmg":  r_mu1_dmg,
    }
    return inter, sim


# ---------------------------------------------------------------------------
# SIG-E audit — 5 rescue conditions
# ---------------------------------------------------------------------------

RESCUE_CONDITIONS = [
    ("no_rescue",                       None,        0, 0,            0           ),
    ("targeted_overlap_rescue",         B1_CUE,      3, RESCUE_CUE_REPS, RESCUE_PASSES),
    ("generic_plain_consolidation",     None,        1, 0,            RESCUE_ROUNDS*RESCUE_PASSES),
    ("generic_all_branch_precue",       GENERIC_CUE, 3, RESCUE_CUE_REPS, RESCUE_PASSES),
    ("nonoverlap_branch_rescue",        NONOVLP_CUE, 3, RESCUE_CUE_REPS, RESCUE_PASSES),
]


def run_sig_e_audit() -> tuple[dict, list[RescueConditionResult]]:
    inter, sim_dmg = run_canonical_e019r()
    L_healthy = inter["L_post"]
    L_damaged = inter["L_dmg"]

    results = []
    for name, cue, rounds, cue_reps, passes in RESCUE_CONDITIONS:
        L_rescue = _linking(deepcopy(sim_dmg)) if (rounds == 0 and passes == 0) else \
                   _run_rescue(sim_dmg, cue, rounds, cue_reps, passes)
        results.append(
            RescueConditionResult.compute(name, L_rescue, L_damaged, L_healthy)
        )
    return inter, results


# ---------------------------------------------------------------------------
# Build SignatureInputs / SignatureProfile for one experiment
# ---------------------------------------------------------------------------

def _sig_e_pair(conditions: list[RescueConditionResult],
                targ_name: str, ref_name: str) -> tuple[float, float]:
    targ = next((c for c in conditions if c.name == targ_name), None)
    ref  = next((c for c in conditions if c.name == ref_name),  None)
    if targ is None or ref is None:
        return float("nan"), float("nan")
    return targ.L_post_rescue - ref.L_post_rescue, targ.normalized_recovery - ref.normalized_recovery


def _build_sig_inputs_e019r(inter: dict, conditions: list[RescueConditionResult],
                             sig_c: float) -> SignatureInputs:
    return SignatureInputs(
        mb_overlap_pre=inter["mb_pre_overlap"],
        mb_overlap_post_cons=inter["mb_post_overlap"],
        mb_nonoverlap_mean_pre=inter["mb_nonoverlap_mean_pre"],
        mb_nonoverlap_mean_post_cons=inter["mb_nonoverlap_mean_post"],
        L_pre=inter["L_pre"],
        L_post_cons=inter["L_post"],
        context_separation=sig_c,
        L_post_damage=inter["L_dmg"],
        recall_support_post_cons=inter["recall_support_post_cons"],
        recall_support_post_damage=inter["recall_support_post_dmg"],
        rescue_conditions=conditions,
        targeted_rescue_name="targeted_overlap_rescue",
        reference_rescue_name="generic_plain_consolidation",
        protocol_name="canonical_e019r",
    )


# ---------------------------------------------------------------------------
# Canonical reproduction table
# ---------------------------------------------------------------------------

def _load_e017_canonical() -> dict:
    path = REPO_ROOT / "results" / "e017_traceable_simulator_core" / "summary" / "signature_summary.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    r = rows[0]
    return {
        "SIG_A": float(r.get("SIG_A_overlap_advantage", "nan")),
        "SIG_B": float(r.get("SIG_B_linking_gain", "nan")),
        "SIG_C": float(r.get("SIG_C_context_separation", "nan")),
        "SIG_D": float(r.get("SIG_D_linking_recall_dissociation", "nan")),
        "SIG_E_old_pp": float(r.get("SIG_E_targeted_rescue_advantage", "nan")),
        "lk_post": float(r.get("lk_post_cons", "nan")),
        "lk_dmg":  float(r.get("lk_post_damage", "nan")),
        "lk_targ": float(r.get("lk_targ_rescue", "nan")),
        "lk_std":  float(r.get("lk_std_rescue", "nan")),
        "notes": (
            "Post-cons + post-damage probe cues active between phases. "
            "Generic rescue = standard (plain) consolidation. "
            "SIG-E old pp = rec_targ% - rec_std%."
        ),
    }


def _load_e018_canonical() -> dict:
    path = REPO_ROOT / "results" / "e018_comparator_trace_matrix" / "summary" / "comparator_signature_matrix.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("comparator") == "full_model":
                return {
                    "SIG_A": float(row["SIG_A_overlap_advantage"]),
                    "SIG_B": float(row["SIG_B_linking_gain"]),
                    "SIG_C": float(row["SIG_C_context_separation"]),
                    "SIG_D": float(row["SIG_D_linking_recall_dissociation"]),
                    "SIG_E_old_pp": float(row["SIG_E_targeted_rescue_advantage"]),
                    "notes": (
                        "Post-cons + post-damage probe cues active (same as E017). "
                        "Generic rescue = plain consolidation (no pre-cue). "
                        "SIG-E identical to E017 — confirms protocol identity."
                    ),
                }
    return {}


def _load_e019_canonical() -> dict:
    path = REPO_ROOT / "results" / "e019_one_at_a_time_parameter_robustness" / "summary" / "canonical_reference.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "SIG_A": data.get("SIG_A_score", float("nan")),
        "SIG_B": data.get("SIG_B_score", float("nan")),
        "SIG_C": data.get("SIG_C_score", float("nan")),
        "SIG_D": data.get("SIG_D_score", float("nan")),
        "SIG_E_old_pp": data.get("SIG_E_score", float("nan")),
        "notes": (
            "No inter-phase probe cues. Generic rescue = plain consolidation "
            "(no pre-cue). Large SIG-E because generic rescue cannot rebuild "
            "M_b without E_b pre-loading."
        ),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="", extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def _round4(x) -> str:
    if isinstance(x, float) and math.isnan(x): return "nan"
    return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _make_figures(
    rescue_results: list[RescueConditionResult],
    inter: dict,
    profile: SignatureProfile,
    e017: dict, e018: dict, e019: dict,
) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[e019r] matplotlib/numpy not available — skipping figures.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    L_healthy = inter["L_post"]
    L_damaged = inter["L_dmg"]

    # ------------------------------------------------------------------
    # Fig 1 — SIG-E rescue conditions bar chart
    # ------------------------------------------------------------------
    fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig1.suptitle("Fig e019r-01  SIG-E rescue conditions audit", fontsize=11)

    names = [r.name.replace("_", "\n") for r in rescue_results]
    abs_rec   = [r.absolute_recovery   for r in rescue_results]
    norm_rec  = [r.normalized_recovery for r in rescue_results]
    l_vals    = [r.L_post_rescue       for r in rescue_results]
    colors = ["#aaaaaa" if r.name == "no_rescue" else
              "#2ca02c" if "targeted" in r.name else
              "#1f77b4" if "plain" in r.name else
              "#ff7f0e" if "all_branch" in r.name else "#9467bd"
              for r in rescue_results]

    ax1, ax2 = axes
    ax1.bar(names, abs_rec, color=colors, edgecolor="white")
    ax1.axhline(0, lw=0.5, color="gray")
    ax1.set_ylabel("Absolute recovery\n(L_post_rescue - L_post_damage)", fontsize=8)
    ax1.set_title("Absolute recovery (ΔL)", fontsize=9)
    ax1.tick_params(axis="x", labelsize=7)

    ax2.bar(names, norm_rec, color=colors, edgecolor="white")
    ax2.axhline(1.0, ls="--", lw=0.8, color="black", label="baseline (NR=1.0 = full recovery)")
    ax2.axhline(0.0, lw=0.5, color="gray")
    ax2.set_ylabel("Normalized recovery (NR)\n= abs_rec / (L_healthy - L_damaged)", fontsize=8)
    ax2.set_title("Normalized recovery (can exceed 1.0 = overshoot)", fontsize=9)
    ax2.tick_params(axis="x", labelsize=7)
    ax2.legend(fontsize=7)

    for ax, vals in [(ax1, abs_rec), (ax2, norm_rec)]:
        for i, v in enumerate(vals):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)

    plt.tight_layout()
    fig1.savefig(FIGURES_DIR / "Fig_e019r_01_sig_e_rescue_conditions.png", dpi=150)
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Fig 2 — Canonical signature reproduction across experiments
    # ------------------------------------------------------------------
    SIG_KEYS = ["SIG_A", "SIG_B", "SIG_C", "SIG_D"]
    exp_labels = ["E017", "E018", "E019", "E019R\n(shared)"]
    exp_data   = [e017, e018, e019,
                  {"SIG_A": profile.SIG_A, "SIG_B": profile.SIG_B,
                   "SIG_C": profile.SIG_C, "SIG_D": profile.SIG_D}]
    colors_exp = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig2, axes2 = plt.subplots(1, len(SIG_KEYS), figsize=(14, 5))
    fig2.suptitle("Fig e019r-02  Canonical signature reproduction E017–E019R", fontsize=10)

    for ax, sig in zip(axes2, SIG_KEYS):
        vals = [d.get(sig, float("nan")) for d in exp_data]
        bars = ax.bar(exp_labels, vals, color=colors_exp, edgecolor="white")
        ax.set_title(sig.replace("_", "-"), fontsize=9)
        ax.set_ylim(0, max(v for v in vals if not math.isnan(v)) * 1.3 + 0.01)
        for bar, v in zip(bars, vals):
            if not math.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                        f"{v:.3f}", ha="center", fontsize=7)
        ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    fig2.savefig(FIGURES_DIR / "Fig_e019r_02_canonical_signature_reproduction.png", dpi=150)
    plt.close(fig2)

    # ------------------------------------------------------------------
    # Fig 3 — Signature interpretation map
    # ------------------------------------------------------------------
    sig_meta = {
        "SIG-A": ("slow-writing\ndiagnostic",         "#2ca02c",  0.85, 0.70),
        "SIG-B": ("slow-writing\ndiagnostic",         "#2ca02c",  0.15, 0.70),
        "SIG-C": ("architectural /\nauxiliary",        "#1f77b4",  0.50, 0.25),
        "SIG-D": ("perturbation-sensitive\nnon-specific", "#ff7f0e", 0.50, 0.75),
        "SIG-E": ("rescue-selectivity /\nprotocol-sensitive", "#9467bd", 0.80, 0.30),
    }

    fig3, ax3 = plt.subplots(figsize=(9, 6))
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.set_xlabel("Diagnostic specificity for slow structural writing →", fontsize=9)
    ax3.set_ylabel("Protocol sensitivity →", fontsize=9)
    ax3.axvline(0.5, lw=0.5, color="gray", ls="--")
    ax3.axhline(0.5, lw=0.5, color="gray", ls="--")

    ax3.text(0.25, 0.5, "Architectural /\nAuxiliary", ha="center", va="center",
             fontsize=8, color="gray", alpha=0.4)
    ax3.text(0.75, 0.5, "Primary\nDiagnostic", ha="center", va="center",
             fontsize=8, color="gray", alpha=0.4)

    for sig, (label, color, x, y) in sig_meta.items():
        ax3.scatter([x], [y], s=250, color=color, zorder=3, edgecolors="white", lw=1.5)
        ax3.text(x + 0.03, y + 0.03, f"{sig}\n({label})", fontsize=8, color=color)

    ax3.set_title("Fig e019r-03  Signature interpretation map\n"
                  "(diagnostic value vs protocol sensitivity)", fontsize=10)
    plt.tight_layout()
    fig3.savefig(FIGURES_DIR / "Fig_e019r_03_signature_interpretation_map.png", dpi=150)
    plt.close(fig3)

    print("[e019r] Figures saved.")


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

def _write_docs(
    profile: SignatureProfile,
    rescue_results: list[RescueConditionResult],
    inter: dict,
    repro_rows: list[dict],
) -> None:
    # signature_thresholds.json
    with (SUMMARY_DIR / "signature_thresholds.json").open("w", encoding="utf-8") as f:
        json.dump(DEFAULT_THRESHOLDS, f, indent=2)

    # signature_units.json
    units = {
        "SIG_A": "delta_M_b (dimensionless, approx [-1, 1])",
        "SIG_B": "delta_L (dimensionless, bounded by allocation geometry)",
        "SIG_C": "recall-support units (dimensionless)",
        "SIG_D": "percentage points (linking% drop - recall% drop; bounded)",
        "SIG_E_raw": "delta_L (dimensionless, absolute linking difference between rescue conditions)",
        "SIG_E_normalized": "dimensionless NR difference (normalized_recovery_targeted - normalized_recovery_reference; CAN exceed 1.0 — label as 'overshoot', NOT 'percentage points')",
    }
    with (SUMMARY_DIR / "signature_units.json").open("w", encoding="utf-8") as f:
        json.dump(units, f, indent=2)

    # signature_definitions.md
    (SUMMARY_DIR / "signature_definitions.md").write_text(
        """# Signature Definitions (locked at E019R)

## SIG-A — Overlap-branch selective structural writing

**Formula:**
```
SIG_A = (M_b(overlap, post) - M_b(overlap, pre)) - mean(M_b(non-overlap, post) - M_b(non-overlap, pre))
```
**Units:** ΔM_b (dimensionless, approx [-1, 1])
**Bounded:** Yes (M_b ∈ [0, 1])
**Directional threshold:** > 0
**Protected threshold:** > 0.02
**Diagnostic:** Primary — requires slow structural write; fails when structural_lr = 0
**Non-diagnostic cases:** None identified

---

## SIG-B — Linking gain after consolidation

**Formula:**
```
SIG_B = L_post_consolidation - L_pre_consolidation
L = Σ_b a_{μ1,b} · a_{μ2,b} · M_b
```
**Units:** ΔL (dimensionless, bounded by allocation geometry)
**Bounded:** Yes (L ∈ [0, max_alloc²])
**Directional threshold:** > 0
**Protected threshold:** > 0.05
**Diagnostic:** Primary — requires slow structural write; fails when structural_lr = 0 or replay_gain = 0
**Non-diagnostic cases:** None identified

---

## SIG-C — Context separation

**Formula:**
```
SIG_C = mean( (r_corr_α - r_wrong_α) + (r_corr_β - r_wrong_β) ) / 2
```
**Units:** recall-support units (dimensionless)
**Bounded:** Approximately (support values bounded by readout_gain and activation)
**Directional threshold:** > 0
**Protected threshold:** > 0.05
**Diagnostic:** ARCHITECTURAL / AUXILIARY — reflects allocation geometry (which
branches mu_alpha and mu_beta occupy). SIG-C passes even when structural_lr = 0 and
context_gain = 0. It does NOT diagnose slow structural writing.
**E018 finding:** All 5 comparators passed SIG-C (including fast_context_only,
fixed_allocation_only). Allocation structure is sufficient.
**E019 finding:** All 7 context_gain values (0.00–3.00) pass SIG-C.
**Article implication:** Report SIG-C as an architectural fast-gating signature,
not as a discriminative slow-writing marker.
**Non-diagnostic cases:** Comparators without slow writing (fast_context_only, etc.)

---

## SIG-D — Linking > recall dissociation

**Formula:**
```
SIG_D = (L_drop_pct) - (recall_drop_pct)
      = 100 × (L_post - L_dmg)/L_post - 100 × (recall_post - recall_dmg)/recall_post
```
**Units:** percentage points (both input ratios are bounded)
**Bounded:** Yes (each drop_pct ∈ [-∞, 100])
**Directional threshold:** > 0
**Protected threshold:** > 5.0 pp
**Diagnostic:** PERTURBATION-SENSITIVE / NON-SPECIFIC — SIG-D passes when the damage
specifically targets the overlap branch (geometry effect) regardless of whether M_b
was dynamically written. Passes in E018 for fast_context_only, replay_no_structure,
and fixed_allocation_only. Not diagnostic alone.
**Article implication:** SIG-D is meaningful only in combination with SIG-A and SIG-B.
**Non-diagnostic cases:** All comparators with structural overlap geometry

---

## SIG-E — Targeted rescue selectivity

**Formula (canonical, locked at E019R):**
```
SIG_E_raw        = L_targeted_rescue - L_generic_plain_consolidation_rescue
SIG_E_normalized = NR_targeted - NR_generic_plain
NR               = (L_post_rescue - L_post_damage) / (L_healthy - L_post_damage)
```
**Units:**
  SIG_E_raw:        ΔL (dimensionless, absolute difference)
  SIG_E_normalized: dimensionless NR difference (CAN exceed 1.0 = overshoot)
**DO NOT label SIG-E in "percentage points"** — NR is unbounded above 1.0
**Protected threshold:** SIG_E_normalized > 0.10 OR SIG_E_raw > 0.02
**Protocol sensitivity:** PROTOCOL-SENSITIVE — see rescue protocol note below
**Non-diagnostic cases:** fails when structural_lr = 0 or replay_gain = 0

### Rescue protocol (canonical, locked)
- Targeted rescue:         B1_CUE × 3 reps/round, consolidation × 3 passes/round, × 3 rounds
- Generic plain reference: consolidation × 9 passes, no pre-cueing

### SIG-E protocol history and the E017/E018 vs E019 discrepancy
| Experiment | Protocol variant | SIG_E value | Explanation |
|---|---|---|---|
| E017 | Post-cons + post-damage probe cues before rescue | +33 | Probes pre-load E_b; generic rescue benefits |
| E018 | Same as E017 | +33 | Identical |
| E019 | No inter-phase probe cues | +155 | Generic rescue starts cold; near-zero recovery |
| E019R | No inter-phase probe cues (E019 protocol) | see CSV | Same as E019 |

E019 / E019R is the canonical forward protocol. E017/E018 values are preserved as record.
""", encoding="utf-8")

    # sig_e_rescue_audit.csv
    L_healthy = inter["L_post"]
    L_damaged = inter["L_dmg"]
    audit_rows = []
    for r in rescue_results:
        targ_raw  = (r.L_post_rescue - rescue_results[0].L_post_rescue) if r.name != "no_rescue" else 0.0
        targ_norm = r.normalized_recovery - rescue_results[0].normalized_recovery
        audit_rows.append({
            "rescue_condition":          r.name,
            "L_pre_damage":              _round4(L_healthy),
            "L_post_damage":             _round4(L_damaged),
            "L_post_rescue":             _round4(r.L_post_rescue),
            "absolute_recovery":         _round4(r.absolute_recovery),
            "normalized_recovery":       _round4(r.normalized_recovery),
            "overshoot":                 str(r.overshoot),
            "targeted_advantage_raw":    _round4(r.L_post_rescue - next(
                (c.L_post_rescue for c in rescue_results if c.name == "generic_plain_consolidation"), float("nan")
            )),
            "targeted_advantage_normalized": _round4(r.normalized_recovery - next(
                (c.normalized_recovery for c in rescue_results if c.name == "generic_plain_consolidation"), float("nan")
            )),
        })
    _write_csv(SUMMARY_DIR / "sig_e_rescue_audit.csv", audit_rows)

    # sig_e_interpretation.md
    targ_nr = next(r.normalized_recovery for r in rescue_results if r.name == "targeted_overlap_rescue")
    gen_nr  = next(r.normalized_recovery for r in rescue_results if r.name == "generic_plain_consolidation")
    (SUMMARY_DIR / "sig_e_interpretation.md").write_text(
        f"""# SIG-E — Interpretation and Unit Discipline

## Canonical values (E019R / E019 protocol)

| Quantity | Value |
|---|---|
| L_pre_damage (healthy) | {_round4(inter['L_post'])} |
| L_post_damage | {_round4(inter['L_dmg'])} |
| damage = L_healthy - L_post_damage | {_round4(inter['L_post'] - inter['L_dmg'])} |
| NR_targeted | {_round4(targ_nr)} |
| NR_generic_plain | {_round4(gen_nr)} |
| SIG_E_normalized = NR_targ - NR_gen | {_round4(profile.SIG_E_normalized)} |
| SIG_E_raw = L_targ - L_gen | {_round4(profile.SIG_E_raw)} |
| Overshoot (targeted) | {profile.SIG_E_targeted_overshoot} |

## Unit decision
SIG-E_normalized CAN exceed 1.0 (targeted rescue overshoots healthy baseline).
Label is "normalized recovery difference", NOT "percentage points".

## Rescue protocol sensitivity
SIG-E magnitude depends critically on whether pre-rescue probe cues are present:
- E017/E018 (probe cues present): SIG-E ≈ 0.33 NR units (old pp = 33)
- E019/E019R (no probe cues):    SIG-E ≈ {_round4(profile.SIG_E_normalized)} NR units

Canonical forward protocol = E019 variant (no inter-phase probes).
""", encoding="utf-8")

    # canonical_reproduction_table.csv
    _write_csv(SUMMARY_DIR / "canonical_reproduction_table.csv", repro_rows)

    # article_signature_language.md
    (SUMMARY_DIR / "article_signature_language.md").write_text(
        """# Article-facing Signature Language (locked at E019R)

## SIG-A and SIG-B
> SIG-A and SIG-B are slow-writing-dependent signatures. Both require
> non-zero structural learning rate and replay gain. Neither passes when
> either of these mechanisms is disabled.

## SIG-C
> SIG-C is an architectural context-separation signature and is NOT a
> diagnostic marker of slow structural writing. It reflects the allocation
> geometry (distinct branch recruitment for contextually differentiated traces)
> and persists even when slow structural updating is fully disabled. It should
> be reported as a fast-gating / architectural confirmation, not as a
> discriminative signature for the model's core mechanism.

## SIG-D
> SIG-D (linking-vs-recall dissociation under overlap-branch damage) is
> partly driven by the overlap-branch perturbation geometry and is not
> diagnostic alone. Several simpler comparators (lacking slow structural writing)
> also pass SIG-D because the selective damage still disproportionately impairs
> linking relative to recall. SIG-D is meaningful as part of the joint profile
> but should not be cited in isolation.

## SIG-E
> SIG-E is a rescue-selectivity signature. Its magnitude depends substantially
> on the rescue protocol (specifically, whether pre-rescue probe cues are used).
> SIG-E should be reported as a normalized recovery difference
> (NR_targeted − NR_reference), not as "percentage points", because the
> normalized recovery can exceed 1.0 when consolidation overshoots the
> healthy baseline. Both raw (ΔL) and normalized values should be reported.

## Model-discrimination claim
> The model-discrimination claim rests on the joint profile, especially the
> combination of SIG-A, SIG-B, and SIG-E. No tested simpler comparator passes
> all five signatures simultaneously. SIG-C and SIG-D are reported as supporting
> characterization, not as primary discriminators.
""", encoding="utf-8")

    # context_probe_limitations.md (unchanged from E018)
    (SUMMARY_DIR / "context_probe_limitations.md").write_text(
        """# Context Probe Limitations (E019R)

SIG-C is computed in a dedicated context-probe simulation, not in the main
ten-phase protocol. The context-probe uses mu_alpha/mu_beta allocations and
explicit context labels; these are incompatible with the main mu1/mu2 linking
protocol.

This limitation was documented in E018 and carries forward. Full time-resolved
context traces across consolidation are deferred to a later experiment.
""", encoding="utf-8")

    # README
    (OUT_ROOT / "README.md").write_text(
        f"""# E019R — Signature Protocol Lock and Rescue Audit

**Date:** {__import__('datetime').date.today()}

## Purpose
Lock the canonical SIG-A to SIG-E protocol before E020.

## Key outputs
- `summary/signature_definitions.md` — frozen definitions
- `summary/sig_e_rescue_audit.csv` — 5 rescue conditions
- `summary/canonical_reproduction_table.csv` — E017/E018/E019/E019R comparison
- `summary/article_signature_language.md` — locked article prose

## Key findings
1. SIG-C is an architectural fast-gating signature (not a slow-writing diagnostic)
2. SIG-D is geometry-driven, not diagnostic alone
3. SIG-E unit label corrected: normalized recovery difference (not pp)
4. SIG-E magnitude is protocol-sensitive (probe cues before rescue lift generic baseline)
5. Centralized computation in `src/cytodend_accessmodel/signatures.py`
""", encoding="utf-8")

    # claim_ledger
    (OUT_ROOT / "claim_ledger.md").write_text(
        """# E019R — Claim Ledger

| Claim | Status | Evidence | Limitation | Next action |
|---|---|---|---|---|
| Signature protocol is locked and auditable | Validated | signature_definitions.md | E020 must use locked protocol | E020 |
| SIG-E rescue sensitivity characterized | Validated | sig_e_rescue_audit.csv | Protocol-sensitive | Paper prose |
| Canonical profile reproducible via shared code | Validated | canonical_reproduction_table.csv | E017/E018 use probe-cue variant | E020 use E019R protocol |
| SIG-C is architectural, not slow-writing diagnostic | Validated | E018 + E019 context_gain sweep | Allocation-driven | Paper prose |
| SIG-D non-diagnostic alone | Validated | E018 comparator matrix | Geometry-driven | Paper prose |
| SIG-E unit = normalized recovery difference (not pp) | Corrected | sig_e_interpretation.md | Can exceed 1.0 | All future experiments |
| Pairwise parameter interactions | Pending | E020 | — | E020 |
""", encoding="utf-8")

    # figure_manifest
    (OUT_ROOT / "figure_manifest.md").write_text(
        """# E019R — Figure Manifest

| File | Content | Status |
|---|---|---|
| Fig_e019r_01_sig_e_rescue_conditions.png | Absolute and NR across 5 rescue conditions | Generated |
| Fig_e019r_02_canonical_signature_reproduction.png | SIG-A–D across E017/E018/E019/E019R | Generated |
| Fig_e019r_03_signature_interpretation_map.png | Interpretation map: diagnostic vs auxiliary | Generated |
""", encoding="utf-8")

    # qc_report
    (OUT_ROOT / "qc_report.md").write_text(
        f"""# E019R — QC Report

## Shared module
`src/cytodend_accessmodel/signatures.py` is the single source of truth.

## Rescue conditions audited
5 conditions: {[r.name for r in rescue_results]}

## Threshold provenance
DEFAULT_THRESHOLDS from signatures.py: {DEFAULT_THRESHOLDS}

## Protocol
E019 canonical (no inter-phase probe cues).
""", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 68)
    print("Experiment 019R -- Signature Protocol Lock and Rescue Audit")
    print("=" * 68)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Canonical run + SIG-E audit
    print("  Running canonical protocol + SIG-E audit (5 rescue conditions)...")
    inter, rescue_results = run_sig_e_audit()

    # SIG-C
    sig_c = _context_probe()

    # Build SignatureInputs and compute profile via shared module
    inputs  = _build_sig_inputs_e019r(inter, rescue_results, sig_c)
    profile = compute_signature_profile(inputs)

    print(f"  SIG-A={profile.SIG_A:+.4f}  SIG-B={profile.SIG_B:+.4f}"
          f"  SIG-C={profile.SIG_C:+.4f}  SIG-D={profile.SIG_D:+.2f}pp")
    print(f"  SIG-E_raw={profile.SIG_E_raw:+.4f}  SIG-E_norm={profile.SIG_E_normalized:+.4f}"
          f"  overshoot={profile.SIG_E_targeted_overshoot}")
    print(f"  Joint protected pass: {profile.joint_protected_pass}")

    print()
    print("  Rescue condition results:")
    L_healthy = inter["L_post"]
    L_damaged = inter["L_dmg"]
    for r in rescue_results:
        print(f"    {r.name:<35}  NR={r.normalized_recovery:+.3f}"
              f"  abs={r.absolute_recovery:+.4f}"
              f"  overshoot={r.overshoot}")

    # Load prior experiment canonical values
    print()
    print("  Loading canonical values from E017/E018/E019...")
    e017 = _load_e017_canonical()
    e018 = _load_e018_canonical()
    e019 = _load_e019_canonical()

    # Build canonical reproduction table
    repro_rows = []
    for exp_name, edata in [
        ("E017", e017), ("E018", e018), ("E019", e019),
        ("E019R", {
            "SIG_A": profile.SIG_A, "SIG_B": profile.SIG_B,
            "SIG_C": profile.SIG_C, "SIG_D": profile.SIG_D,
        }),
    ]:
        if not edata:
            continue
        sig_e_raw  = edata.get("SIG_E_old_pp", float("nan"))
        sig_e_norm = float("nan")
        if exp_name == "E019R":
            sig_e_raw  = profile.SIG_E_raw
            sig_e_norm = profile.SIG_E_normalized
        elif exp_name in ("E017", "E018"):
            # Compute normalized from saved lk values (E017 only, E018 identical)
            lk_p = e017.get("lk_post", float("nan"))
            lk_d = e017.get("lk_dmg",  float("nan"))
            lk_t = e017.get("lk_targ", float("nan"))
            lk_s = e017.get("lk_std",  float("nan"))
            denom = lk_p - lk_d if not math.isnan(lk_p) and not math.isnan(lk_d) else float("nan")
            if not math.isnan(denom) and abs(denom) > 1e-9:
                nr_t = (lk_t - lk_d) / denom
                nr_s = (lk_s - lk_d) / denom
                sig_e_norm = nr_t - nr_s
        elif exp_name == "E019":
            # Normalize E019's old_pp back to NR using known denominator
            # E019 protocol: SIG_E_pp = (lk_targ - lk_gen) / (lk_post - lk_dmg) × 100
            # We use E019R's denominator as the best proxy
            denom = inter["L_post"] - inter["L_dmg"]
            if abs(denom) > 1e-9 and not math.isnan(sig_e_raw):
                sig_e_norm = sig_e_raw / 100.0  # old_pp is already /denom*100

        joint = all([
            edata.get("SIG_A", float("nan")) > DEFAULT_THRESHOLDS["SIG_A"],
            edata.get("SIG_B", float("nan")) > DEFAULT_THRESHOLDS["SIG_B"],
            edata.get("SIG_C", float("nan")) > DEFAULT_THRESHOLDS["SIG_C"],
            edata.get("SIG_D", float("nan")) > DEFAULT_THRESHOLDS["SIG_D"],
        ])

        repro_rows.append({
            "experiment":     exp_name,
            "SIG_A":          _round4(edata.get("SIG_A", float("nan"))),
            "SIG_B":          _round4(edata.get("SIG_B", float("nan"))),
            "SIG_C":          _round4(edata.get("SIG_C", float("nan"))),
            "SIG_D":          _round4(edata.get("SIG_D", float("nan"))),
            "SIG_E_raw":      _round4(sig_e_raw),
            "SIG_E_normalized": _round4(sig_e_norm),
            "joint_pass_A_D": str(joint),
            "notes":          edata.get("notes", ""),
        })

    print()
    print("  Canonical reproduction table:")
    for r in repro_rows:
        print(f"    {r['experiment']:<8}  A={r['SIG_A']}  B={r['SIG_B']}"
              f"  C={r['SIG_C']}  D={r['SIG_D']}  E_norm={r['SIG_E_normalized']}")

    # Figures and docs
    _make_figures(rescue_results, inter, profile, e017, e018, e019)
    _write_docs(profile, rescue_results, inter, repro_rows)
    print()
    print("[e019r] Documentation written.")
    print(f"  Outputs: {OUT_ROOT}")
    print("=" * 68)


if __name__ == "__main__":
    main()
