"""
Generate all figures for the executable paper:
  figure_E1_architecture.svg          -- conceptual block diagram (SVG patches)
  figure_E2_branch_writing.svg        -- branch M_b + linking + recall (3-panel)
  figure_E3_context_recall.svg        -- context disambiguation bar chart
  figure_E4_three_factor.svg          -- timing / replay / modulatory (4-condition)
  figure_E5_ablation.svg              -- structural gate ablation 3-panel
  figure_E6_comparator_heatmap.svg    -- pass/fail signature heatmap (colour)
  figure_E7_motif_family.svg          -- linking gain by motif + hub inset

Run from repo root:
    python experiments/gen_figures_executable.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import random
from copy import deepcopy
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from cytodend_accessmodel.simulator import CytodendAccessModelSimulator, _sigmoid, _clamp01
from cytodend_accessmodel.contracts import (
    ConsolidationWindow,
    DynamicsParameters,
    EngramTrace,
    StructuralState,
    TraceAllocation,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTDIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    ".article",
    "Executable Structural Accessibility - A Biologically Constrained Cytoskeletal-Dendritic Model of Memory Linking",
    "figures",
)
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Palette
C_SLOW      = "#2B5C8A"   # deep blue  – slow structural layer
C_SLOW_POST = "#1A3A5C"   # darker blue – post-consolidation
C_FAST      = "#E07B39"   # amber – fast layer
C_OVERLAP   = "#D4A017"   # gold – overlap branch
C_FULL      = "#2B5C8A"   # full model
C_ABLATED   = "#C0392B"   # ablated / wrong
C_CORRECT   = "#27AE60"   # correct context / pass
C_WRONG     = "#E74C3C"   # wrong context / fail
C_NEUTRAL   = "#95A5A6"   # unrelated / neutral
C_PRE       = "#A8C6DE"   # light blue – pre-consolidation
C_POST      = "#1A5276"   # dark blue  – post-consolidation

# ---------------------------------------------------------------------------
# Shared simulation helpers
# ---------------------------------------------------------------------------
BRANCH_IDS = ["b0", "b1", "b2", "b3"]
MU1_ALLOC = TraceAllocation("mu1", {"b0": 0.90, "b1": 0.85, "b2": 0.05, "b3": 0.05})
MU2_ALLOC = TraceAllocation("mu2", {"b0": 0.05, "b1": 0.85, "b2": 0.90, "b3": 0.05})
MU1_CUE   = {"b0": 1.0, "b1": 0.8, "b2": 0.0, "b3": 0.0}
MU2_CUE   = {"b0": 0.0, "b1": 0.8, "b2": 1.0, "b3": 0.0}
AMBIG_CUE = {"b0": 0.5, "b1": 0.5, "b2": 0.5, "b3": 0.5}
B1_CUE    = {"b0": 0.0, "b1": 1.0, "b2": 0.0, "b3": 0.0}

BASE_PARAMS = DynamicsParameters(
    structural_lr=0.18, replay_gain=0.80, eligibility_decay=0.12,
    structural_decay=0.005, structural_gain=6.0, structural_max=1.0,
    translation_decay=0.05, sleep_gain=0.0, structural_noise=0.0,
)


def _build(params=None) -> CytodendAccessModelSimulator:
    p = params or BASE_PARAMS
    sim = CytodendAccessModelSimulator.from_branch_ids(BRANCH_IDS, parameters=p)
    sim.traces["mu1"] = EngramTrace("mu1", MU1_ALLOC)
    sim.traces["mu2"] = EngramTrace("mu2", MU2_ALLOC)
    return sim


def _encode(sim):
    for _ in range(2): sim.apply_cue(MU1_CUE)
    for _ in range(2): sim.apply_cue(MU2_CUE)


def _consolidate(sim, n=9, drive=1.0, replay=None):
    win = ConsolidationWindow(
        replay_trace_ids=replay if replay is not None else ["mu1", "mu2"],
        modulatory_drive=drive,
    )
    for _ in range(n): sim.run_consolidation(win)


def _mb(sim) -> dict: return {b: sim.branches[b].structural.accessibility for b in BRANCH_IDS}


def _linking(sim) -> float:
    return sum(MU1_ALLOC.branch_weights.get(b,0)*MU2_ALLOC.branch_weights.get(b,0)
               *sim.branches[b].structural.accessibility for b in BRANCH_IDS)


def _recall(sim, cue, tid) -> float:
    sim.apply_cue(cue)
    rmap = {rs.trace_id: rs for rs in sim.compute_recall_supports()}
    return rmap[tid].support if tid in rmap else 0.0


def _recovery_pct(post, dmg, healthy):
    d = healthy - dmg
    return (post - dmg) / d * 100.0 if abs(d) > 1e-9 else 0.0


# ============================================================================
# Figure E1 – Simulator Architecture  (pure matplotlib patch diagram)
# ============================================================================

def fig_e1():
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, color, label, fontsize=8, text_color="white", style="round,pad=0.05"):
        rect = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=0.8,
                               edgecolor="white", facecolor=color, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold", zorder=4,
                wrap=True)

    def arrow(x1, y1, x2, y2, color="#555555", lw=1.2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                    zorder=5)

    # Background panels
    bg_slow = FancyBboxPatch((0.3, 0.3), 9.4, 1.6, boxstyle="round,pad=0.1",
                              linewidth=1, edgecolor=C_SLOW, facecolor="#EAF2FB", zorder=1)
    bg_fast = FancyBboxPatch((0.3, 2.1), 9.4, 1.7, boxstyle="round,pad=0.1",
                              linewidth=1, edgecolor=C_FAST, facecolor="#FEF5EC", zorder=1)
    ax.add_patch(bg_slow)
    ax.add_patch(bg_fast)
    ax.text(0.5, 1.95, "Slow structural layer  (M_b, E_b, P_b)", fontsize=7.5,
            color=C_SLOW, fontweight="bold")
    ax.text(0.5, 3.85, "Fast access layer  (x_b, s_i, A_b)", fontsize=7.5,
            color="#A04000", fontweight="bold")

    # --- Slow layer boxes ---
    box(0.5, 0.5, 1.8, 1.0, C_SLOW, "Slow structural\naccessibility\nM_b", fontsize=7.5)
    box(2.6, 0.5, 1.6, 1.0, "#34495E", "Eligibility\ntrace  E_b", fontsize=7.5)
    box(4.5, 0.5, 1.8, 1.0, "#1A6A4A", "Translation\nreadiness  P_b", fontsize=7.5)
    box(6.6, 0.5, 2.0, 1.0, "#5D3A8A", "Replay-like\nconsolidation", fontsize=7.5)

    # arrows in slow layer
    arrow(6.6, 1.0, 6.3, 1.0)  # replay -> P_b
    arrow(5.7, 1.5, 1.5, 1.5)  # P_b -> M_b (top)
    arrow(3.5, 1.5, 1.5, 1.5)  # E_b -> M_b

    # --- Fast layer boxes ---
    box(0.5, 2.3, 1.8, 1.2, C_FAST, "Branch\nactivation  x_b", fontsize=7.5, text_color="white")
    box(2.6, 2.3, 1.6, 1.2, "#B7770D", "Spine access\ns_i", fontsize=7.5)
    box(4.5, 2.3, 1.8, 1.2, "#1F618D", "Effective\naccess  A_b", fontsize=7.5)
    box(6.6, 2.3, 2.0, 1.2, "#117A65", "Context\nbias", fontsize=7.5)

    # arrows fast layer
    arrow(2.6, 2.9, 2.3, 2.9)  # spines -> x_b
    arrow(4.5, 2.9, 4.3, 2.9)  # A_b -> x_b
    arrow(6.6, 2.9, 6.3, 2.9)  # context -> A_b

    # --- Trace allocation (top) ---
    box(0.9, 4.2, 3.8, 1.2, "#6C3483",
        "Trace allocation  a_{mu,b}\n(mu1: b0+b1 | mu2: b1+b2)", fontsize=7.5)
    box(5.0, 4.2, 4.0, 1.2, "#1B4F72",
        "Recall support  R_mu\n= sum_b a_{mu,b} * x_b", fontsize=7.5)

    # arrows between layers
    arrow(2.8, 4.2, 2.0, 3.5)   # allocation -> branch activation
    arrow(1.9, 2.3, 1.9, 1.5)   # activation -> eligibility
    arrow(3.7, 1.5, 3.7, 2.3)   # E_b feedback to fast
    arrow(7.0, 2.3, 7.0, 1.5)   # consolidation trigger
    arrow(2.4, 3.5, 5.0, 4.8)   # x_b -> recall
    arrow(1.3, 1.5, 1.3, 2.3)   # M_b -> A_b (via slow_access)

    ax.set_title("Figure E1 — cytodend_accessmodel simulator architecture",
                 fontsize=9.5, fontweight="bold", pad=8)

    path = os.path.join(OUTDIR, "figure_E1_architecture.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# Figure E2 – Branch-specific writing + linking emergence  (3-panel)
# ============================================================================

def _run_exp001():
    sim = _build()
    mb_pre = _mb(sim)
    lk_pre = _linking(sim)
    _encode(sim)
    _consolidate(sim, n=9)
    mb_post = _mb(sim)
    lk_post = _linking(sim)
    # recall support pre/post
    sim_pre = _build()
    _encode(sim_pre)
    r_pre_cued     = _recall(deepcopy(sim_pre), MU1_CUE, "mu1")
    r_pre_linked   = _recall(deepcopy(sim_pre), MU1_CUE, "mu2")
    r_post_cued    = _recall(deepcopy(sim), MU1_CUE, "mu1")
    r_post_linked  = _recall(deepcopy(sim), MU1_CUE, "mu2")
    return dict(mb_pre=mb_pre, mb_post=mb_post, lk_pre=lk_pre, lk_post=lk_post,
                r_pre_cued=r_pre_cued, r_pre_linked=r_pre_linked,
                r_post_cued=r_post_cued, r_post_linked=r_post_linked)


def fig_e2():
    d = _run_exp001()
    branches = ["b0", "b1", "b2", "b3"]
    labels   = ["b0\n(mu1)", "b1\n(overlap)", "b2\n(mu2)", "b3\n(unrelated)"]
    colors_pre  = [C_PRE]*4
    colors_post = [C_FULL, C_OVERLAP, C_FULL, C_NEUTRAL]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8),
                             gridspec_kw={"width_ratios": [2.5, 1, 1.5]})
    fig.suptitle("Figure E2 — Branch-specific structural writing and linking emergence",
                 fontsize=9, fontweight="bold", y=1.02)

    # -- Panel A: M_b per branch pre/post --
    ax = axes[0]
    x = np.arange(4)
    w = 0.35
    ax.bar(x - w/2, [d["mb_pre"][b]  for b in branches], w,
           color=C_PRE,  label="Pre-consolidation", edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, [d["mb_post"][b] for b in branches], w,
           color=colors_post, label="Post-consolidation", edgecolor="white", linewidth=0.5)

    # delta labels
    for i, b in enumerate(branches):
        delta = d["mb_post"][b] - d["mb_pre"][b]
        sign = "+" if delta >= 0 else ""
        ax.text(x[i] + w/2, d["mb_post"][b] + 0.015,
                f"{sign}{delta:.3f}", ha="center", va="bottom", fontsize=6.5,
                color=colors_post[i], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Structural accessibility  M_b")
    ax.set_ylim(0, 0.92)
    ax.set_title("A  Branch structural accessibility", fontsize=8.5, loc="left")
    # Legend
    ax.bar(0, 0, color=C_PRE, label="Pre")
    handles = [mpatches.Patch(color=C_PRE, label="Pre"),
               mpatches.Patch(color=C_FULL, label="Post (active)"),
               mpatches.Patch(color=C_OVERLAP, label="Post (overlap b1)"),
               mpatches.Patch(color=C_NEUTRAL, label="Post (unrelated)")]
    ax.legend(handles=handles, fontsize=6.5, loc="upper left",
              framealpha=0.8, borderpad=0.4, handlelength=1.2)

    # -- Panel B: linking metric pre/post --
    ax = axes[1]
    vals = [d["lk_pre"], d["lk_post"]]
    clrs = [C_PRE, C_POST]
    bars = ax.bar(["Pre", "Post"], vals, color=clrs, width=0.5,
                  edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    chg = (d["lk_post"] - d["lk_pre"]) / d["lk_pre"] * 100
    ax.annotate(f"+{chg:.1f}%",
                xy=(1, d["lk_post"]), xytext=(0.5, d["lk_post"] + 0.06),
                ha="center", fontsize=8, color=C_POST, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=C_POST, lw=0.8))
    ax.set_ylabel("Linking metric  L(mu1,mu2)")
    ax.set_ylim(0, 0.85)
    ax.set_title("B  Linking metric", fontsize=8.5, loc="left")

    # -- Panel C: recall support pre/post --
    ax = axes[2]
    x = np.arange(2)
    w = 0.32
    pre_vals  = [d["r_pre_cued"],   d["r_pre_linked"]]
    post_vals = [d["r_post_cued"],  d["r_post_linked"]]
    ax.bar(x - w/2, pre_vals,  w, color=C_PRE,  edgecolor="white", linewidth=0.5, label="Pre")
    ax.bar(x + w/2, post_vals, w, color=[C_FULL, C_OVERLAP], edgecolor="white",
           linewidth=0.5, label="Post")
    for i, (pv, pov) in enumerate(zip(post_vals, post_vals)):
        ax.text(x[i] + w/2, pov + 0.02, f"{pov:.3f}",
                ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Cued\n(mu1)", "Linked\n(mu2)"], fontsize=7.5)
    ax.set_ylabel("Recall support  R_mu")
    ax.set_ylim(0, 1.55)
    ax.set_title("C  Recall support", fontsize=8.5, loc="left")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.8)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "figure_E2_branch_writing.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# Figure E3 – Context-sensitive recall under ambiguous cue
# ============================================================================

def _run_exp002():
    ALPHA_ALLOC = TraceAllocation("mu_alpha", {"b0": 0.90, "b1": 0.05, "b2": 0.05, "b3": 0.0})
    BETA_ALLOC  = TraceAllocation("mu_beta",  {"b0": 0.05, "b1": 0.05, "b2": 0.90, "b3": 0.0})
    ALPHA_BIAS  = {"b0": 0.5, "b1": 0.5, "b2": -0.5, "b3": -0.5}
    BETA_BIAS   = {"b0": -0.5, "b1": -0.5, "b2": 0.5, "b3": 0.5}

    p = deepcopy(BASE_PARAMS)
    sim = CytodendAccessModelSimulator.from_branch_ids(BRANCH_IDS, parameters=p)
    sim.traces["mu_alpha"] = EngramTrace("mu_alpha", ALPHA_ALLOC, context="alpha")
    sim.traces["mu_beta"]  = EngramTrace("mu_beta",  BETA_ALLOC,  context="beta")

    for _ in range(2): sim.apply_cue({"b0": 1.0, "b1": 0.0, "b2": 0.0, "b3": 0.0}, context="alpha")
    for _ in range(2): sim.apply_cue({"b0": 0.0, "b1": 0.0, "b2": 1.0, "b3": 0.0}, context="beta")
    win = ConsolidationWindow(replay_trace_ids=["mu_alpha","mu_beta"], modulatory_drive=1.0)
    for _ in range(9): sim.run_consolidation(win)

    def probe(ctx, bias):
        s = deepcopy(sim)
        s.apply_cue(AMBIG_CUE, context=ctx, context_bias=bias)
        rmap = {rs.trace_id: rs for rs in s.compute_recall_supports()}
        return (rmap.get("mu_alpha", type("X",(),{"support":0.0})()).support,
                rmap.get("mu_beta",  type("X",(),{"support":0.0})()).support)

    ra_alpha, rb_alpha = probe("alpha", ALPHA_BIAS)
    ra_beta,  rb_beta  = probe("beta",  BETA_BIAS)
    # no context: no bias
    s_none = deepcopy(sim)
    s_none.apply_cue(AMBIG_CUE)
    rmap_none = {rs.trace_id: rs for rs in s_none.compute_recall_supports()}
    ra_none = rmap_none.get("mu_alpha", type("X",(),{"support":0.0})()).support
    rb_none = rmap_none.get("mu_beta",  type("X",(),{"support":0.0})()).support

    return dict(alpha=(ra_alpha, rb_alpha), beta=(ra_beta, rb_beta), none=(ra_none, rb_none))


def fig_e3():
    d = _run_exp002()
    contexts = ["Alpha\ncontext", "Beta\ncontext", "No\ncontext"]
    r_alpha = [d["alpha"][0], d["beta"][0], d["none"][0]]
    r_beta  = [d["alpha"][1], d["beta"][1], d["none"][1]]

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x = np.arange(3)
    w = 0.32
    b1 = ax.bar(x - w/2, r_alpha, w, color=C_CORRECT, label="R(mu_alpha)", edgecolor="white")
    b2 = ax.bar(x + w/2, r_beta,  w, color=C_SLOW,    label="R(mu_beta)",  edgecolor="white")

    for bar, v in zip(b1, r_alpha):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold", color=C_CORRECT)
    for bar, v in zip(b2, r_beta):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold", color=C_SLOW)

    # winner arrows
    for i in range(3):
        w_val  = "mu_alpha" if r_alpha[i] > r_beta[i] else "mu_beta"
        clr    = C_CORRECT if w_val == "mu_alpha" else C_SLOW
        winner = r_alpha[i] if w_val == "mu_alpha" else r_beta[i]
        ax.text(x[i], winner + 0.045, "winner", ha="center", va="bottom",
                fontsize=6.5, color=clr, fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(contexts)
    ax.set_ylabel("Recall support")
    ax.set_ylim(0, 0.65)
    ax.legend(loc="upper right", framealpha=0.8)
    ax.set_title("Figure E3 — Context-sensitive recall under ambiguous cue",
                 fontsize=9, fontweight="bold")

    # Annotations: "correct context selects correct trace" / "recency bias"
    ax.annotate("correct context\nselects correct trace",
                xy=(0.5, 0.47), ha="center", fontsize=7, color="#555555",
                style="italic")
    ax.annotate("recency bias\n(mu_beta more recent)",
                xy=(2, 0.43), ha="center", fontsize=7, color="#888888",
                style="italic")

    fig.tight_layout()
    path = os.path.join(OUTDIR, "figure_E3_context_recall.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# Figure E4 – Three-factor dependence
# ============================================================================

def _run_exp003():
    results = {}

    def run_cond(label, gap=0, replay=None, drive=1.0):
        sim = _build()
        _encode(sim)
        # spacing gap
        for _ in range(gap):
            sim.apply_cue({b: 0.0 for b in BRANCH_IDS})
        _consolidate(sim, n=9, drive=drive,
                     replay=replay if replay else ["mu1","mu2"])
        return {"mb": _mb(sim), "lk": _linking(sim)}

    results["immediate_joint"]   = run_cond("immediate + joint", gap=0)
    results["spaced_joint"]      = run_cond("spaced (12) + joint", gap=12)
    results["immediate_mu1only"] = run_cond("immediate + mu1 only", gap=0, replay=["mu1"])
    results["modulator_zero"]    = run_cond("immediate + drive=0", gap=0, drive=0.0)
    return results


def fig_e4():
    d = _run_exp003()
    conds  = ["immediate\njoint", "spaced 12\njoint", "immediate\nmu1 only", "drive = 0"]
    keys   = ["immediate_joint","spaced_joint","immediate_mu1only","modulator_zero"]
    colors = [C_FULL, C_SLOW, C_FAST, C_ABLATED]

    mb_b1  = [d[k]["mb"]["b1"] for k in keys]
    mb_b0  = [d[k]["mb"]["b0"] for k in keys]
    lk_val = [d[k]["lk"]       for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.suptitle("Figure E4 — Three-factor consolidation dependence",
                 fontsize=9, fontweight="bold", y=1.01)

    x = np.arange(4)
    w = 0.35

    # --- Panel A: M_b values ---
    ax = axes[0]
    bars1 = ax.bar(x - w/2, mb_b1, w, color=colors, edgecolor="white", linewidth=0.5,
                   label="b1 (overlap)")
    bars2 = ax.bar(x + w/2, mb_b0, w, color=colors, edgecolor="white", linewidth=0.5,
                   alpha=0.45, label="b0 (mu1 primary)", hatch="//")
    for bar, v in zip(bars1, mb_b1):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=7.5)
    ax.set_ylabel("Structural accessibility  M_b")
    ax.set_ylim(0, 0.88)
    ax.set_title("A  Branch structural accessibility", fontsize=8.5, loc="left")
    ax.legend(fontsize=7)

    # --- Panel B: linking metric ---
    ax = axes[1]
    bars = ax.bar(x, lk_val, color=colors, edgecolor="white", linewidth=0.5)
    # Reference line = modulatory=0 value
    ax.axhline(lk_val[-1], color=C_ABLATED, linestyle="--", lw=0.8, alpha=0.6,
               label="drive=0 baseline")
    for bar, v in zip(bars, lk_val):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.004, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=7.5)
    ax.set_ylabel("Linking metric  L(mu1,mu2)")
    ax.set_ylim(0, 0.75)
    ax.set_title("B  Linking metric by condition", fontsize=8.5, loc="left")
    ax.legend(fontsize=7)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "figure_E4_three_factor.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# Figure E5 – Structural gate ablation (3-panel)
# ============================================================================

def _run_exp014():
    ABLATED = DynamicsParameters(
        structural_lr=0.0, replay_gain=0.80, eligibility_decay=0.12,
        structural_decay=0.005, structural_gain=6.0, structural_max=1.0,
        translation_decay=0.05, sleep_gain=0.0, structural_noise=0.0,
    )

    res = {}
    for label, params in [("Full model", BASE_PARAMS), ("LR ablation\n(structural_lr=0)", ABLATED)]:
        sim = _build(params)
        mb_pre = _mb(sim)
        lk_pre = _linking(sim)
        _encode(sim)
        _consolidate(sim, n=9)
        mb_post = _mb(sim)
        lk_post = _linking(sim)
        h_link = _linking(sim)

        # Rescue protocol
        sim2 = _build(params)
        _encode(sim2)
        _consolidate(sim2, n=9)
        h_link2 = _linking(sim2)
        sim2.branches["b1"].structural.decay_rate = 0.030
        win_null = ConsolidationWindow(replay_trace_ids=[], modulatory_drive=0.0)
        for _ in range(9): sim2.run_consolidation(win_null)
        d_link = _linking(sim2)

        sim_std = deepcopy(sim2)
        _consolidate(sim_std, n=9)
        link_std = _linking(sim_std)

        sim_ovlp = deepcopy(sim2)
        for _ in range(3):
            for _ in range(3): sim_ovlp.apply_cue(B1_CUE)
            _consolidate(sim_ovlp, n=3)
        link_ovlp = _linking(sim_ovlp)

        rec_std  = _recovery_pct(link_std,  d_link, h_link2)
        rec_ovlp = _recovery_pct(link_ovlp, d_link, h_link2)

        res[label] = dict(
            delta_mb1 = mb_post["b1"] - mb_pre["b1"],
            lk_change_pct = (lk_post - lk_pre) / max(abs(lk_pre),1e-9) * 100,
            rescue_advantage = rec_ovlp - rec_std,
        )
    return res


def fig_e5():
    d = _run_exp014()
    labels = list(d.keys())
    metrics = [
        ("delta_mb1",        "Delta M_b1 (overlap branch)", [C_FULL, C_ABLATED], None),
        ("lk_change_pct",    "Linking change (%)",          [C_FULL, C_ABLATED], 0),
        ("rescue_advantage", "Targeted rescue\nadvantage (%)", [C_FULL, C_ABLATED], 0),
    ]
    panel_labels = ["A  Overlap branch strengthening",
                    "B  Linking growth",
                    "C  Targeted rescue selectivity"]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2))
    fig.suptitle("Figure E5 — Structural gate ablation: three signatures collapse simultaneously",
                 fontsize=9, fontweight="bold", y=1.02)

    for ax, (key, ylabel, colors, hline), plabel in zip(axes, metrics, panel_labels):
        vals = [d[label][key] for label in labels]
        bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            sign = "+" if v >= 0 else ""
            ax.text(bar.get_x()+bar.get_width()/2,
                    (v + max(abs(v)*0.03, 0.005)) * (1 if v >= 0 else -1),
                    f"{sign}{v:.1f}" + ("%" if "pct" in key or "adv" in key.lower() else ""),
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8, fontweight="bold")
        if hline is not None:
            ax.axhline(hline, color="#aaaaaa", linestyle="--", lw=0.8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(plabel, fontsize=8.5, loc="left")
        ax.tick_params(axis="x", labelsize=8)

    # Shared label for ablation outcome
    axes[0].set_ylim(-0.08, 0.42)
    axes[1].set_ylim(-15, 80)
    axes[2].set_ylim(-20, 200)

    for ax in axes:
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "figure_E5_ablation.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# Figure E6 – Comparator-baseline signature heatmap  (COLOUR)
# ============================================================================

def fig_e6():
    # Data from exp015 results
    baselines = ["Full model", "Fast context\nonly", "Replay no\nstructure",
                 "Random\nslow drift", "Fixed allocation\nonly"]
    sigs      = ["SIG-A\noverlap\nstrength", "SIG-B\nlinking\ngain",
                 "SIG-C\ncontext\nsep", "SIG-D\nlinking>\nrecall",
                 "SIG-E\ntargeted\nrescue"]

    # 1=PASS, 0=FAIL
    data = np.array([
        [1, 1, 1, 1, 1],   # full_model
        [0, 0, 1, 1, 0],   # fast_context_only
        [0, 0, 1, 1, 0],   # replay_no_structure
        [0, 0, 1, 1, 0],   # random_slow_drift
        [0, 0, 1, 1, 0],   # fixed_allocation_only
    ], dtype=float)

    # Quantitative values for annotation
    sig_a_vals = ["+0.068", "0.000", "0.000", "-0.088", "0.000"]
    sig_b_vals = ["+58.6%", "+0.0%", "-4.4%", "-15.9%", "+0.0%"]
    sig_c_vals = ["+0.155", "+0.149", "+0.148", "+0.150", "+0.149"]
    sig_d_vals = ["+21.0", "+19.3", "+19.2", "+19.1", "+19.3"]
    sig_e_vals = ["+154.7%", "0.0%", "0.0%", "-6.2%", "0.0%"]
    annot = np.array([sig_a_vals, sig_b_vals, sig_c_vals, sig_d_vals, sig_e_vals]).T

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    cmap = matplotlib.colors.ListedColormap(["#E74C3C", "#27AE60"])
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(sigs, fontsize=7.5)
    ax.set_yticks(np.arange(5))
    ax.set_yticklabels(baselines, fontsize=8)

    for i in range(5):
        for j in range(5):
            pf = "PASS" if data[i, j] == 1 else "FAIL"
            tc = "white"
            ax.text(j, i, f"{pf}\n{annot[i,j]}", ha="center", va="center",
                    fontsize=6.8, fontweight="bold", color=tc)

    # Grid lines
    for x_pos in np.arange(-0.5, 5.5, 1):
        ax.axvline(x_pos, color="white", lw=1.2)
    for y_pos in np.arange(-0.5, 5.5, 1):
        ax.axhline(y_pos, color="white", lw=1.2)

    # Legend
    pass_patch = mpatches.Patch(color="#27AE60", label="PASS")
    fail_patch = mpatches.Patch(color="#E74C3C", label="FAIL")
    ax.legend(handles=[pass_patch, fail_patch], loc="lower right",
              bbox_to_anchor=(1.0, -0.02), fontsize=8, framealpha=0.9)

    ax.set_title("Figure E6 — Comparator-baseline joint signature profile",
                 fontsize=9, fontweight="bold", pad=10)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    ax.xaxis.set_ticks_position("bottom")

    fig.tight_layout()
    path = os.path.join(OUTDIR, "figure_E6_comparator_heatmap.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# Figure E7 – Motif-family + hub ablation inset
# ============================================================================

def _run_exp016():
    from itertools import combinations as comb

    PARAMS_16 = DynamicsParameters(
        structural_lr=0.18, replay_gain=0.80, eligibility_decay=0.12,
        structural_decay=0.005, structural_gain=6.0, structural_max=1.0,
        translation_decay=0.05, sleep_gain=0.0, structural_noise=0.0,
    )

    def make_trace(tid, branches, weights):
        return EngramTrace(tid, TraceAllocation(tid, weights))

    def build_sim(branches, traces):
        sim = CytodendAccessModelSimulator.from_branch_ids(branches, parameters=PARAMS_16)
        for t in traces: sim.add_trace(t)
        return sim

    def encode_all(sim, traces):
        for t in traces:
            cue = dict(t.allocation.branch_weights)
            for _ in range(2): sim.apply_cue(cue)

    def consolidate_all(sim, n=9):
        win = ConsolidationWindow(modulatory_drive=1.0)
        for _ in range(n): sim.run_consolidation(win)

    def mean_linking(sim, traces):
        pairs = list(comb(traces, 2))
        if not pairs: return 0.0
        return sum(
            sum(t1.allocation.branch_weights.get(b,0)*t2.allocation.branch_weights.get(b,0)
                *sim.branches[b].structural.accessibility for b in sim.branches)
            for t1,t2 in pairs
        ) / len(pairs)

    def motif_no_overlap():
        branches = [f"b{i:02d}" for i in range(12)]
        traces = []
        for k in range(6):
            tid, b0, b1 = f"mu{k}", branches[k*2], branches[k*2+1]
            w = {b: 0.02 for b in branches}; w[b0]=0.90; w[b1]=0.85
            traces.append(make_trace(tid, branches, w))
        return branches, traces, []

    def motif_weak():
        branches = [f"b{i:02d}" for i in range(12)]
        shared = [f"b{i:02d}" for i in range(6,11)]
        traces = []
        for k in range(6):
            w = {b: 0.02 for b in branches}; w[branches[k]] = 0.90
            if k < 5: w[shared[k]] = 0.30
            if k > 0: w[shared[k-1]] = 0.30
            traces.append(make_trace(f"mu{k}", branches, w))
        return branches, traces, shared

    def motif_strong():
        branches = [f"b{i:02d}" for i in range(10)]
        configs = [("mu0","b00","b08"),("mu1","b01","b08"),("mu2","b02","b09"),("mu3","b03","b09")]
        traces = []
        for tid,p,ovb in configs:
            w = {b:0.02 for b in branches}; w[p]=0.90; w[ovb]=0.85
            traces.append(make_trace(tid, branches, w))
        return branches, traces, ["b08","b09"]

    def motif_chain():
        branches = [f"b{i:02d}" for i in range(10)]
        shared = [f"b{i:02d}" for i in range(5,9)]
        traces = []
        for k in range(5):
            w = {b:0.02 for b in branches}; w[branches[k]]=0.90
            if k<4: w[shared[k]]=0.60
            if k>0: w[shared[k-1]]=0.60
            traces.append(make_trace(f"mu{k}", branches, w))
        return branches, traces, shared

    def motif_hub():
        branches = [f"b{i:02d}" for i in range(12)]
        hub = "b11"
        traces = []
        for k in range(6):
            w = {b:0.02 for b in branches}; w[branches[k]]=0.90; w[hub]=0.50
            traces.append(make_trace(f"mu{k}", branches, w))
        return branches, traces, [hub]

    results = {}
    for name, builder in [("no_overlap",motif_no_overlap),("weak",motif_weak),
                            ("strong",motif_strong),("chain",motif_chain),("hub",motif_hub)]:
        br, tr, ov = builder()
        sim_pre = build_sim(br, tr)
        sim_post = build_sim(br, tr)
        encode_all(sim_post, tr)
        consolidate_all(sim_post, 9)
        lk_pre = mean_linking(sim_pre, tr)
        lk_post = mean_linking(sim_post, tr)
        results[name] = {"lk_pre": lk_pre, "lk_post": lk_post,
                         "lk_chg": (lk_post-lk_pre)/max(abs(lk_pre),1e-9)*100}
        if name == "hub":
            hub = "b11"
            ctrl = "b00"
            pairs = list(comb(tr, 2))
            def mp(s): return sum(
                sum(t1.allocation.branch_weights.get(b,0)*t2.allocation.branch_weights.get(b,0)
                    *s.branches[b].structural.accessibility for b in s.branches)
                for t1,t2 in pairs
            ) / len(pairs)
            sh = deepcopy(sim_post); sh.branches[hub].structural.accessibility = 0.0
            sc = deepcopy(sim_post); sc.branches[ctrl].structural.accessibility = 0.0
            results["hub"]["hub_drop"] = (lk_post - mp(sh)) / max(abs(lk_post),1e-9) * 100
            results["hub"]["ctrl_drop"] = (lk_post - mp(sc)) / max(abs(lk_post),1e-9) * 100

    return results


def fig_e7():
    d = _run_exp016()
    motifs  = ["no_overlap", "weak", "chain", "strong", "hub"]
    labels  = ["No\noverlap", "Weak\noverlap", "Chain\noverlap", "Strong\noverlap", "Hub\noverlap"]
    chg_vals = [d[m]["lk_chg"] for m in motifs]

    colors_bar = [C_NEUTRAL, "#7FB3D3", "#2E86C1", C_FULL, C_OVERLAP]

    fig, ax_main = plt.subplots(figsize=(5.5, 3.4), constrained_layout=False)
    bars = ax_main.bar(labels, chg_vals, color=colors_bar, width=0.6,
                       edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, chg_vals):
        ax_main.text(bar.get_x()+bar.get_width()/2, v+0.5, f"+{v:.1f}%",
                     ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax_main.set_ylabel("Linking gain after consolidation (%)")
    ax_main.set_ylim(0, 70)
    ax_main.set_title("Figure E7 — Linking gain tracks overlap structure\nacross five motif families",
                      fontsize=9, fontweight="bold")

    # --- Inset: hub ablation ---
    ax_inset = fig.add_axes([0.63, 0.52, 0.33, 0.40])
    hub_vals  = [d["hub"]["hub_drop"], d["hub"]["ctrl_drop"]]
    inset_colors = [C_OVERLAP, C_NEUTRAL]
    b_ins = ax_inset.bar(["Hub\nbranch", "Control\nbranch"], hub_vals,
                          color=inset_colors, width=0.5, edgecolor="white")
    for bar, v in zip(b_ins, hub_vals):
        ax_inset.text(bar.get_x()+bar.get_width()/2, v+0.5, f"{v:.1f}%",
                      ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax_inset.set_ylabel("Linking drop\nafter ablation (%)", fontsize=6.5)
    ax_inset.set_title("Hub bottleneck\n(hub_overlap motif)", fontsize=6.5, fontweight="bold")
    ax_inset.set_ylim(0, 105)
    ax_inset.tick_params(labelsize=6.5)
    ax_inset.spines["top"].set_visible(False)
    ax_inset.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.12)
    path = os.path.join(OUTDIR, "figure_E7_motif_family.svg")
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    random.seed(42)
    print(f"\nGenerating figures -> {OUTDIR}\n")
    fig_e1()
    fig_e2()
    fig_e3()
    fig_e4()
    fig_e5()
    fig_e6()
    fig_e7()
    print("\nAll figures saved.\n")
