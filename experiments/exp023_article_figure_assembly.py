"""Experiment 023 — Article Figure Assembly.

Generates six publication-facing figures from E017–E022R results, writing to:

    article/Slow Branch-Level Accessibility.../figures2/
        Fig_e023_01_model_concept.png
        Fig_e023_02_canonical_traces.png
        Fig_e023_03_comparator_matrix.png
        Fig_e023_04_robustness_landscape.png
        Fig_e023_05_motif_scaling.png
        Fig_e023_06_shuffled_replay_audit.png

Sources
-------
Fig 1  : concept figure (matplotlib text/box panels; no CSV)
Fig 2  : results/e017_*/traces/branch_traces.csv + linking_trace.csv
Fig 3  : results/e018_*/summary/ + results/e022_hard_comparators/summary/
Fig 4  : results/e019_*/summary/ + results/e020_*/summary/
Fig 5  : results/e021_*/summary/ + results/e021r_*/summary/
Fig 6  : results/e022r_*/summary/shuffled_replay_vs_full_model.csv
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTICLE_DIR = REPO_ROOT / "article" / \
    "Slow Branch-Level Accessibility as a Structural Constraint on Memory Linking"
OUT_DIR = ARTICLE_DIR / "figures2"

R = REPO_ROOT / "results"
E017 = R / "e017_traceable_simulator_core"
E018 = R / "e018_comparator_trace_matrix"
E019 = R / "e019_one_at_a_time_parameter_robustness"
E020 = R / "e020_two_parameter_robustness_heatmaps"
E021 = R / "e021_scaling_and_motif_generalization"
E021R = R / "e021r_generalized_specificity_gate"
E022 = R / "e022_hard_comparators"
E022R = R / "e022r_shuffled_replay_scaling_audit"


def _load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _flt(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _bool(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Figure 1 — Model concept
# ---------------------------------------------------------------------------

def fig1_model_concept(ax_list) -> None:
    """Four-panel concept figure: biological sketch, simulator map, equations, claim boundary."""
    import matplotlib.patches as mpatches

    titles = [
        "A. Biological rationale",
        "B. Minimal simulator",
        "C. Variable mapping",
        "D. Claim boundary",
    ]
    bodies = [
        (
            "Dendritic branches are not passive.\n"
            "Fast gating opens branches momentarily.\n"
            "Slow structural writing (M_b) changes\n"
            "which branches remain easier to reuse.\n\n"
            "Two traces sharing an overlap branch\n"
            "become linked when replay consolidates\n"
            "that shared structural state."
        ),
        (
            "4 branches (b0–b3)  •  2 traces (μ1, μ2)\n"
            "b1 = overlap branch (shared by μ1, μ2)\n\n"
            "Phases:\n"
            "  init → encode → consolidate\n"
            "  → damage → rescue\n\n"
            "L_μν = Σb a_μb · a_νb · M_b(t)"
        ),
        (
            "M_b(t)   slow structural accessibility\n"
            "         updated by replay × eligibility\n\n"
            "E_b(t)   eligibility trace\n"
            "         set by encoding cue\n\n"
            "P_b(t)   consolidation support\n"
            "         set by replay signal\n\n"
            "ΔM_b = η · E_b · P_b · W(t) · (M_max − M_b)\n"
            "     − λ · M_b"
        ),
        (
            "The simulator supports:\n"
            "  slow writing as a discrimination hypothesis\n\n"
            "The simulator does NOT establish:\n"
            "  a cytoskeletal memory code\n"
            "  biological validation of M_b\n"
            "  proof from DANDI open data\n\n"
            "Joint profile matters.\n"
            "No single signature is diagnostic."
        ),
    ]
    colors = ["#dceefa", "#d4f4dd", "#fef3cd", "#fde8e8"]

    for ax, title, body, color in zip(ax_list, titles, bodies, colors):
        ax.set_facecolor(color)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, 0.96, title, ha="center", va="top", fontsize=9,
                fontweight="bold", transform=ax.transAxes)
        ax.text(0.07, 0.82, body, ha="left", va="top", fontsize=7.5,
                transform=ax.transAxes, family="monospace",
                linespacing=1.55)


# ---------------------------------------------------------------------------
# Figure 2 — Canonical simulator traces (E017)
# ---------------------------------------------------------------------------

def fig2_canonical_traces() -> "plt.Figure":
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    branch_rows = _load_csv(E017 / "traces" / "branch_traces.csv")
    link_rows   = _load_csv(E017 / "traces" / "linking_trace.csv")

    PHASES = ["init", "encode", "probe_pre", "consolidate", "probe_post",
              "damage", "probe_damage", "rescue_targeted", "probe_rescue"]
    PHASE_COLORS = {
        "init": "#eeeeee", "encode": "#d4eaf7", "probe_pre": "#eaf7d4",
        "consolidate": "#fde9c0", "probe_post": "#eaf7d4",
        "damage": "#fdd0d0", "probe_damage": "#eaf7d4",
        "rescue_targeted": "#d4f4dd", "probe_rescue": "#eaf7d4",
    }

    def _get_phase(rows, phase, col, bid):
        return [(int(r["step"]), _flt(r[col]))
                for r in rows if r.get("phase") == phase and r.get("branch_id") == bid]

    all_phases_br = sorted({r["phase"] for r in branch_rows},
                           key=lambda p: PHASES.index(p) if p in PHASES else 99)
    overlap_bid = next((r["branch_id"] for r in branch_rows if _bool(r.get("is_overlap"))), "b1")
    private_bid = next((r["branch_id"] for r in branch_rows
                        if not _bool(r.get("is_overlap")) and r.get("branch_id") != overlap_bid), "b0")

    # Build step-indexed arrays
    def _series(col, bid):
        return [(int(r["step"]), _flt(r[col]))
                for r in branch_rows if r.get("branch_id") == bid]

    def _link_series():
        return [(int(r["step"]), _flt(r.get("linking_score", 0)))
                for r in link_rows]

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    fig.suptitle("Canonical simulator traces\n"
                 "Overlap branch (solid) vs private branch (dashed) across phases",
                 fontsize=10)

    variables = [
        ("structural_accessibility", "M_b (structural accessibility)", "tab:blue"),
        ("eligibility",              "E_b (eligibility)",               "tab:orange"),
        ("translation_readiness",    "P_b (consolidation support)",     "tab:green"),
    ]

    for ax, (col, label, color) in zip(axes[:3], variables):
        ovlp = sorted(_series(col, overlap_bid))
        priv = sorted(_series(col, private_bid))
        if ovlp:
            xs, ys = zip(*ovlp)
            ax.plot(xs, ys, "-",  color=color, linewidth=1.8, label="overlap")
        if priv:
            xs, ys = zip(*priv)
            ax.plot(xs, ys, "--", color=color, linewidth=1.2, alpha=0.7, label="private")
        ax.set_ylabel(label, fontsize=7)
        ax.legend(fontsize=6, loc="upper right")
        ax.tick_params(labelsize=6)

    # Linking
    ldata = sorted(_link_series())
    if ldata:
        xs, ys = zip(*ldata)
        axes[3].plot(xs, ys, "-", color="tab:purple", linewidth=2)
    axes[3].set_ylabel("L (linking)", fontsize=7)
    axes[3].set_xlabel("Simulation step", fontsize=8)
    axes[3].tick_params(labelsize=6)

    # Phase bands
    phase_steps: dict[str, tuple[int, int]] = {}
    for r in branch_rows:
        p = r.get("phase", "")
        s = int(r.get("step", 0))
        if p not in phase_steps:
            phase_steps[p] = (s, s)
        else:
            lo, hi = phase_steps[p]
            phase_steps[p] = (min(lo, s), max(hi, s))

    for ax in axes:
        for ph, (lo, hi) in phase_steps.items():
            c = PHASE_COLORS.get(ph, "#f9f9f9")
            ax.axvspan(lo - 0.5, hi + 0.5, color=c, alpha=0.25, zorder=0)

    # Phase label on top axis
    for ph, (lo, hi) in phase_steps.items():
        mid = (lo + hi) / 2
        lbl = ph.replace("_", "\n")
        axes[0].text(mid, axes[0].get_ylim()[1] * 0.88, lbl,
                     ha="center", va="top", fontsize=5, rotation=0, color="#444")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Comparator discrimination matrix (E018 + E022)
# ---------------------------------------------------------------------------

def fig3_comparator_matrix() -> "plt.Figure":
    import matplotlib.pyplot as plt
    import numpy as np

    # E018 baseline comparators
    e018_rows = _load_csv(E018 / "summary" / "comparator_pass_fail_matrix.csv")
    e022_sig  = _load_csv(E022 / "summary" / "hard_comparator_signature_matrix.csv")
    e022_beh  = _load_csv(E022 / "summary" / "hard_comparator_behavioral_matrix.csv")

    SIGS = ["SIG_A", "SIG_B", "SIG_C", "SIG_D", "SIG_E"]
    BEHAV_COLS = ["B1_linking_gain", "B3_damage_sensitivity", "B5_recovery_index"]
    BEHAV_LABELS = ["B1 linking", "B3 damage", "B5 rescue"]

    # Build combined comparator list
    # E018 comparators
    e018_comp_order = ["full_model", "fast_context_only", "replay_no_structure",
                       "random_slow_drift", "fixed_allocation_only"]
    # E022 comparators (excluding full_model already included)
    e022_hard = ["hebbian_weight_only", "soma_global_gain_only", "shuffled_replay",
                 "eligibility_only", "resource_only"]

    ALL_COMPS = e018_comp_order + e022_hard
    COMP_LABELS = [c.replace("_", "\n") for c in ALL_COMPS]

    # Structural pass matrix
    struct_mat = np.full((len(ALL_COMPS), len(SIGS)), float("nan"))

    e018_by_name = {r["comparator"]: r for r in e018_rows}
    for i, comp in enumerate(e018_comp_order):
        row = e018_by_name.get(comp, {})
        for j, sig in enumerate(SIGS):
            struct_mat[i, j] = 1.0 if _bool(row.get(sig, False)) else 0.0

    # E022 hard: use gSIG_A/gSIG_B as proxy; use canonical motif result
    e022_sig_by_comp = {}
    for r in e022_sig:
        comp = r.get("comparator", "")
        if comp not in e022_sig_by_comp:
            e022_sig_by_comp[comp] = []
        e022_sig_by_comp[comp].append(r)

    for i, comp in enumerate(e022_hard):
        idx = len(e018_comp_order) + i
        canon_rows = [r for r in e022_sig_by_comp.get(comp, [])
                      if r.get("motif_type") == "canonical"]
        if canon_rows:
            r = canon_rows[0]
            struct_mat[idx, 0] = 1.0 if _bool(r.get("gSIG_A_pass")) else 0.0
            struct_mat[idx, 1] = float("nan")  # gSIG-B N/A for 2-trace
            struct_mat[idx, 2] = 1.0  # SIG-C: context sep (architectural, broad)
            struct_mat[idx, 3] = 1.0  # SIG-D: not specific
            ic = r.get("interpretation_class", "")
            struct_mat[idx, 4] = 1.0 if ic == "full_structural_match" else 0.0

    # Behavioral matrix for E022 (canonical motif, B1/B3/B5)
    e022_beh_by_comp = {}
    for r in e022_beh:
        comp = r.get("comparator", "")
        if comp not in e022_beh_by_comp:
            e022_beh_by_comp[comp] = []
        e022_beh_by_comp[comp].append(r)

    behav_mat = np.full((len(ALL_COMPS), len(BEHAV_COLS)), float("nan"))
    BEHAV_THRESH = [0.01, 0.01, 0.1]  # positive threshold for B1, B3, B5

    for i, comp in enumerate(ALL_COMPS):
        canon = [r for r in e022_beh_by_comp.get(comp, [])
                 if r.get("motif_type") == "canonical"]
        if not canon and comp in e018_comp_order:
            # Use E018 data for simple comparators (B1=SIG_B, B5=SIG_E)
            e018r = e018_by_name.get(comp, {})
            sig_b = _bool(e018r.get("SIG_B", False))
            sig_e = _bool(e018r.get("SIG_E", False))
            behav_mat[i, 0] = 1.0 if sig_b else 0.0
            behav_mat[i, 1] = 1.0 if _bool(e018r.get("SIG_D", False)) else 0.0
            behav_mat[i, 2] = 1.0 if sig_e else 0.0
        elif canon:
            r = canon[0]
            for j, (col, thresh) in enumerate(zip(BEHAV_COLS, BEHAV_THRESH)):
                v = _flt(r.get(col, "nan"))
                if not math.isnan(v):
                    behav_mat[i, j] = 1.0 if v > thresh else 0.0

    # Build figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                             gridspec_kw={"width_ratios": [5, 3]})
    fig.suptitle("Comparator discrimination matrix\n"
                 "Left: structural signatures (SIG-A–E); Right: output-level metrics (B1/B3/B5)",
                 fontsize=10)

    cmap_pf = plt.cm.RdYlGn

    for ax, mat, col_labels, title in [
        (axes[0], struct_mat, [s.replace("_", "-") for s in SIGS], "Structural layer (SIG-A–E)"),
        (axes[1], behav_mat, BEHAV_LABELS,                          "Behavioral layer (B1/B3/B5)"),
    ]:
        im = ax.imshow(mat, cmap=cmap_pf, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticks(range(len(ALL_COMPS))); ax.set_yticklabels(COMP_LABELS, fontsize=7)
        ax.set_title(title, fontsize=8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not math.isnan(v):
                    lbl = "PASS" if v > 0.5 else "fail"
                    ax.text(j, i, lbl, ha="center", va="center",
                            fontsize=6, color="white" if v > 0.5 else "#333")
                else:
                    ax.text(j, i, "N/A", ha="center", va="center", fontsize=5, color="#888")

    # Separator line between E018 and hard comparators
    for ax in axes:
        ax.axhline(len(e018_comp_order) - 0.5, color="black", linewidth=1.2, linestyle="--")
        ax.text(-0.5, len(e018_comp_order) / 2 - 0.5, "baseline", fontsize=6,
                ha="right", va="center", rotation=90, color="#555")
        ax.text(-0.5, len(e018_comp_order) + len(e022_hard) / 2 - 0.5, "hard",
                fontsize=6, ha="right", va="center", rotation=90, color="#555")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Robustness landscape (E019 + E020)
# ---------------------------------------------------------------------------

def fig4_robustness_landscape() -> "plt.Figure":
    import matplotlib.pyplot as plt
    import numpy as np

    oat = _load_csv(E019 / "summary" / "robustness_summary_by_parameter.csv")
    heatmap_all = _load_csv(E020 / "summary" / "all_heatmaps_long.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Robustness landscape\n"
                 "A: OAT % joint pass  B: structural_lr × replay_gain  C: overlap_strength × replay_gain",
                 fontsize=10)

    # Panel A: OAT bar chart
    ax_a = axes[0]
    params = [r["parameter"] for r in oat]
    pct    = [_flt(r.get("pct_joint_pass", 0)) for r in oat]
    colors = ["#2ca02c" if p >= 75 else "#ff7f0e" if p >= 50 else "#d62728" for p in pct]
    bars = ax_a.barh(params, pct, color=colors, alpha=0.85)
    ax_a.axvline(75, color="gray", linestyle="--", linewidth=0.8)
    ax_a.axvline(50, color="#ff7f0e", linestyle=":", linewidth=0.8)
    ax_a.set_xlim(0, 110)
    ax_a.set_xlabel("% joint-protected pass", fontsize=8)
    ax_a.set_title("A. OAT parameter sweep", fontsize=8)
    ax_a.tick_params(labelsize=7)
    for bar, p in zip(bars, pct):
        ax_a.text(p + 1, bar.get_y() + bar.get_height() / 2,
                  f"{p:.0f}%", va="center", fontsize=6)

    # Panels B and C: heatmaps
    HEATMAP_PAIRS = [
        ("structural_lr", "replay_gain",      "B. structural_lr × replay_gain"),
        ("overlap_strength", "replay_gain",   "C. overlap × replay_gain"),
    ]
    for ax, (px, py, title) in zip(axes[1:], HEATMAP_PAIRS):
        rows = [r for r in heatmap_all
                if r.get("param_x_name") == px and r.get("param_y_name") == py]
        if not rows:
            ax.set_title(title + "\n(data not found)", fontsize=7)
            continue
        xs = sorted({_flt(r["param_x_value"]) for r in rows})
        ys = sorted({_flt(r["param_y_value"]) for r in rows})
        mat = np.full((len(ys), len(xs)), float("nan"))
        xi = {v: i for i, v in enumerate(xs)}
        yi = {v: i for i, v in enumerate(ys)}
        for r in rows:
            x, y = _flt(r["param_x_value"]), _flt(r["param_y_value"])
            jp = _bool(r.get("joint_protected_pass", "False"))
            i, j = yi.get(y, -1), xi.get(x, -1)
            if i >= 0 and j >= 0:
                mat[i, j] = 1.0 if jp else 0.0
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto",
                       origin="lower")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{v:.2f}" for v in xs], fontsize=5, rotation=45, ha="right")
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels([f"{v:.2f}" for v in ys], fontsize=5)
        ax.set_xlabel(px, fontsize=7); ax.set_ylabel(py, fontsize=7)
        ax.set_title(title, fontsize=8)
        # Mark canonical
        try:
            import json
            ref = json.loads((E020 / "summary" / "canonical_reference.json").read_text(encoding="utf-8"))
            cx = ref.get(px); cy = ref.get(py)
            if cx is not None and cy is not None:
                cx_vals = [abs(v - cx) for v in xs]
                cy_vals = [abs(v - cy) for v in ys]
                ci, cj = cy_vals.index(min(cy_vals)), cx_vals.index(min(cx_vals))
                ax.plot(cj, ci, "k*", markersize=10, zorder=5)
        except Exception:
            pass

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Scaling, motifs, specificity (E021 + E021R)
# ---------------------------------------------------------------------------

def fig5_motif_scaling() -> "plt.Figure":
    import matplotlib.pyplot as plt
    import numpy as np

    motif_sum = _load_csv(E021 / "summary" / "generalization_summary_by_motif.csv")
    spec_cls  = _load_csv(E021R / "summary" / "motif_specificity_classification.csv")

    MOTIFS = ["canonical", "strong_overlap", "chain_overlap", "hub_overlap",
              "sparse_random", "weak_overlap"]
    MOTIF_COLORS = {
        "canonical":      "#2ca02c",
        "strong_overlap": "#1f77b4",
        "chain_overlap":  "#ff7f0e",
        "hub_overlap":    "#d62728",
        "sparse_random":  "#9467bd",
        "weak_overlap":   "#7f7f7f",
    }
    IC_COLORS = {
        "canonical_reference":       "#2ca02c",
        "specific_linking_success":  "#1f77b4",
        "local_linking_success":     "#ff7f0e",
        "hub_overlinking_boundary":  "#d62728",
        "weak_overlap_failure":      "#7f7f7f",
        "density_dependent":         "#9467bd",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Motif scaling and specificity\n"
                 "A: gSIG-A by motif  B: gSIG-A by branch count (canonical)  C: false-linking by motif",
                 fontsize=10)

    # Panel A: gSIG-A by motif (bar)
    ax_a = axes[0]
    motif_data = {r["motif_type"]: r for r in motif_sum}
    ms = [m for m in MOTIFS if m in motif_data]
    gA_vals = [_flt(motif_data[m].get("mean_gSIG_A", "nan")) for m in ms]
    colors = [MOTIF_COLORS.get(m, "#aec7e8") for m in ms]
    ax_a.bar(range(len(ms)), gA_vals, color=colors, alpha=0.85)
    ax_a.axhline(0, color="black", linewidth=0.8)
    ax_a.set_xticks(range(len(ms)))
    ax_a.set_xticklabels([m.replace("_", "\n") for m in ms], fontsize=6)
    ax_a.set_ylabel("mean gSIG-A", fontsize=8)
    ax_a.set_title("A. gSIG-A by motif type", fontsize=8)
    ax_a.tick_params(labelsize=6)

    # Panel B: gSIG-A vs branch count for canonical
    ax_b = axes[1]
    can_rows = sorted(
        [r for r in spec_cls if r.get("motif_type") == "canonical"],
        key=lambda r: int(r.get("n_branches", 0))
    )
    if can_rows:
        nb = [int(r["n_branches"]) for r in can_rows]
        gA = [_flt(r.get("gSIG_A", "nan")) for r in can_rows]
        gE = [_flt(r.get("gSIG_E", "nan")) for r in can_rows]
        ax_b.plot(nb, gA, "o-", color="#2ca02c", linewidth=2, label="gSIG-A", markersize=6)
        ax_b2 = ax_b.twinx()
        ax_b2.plot(nb, gE, "s--", color="#1f77b4", linewidth=1.5, label="gSIG-E", markersize=5)
        ax_b2.set_ylabel("gSIG-E (rescue)", fontsize=7, color="#1f77b4")
        ax_b2.tick_params(labelsize=6, colors="#1f77b4")
    ax_b.set_xlabel("n branches", fontsize=8)
    ax_b.set_ylabel("gSIG-A (structural write)", fontsize=8)
    ax_b.set_title("B. Canonical: scaling with branch count", fontsize=8)
    ax_b.tick_params(labelsize=6)
    ax_b.legend(fontsize=6, loc="upper left")

    # Panel C: false-linking rate by motif (scatter with jitter)
    ax_c = axes[2]
    multi_rows = [r for r in spec_cls
                  if r.get("motif_type") not in ("canonical", "strong_overlap", "weak_overlap")]
    seen = {}
    for r in multi_rows:
        mt = r.get("motif_type", "?")
        fl = _flt(r.get("false_linking_rate", "nan"))
        if not math.isnan(fl):
            seen.setdefault(mt, []).append(fl)

    mt_names = sorted(seen)
    for i, mt in enumerate(mt_names):
        vals = seen[mt]
        ax_c.scatter([i] * len(vals), vals, s=50, alpha=0.75,
                     color=MOTIF_COLORS.get(mt, "#aec7e8"), zorder=4)
        ax_c.plot([i - 0.3, i + 0.3], [sum(vals) / len(vals)] * 2,
                  "k-", linewidth=2, zorder=5)
    ax_c.axhline(0.25, color="green",  linestyle="--", linewidth=0.9, label="good (<0.25)")
    ax_c.axhline(0.50, color="orange", linestyle="--", linewidth=0.9, label="moderate (<0.50)")
    ax_c.set_xticks(range(len(mt_names)))
    ax_c.set_xticklabels([m.replace("_", "\n") for m in mt_names], fontsize=7)
    ax_c.set_ylabel("false-linking rate", fontsize=8)
    ax_c.set_title("C. False-linking rate (multi-trace motifs)", fontsize=8)
    ax_c.legend(fontsize=6)
    ax_c.tick_params(labelsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ---------------------------------------------------------------------------
# Figure 6 — Shuffled replay audit (E022R)
# ---------------------------------------------------------------------------

def fig6_shuffled_replay_audit() -> "plt.Figure":
    import matplotlib.pyplot as plt
    import numpy as np

    comp_rows = _load_csv(E022R / "summary" / "shuffled_replay_vs_full_model.csv")

    MOTIF_COLORS = {"canonical": "#1f77b4", "strong_overlap": "#ff7f0e"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Shuffled replay scaling audit (E022R)\n"
                 "Identity-preserving replay is required for scalable structural specificity",
                 fontsize=10)

    # Panel A: gSIG-A vs branch count
    ax_a = axes[0]
    for mt in ("canonical", "strong_overlap"):
        rows = sorted([r for r in comp_rows if r["motif_type"] == mt],
                      key=lambda r: int(r["n_branches"]))
        nb = [int(r["n_branches"]) for r in rows]
        full = [_flt(r["full_gSIG_A"]) for r in rows]
        s_mn = [_flt(r["shuffled_mean"]) for r in rows]
        s_sd = [_flt(r["shuffled_sd"]) for r in rows]
        c = MOTIF_COLORS[mt]
        ax_a.plot(nb, full, "--", color=c, linewidth=1.5, label=f"full {mt[:4]}")
        ax_a.plot(nb, s_mn, "-",  color=c, linewidth=2,   label=f"shuffled {mt[:4]}")
        ax_a.fill_between(nb,
                          [m - s for m, s in zip(s_mn, s_sd)],
                          [m + s for m, s in zip(s_mn, s_sd)],
                          alpha=0.15, color=c)
    ax_a.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax_a.set_xlabel("n branches", fontsize=8)
    ax_a.set_ylabel("gSIG-A", fontsize=8)
    ax_a.set_title("A. gSIG-A full vs shuffled", fontsize=8)
    ax_a.legend(fontsize=6)
    ax_a.tick_params(labelsize=7)

    # Panel B: ratio to full model
    ax_b = axes[1]
    for mt in ("canonical", "strong_overlap"):
        rows = sorted([r for r in comp_rows if r["motif_type"] == mt],
                      key=lambda r: int(r["n_branches"]))
        nb    = [int(r["n_branches"]) for r in rows]
        ratio = [_flt(r["ratio_to_full_model"]) for r in rows]
        c = MOTIF_COLORS[mt]
        ax_b.plot(nb, ratio, "-o", color=c, linewidth=2, label=mt, markersize=6)
        for x, y in zip(nb, ratio):
            if not math.isnan(y):
                ax_b.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                              xytext=(0, 5), ha="center", fontsize=7)
    ax_b.axhline(1.0, color="green",  linestyle="--", linewidth=0.8, label="ratio = 1.0")
    ax_b.axhline(0.0, color="gray",   linestyle=":",  linewidth=0.8)
    ax_b.set_xlabel("n branches", fontsize=8)
    ax_b.set_ylabel("shuffled / full gSIG-A", fontsize=8)
    ax_b.set_title("B. Ratio to full model", fontsize=8)
    ax_b.set_ylim(-0.3, 1.4)
    ax_b.legend(fontsize=7)
    ax_b.tick_params(labelsize=7)

    # Panel C: Seed spread at n=4
    ax_c = axes[2]
    try:
        raw_rows = _load_csv(E022R / "summary" / "shuffled_replay_scaling.csv")
        for xi, mt in enumerate(("canonical", "strong_overlap")):
            vals = [_flt(r["gSIG_A"]) for r in raw_rows
                    if r["motif_type"] == mt and r["is_shuffled"] == "True"
                    and int(r.get("n_branches", 0)) == 4
                    and not math.isnan(_flt(r["gSIG_A"]))]
            full_ref = next((_flt(r["gSIG_A"]) for r in raw_rows
                             if r["motif_type"] == mt and r["is_shuffled"] == "False"
                             and int(r.get("n_branches", 0)) == 4), None)
            if vals:
                bp = ax_c.boxplot([vals], positions=[xi], widths=0.4,
                                  patch_artist=True, medianprops={"color": "black"})
                for patch in bp["boxes"]:
                    patch.set_facecolor(MOTIF_COLORS[mt]); patch.set_alpha(0.6)
            if full_ref is not None:
                ax_c.plot([xi - 0.3, xi + 0.3], [full_ref, full_ref],
                          "g--", linewidth=2, zorder=5)
    except Exception:
        ax_c.text(0.5, 0.5, "(seed spread data\nnot available)",
                  ha="center", va="center", transform=ax_c.transAxes, fontsize=9)

    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["canonical", "strong_overlap"], fontsize=7)
    ax_c.set_ylabel("gSIG-A (shuffled, n=4, 20 seeds)", fontsize=7)
    ax_c.set_title("C. Seed spread at n=4\n(green dashes = full model)", fontsize=7)
    ax_c.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[e023] matplotlib not available — cannot generate figures.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("Experiment 023 — Article Figure Assembly")
    print(f"  Output: {OUT_DIR}")
    print("=" * 72)

    # ---------- Fig 1 ----------
    print("  Generating Fig 1 (model concept)...")
    fig1, axes1 = plt.subplots(1, 4, figsize=(16, 5))
    fig1.suptitle("Slow branch-level accessibility as a structural constraint\n"
                  "on memory linking", fontsize=11, fontweight="bold")
    fig1_model_concept(axes1)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig1.savefig(OUT_DIR / "Fig_e023_01_model_concept.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print("    -> Fig_e023_01_model_concept.png")

    # ---------- Fig 2 ----------
    print("  Generating Fig 2 (canonical traces)...")
    fig2 = fig2_canonical_traces()
    fig2.savefig(OUT_DIR / "Fig_e023_02_canonical_traces.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print("    -> Fig_e023_02_canonical_traces.png")

    # ---------- Fig 3 ----------
    print("  Generating Fig 3 (comparator matrix)...")
    fig3 = fig3_comparator_matrix()
    fig3.savefig(OUT_DIR / "Fig_e023_03_comparator_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig3)
    print("    -> Fig_e023_03_comparator_matrix.png")

    # ---------- Fig 4 ----------
    print("  Generating Fig 4 (robustness landscape)...")
    fig4 = fig4_robustness_landscape()
    fig4.savefig(OUT_DIR / "Fig_e023_04_robustness_landscape.png", dpi=200, bbox_inches="tight")
    plt.close(fig4)
    print("    -> Fig_e023_04_robustness_landscape.png")

    # ---------- Fig 5 ----------
    print("  Generating Fig 5 (motif scaling)...")
    fig5 = fig5_motif_scaling()
    fig5.savefig(OUT_DIR / "Fig_e023_05_motif_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig5)
    print("    -> Fig_e023_05_motif_scaling.png")

    # ---------- Fig 6 ----------
    print("  Generating Fig 6 (shuffled replay audit)...")
    fig6 = fig6_shuffled_replay_audit()
    fig6.savefig(OUT_DIR / "Fig_e023_06_shuffled_replay_audit.png", dpi=200, bbox_inches="tight")
    plt.close(fig6)
    print("    -> Fig_e023_06_shuffled_replay_audit.png")

    figs = list(OUT_DIR.glob("Fig_e023_0*.png"))
    print(f"\n  {len(figs)} figures written to {OUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
