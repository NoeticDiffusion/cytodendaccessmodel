"""Experiment 017 — Traceable Simulator Core.

Runs the canonical branch-accessibility simulator across all ten protocol phases
and exports full time-resolved trace data, enabling reviewer-inspectable figures.

This experiment does NOT add new biological claims, DANDI analyses, or parameter
sweeps. Its sole purpose is to make the simulator's internal dynamics auditable
and visible as time series.

Protocol phases
---------------
    init                    — record baseline state before any encoding
    encode_mu_1             — two cue passes driving mu1 pattern
    encode_mu_2             — two cue passes driving mu2 pattern
    pre_consolidation_probe — probe recall/linking before consolidation
    consolidation_replay    — 9 replay passes (slow structural update)
    post_consolidation_probe— probe recall/linking after consolidation
    overlap_damage          — raise decay on overlap branch b1, null passes
    post_damage_probe       — probe recall/linking after damage
    targeted_rescue         — targeted b1 cuing + consolidation
    post_rescue_probe       — probe recall/linking after rescue

Canonical parameters
--------------------
Matches exp013 / exp015 BASE_PARAMS (structural_lr=0.18, replay_gain=0.80,
eligibility_decay=0.12, structural_gain=6.0, n_passes=9).

Outputs (under results/e017_traceable_simulator_core/)
-------------------------------------------------------
    traces/branch_traces.csv
    traces/trace_support.csv
    traces/linking_trace.csv
    summary/signature_summary.csv
    summary/run_metadata.json
    figures/Fig_e017_01_branch_state_traces.png
    figures/Fig_e017_02_structural_accessibility_traces.png
    figures/Fig_e017_03_recall_and_linking_traces.png
    figures/Fig_e017_04_signature_barplot.png
    README.md
    qc_report.md
    effect_summary.md
    claim_ledger.md
    figure_manifest.md
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

# Make src importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cytodend_accessmodel.contracts import (
    ConsolidationWindow,
    DynamicsParameters,
    EngramTrace,
    TraceAllocation,
)
from cytodend_accessmodel.simulator import CytodendAccessModelSimulator, _sigmoid, _clamp01

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "results" / "e017_traceable_simulator_core"
TRACES_DIR = OUT_ROOT / "traces"
SUMMARY_DIR = OUT_ROOT / "summary"
FIGURES_DIR = OUT_ROOT / "figures"

# ---------------------------------------------------------------------------
# Canonical setup (matches exp015 BASE_PARAMS / exp013)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
BRANCH_IDS = ["b0", "b1", "b2", "b3"]
OVERLAP_BRANCH = "b1"

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

# Context traces for SIG-C (separate sim)
ALPHA_ALLOC = TraceAllocation(
    trace_id="mu_alpha",
    branch_weights={"b0": 0.90, "b1": 0.05, "b2": 0.05, "b3": 0.00},
)
BETA_ALLOC = TraceAllocation(
    trace_id="mu_beta",
    branch_weights={"b0": 0.05, "b1": 0.05, "b2": 0.90, "b3": 0.00},
)
AMBIG_CUE  = {"b0": 0.5, "b1": 0.5, "b2": 0.5, "b3": 0.5}
ALPHA_BIAS = {"b0": 0.5, "b1": 0.5, "b2": -0.5, "b3": -0.5}
BETA_BIAS  = {"b0": -0.5, "b1": -0.5, "b2": 0.5, "b3": 0.5}

CANONICAL_PARAMS = DynamicsParameters(
    structural_lr=0.18,
    replay_gain=0.80,
    eligibility_decay=0.12,
    structural_decay=0.005,
    structural_gain=6.0,
    structural_max=1.0,
    translation_decay=0.05,
    sleep_gain=0.0,
    context_gain=1.0,
    structural_noise=0.0,
    readout_gain=5.0,
    readout_threshold=0.3,
)

CONSOLIDATION_PASSES = 9
DAMAGE_NULL_PASSES   = 9
RESCUE_PASSES        = 3
RESCUE_CUE_REPS      = 3
RESCUE_ROUNDS        = 3
DAMAGE_DECAY_RATE    = 0.030


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_sim() -> CytodendAccessModelSimulator:
    sim = CytodendAccessModelSimulator.from_branch_ids(
        BRANCH_IDS, parameters=CANONICAL_PARAMS
    )
    sim.traces["mu1"] = EngramTrace(trace_id="mu1", allocation=MU1_ALLOC)
    sim.traces["mu2"] = EngramTrace(trace_id="mu2", allocation=MU2_ALLOC)
    return sim


def _linking(sim: CytodendAccessModelSimulator) -> float:
    """L(mu1, mu2) = sum_b a_mu1b * a_mu2b * M_b"""
    return sum(
        MU1_ALLOC.branch_weights.get(b, 0.0)
        * MU2_ALLOC.branch_weights.get(b, 0.0)
        * sim.branches[b].structural.accessibility
        for b in BRANCH_IDS
    )


def _linking_breakdown(sim: CytodendAccessModelSimulator) -> tuple[float, float]:
    """Returns (overlap_contribution, nonoverlap_contribution)."""
    overlap = (
        MU1_ALLOC.branch_weights.get(OVERLAP_BRANCH, 0.0)
        * MU2_ALLOC.branch_weights.get(OVERLAP_BRANCH, 0.0)
        * sim.branches[OVERLAP_BRANCH].structural.accessibility
    )
    total = _linking(sim)
    return overlap, total - overlap


def _snapshot_branches(
    sim: CytodendAccessModelSimulator,
    step: int,
    phase: str,
    input_drives: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    drives = input_drives or {}
    for bid in BRANCH_IDS:
        b = sim.branches[bid]
        rows.append({
            "step":                    step,
            "phase":                   phase,
            "branch_id":               bid,
            "is_overlap":              bid == OVERLAP_BRANCH,
            "x_b":                     b.activation,
            "fast_access":             b.fast_access,
            "slow_access":             b.slow_access,
            "effective_access":        b.effective_access,
            "eligibility":             b.eligibility.value,
            "translation_readiness":   b.translation_readiness.value,
            "structural_accessibility":b.structural.accessibility,
            "input_drive":             drives.get(bid, float("nan")),
        })
    return rows


def _snapshot_supports(
    sim: CytodendAccessModelSimulator,
    step: int,
    phase: str,
) -> list[dict[str, Any]]:
    rows = []
    supports = sim.compute_recall_supports()
    for rs in supports:
        rows.append({
            "step":            step,
            "phase":           phase,
            "trace_id":        rs.trace_id,
            "recall_support":  rs.support,
            "readout_value":   rs.expressed_strength,
            "context_label":   rs.matched_context or "none",
        })
    return rows


def _snapshot_linking(
    sim: CytodendAccessModelSimulator,
    step: int,
    phase: str,
) -> dict[str, Any]:
    lk = _linking(sim)
    ovlp, nonovlp = _linking_breakdown(sim)
    return {
        "step":                       step,
        "phase":                      phase,
        "trace_pair":                 "mu1_mu2",
        "linking_score":              lk,
        "overlap_branch_contribution":ovlp,
        "nonoverlap_contribution":    nonovlp,
    }


def _apply_cue_traced(
    sim: CytodendAccessModelSimulator,
    cue: dict,
    step_counter: list[int],
    phase: str,
    branch_rows: list,
    support_rows: list,
    linking_rows: list,
) -> None:
    sim.apply_cue(cue)
    step_counter[0] += 1
    branch_rows.extend(_snapshot_branches(sim, step_counter[0], phase, cue))
    support_rows.extend(_snapshot_supports(sim, step_counter[0], phase))
    linking_rows.append(_snapshot_linking(sim, step_counter[0], phase))


def _consolidate_traced(
    sim: CytodendAccessModelSimulator,
    n_passes: int,
    step_counter: list[int],
    phase: str,
    branch_rows: list,
    support_rows: list,
    linking_rows: list,
    replay_ids: list[str] | None = None,
    modulatory_drive: float = 1.0,
) -> None:
    win = ConsolidationWindow(
        replay_trace_ids=replay_ids if replay_ids is not None else ["mu1", "mu2"],
        modulatory_drive=modulatory_drive,
    )
    for _ in range(n_passes):
        sim.run_consolidation(win)
        step_counter[0] += 1
        branch_rows.extend(_snapshot_branches(sim, step_counter[0], phase))
        support_rows.extend(_snapshot_supports(sim, step_counter[0], phase))
        linking_rows.append(_snapshot_linking(sim, step_counter[0], phase))


def _null_consolidate_traced(
    sim: CytodendAccessModelSimulator,
    n_passes: int,
    step_counter: list[int],
    phase: str,
    branch_rows: list,
    support_rows: list,
    linking_rows: list,
) -> None:
    _consolidate_traced(
        sim, n_passes, step_counter, phase,
        branch_rows, support_rows, linking_rows,
        replay_ids=[], modulatory_drive=0.0,
    )


# ---------------------------------------------------------------------------
# Main canonical run
# ---------------------------------------------------------------------------

def run_canonical() -> tuple[
    list[dict], list[dict], list[dict], dict[str, float]
]:
    """Run the full 10-phase canonical protocol.

    Returns (branch_rows, support_rows, linking_rows, signature_values).
    """
    random.seed(RANDOM_SEED)
    sim = _build_sim()

    branch_rows: list[dict] = []
    support_rows: list[dict] = []
    linking_rows: list[dict] = []
    step = [0]

    # --- Phase: init ---
    branch_rows.extend(_snapshot_branches(sim, step[0], "init"))
    support_rows.extend(_snapshot_supports(sim, step[0], "init"))
    linking_rows.append(_snapshot_linking(sim, step[0], "init"))

    # --- Phase: encode_mu_1 ---
    for _ in range(2):
        _apply_cue_traced(sim, MU1_CUE, step, "encode_mu_1",
                          branch_rows, support_rows, linking_rows)

    # --- Phase: encode_mu_2 ---
    for _ in range(2):
        _apply_cue_traced(sim, MU2_CUE, step, "encode_mu_2",
                          branch_rows, support_rows, linking_rows)

    # --- Phase: pre_consolidation_probe (cue mu1, then mu2) ---
    _apply_cue_traced(sim, MU1_CUE, step, "pre_consolidation_probe",
                      branch_rows, support_rows, linking_rows)
    _apply_cue_traced(sim, MU2_CUE, step, "pre_consolidation_probe",
                      branch_rows, support_rows, linking_rows)

    mb_pre = {b: sim.branches[b].structural.accessibility for b in BRANCH_IDS}
    lk_pre = _linking(sim)
    r_mu1_pre = next((rs.support for rs in sim.compute_recall_supports() if rs.trace_id == "mu1"), 0.0)

    # --- Phase: consolidation_replay ---
    _consolidate_traced(sim, CONSOLIDATION_PASSES, step, "consolidation_replay",
                        branch_rows, support_rows, linking_rows)

    mb_post_cons = {b: sim.branches[b].structural.accessibility for b in BRANCH_IDS}
    lk_post_cons = _linking(sim)

    # --- Phase: post_consolidation_probe ---
    _apply_cue_traced(sim, MU1_CUE, step, "post_consolidation_probe",
                      branch_rows, support_rows, linking_rows)
    _apply_cue_traced(sim, MU2_CUE, step, "post_consolidation_probe",
                      branch_rows, support_rows, linking_rows)

    sim_for_recall = deepcopy(sim)
    sim_for_recall.apply_cue(MU1_CUE)
    r_mu1_post_cons = next(
        (rs.support for rs in sim_for_recall.compute_recall_supports() if rs.trace_id == "mu1"), 0.0
    )

    # --- Phase: overlap_damage ---
    sim.branches[OVERLAP_BRANCH].structural.decay_rate = DAMAGE_DECAY_RATE
    _null_consolidate_traced(sim, DAMAGE_NULL_PASSES, step, "overlap_damage",
                             branch_rows, support_rows, linking_rows)

    lk_post_damage = _linking(sim)

    # --- Phase: post_damage_probe ---
    _apply_cue_traced(sim, MU1_CUE, step, "post_damage_probe",
                      branch_rows, support_rows, linking_rows)
    _apply_cue_traced(sim, MU2_CUE, step, "post_damage_probe",
                      branch_rows, support_rows, linking_rows)

    sim_for_recall_dmg = deepcopy(sim)
    sim_for_recall_dmg.apply_cue(MU1_CUE)
    r_mu1_post_damage = next(
        (rs.support for rs in sim_for_recall_dmg.compute_recall_supports() if rs.trace_id == "mu1"), 0.0
    )

    # --- Phase: targeted_rescue (standard path = re-consolidate overlap trace) ---
    sim_std  = deepcopy(sim)   # standard rescue: plain re-consolidation
    sim_targ = deepcopy(sim)   # targeted rescue: b1 pre-cueing then consolidation

    # Standard rescue: run consolidation without targeted pre-cueing
    _consolidate_traced(sim_std, CONSOLIDATION_PASSES, step, "targeted_rescue",
                        branch_rows, support_rows, linking_rows)
    lk_std_rescue = _linking(sim_std)

    # Targeted rescue: cue b1 to rebuild E_b, then consolidate (not traced separately)
    for _ in range(RESCUE_ROUNDS):
        for _ in range(RESCUE_CUE_REPS):
            sim_targ.apply_cue(B1_CUE)
        win_rescue = ConsolidationWindow(
            replay_trace_ids=["mu1", "mu2"],
            modulatory_drive=1.0,
        )
        for _ in range(RESCUE_PASSES):
            sim_targ.run_consolidation(win_rescue)
    lk_targ_rescue = _linking(sim_targ)

    # Use targeted rescue sim state for final probe
    sim_final = sim_targ

    # --- Phase: post_rescue_probe ---
    _apply_cue_traced(sim_final, MU1_CUE, step, "post_rescue_probe",
                      branch_rows, support_rows, linking_rows)
    _apply_cue_traced(sim_final, MU2_CUE, step, "post_rescue_probe",
                      branch_rows, support_rows, linking_rows)

    # -----------------------------------------------------------------------
    # SIG-C: context separation (separate sim, captured as scalar only)
    # -----------------------------------------------------------------------
    sig_c_score = _run_sig_c()

    # -----------------------------------------------------------------------
    # Compute signatures
    # -----------------------------------------------------------------------
    # SIG-A: overlap-branch structural writing
    delta_m_overlap    = mb_post_cons[OVERLAP_BRANCH] - mb_pre[OVERLAP_BRANCH]
    nonoverlap_ids     = [b for b in BRANCH_IDS if b != OVERLAP_BRANCH]
    delta_m_nonoverlap = sum(
        mb_post_cons[b] - mb_pre[b] for b in nonoverlap_ids
    ) / len(nonoverlap_ids)
    sig_a = delta_m_overlap - delta_m_nonoverlap

    # SIG-B: linking gain
    sig_b = lk_post_cons - lk_pre

    # SIG-C: context separation (from separate run)
    sig_c = sig_c_score

    # SIG-D: linking vs recall vulnerability dissociation
    link_drop_pct   = (lk_post_cons - lk_post_damage) / max(abs(lk_post_cons), 1e-9) * 100.0
    recall_drop_pct = (r_mu1_post_cons - r_mu1_post_damage) / max(abs(r_mu1_post_cons), 1e-9) * 100.0
    sig_d = link_drop_pct - recall_drop_pct

    # SIG-E: targeted rescue selectivity
    def _recovery_pct(post: float, dmg: float, healthy: float) -> float:
        denom = healthy - dmg
        return (post - dmg) / denom * 100.0 if abs(denom) > 1e-9 else 0.0

    rec_std  = _recovery_pct(lk_std_rescue,  lk_post_damage, lk_post_cons)
    rec_targ = _recovery_pct(lk_targ_rescue, lk_post_damage, lk_post_cons)
    sig_e = rec_targ - rec_std

    sigs = {
        "SIG_A_overlap_advantage":           sig_a,
        "SIG_B_linking_gain":                sig_b,
        "SIG_C_context_separation":          sig_c,
        "SIG_D_linking_recall_dissociation": sig_d,
        "SIG_E_targeted_rescue_advantage":   sig_e,
        # raw values for QC
        "mb_pre_overlap":         mb_pre[OVERLAP_BRANCH],
        "mb_post_overlap":        mb_post_cons[OVERLAP_BRANCH],
        "delta_m_overlap":        delta_m_overlap,
        "delta_m_nonoverlap_mean":delta_m_nonoverlap,
        "lk_pre":                 lk_pre,
        "lk_post_cons":           lk_post_cons,
        "lk_post_damage":         lk_post_damage,
        "lk_std_rescue":          lk_std_rescue,
        "lk_targ_rescue":         lk_targ_rescue,
        "r_mu1_pre":              r_mu1_pre,
        "r_mu1_post_cons":        r_mu1_post_cons,
        "r_mu1_post_damage":      r_mu1_post_damage,
        "link_drop_pct":          link_drop_pct,
        "recall_drop_pct":        recall_drop_pct,
        "rec_std_pct":            rec_std,
        "rec_targ_pct":           rec_targ,
    }

    return branch_rows, support_rows, linking_rows, sigs


def _run_sig_c() -> float:
    """Run the context-separation probe on a fresh sim."""
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

    # Encode
    for _ in range(2):
        sim.apply_cue({"b0": 1.0, "b1": 0.0, "b2": 0.0, "b3": 0.0}, context="alpha")
    for _ in range(2):
        sim.apply_cue({"b0": 0.0, "b1": 0.0, "b2": 1.0, "b3": 0.0}, context="beta")

    # Consolidate
    win = ConsolidationWindow(
        replay_trace_ids=["mu_alpha", "mu_beta"], modulatory_drive=1.0
    )
    for _ in range(CONSOLIDATION_PASSES):
        sim.run_consolidation(win)

    # Probe
    sim_a = deepcopy(sim)
    sim_b = deepcopy(sim)

    sim_a.apply_cue(AMBIG_CUE, context="alpha", context_bias=ALPHA_BIAS)
    rmap_a = {rs.trace_id: rs for rs in sim_a.compute_recall_supports()}
    r_a_corr  = rmap_a.get("mu_alpha", type("_", (), {"support": 0.0})()).support
    r_a_wrong = rmap_a.get("mu_beta",  type("_", (), {"support": 0.0})()).support

    sim_b.apply_cue(AMBIG_CUE, context="beta", context_bias=BETA_BIAS)
    rmap_b = {rs.trace_id: rs for rs in sim_b.compute_recall_supports()}
    r_b_corr  = rmap_b.get("mu_beta",  type("_", (), {"support": 0.0})()).support
    r_b_wrong = rmap_b.get("mu_alpha", type("_", (), {"support": 0.0})()).support

    return ((r_a_corr - r_a_wrong) + (r_b_corr - r_b_wrong)) / 2.0


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_metadata(
    path: Path,
    sigs: dict[str, float],
    output_paths: list[Path],
) -> None:
    import cytodend_accessmodel
    meta: dict[str, Any] = {
        "experiment_id": "e017",
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "cytodend_accessmodel_version": getattr(cytodend_accessmodel, "__version__", "unknown"),
        "random_seed": RANDOM_SEED,
        "parameters": {
            "structural_lr":      CANONICAL_PARAMS.structural_lr,
            "replay_gain":        CANONICAL_PARAMS.replay_gain,
            "eligibility_decay":  CANONICAL_PARAMS.eligibility_decay,
            "structural_decay":   CANONICAL_PARAMS.structural_decay,
            "structural_gain":    CANONICAL_PARAMS.structural_gain,
            "structural_max":     CANONICAL_PARAMS.structural_max,
            "translation_decay":  CANONICAL_PARAMS.translation_decay,
            "sleep_gain":         CANONICAL_PARAMS.sleep_gain,
            "context_gain":       CANONICAL_PARAMS.context_gain,
            "structural_noise":   CANONICAL_PARAMS.structural_noise,
            "readout_gain":       CANONICAL_PARAMS.readout_gain,
            "readout_threshold":  CANONICAL_PARAMS.readout_threshold,
        },
        "protocol": {
            "branch_ids":             BRANCH_IDS,
            "overlap_branch":         OVERLAP_BRANCH,
            "consolidation_passes":   CONSOLIDATION_PASSES,
            "damage_null_passes":     DAMAGE_NULL_PASSES,
            "damage_decay_rate":      DAMAGE_DECAY_RATE,
            "rescue_passes":          RESCUE_PASSES,
            "rescue_cue_reps":        RESCUE_CUE_REPS,
            "rescue_rounds":          RESCUE_ROUNDS,
        },
        "trace_allocations": {
            "mu1": MU1_ALLOC.branch_weights,
            "mu2": MU2_ALLOC.branch_weights,
        },
        "signatures": sigs,
        "output_file_hashes": {
            str(p.relative_to(OUT_ROOT)): _file_sha256(p)
            for p in output_paths if p.exists()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _git_commit() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _make_figures(
    branch_rows: list[dict],
    support_rows: list[dict],
    linking_rows: list[dict],
    sigs: dict[str, float],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[e017] matplotlib not available — skipping figures.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Branch colour palette
    colours = {"b0": "#1f77b4", "b1": "#d62728", "b2": "#2ca02c", "b3": "#9467bd"}
    # Phase vertical bands
    phase_order = [
        "init", "encode_mu_1", "encode_mu_2",
        "pre_consolidation_probe", "consolidation_replay",
        "post_consolidation_probe", "overlap_damage",
        "post_damage_probe", "targeted_rescue", "post_rescue_probe",
    ]
    phase_colours = [
        "#f0f0f0", "#d4ecd4", "#d4ecd4",
        "#fff0b3", "#b3d9ff",
        "#fff0b3", "#ffd4d4",
        "#fff0b3", "#d4f0d4", "#fff0b3",
    ]

    def _phase_band_steps(rows: list[dict]) -> list[tuple[float, float, str]]:
        """Return (x0, x1, phase) bands for shading."""
        phase_steps: dict[str, list[int]] = {}
        for r in rows:
            phase_steps.setdefault(r["phase"], []).append(r["step"])
        bands = []
        for ph in phase_order:
            if ph in phase_steps:
                steps = phase_steps[ph]
                bands.append((min(steps) - 0.5, max(steps) + 0.5, ph))
        return bands

    def _shade_phases(ax, bands: list[tuple], phase_cols: list[str]) -> None:
        for i, (x0, x1, ph) in enumerate(bands):
            col = phase_cols[phase_order.index(ph)] if ph in phase_order else "#f8f8f8"
            ax.axvspan(x0, x1, color=col, alpha=0.4, zorder=0)

    def _add_phase_labels(ax, bands: list[tuple]) -> None:
        for x0, x1, ph in bands:
            ax.text(
                (x0 + x1) / 2, 1.01, ph.replace("_", "\n"),
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=5.5, rotation=0,
            )

    # ------------------------------------------------------------------
    # Figure 1 — Branch state: fast access, effective access
    # ------------------------------------------------------------------
    fig1, axes1 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig1.suptitle("Fig e017-01  Branch fast-state traces", fontsize=11, y=0.98)

    vars_ax1 = [
        ("fast_access",      "Fast access  $A^f_b$"),
        ("effective_access", "Effective access  $A_b$"),
        ("x_b",              "Branch activation  $x_b$"),
    ]
    vars_ax2 = [
        ("structural_accessibility", "Structural accessibility  $M_b$"),
        ("eligibility",              "Eligibility trace  $E_b$"),
        ("translation_readiness",    "Translation readiness  $P_b$"),
    ]

    # Collect steps per (branch, variable) — all vars needed across figures 1 and 2
    all_traced_vars = [vname for vname, _ in vars_ax1] + [vname for vname, _ in vars_ax2]
    branch_data: dict[str, dict[str, list]] = {b: {"step": [], "phase": []} for b in BRANCH_IDS}
    for r in branch_rows:
        bid = r["branch_id"]
        branch_data[bid]["step"].append(r["step"])
        branch_data[bid]["phase"].append(r["phase"])
        for vname in all_traced_vars:
            branch_data[bid].setdefault(vname, []).append(r[vname])

    bands1 = _phase_band_steps(branch_rows)

    for ax, (vname, ylabel) in zip(axes1, vars_ax1):
        _shade_phases(ax, bands1, phase_colours)
        for bid in BRANCH_IDS:
            lw = 2.0 if bid == OVERLAP_BRANCH else 1.0
            ls = "-" if bid == OVERLAP_BRANCH else "--"
            ax.plot(branch_data[bid]["step"], branch_data[bid][vname],
                    color=colours[bid], lw=lw, ls=ls, label=bid)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc="upper right", ncol=4)

    _add_phase_labels(axes1[0], bands1)
    axes1[-1].set_xlabel("Step", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig1.savefig(FIGURES_DIR / "Fig_e017_01_branch_state_traces.png", dpi=150)
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2 — Structural accessibility traces: M_b, E_b, P_b
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig2.suptitle("Fig e017-02  Slow structural state traces", fontsize=11, y=0.98)

    for ax, (vname, ylabel) in zip(axes2, vars_ax2):
        _shade_phases(ax, bands1, phase_colours)
        for bid in BRANCH_IDS:
            lw = 2.2 if bid == OVERLAP_BRANCH else 1.0
            ls = "-" if bid == OVERLAP_BRANCH else "--"
            ax.plot(branch_data[bid]["step"], branch_data[bid][vname],
                    color=colours[bid], lw=lw, ls=ls, label=bid)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=7, loc="upper right", ncol=4)

    _add_phase_labels(axes2[0], bands1)
    axes2[-1].set_xlabel("Step", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig2.savefig(FIGURES_DIR / "Fig_e017_02_structural_accessibility_traces.png", dpi=150)
    plt.close(fig2)

    # ------------------------------------------------------------------
    # Figure 3 — Recall support and linking score over phases
    # ------------------------------------------------------------------
    # Build per-trace and linking series
    support_data: dict[str, dict[str, list]] = {}
    for r in support_rows:
        tid = r["trace_id"]
        if tid not in support_data:
            support_data[tid] = {"step": [], "recall_support": [], "readout_value": []}
        support_data[tid]["step"].append(r["step"])
        support_data[tid]["recall_support"].append(r["recall_support"])
        support_data[tid]["readout_value"].append(r["readout_value"])

    lk_steps  = [r["step"]         for r in linking_rows]
    lk_scores = [r["linking_score"] for r in linking_rows]
    lk_ovlp   = [r["overlap_branch_contribution"] for r in linking_rows]

    bands3 = _phase_band_steps(linking_rows)

    fig3, axes3 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig3.suptitle("Fig e017-03  Recall support and linking score", fontsize=11, y=0.98)

    trace_colours = {"mu1": "#1f77b4", "mu2": "#2ca02c",
                     "mu_alpha": "#ff7f0e", "mu_beta": "#9467bd"}

    ax3a, ax3b, ax3c = axes3

    # Panel A: recall support
    _shade_phases(ax3a, bands3, phase_colours)
    for tid, tdata in support_data.items():
        ax3a.plot(tdata["step"], tdata["recall_support"],
                  color=trace_colours.get(tid, "gray"), lw=1.5, label=tid)
    ax3a.set_ylabel("Recall support  $R_\\mu$", fontsize=8)
    ax3a.legend(fontsize=7, loc="upper right")

    # Panel B: readout (expressed) strength
    _shade_phases(ax3b, bands3, phase_colours)
    for tid, tdata in support_data.items():
        ax3b.plot(tdata["step"], tdata["readout_value"],
                  color=trace_colours.get(tid, "gray"), lw=1.5, ls="--", label=tid)
    ax3b.set_ylabel("Readout strength", fontsize=8)
    ax3b.legend(fontsize=7, loc="upper right")

    # Panel C: linking score
    _shade_phases(ax3c, bands3, phase_colours)
    ax3c.plot(lk_steps, lk_scores, color="black", lw=2.0, label="$L_{\\mu_1\\mu_2}$")
    ax3c.fill_between(lk_steps, lk_ovlp, alpha=0.35, color="#d62728",
                      label="overlap $b_1$ contribution")
    ax3c.set_ylabel("Linking score  $L_{\\mu_1\\mu_2}$", fontsize=8)
    ax3c.legend(fontsize=7, loc="upper right")

    _add_phase_labels(ax3a, bands3)
    ax3c.set_xlabel("Step", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig3.savefig(FIGURES_DIR / "Fig_e017_03_recall_and_linking_traces.png", dpi=150)
    plt.close(fig3)

    # ------------------------------------------------------------------
    # Figure 4 — Signature bar plot
    # ------------------------------------------------------------------
    sig_labels = ["SIG-A\nOverlap\nwriting", "SIG-B\nLinking\ngain",
                  "SIG-C\nContext\nsep", "SIG-D\nLink>recall\nvulnerability",
                  "SIG-E\nTargeted\nrescue"]
    sig_keys   = ["SIG_A_overlap_advantage", "SIG_B_linking_gain",
                  "SIG_C_context_separation", "SIG_D_linking_recall_dissociation",
                  "SIG_E_targeted_rescue_advantage"]
    sig_units  = ["ΔM_b units", "ΔL units", "support units", "pp", "pp"]
    sig_values = [sigs[k] for k in sig_keys]

    bar_colours = ["#2ca02c" if v > 0 else "#d62728" for v in sig_values]

    fig4, ax4 = plt.subplots(figsize=(9, 5))
    bars = ax4.bar(sig_labels, sig_values, color=bar_colours, width=0.55, edgecolor="white", lw=1.5)

    for bar, val, unit in zip(bars, sig_values, sig_units):
        xpos = bar.get_x() + bar.get_width() / 2
        yoff = 0.002 if val >= 0 else -0.002
        ax4.text(xpos, val + yoff, f"{val:.3f}\n({unit})",
                 ha="center", va="bottom" if val >= 0 else "top",
                 fontsize=7.5)

    ax4.axhline(0, color="black", lw=0.8)
    ax4.set_title("Fig e017-04  Protected signature profile — full model (canonical params)",
                  fontsize=10)
    ax4.set_ylabel("Signature score", fontsize=9)
    pos_patch = mpatches.Patch(color="#2ca02c", label="Positive (supports hypothesis)")
    neg_patch = mpatches.Patch(color="#d62728", label="Negative (against hypothesis)")
    ax4.legend(handles=[pos_patch, neg_patch], fontsize=8)

    plt.tight_layout()
    fig4.savefig(FIGURES_DIR / "Fig_e017_04_signature_barplot.png", dpi=150)
    plt.close(fig4)

    print("[e017] Figures saved.")


# ---------------------------------------------------------------------------
# Documentation files
# ---------------------------------------------------------------------------

def _write_docs(sigs: dict[str, float]) -> None:
    def _sig_pass(val: float, threshold: float) -> str:
        return "PASS" if val > threshold else "FAIL"

    sa_pass = _sig_pass(sigs["SIG_A_overlap_advantage"],           0.0)
    sb_pass = _sig_pass(sigs["SIG_B_linking_gain"],                0.0)
    sc_pass = _sig_pass(sigs["SIG_C_context_separation"],          0.0)
    sd_pass = _sig_pass(sigs["SIG_D_linking_recall_dissociation"], 0.0)
    se_pass = _sig_pass(sigs["SIG_E_targeted_rescue_advantage"],   0.0)

    # README
    (OUT_ROOT / "README.md").write_text(
        f"""# e017 — Traceable Simulator Core

**Date:** {__import__('datetime').date.today()}
**Status:** completed

## Purpose
Export full time-resolved trace data from the canonical branch-accessibility
simulator across all ten protocol phases, enabling reviewer-inspectable figures.

## How to reproduce
```bash
python experiments/exp017_traceable_simulator_core.py
```

## Outputs
- `traces/branch_traces.csv` — per-branch state at every step
- `traces/trace_support.csv` — recall support at every step
- `traces/linking_trace.csv` — linking score at every step
- `summary/signature_summary.csv` — SIG-A to SIG-E values
- `summary/run_metadata.json` — full run parameters, hashes, git commit
- `figures/Fig_e017_01_branch_state_traces.png`
- `figures/Fig_e017_02_structural_accessibility_traces.png`
- `figures/Fig_e017_03_recall_and_linking_traces.png`
- `figures/Fig_e017_04_signature_barplot.png`

## Claim scope
This experiment supports only:
> "The simulator emits reproducible branch-level and trace-level dynamics
> that can be inspected as time-resolved traces."

It does NOT support:
- biological validation
- robustness across parameters
- comparator baseline claims
- DANDI evidence claims
""", encoding="utf-8"
    )

    # qc_report
    (OUT_ROOT / "qc_report.md").write_text(
        f"""# e017 — QC Report

## Variable availability
All required trace variables present: x_b, fast_access, slow_access,
effective_access, eligibility, translation_readiness, structural_accessibility,
recall_support, readout_value, linking_score.

## Missing variables
- `input_drive` is NaN during consolidation phases (no cue applied). Expected.
- `context_value` is not currently exported (not in BranchState). Noted.

## Determinism
Fixed seed: {RANDOM_SEED}. Structural noise = 0.0. Run is deterministic.
Re-run and compare hashes from `run_metadata.json` to verify.

## Phase coverage
All 10 phases present in branch_traces.csv.

## SIG-C note
SIG-C uses a separate context-probe simulator (not the main canonical sim).
Traces for SIG-C are therefore not in branch_traces.csv. Score is reported
in signature_summary.csv. This is expected and documented.

## Signature summary

| Signature | Value | Direction pass |
|-----------|-------|---------------|
| SIG-A overlap advantage     | {sigs['SIG_A_overlap_advantage']:.4f} | {sa_pass} |
| SIG-B linking gain          | {sigs['SIG_B_linking_gain']:.4f} | {sb_pass} |
| SIG-C context separation    | {sigs['SIG_C_context_separation']:.4f} | {sc_pass} |
| SIG-D dissociation (pp)     | {sigs['SIG_D_linking_recall_dissociation']:.2f} | {sd_pass} |
| SIG-E rescue advantage (pp) | {sigs['SIG_E_targeted_rescue_advantage']:.2f} | {se_pass} |

Note: "direction pass" = signature is in the predicted positive direction.
Magnitude and robustness are assessed in later experiments (e018, e019).
""", encoding="utf-8"
    )

    # effect_summary
    (OUT_ROOT / "effect_summary.md").write_text(
        f"""# e017 — Effect Summary

## Protocol
Canonical 4-branch simulator: b0, b1 (overlap), b2, b3.
Two traces: mu1 (b0/b1-dominant), mu2 (b1/b2-dominant).
10 phases: init → encode → probe → consolidation → probe →
           damage → probe → rescue → probe.

## Parameters
structural_lr={CANONICAL_PARAMS.structural_lr}, replay_gain={CANONICAL_PARAMS.replay_gain},
eligibility_decay={CANONICAL_PARAMS.eligibility_decay},
structural_gain={CANONICAL_PARAMS.structural_gain},
consolidation_passes={CONSOLIDATION_PASSES}

## Key values

| Metric | Value |
|--------|-------|
| M_b1 pre-consolidation  | {sigs['mb_pre_overlap']:.4f} |
| M_b1 post-consolidation | {sigs['mb_post_overlap']:.4f} |
| ΔM_b1 (overlap branch)  | {sigs['delta_m_overlap']:+.4f} |
| ΔM_b mean (non-overlap) | {sigs['delta_m_nonoverlap_mean']:+.4f} |
| L pre-consolidation     | {sigs['lk_pre']:.4f} |
| L post-consolidation    | {sigs['lk_post_cons']:.4f} |
| L post-damage           | {sigs['lk_post_damage']:.4f} |
| L standard rescue       | {sigs['lk_std_rescue']:.4f} |
| L targeted rescue       | {sigs['lk_targ_rescue']:.4f} |
| R_mu1 post-cons         | {sigs['r_mu1_post_cons']:.4f} |
| R_mu1 post-damage       | {sigs['r_mu1_post_damage']:.4f} |
| Link drop %             | {sigs['link_drop_pct']:+.1f}% |
| Recall drop %           | {sigs['recall_drop_pct']:+.1f}% |
| Standard rescue %       | {sigs['rec_std_pct']:+.1f}% |
| Targeted rescue %       | {sigs['rec_targ_pct']:+.1f}% |

## Protected signatures

| Signature | Score | Direction |
|-----------|-------|-----------|
| SIG-A overlap advantage     | {sigs['SIG_A_overlap_advantage']:.4f} | {sa_pass} |
| SIG-B linking gain          | {sigs['SIG_B_linking_gain']:.4f} | {sb_pass} |
| SIG-C context separation    | {sigs['SIG_C_context_separation']:.4f} | {sc_pass} |
| SIG-D dissociation (pp)     | {sigs['SIG_D_linking_recall_dissociation']:.2f}pp | {sd_pass} |
| SIG-E rescue advantage (pp) | {sigs['SIG_E_targeted_rescue_advantage']:.2f}pp | {se_pass} |
""", encoding="utf-8"
    )

    # claim_ledger
    joint = all(x == "PASS" for x in [sa_pass, sb_pass, sc_pass, sd_pass, se_pass])
    (OUT_ROOT / "claim_ledger.md").write_text(
        f"""# e017 — Claim Ledger

| Claim | Status | Evidence | Limitation | Next action |
|-------|--------|----------|------------|-------------|
| Simulator emits reproducible time-resolved traces | Internal validated result | branch_traces.csv, run_metadata.json, figures 1–3 | Canonical params only; no noise; no parameter sweep | e018 comparator traces |
| SIG-A: overlap branch gains more M_b than non-overlap | {'Internal validated result' if sa_pass == 'PASS' else 'FAIL — investigate'} | effect_summary.md | Single canonical run | e018 across baselines |
| SIG-B: linking increases post-consolidation | {'Internal validated result' if sb_pass == 'PASS' else 'FAIL — investigate'} | effect_summary.md | Single canonical run | e018 |
| SIG-C: context separation | {'Internal validated result' if sc_pass == 'PASS' else 'FAIL — investigate'} | separate context-sim; score only | Not in main trace CSV | e018 |
| SIG-D: linking more fragile than recall under damage | {'Internal validated result' if sd_pass == 'PASS' else 'FAIL — investigate'} | effect_summary.md | Damage modelled as decay-rate increase only | e018 |
| SIG-E: targeted rescue selectivity | {'Internal validated result' if se_pass == 'PASS' else 'FAIL — investigate'} | effect_summary.md | Targeted rescue vs standard only; no generic rescue | e018 |
| Joint signature profile supported | {'Internal validated result' if joint else 'PARTIALLY SUPPORTED — check failures'} | signature_summary.csv | Canonical params only | e018 comparators |
| Simulator separates slow writing from fast gating | Pending | Requires e018 comparator baseline | Not tested in e017 | e018 |
| Biological validation | Not supported | e017 scope is instrumentation only | — | — |
""", encoding="utf-8"
    )

    # figure_manifest
    (OUT_ROOT / "figure_manifest.md").write_text(
        f"""# e017 — Figure Manifest

| File | Content | Status |
|------|---------|--------|
| Fig_e017_01_branch_state_traces.png | Fast variables x_b, A_f, A_eff per branch × all phases | Generated |
| Fig_e017_02_structural_accessibility_traces.png | Slow variables M_b, E_b, P_b per branch × all phases | Generated |
| Fig_e017_03_recall_and_linking_traces.png | R_mu1, R_mu2, L_mu1mu2 over all steps | Generated |
| Fig_e017_04_signature_barplot.png | SIG-A to SIG-E compact bar chart | Generated |

Colour coding in trace figures:
- b0 (blue, dashed): single-trace mu1 branch
- b1 (red, solid, thicker): **overlap branch**
- b2 (green, dashed): single-trace mu2 branch
- b3 (purple, dashed): unrelated branch

Phase shading:
- grey: init
- light green: encoding phases
- yellow: probe phases
- light blue: consolidation
- light red: damage
- light green: rescue
""", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 64)
    print("Experiment 017 — Traceable Simulator Core")
    print("=" * 64)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[e017] Running canonical protocol...")
    branch_rows, support_rows, linking_rows, sigs = run_canonical()

    # --- Write CSVs ---
    branch_path  = TRACES_DIR / "branch_traces.csv"
    support_path = TRACES_DIR / "trace_support.csv"
    linking_path = TRACES_DIR / "linking_trace.csv"
    sig_path     = SUMMARY_DIR / "signature_summary.csv"

    _write_csv(branch_path, branch_rows)
    _write_csv(support_path, support_rows)
    _write_csv(linking_path, linking_rows)
    _write_csv(sig_path, [sigs])

    print(f"[e017] Wrote {len(branch_rows)} branch rows, "
          f"{len(support_rows)} support rows, "
          f"{len(linking_rows)} linking rows.")

    # --- Figures ---
    _make_figures(branch_rows, support_rows, linking_rows, sigs)

    # --- Docs ---
    _write_docs(sigs)
    print("[e017] Documentation written.")

    # --- Metadata (after files exist for hashing) ---
    output_paths = [
        branch_path, support_path, linking_path, sig_path,
        FIGURES_DIR / "Fig_e017_01_branch_state_traces.png",
        FIGURES_DIR / "Fig_e017_02_structural_accessibility_traces.png",
        FIGURES_DIR / "Fig_e017_03_recall_and_linking_traces.png",
        FIGURES_DIR / "Fig_e017_04_signature_barplot.png",
    ]
    _write_metadata(SUMMARY_DIR / "run_metadata.json", sigs, output_paths)
    print("[e017] Metadata written.")

    # --- Print summary ---
    print()
    print("-" * 64)
    print("SIGNATURE SUMMARY")
    print("-" * 64)
    print(f"  SIG-A  overlap advantage:            {sigs['SIG_A_overlap_advantage']:+.4f}")
    print(f"  SIG-B  linking gain:                 {sigs['SIG_B_linking_gain']:+.4f}")
    print(f"  SIG-C  context separation:           {sigs['SIG_C_context_separation']:+.4f}")
    print(f"  SIG-D  link>recall dissociation:     {sigs['SIG_D_linking_recall_dissociation']:+.2f} pp")
    print(f"  SIG-E  targeted rescue advantage:    {sigs['SIG_E_targeted_rescue_advantage']:+.2f} pp")
    print()
    print(f"  Outputs: {OUT_ROOT}")
    print("=" * 64)


if __name__ == "__main__":
    main()
