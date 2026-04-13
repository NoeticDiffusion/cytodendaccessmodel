"""Experiment 004: Robustness over seeds and parameters.

Reruns the three core metrics from experiments 001-003 across:
  - 6 swept parameters (one-at-a-time, 5 values each)
  - 10 stochastic seeds per (parameter, value) combination

A small background structural noise (noise_scale=0.005) is active for all
runs so that seeds produce genuinely different trajectories.

Core metrics tested:
  M1  overlap_advantage   [exp001]  b1_delta_M_b > b3_delta_M_b
  M2  linking_growth      [exp001]  L_post > L_pre
  M3  context_separation  [exp002]  R_correct > R_wrong
  M4  context_gap_widens  [exp002]  gap_post > gap_pre
  M5  timing_effect       [exp003]  b0_M_b(immediate) > b0_M_b(spaced)
  M6  replay_required     [exp003]  L(joint) > L(no_replay)

Acceptance criterion (from Diary 004):
  Sign and ordering of main effects must remain stable across seeds
  and local parameter sweeps, even if exact magnitudes vary.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Callable

from cytodend_accessmodel import (
    ConsolidationWindow,
    CytodendAccessModelSimulator,
    DynamicsParameters,
    EngramTrace,
    TraceAllocation,
)

# ---------------------------------------------------------------------------
# Shared topology (same as exp001 / exp003)
# ---------------------------------------------------------------------------
BRANCH_IDS = ["b0", "b1", "b2", "b3"]

MU1_ALLOC = TraceAllocation("mu1", {"b0": 0.90, "b1": 0.85, "b2": 0.05, "b3": 0.05})
MU2_ALLOC = TraceAllocation("mu2", {"b0": 0.05, "b1": 0.85, "b2": 0.90, "b3": 0.05})
ALPHA_ALLOC = TraceAllocation("mu_alpha", {"b0": 0.90, "b1": 0.80, "b2": 0.05, "b3": 0.05})
BETA_ALLOC  = TraceAllocation("mu_beta",  {"b0": 0.05, "b1": 0.05, "b2": 0.80, "b3": 0.90})

MU1_CUE      = {"b0": 1.0,  "b1": 0.9,  "b2": 0.05, "b3": 0.0}
MU2_CUE      = {"b0": 0.05, "b1": 0.9,  "b2": 1.0,  "b3": 0.0}
ALPHA_CUE    = {"b0": 1.0,  "b1": 0.9,  "b2": 0.05, "b3": 0.0}
BETA_CUE     = {"b0": 0.0,  "b1": 0.05, "b2": 0.9,  "b3": 1.0}
PARTIAL_CUE  = {"b0": 0.4,  "b1": 0.4,  "b2": 0.4,  "b3": 0.4}
NULL_CUE     = {b: 0.0 for b in BRANCH_IDS}

N_SEEDS       = 10
N_CONSOLIDATE = 3

# Baseline parameters (noise active so seeds differ)
DEFAULTS = DynamicsParameters(
    fast_gain=2.0, structural_gain=2.0,
    eligibility_decay=0.10, translation_decay=0.05,
    structural_lr=0.20, structural_decay=0.005, structural_max=1.0,
    replay_gain=1.2, sleep_gain=0.8,
    readout_gain=4.0, readout_threshold=0.4,
    context_mismatch_penalty=0.35,
    structural_noise=0.005,
)


# ---------------------------------------------------------------------------
# Simulator factory helpers
# ---------------------------------------------------------------------------

def _sim_exp001(params: DynamicsParameters) -> CytodendAccessModelSimulator:
    sim = CytodendAccessModelSimulator.from_branch_ids(BRANCH_IDS, spines_per_branch=3, parameters=params)
    sim.add_trace(EngramTrace("mu1", MU1_ALLOC))
    sim.add_trace(EngramTrace("mu2", MU2_ALLOC))
    return sim


def _consolidate(sim: CytodendAccessModelSimulator, replay_ids: tuple[str, ...], modulatory_drive: float = 1.0) -> None:
    for i in range(N_CONSOLIDATE):
        sim.run_consolidation(ConsolidationWindow(
            window_id=f"s{i}", replay_trace_ids=replay_ids, modulatory_drive=modulatory_drive,
        ))


def _linking(sim: CytodendAccessModelSimulator) -> float:
    return sum(
        MU1_ALLOC.branch_weights.get(b, 0) * MU2_ALLOC.branch_weights.get(b, 0)
        * sim.branches[b].structural.accessibility
        for b in BRANCH_IDS
    )


# ---------------------------------------------------------------------------
# Metric functions (return bool)
# ---------------------------------------------------------------------------

def metric_overlap_advantage(params: DynamicsParameters, **_) -> bool:
    """M1: b1_delta_M_b > b3_delta_M_b after consolidation."""
    sim = _sim_exp001(params)
    pre = {b: sim.branches[b].structural.accessibility for b in BRANCH_IDS}
    for _ in range(2): sim.apply_cue(MU1_CUE)
    for _ in range(2): sim.apply_cue(MU2_CUE)
    _consolidate(sim, ("mu1", "mu2"))
    delta = {b: sim.branches[b].structural.accessibility - pre[b] for b in BRANCH_IDS}
    return delta["b1"] > delta["b3"]


def metric_linking_growth(params: DynamicsParameters, **_) -> bool:
    """M2: L_post > L_pre."""
    sim = _sim_exp001(params)
    for _ in range(2): sim.apply_cue(MU1_CUE)
    for _ in range(2): sim.apply_cue(MU2_CUE)
    sim.apply_cue(MU1_CUE); sim.apply_cue(MU2_CUE)  # pre-consolidation recall
    l_pre = _linking(sim)
    _consolidate(sim, ("mu1", "mu2"))
    l_post = _linking(sim)
    return l_post > l_pre


def metric_context_separation(params: DynamicsParameters, ctx_bias_scale: float = 1.0, **_) -> bool:
    """M3: R_correct > R_wrong after encoding in context."""
    ab = {"b0": 0.7 * ctx_bias_scale, "b1": 0.6 * ctx_bias_scale, "b2": 0.0, "b3": 0.0}
    bb = {"b0": 0.0, "b1": 0.0, "b2": 0.6 * ctx_bias_scale, "b3": 0.7 * ctx_bias_scale}
    sim = CytodendAccessModelSimulator.from_branch_ids(BRANCH_IDS, spines_per_branch=3, parameters=params)
    sim.add_trace(EngramTrace("mu_alpha", ALPHA_ALLOC, context="alpha"))
    sim.add_trace(EngramTrace("mu_beta",  BETA_ALLOC,  context="beta"))
    for _ in range(3): sim.apply_cue(ALPHA_CUE, context="alpha", context_bias=ab)
    for _ in range(3): sim.apply_cue(BETA_CUE,  context="beta",  context_bias=bb)
    _consolidate(sim, ("mu_alpha", "mu_beta"))
    sim.apply_cue(PARTIAL_CUE, context="alpha", context_bias=ab)
    supports = {rs.trace_id: rs.support for rs in sim.compute_recall_supports()}
    return supports.get("mu_alpha", 0) > supports.get("mu_beta", 0)


def metric_context_gap_widens(params: DynamicsParameters, ctx_bias_scale: float = 1.0, **_) -> bool:
    """M4: consolidation widens the correct-vs-wrong context gap."""
    ab = {"b0": 0.7 * ctx_bias_scale, "b1": 0.6 * ctx_bias_scale, "b2": 0.0, "b3": 0.0}
    bb = {"b0": 0.0, "b1": 0.0, "b2": 0.6 * ctx_bias_scale, "b3": 0.7 * ctx_bias_scale}
    sim = CytodendAccessModelSimulator.from_branch_ids(BRANCH_IDS, spines_per_branch=3, parameters=params)
    sim.add_trace(EngramTrace("mu_alpha", ALPHA_ALLOC, context="alpha"))
    sim.add_trace(EngramTrace("mu_beta",  BETA_ALLOC,  context="beta"))
    for _ in range(3): sim.apply_cue(ALPHA_CUE, context="alpha", context_bias=ab)
    for _ in range(3): sim.apply_cue(BETA_CUE,  context="beta",  context_bias=bb)

    def _gap(s):
        s.apply_cue(PARTIAL_CUE, context="alpha", context_bias=ab)
        sup = {rs.trace_id: rs.support for rs in s.compute_recall_supports()}
        return sup.get("mu_alpha", 0) - sup.get("mu_beta", 0)

    gap_pre = _gap(sim)
    _consolidate(sim, ("mu_alpha", "mu_beta"))
    gap_post = _gap(sim)
    return gap_post > gap_pre


def metric_timing_effect(params: DynamicsParameters, gap: int = 12, **_) -> bool:
    """M5: b0 M_b is higher with immediate encoding than with gap encoding."""
    def _run(g: int) -> float:
        s = _sim_exp001(params)
        for _ in range(2): s.apply_cue(MU1_CUE)
        for _ in range(g): s.apply_cue(NULL_CUE)
        for _ in range(2): s.apply_cue(MU2_CUE)
        _consolidate(s, ("mu1", "mu2"))
        return s.branches["b0"].structural.accessibility

    return _run(0) > _run(gap)


def metric_replay_required(params: DynamicsParameters, **_) -> bool:
    """M6: L(joint replay) > L(modulatory_drive=0)."""
    def _run(mod_drive: float) -> float:
        s = _sim_exp001(params)
        for _ in range(2): s.apply_cue(MU1_CUE)
        for _ in range(2): s.apply_cue(MU2_CUE)
        _consolidate(s, ("mu1", "mu2"), modulatory_drive=mod_drive)
        return _linking(s)

    return _run(1.0) > _run(0.0)


# ---------------------------------------------------------------------------
# Sweep engine
# ---------------------------------------------------------------------------

MetricFn = Callable[..., bool]

METRICS: list[tuple[str, MetricFn]] = [
    ("M1 overlap_advantage",  metric_overlap_advantage),
    ("M2 linking_growth",     metric_linking_growth),
    ("M3 context_separation", metric_context_separation),
    ("M4 context_gap_widens", metric_context_gap_widens),
    ("M5 timing_effect",      metric_timing_effect),
    ("M6 replay_required",    metric_replay_required),
]

SWEEP_DEFS: list[tuple[str, list, dict]] = [
    # (param_name, values, extra_kwargs)
    ("structural_lr",     [0.08, 0.12, 0.20, 0.28, 0.35], {}),
    ("replay_gain",       [0.5,  0.8,  1.2,  1.6,  2.0],  {}),
    ("eligibility_decay", [0.05, 0.08, 0.10, 0.15, 0.20], {}),
    ("structural_noise",  [0.0,  0.005, 0.010, 0.015, 0.020], {}),
    ("gap",               [0,    4,    8,    12,   16],    {"gap": None}),   # exp003-specific
    ("ctx_bias_scale",    [0.3,  0.5,  0.7,  0.9,  1.1],  {"ctx_bias_scale": None}),  # exp002-specific
]


def sweep_one(
    metric_fn: MetricFn,
    param_name: str,
    values: list,
    extra_kwargs: dict,
) -> list[float]:
    """Run metric over param values; return pass rates (0–1)."""
    pass_rates: list[float] = []
    for val in values:
        # Build params (structural-noise is set via DynamicsParameters)
        if param_name == "structural_noise":
            params = replace(DEFAULTS, structural_noise=val)
        elif param_name in ("gap", "ctx_bias_scale"):
            params = DEFAULTS  # these are kwargs, not DynamicsParameters fields
        else:
            params = replace(DEFAULTS, **{param_name: val})

        kwargs: dict = {}
        if "gap" in extra_kwargs:
            kwargs["gap"] = val
        if "ctx_bias_scale" in extra_kwargs:
            kwargs["ctx_bias_scale"] = val

        passes = 0
        for seed in range(N_SEEDS):
            random.seed(seed)
            try:
                if metric_fn(params, **kwargs):
                    passes += 1
            except Exception:
                pass  # count as failure
        pass_rates.append(passes / N_SEEDS)
    return pass_rates


# ---------------------------------------------------------------------------
# Metric relevance filter (some sweeps only meaningful for specific metrics)
# ---------------------------------------------------------------------------

_METRIC_PARAM_RELEVANCE: dict[str, set[str]] = {
    "gap":           {"M5 timing_effect"},
    "ctx_bias_scale":{"M3 context_separation", "M4 context_gap_widens"},
}

def _is_relevant(metric_name: str, param_name: str) -> bool:
    if param_name in _METRIC_PARAM_RELEVANCE:
        return metric_name in _METRIC_PARAM_RELEVANCE[param_name]
    return True  # global params relevant to all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    print("=" * 78)
    print("Experiment 004: Robustness Over Seeds and Parameters")
    print(f"  {N_SEEDS} seeds × 5 values × {len(SWEEP_DEFS)} params × {len(METRICS)} metrics")
    print("=" * 78)

    all_results: dict[tuple[str, str], list[float]] = {}

    for param_name, values, extra_kwargs in SWEEP_DEFS:
        print(f"\n--- Sweeping: {param_name} = {values} ---")
        for metric_name, metric_fn in METRICS:
            if not _is_relevant(metric_name, param_name):
                continue
            rates = sweep_one(metric_fn, param_name, values, extra_kwargs)
            all_results[(param_name, metric_name)] = rates
            rate_str = "  ".join(f"{r*100:5.0f}%" for r in rates)
            print(f"  {metric_name:<28}  [{rate_str}]")

    # ------------------------------------------------------------------
    # Summary: minimum pass rate per metric across all swept values
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY: Minimum pass rate per metric across all parameter sweeps")
    print("=" * 78)
    print(f"  {'metric':<28}  {'min_pass_rate':>14}  {'verdict':>12}")

    all_pass = True
    for metric_name, _ in METRICS:
        relevant_rates: list[float] = []
        for (param_name, mn), rates in all_results.items():
            if mn == metric_name:
                relevant_rates.extend(rates)
        if not relevant_rates:
            continue
        min_rate = min(relevant_rates)
        verdict = "ROBUST" if min_rate >= 0.80 else ("MARGINAL" if min_rate >= 0.60 else "FRAGILE")
        if min_rate < 0.80:
            all_pass = False
        print(f"  {metric_name:<28}  {min_rate*100:>13.0f}%  {verdict:>12}")

    # ------------------------------------------------------------------
    # Per-parameter stability report
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Per-parameter minimum pass rate (across all relevant metrics)")
    print("=" * 78)
    print(f"  {'parameter':<22}  {'values':<40}  min_rate")
    for param_name, values, extra_kwargs in SWEEP_DEFS:
        relevant = [rates for (pn, mn), rates in all_results.items() if pn == param_name]
        if not relevant:
            continue
        flat = [r for rlist in relevant for r in rlist]
        min_r = min(flat) if flat else 0.0
        val_str = str(values)[:38]
        print(f"  {param_name:<22}  {val_str:<40}  {min_r*100:.0f}%")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if all_pass:
        print("  All metrics ROBUST (≥ 80% pass rate across all swept values and seeds).")
        print("  The three core mechanisms are not fragile to local parameter variation.")
    else:
        print("  Some metrics fall below 80%. See per-metric table above.")
        print("  Investigate fragile combinations before scaling to richer experiments.")


if __name__ == "__main__":
    run()
