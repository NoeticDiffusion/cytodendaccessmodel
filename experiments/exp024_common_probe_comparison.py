"""E024 — common-probe model comparison.

This experiment implements the frozen protocol in
``project/neural_computation/model_comparison_protocol.md``. It writes only to
``results/e024_common_probe_comparison`` and refuses to overwrite that folder.
E017–E023 outputs and the existing E024 concept-figure pack are untouched.

The primary score is the change from pre- to post-consolidation in target-trace
readout for a one-hot shared-branch cue relative to a matched source-private
branch cue. Every family uses the same final sigmoid readout. Mechanism-specific
state diagnostics are exported separately from the primary score.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cytodend_accessmodel.contracts import (  # noqa: E402
    ConsolidationWindow,
    DynamicsParameters,
    EngramTrace,
    TraceAllocation,
)
from cytodend_accessmodel.motifs import MotifSpec, alloc_to_cue, build_motif  # noqa: E402
from cytodend_accessmodel.simulator import CytodendAccessModelSimulator  # noqa: E402


OUT_ROOT = REPO_ROOT / "results" / "e024_common_probe_comparison"
META_DIR = OUT_ROOT / "metadata"
SUMMARY_DIR = OUT_ROOT / "summary"
TRACE_DIR = OUT_ROOT / "traces"
FIGURE_DIR = OUT_ROOT / "figures"
PROTOCOL_PATH = REPO_ROOT / "project" / "neural_computation" / "model_comparison_protocol.md"
NO_REPLAY_SENTINEL = "__E024_EXPLICIT_NO_REPLAY__"

MODEL_ORDER = [
    "full_model",
    "fast_context_only",
    "replay_no_structure",
    "random_slow_drift",
    "fixed_allocation_only",
    "hebbian_weight_only",
    "soma_global_gain_only",
    "shuffled_replay",
    "eligibility_only",
    "resource_only",
]
E018_MODELS = {
    "full_model", "fast_context_only", "replay_no_structure",
    "random_slow_drift", "fixed_allocation_only",
}
MODEL_LABELS = {
    "full_model": "Full model",
    "fast_context_only": "Fast context only",
    "replay_no_structure": "Replay, no structure",
    "random_slow_drift": "Random slow drift",
    "fixed_allocation_only": "Fixed allocation",
    "hebbian_weight_only": "Hebbian weights",
    "soma_global_gain_only": "Global gain",
    "shuffled_replay": "Shuffled replay",
    "eligibility_only": "Eligibility only",
    "resource_only": "Resource only",
}
FAST_GAIN_GRID = (1.0, 1.5, 2.0, 2.5, 3.0)
CALIBRATION_SEEDS = tuple(range(20))
EVALUATION_SEEDS = tuple(range(100, 150))
BOOTSTRAP_REPLICATES = 10_000
RANDOMIZATION_REPLICATES = 100_000
MULTIPLICITY_SEED = 240925
MIN_EFFECT = 0.02
CONSOLIDATION_PASSES = 9
SHUFFLE_SIGMA = 0.025
HEBBIAN_LR = 0.18


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _canonical_params(fast_gain: float = 2.0) -> DynamicsParameters:
    return DynamicsParameters(
        fast_gain=fast_gain,
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


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    source: str
    mode: str
    description: str
    weight_rule: str
    branch_gain_rule: str
    replay_rule: str


MODEL_DEFINITIONS = {
    "full_model": ModelDefinition(
        "full_model", "E018/E022", "standard",
        "Replay-dependent slow branch-level writing.",
        "fixed motif allocation", "1", "all traces; canonical replay and M_b write",
    ),
    "fast_context_only": ModelDefinition(
        "fast_context_only", "E018", "standard",
        "Fast access with M_b fixed at its initial value.",
        "fixed motif allocation", "1", "all traces; structural write and decay disabled",
    ),
    "replay_no_structure": ModelDefinition(
        "replay_no_structure", "E018", "standard",
        "Replay and transient E_b/P_b updates without structural writing.",
        "fixed motif allocation", "1", "all traces; structural write disabled",
    ),
    "random_slow_drift": ModelDefinition(
        "random_slow_drift", "E018", "random_drift",
        "Uncoupled Gaussian M_b drift at sigma 0.025 per pass.",
        "fixed motif allocation", "1", "replay ignored; E_b/P_b held during drift",
    ),
    "fixed_allocation_only": ModelDefinition(
        "fixed_allocation_only", "E018", "standard",
        "Fixed allocation and initial M_b; all dynamic updates disabled.",
        "fixed motif allocation", "1", "all update rates zero",
    ),
    "hebbian_weight_only": ModelDefinition(
        "hebbian_weight_only", "E022", "hebbian",
        "Eligibility-driven persistent branch-to-trace weight updates; M_b frozen.",
        "W_jb += 0.18 * E_b * a_jb per replay pass", "1",
        "all traces; P_b may update but is not a recall gain",
    ),
    "soma_global_gain_only": ModelDefinition(
        "soma_global_gain_only", "E022", "soma_global",
        "Eligibility flattened across branches before global M_b writing.",
        "W_jb = 1 for all j,b", "1", "all traces; canonical parameters",
    ),
    "shuffled_replay": ModelDefinition(
        "shuffled_replay", "E022", "shuffled",
        "E_b and replay allocations shuffled on each consolidation pass.",
        "fixed motif allocation for recall", "1", "all traces; fresh random allocations per pass",
    ),
    "eligibility_only": ModelDefinition(
        "eligibility_only", "E022", "eligibility",
        "Transient eligibility contributes to common probe gain; replay gain and M_b write are disabled.",
        "fixed motif allocation", "1 + E_b", "all replay events offered; replay gain zero; E_b/P_b decay",
    ),
    "resource_only": ModelDefinition(
        "resource_only", "E022", "resource",
        "Replay-built P_b contributes transiently to common probe gain; no M_b write.",
        "fixed motif allocation", "1 + P_b", "all traces; structural write disabled",
    ),
}


@dataclass
class E024Model:
    name: str
    motif: MotifSpec
    params: DynamicsParameters
    simulator: CytodendAccessModelSimulator
    decoder: dict[str, dict[str, float]]
    mode: str
    rng: random.Random
    context_by_trace: dict[str, str | None]
    primary_exclusion: str = ""
    calibration_failed: bool = False


def _parameters_for(name: str, fast_gain: float) -> DynamicsParameters:
    base = _canonical_params(fast_gain)
    if name == "fast_context_only":
        return replace(base, structural_lr=0.0, structural_decay=0.0)
    if name in {"replay_no_structure", "hebbian_weight_only", "resource_only"}:
        return replace(base, structural_lr=0.0)
    if name == "eligibility_only":
        return replace(base, structural_lr=0.0, replay_gain=0.0)
    if name == "fixed_allocation_only":
        return replace(
            base,
            structural_lr=0.0,
            replay_gain=0.0,
            eligibility_decay=0.0,
            structural_decay=0.0,
            translation_decay=0.0,
        )
    return base


def _flat_weights(motif: MotifSpec) -> dict[str, dict[str, float]]:
    return {tid: {bid: 1.0 for bid in motif.branch_ids} for tid in motif.trace_ids}


def build_model(
    name: str,
    motif: MotifSpec,
    *,
    fast_gain: float = 2.0,
    seed: int = 0,
    contexts: bool = False,
    calibration_failed: bool = False,
) -> E024Model:
    if name not in MODEL_DEFINITIONS:
        raise KeyError(f"Unknown E024 model family: {name}")
    params = _parameters_for(name, fast_gain)
    sim = CytodendAccessModelSimulator.from_branch_ids(motif.branch_ids, parameters=params)
    context_by_trace = {
        tid: (f"ctx{idx}" if contexts else None)
        for idx, tid in enumerate(motif.trace_ids)
    }
    decoder = _flat_weights(motif) if name == "soma_global_gain_only" else deepcopy(motif.allocations)
    simulator_allocations = _flat_weights(motif) if name == "soma_global_gain_only" else motif.allocations
    for tid in motif.trace_ids:
        sim.add_trace(
            EngramTrace(
                trace_id=tid,
                allocation=TraceAllocation(tid, dict(simulator_allocations[tid])),
                context=context_by_trace[tid],
            )
        )
    exclusion = "hub_boundary_no_unlinked_pairs" if motif.motif_type == "hub_overlap" else ""
    return E024Model(
        name=name,
        motif=motif,
        params=params,
        simulator=sim,
        decoder=decoder,
        mode=MODEL_DEFINITIONS[name].mode,
        rng=random.Random(seed + 77),
        context_by_trace=context_by_trace,
        primary_exclusion=exclusion,
        calibration_failed=calibration_failed,
    )


def _one_hot(branch_id: str, amplitude: float = 1.0) -> dict[str, float]:
    return {branch_id: float(amplitude)}


def private_branches(motif: MotifSpec, trace_id: str, threshold: float = 0.40) -> list[str]:
    others = [other for other in motif.trace_ids if other != trace_id]
    return [
        bid for bid in motif.branch_ids
        if motif.allocations[trace_id].get(bid, 0.0) > threshold
        and all(motif.allocations[other].get(bid, 0.0) <= threshold for other in others)
    ]


def build_probe_manifest(motif: MotifSpec, seed: int, analysis: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if motif.motif_type == "hub_overlap":
        return [{
            "analysis": analysis,
            "seed": seed,
            "motif_type": motif.motif_type,
            "motif_id": motif.motif_id,
            "source_trace": "",
            "target_trace": "",
            "shared_branches": "",
            "source_private_branches": "",
            "n_probe_combinations": 0,
            "primary_status": "N/A",
            "reason": "hub_boundary_no_unlinked_pairs",
        }]

    for source_idx, source in enumerate(motif.trace_ids):
        for target_idx, target in enumerate(motif.trace_ids):
            if source_idx == target_idx:
                continue
            shared = [
                bid for bid in motif.branch_ids
                if motif.allocations[source].get(bid, 0.0) > 0.40
                and motif.allocations[target].get(bid, 0.0) > 0.40
            ]
            controls = [
                bid for bid in motif.branch_ids
                if motif.allocations[source].get(bid, 0.0) > 0.40
                and motif.allocations[target].get(bid, 0.0) <= 0.40
            ]
            eligible = bool(shared and controls)
            reason = "" if eligible else ("no_shared_branch" if not shared else "no_source_private_control")
            rows.append({
                "analysis": analysis,
                "seed": seed,
                "motif_type": motif.motif_type,
                "motif_id": motif.motif_id,
                "source_trace": source,
                "target_trace": target,
                "shared_branches": ";".join(shared),
                "source_private_branches": ";".join(controls),
                "n_probe_combinations": len(shared) * len(controls) if eligible else 0,
                "primary_status": "defined" if eligible else "N/A",
                "reason": reason,
            })
    if not any(row["primary_status"] == "defined" for row in rows):
        for row in rows:
            row["reason"] = row["reason"] or "no_eligible_pair_in_motif"
    return rows


def _explicit_no_replay(model: E024Model) -> None:
    """Run one decay-only pass while explicitly selecting zero traces.

    The public simulator interprets an empty replay ID sequence as its default
    all-compatible-traces selection. A sentinel that cannot resolve to a trace
    gives an explicit empty selection without changing historical simulator API.
    """
    report = model.simulator.run_consolidation(
        ConsolidationWindow(
            window_id="e024_no_replay",
            replay_trace_ids=(NO_REPLAY_SENTINEL,),
            modulatory_drive=0.0,
            sleep_drive=0.0,
        )
    )
    if report.traces_replayed:
        raise AssertionError(f"Explicit no-replay selected traces: {report.traces_replayed}")


def _flatten_eligibility(model: E024Model) -> None:
    vals = [branch.eligibility.value for branch in model.simulator.branches.values()]
    mean_val = sum(vals) / len(vals) if vals else 0.0
    for branch in model.simulator.branches.values():
        branch.eligibility.value = mean_val


def _shuffle_eligibility(model: E024Model) -> None:
    bids = list(model.motif.branch_ids)
    vals = [model.simulator.branches[bid].eligibility.value for bid in bids]
    model.rng.shuffle(vals)
    for bid, value in zip(bids, vals):
        model.simulator.branches[bid].eligibility.value = value


def _consolidate_one(model: E024Model, pass_index: int = 0) -> None:
    if model.mode == "random_drift":
        for branch in model.simulator.branches.values():
            state = branch.structural
            assert state is not None
            state.accessibility = min(
                state.max_accessibility,
                max(0.0, state.accessibility + model.rng.gauss(0.0, SHUFFLE_SIGMA)),
            )
            branch.slow_access = 1.0 / (1.0 + math.exp(-model.params.structural_gain * state.accessibility))
        return

    if model.mode == "hebbian":
        for bid in model.motif.branch_ids:
            eligibility = model.simulator.branches[bid].eligibility.value
            for tid in model.motif.trace_ids:
                alloc = model.motif.allocations[tid].get(bid, 0.0)
                model.decoder[tid][bid] = model.decoder[tid].get(bid, 0.0) + HEBBIAN_LR * eligibility * alloc

    if model.mode == "soma_global" and pass_index == 0:
        _flatten_eligibility(model)

    if model.mode == "shuffled":
        _shuffle_eligibility(model)
        original = {
            tid: dict(model.simulator.traces[tid].allocation.branch_weights)
            for tid in model.motif.trace_ids
        }
        for tid in model.motif.trace_ids:
            raw = {bid: model.rng.random() for bid in model.motif.branch_ids}
            denom = sum(raw.values()) or 1.0
            model.simulator.traces[tid].allocation.branch_weights = {
                bid: value / denom for bid, value in raw.items()
            }
        try:
            model.simulator.run_consolidation(
                ConsolidationWindow(
                    window_id=f"e024_shuffle_{pass_index}",
                    replay_trace_ids=tuple(model.motif.trace_ids),
                    modulatory_drive=1.0,
                    sleep_drive=0.0,
                )
            )
        finally:
            for tid in model.motif.trace_ids:
                model.simulator.traces[tid].allocation.branch_weights = original[tid]
        return

    model.simulator.run_consolidation(
        ConsolidationWindow(
            window_id=f"e024_replay_{pass_index}",
            replay_trace_ids=tuple(model.motif.trace_ids),
            modulatory_drive=1.0,
            sleep_drive=0.0,
        )
    )


def encode_model(model: E024Model) -> None:
    for tid in model.motif.trace_ids:
        cue = alloc_to_cue(model.motif.allocations[tid])
        context = model.context_by_trace[tid]
        for _ in range(2):
            model.simulator.apply_cue(cue, context=context)


def train_model(model: E024Model, *, consolidate: bool = True) -> E024Model:
    encode_model(model)
    if consolidate:
        for pass_index in range(CONSOLIDATION_PASSES):
            _consolidate_one(model, pass_index)
    return model


def _apply_cue_with_input_gain(
    model: E024Model,
    cue: dict[str, float],
    *,
    context: str | None = None,
    input_gain: dict[str, float] | None = None,
) -> CytodendAccessModelSimulator:
    probe = deepcopy(model.simulator)
    adjusted = dict(cue)
    for bid, gain in (input_gain or {}).items():
        adjusted[bid] = adjusted.get(bid, 0.0) * gain
    probe.apply_cue(adjusted, context=context)
    return probe


def probe_readout(
    model: E024Model,
    cue: dict[str, float],
    *,
    context: str | None = None,
    masked_branches: Iterable[str] = (),
    input_gain: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    probe = _apply_cue_with_input_gain(model, cue, context=context, input_gain=input_gain)
    masked = set(masked_branches)
    branch_gain_mode = MODEL_DEFINITIONS[model.name].branch_gain_rule
    result: dict[str, dict[str, float]] = {}
    for tid in model.motif.trace_ids:
        support = 0.0
        trace = probe.traces[tid]
        for bid, weight in model.decoder[tid].items():
            if bid in masked:
                continue
            branch = probe.branches[bid]
            gain = 1.0
            if branch_gain_mode == "1 + E_b":
                gain += branch.eligibility.value
            elif branch_gain_mode == "1 + P_b":
                gain += branch.translation_readiness.value
            if input_gain and bid in input_gain:
                gain *= input_gain[bid]
            support += float(weight) * branch.activation * gain
        if probe.context is not None and trace.context is not None and trace.context != probe.context:
            support *= max(0.0, 1.0 - model.params.context_mismatch_penalty)
        readout = 1.0 / (1.0 + math.exp(-model.params.readout_gain * (support - model.params.readout_threshold)))
        result[tid] = {"support": support, "readout": readout}
    return result


def _branch_set(row: dict[str, Any], key: str) -> list[str]:
    value = str(row.get(key, ""))
    return [item for item in value.split(";") if item]


def primary_effect(
    pre: E024Model,
    post: E024Model,
    pair_rows: list[dict[str, Any]],
    *,
    collect_raw: bool = False,
    perturbation_gain: dict[str, float] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    if pre.primary_exclusion or post.primary_exclusion:
        return float("nan"), []
    valid = [row for row in pair_rows if row["primary_status"] == "defined"]
    if not valid:
        return float("nan"), []
    all_directions: list[float] = []
    raw_rows: list[dict[str, Any]] = []
    for row in valid:
        source = row["source_trace"]
        target = row["target_trace"]
        shared = _branch_set(row, "shared_branches")
        controls = _branch_set(row, "source_private_branches")
        combo_diffs: list[float] = []
        for shared_bid in shared:
            for control_bid in controls:
                values: dict[str, float] = {}
                for stage, model in (("pre", pre), ("post", post)):
                    shared_map = probe_readout(
                        model,
                        _one_hot(shared_bid),
                        input_gain=perturbation_gain,
                    )
                    control_map = probe_readout(
                        model,
                        _one_hot(control_bid),
                        input_gain=perturbation_gain,
                    )
                    r_shared = shared_map[target]["readout"]
                    r_control = control_map[target]["readout"]
                    values[f"shared_{stage}"] = r_shared
                    values[f"control_{stage}"] = r_control
                value = (values["shared_post"] - values["control_post"]) - (
                    values["shared_pre"] - values["control_pre"]
                )
                combo_diffs.append(value)
                if collect_raw:
                    raw_rows.append({
                        "source_trace": source,
                        "target_trace": target,
                        "shared_branch": shared_bid,
                        "control_branch": control_bid,
                        **values,
                        "D_pair_combo": value,
                    })
        if combo_diffs:
            all_directions.append(sum(combo_diffs) / len(combo_diffs))
    return (sum(all_directions) / len(all_directions) if all_directions else float("nan")), raw_rows


def _private_correct_recall(model: E024Model) -> float:
    values: list[float] = []
    for tid in model.motif.trace_ids:
        for bid in private_branches(model.motif, tid):
            result = probe_readout(model, _one_hot(bid))
            values.append(result[tid]["readout"])
    return sum(values) / len(values) if values else float("nan")


def calibrate_models() -> tuple[dict[str, float], list[dict[str, Any]], dict[str, bool]]:
    motif = build_motif("canonical", n_branches=4, n_traces=2)
    reference_vals: list[float] = []
    for seed in CALIBRATION_SEEDS:
        reference = train_model(build_model("full_model", motif, fast_gain=2.0, seed=seed), consolidate=False)
        reference_vals.append(_private_correct_recall(reference))
    reference_mean = sum(reference_vals) / len(reference_vals)
    rows: list[dict[str, Any]] = []
    selected: dict[str, float] = {}
    failed: dict[str, bool] = {}
    for name in MODEL_ORDER:
        scores: list[tuple[float, float]] = []
        for gain in FAST_GAIN_GRID:
            vals: list[float] = []
            for seed in CALIBRATION_SEEDS:
                model = train_model(build_model(name, motif, fast_gain=gain, seed=seed), consolidate=False)
                vals.append(_private_correct_recall(model))
            mean_score = sum(vals) / len(vals)
            diff = abs(mean_score - reference_mean)
            scores.append((diff, gain))
            rows.append({
                "model": name,
                "fast_gain": gain,
                "calibration_mean_readout": mean_score,
                "full_model_reference": reference_mean,
                "absolute_difference": diff,
            })
        best_diff, best_gain = min(scores, key=lambda item: (item[0], abs(item[1] - 2.0)))
        selected[name] = best_gain
        failed[name] = best_diff > 0.02
        for row in rows:
            if row["model"] == name:
                row["selected"] = math.isclose(row["fast_gain"], best_gain)
                row["baseline_match_failed"] = failed[name]
    return selected, rows, failed


def _build_motifs(seed: int) -> list[MotifSpec]:
    return [
        build_motif("canonical", n_branches=4, n_traces=2, seed=seed),
        build_motif("strong_overlap", n_branches=4, n_traces=2, seed=seed),
        build_motif("chain_overlap", n_branches=8, n_traces=3, seed=seed),
        build_motif("sparse_random", n_branches=8, n_traces=4, seed=seed),
        build_motif("hub_overlap", n_branches=5, n_traces=4, seed=seed),
    ]


def run_context_separation(model_name: str, motif: MotifSpec, seed: int, fast_gain: float) -> float:
    model = train_model(build_model(model_name, motif, fast_gain=fast_gain, seed=seed, contexts=True))
    margins: list[float] = []
    for tid in motif.trace_ids:
        branches = private_branches(motif, tid)
        if not branches:
            continue
        for bid in branches:
            ctx = model.context_by_trace[tid]
            scores = probe_readout(model, _one_hot(bid), context=ctx)
            wrong = max(
                (data["readout"] for other, data in scores.items() if other != tid),
                default=float("nan"),
            )
            if math.isfinite(wrong):
                margins.append(scores[tid]["readout"] - wrong)
    return sum(margins) / len(margins) if margins else float("nan")


def run_branch_perturbation(
    post: E024Model,
    motif: MotifSpec,
    pair_rows: list[dict[str, Any]],
) -> float:
    if post.primary_exclusion:
        return float("nan")
    values: list[float] = []
    for row in pair_rows:
        if row["primary_status"] != "defined":
            continue
        source, target = row["source_trace"], row["target_trace"]
        cue = alloc_to_cue(motif.allocations[source])
        shared = _branch_set(row, "shared_branches")
        controls = _branch_set(row, "source_private_branches")
        base = probe_readout(post, cue)[target]["readout"]
        for b_shared in shared:
            for b_control in controls:
                drop_shared = base - probe_readout(post, cue, masked_branches=(b_shared,))[target]["readout"]
                drop_control = base - probe_readout(post, cue, masked_branches=(b_control,))[target]["readout"]
                values.append(drop_shared - drop_control)
    return sum(values) / len(values) if values else float("nan")


def _primary_under_gain(
    pre: E024Model,
    post: E024Model,
    pair_rows: list[dict[str, Any]],
    gain_branch: str,
) -> float:
    return primary_effect(pre, post, pair_rows, perturbation_gain={gain_branch: 0.5})[0]


def run_recovery_arms(
    pre: E024Model,
    post: E024Model,
    pair_rows: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    valid = [row for row in pair_rows if row["primary_status"] == "defined"]
    if not valid:
        return [{"arm": "targeted", "recovery_gain": float("nan"), "post_recovery_unmasked": float("nan"), "reason": "no_eligible_pair"}]
    out: list[dict[str, Any]] = []
    # Each directed pair gets the same targeted/control challenge, selected by
    # motif order before scores are inspected.
    for row in valid:
        pair = [row]
        shared_bid = _branch_set(row, "shared_branches")[0]
        private_bid = _branch_set(row, "source_private_branches")[0]
        initial_perturbed = _primary_under_gain(pre, post, pair, shared_bid)
        for arm in ("targeted", "control"):
            model = deepcopy(post)
            cue_branch = shared_bid if arm == "targeted" else private_bid
            amplitude = 1.0 if arm == "targeted" else 0.5
            for pass_idx in range(3):
                for _ in range(3):
                    cue = _one_hot(cue_branch, amplitude)
                    if arm == "targeted":
                        cue[shared_bid] *= 0.5
                    model.simulator.apply_cue(cue)
                for _ in range(3):
                    _consolidate_one(model, pass_idx * 3)
            perturbed = primary_effect(pre, model, pair, perturbation_gain={shared_bid: 0.5})[0]
            unmasked = primary_effect(pre, model, pair)[0]
            out.append({
                "arm": arm,
                "source_trace": row["source_trace"],
                "target_trace": row["target_trace"],
                "shared_branch": shared_bid,
                "control_branch": private_bid,
                "pre_recovery_perturbed_D": initial_perturbed,
                "post_recovery_perturbed_D": perturbed,
                "post_recovery_unmasked_D": unmasked,
                "recovery_change_under_perturbation": perturbed - initial_perturbed,
                "cues": 9,
                "consolidation_passes": 9,
                "reason": "",
            })
    return out


def _preflight_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    canonical = build_motif("canonical", n_branches=4, n_traces=2)
    model = train_model(build_model("full_model", canonical, fast_gain=2.0), consolidate=False)
    no_cue = probe_readout(model, {})
    checks.append({
        "check": "zero_input_readout_baseline",
        "status": "pass" if all(abs(v["support"]) < 1e-12 and abs(v["readout"] - 0.18242552380635635) < 1e-12 for v in no_cue.values()) else "fail",
        "detail": "zero raw support maps to sigmoid baseline at threshold 0.3",
    })

    before_step = model.simulator.step_index
    before_e = {bid: b.eligibility.value for bid, b in model.simulator.branches.items()}
    probe_readout(model, _one_hot("b1"))
    isolated = before_step == model.simulator.step_index and before_e == {
        bid: b.eligibility.value for bid, b in model.simulator.branches.items()
    }
    checks.append({
        "check": "probe_state_isolation",
        "status": "pass" if isolated else "fail",
        "detail": "probe operates on a deep copy and leaves the frozen state unchanged",
    })

    no_replay = build_model("eligibility_only", canonical, seed=4)
    no_replay.simulator.branches["b0"].translation_readiness.value = 0.5
    _explicit_no_replay(no_replay)
    expected_p = 0.5 * (1.0 - no_replay.params.translation_decay)
    actual_p = no_replay.simulator.branches["b0"].translation_readiness.value
    checks.append({
        "check": "explicit_no_replay",
        "status": "pass" if abs(actual_p - expected_p) < 1e-12 else "fail",
        "detail": f"P_b decays only: expected={expected_p:.12g}, observed={actual_p:.12g}",
    })

    hub = build_motif("hub_overlap", n_branches=5, n_traces=4, seed=0)
    hub_manifest = build_probe_manifest(hub, 0, "preflight")
    hub_excluded = all(r["primary_status"] == "N/A" and r["reason"] == "hub_boundary_no_unlinked_pairs" for r in hub_manifest)
    checks.append({
        "check": "hub_boundary_NA",
        "status": "pass" if hub_excluded else "fail",
        "detail": "hub motif is registered as a boundary case and omitted from primary mean",
    })

    valid_manifest = build_probe_manifest(canonical, 0, "preflight")
    canonical_pairs = sum(1 for row in valid_manifest if row["primary_status"] == "defined")
    checks.append({
        "check": "canonical_pair_manifest",
        "status": "pass" if canonical_pairs == 2 else "fail",
        "detail": f"expected two directed eligible pairs; observed={canonical_pairs}",
    })

    if any(row["status"] != "pass" for row in checks):
        raise RuntimeError("E024 preflight failed; evaluation matrix was not started")
    return checks


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = (len(sorted_values) - 1) * p
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _bootstrap_interval(values: list[float], seed: int, reps: int = BOOTSTRAP_REPLICATES) -> tuple[float, float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(finite)
    means = []
    for _ in range(reps):
        means.append(sum(finite[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _randomization_pvalue(centered: list[float], *, alternative: str, rng_seed: int) -> float:
    vals = np.asarray([value for value in centered if math.isfinite(value)], dtype=float)
    if vals.size == 0:
        return float("nan")
    observed = float(np.mean(vals))
    rng = np.random.default_rng(rng_seed)
    signs = rng.integers(0, 2, size=(RANDOMIZATION_REPLICATES, vals.size), dtype=np.int8) * 2 - 1
    permuted = np.mean(signs * vals[None, :], axis=1)
    if alternative == "greater":
        exceed = int(np.count_nonzero(permuted >= observed - 1e-15))
    elif alternative == "less":
        exceed = int(np.count_nonzero(permuted <= observed + 1e-15))
    else:
        raise ValueError(alternative)
    return (exceed + 1) / (RANDOMIZATION_REPLICATES + 1)


def _holm(pvalues: list[float]) -> list[float]:
    n = len(pvalues)
    order = sorted(range(n), key=lambda idx: (math.inf if not math.isfinite(pvalues[idx]) else pvalues[idx], idx))
    adjusted = [float("nan")] * n
    running = 0.0
    for rank, idx in enumerate(order):
        if not math.isfinite(pvalues[idx]):
            adjusted[idx] = float("nan")
            continue
        current = min(1.0, (n - rank) * pvalues[idx])
        running = max(running, current)
        adjusted[idx] = running
    return adjusted


def _as_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _write_primary_summaries(primary_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_summary: list[dict[str, Any]] = []
    by_analysis = sorted({row["analysis"] for row in primary_rows})
    for analysis in by_analysis:
        for model in MODEL_ORDER:
            for motif_type in ("canonical", "strong_overlap", "chain_overlap", "sparse_random"):
                vals = [
                    _as_float(row["D"])
                    for row in primary_rows
                    if row["analysis"] == analysis and row["model"] == model and row["motif_type"] == motif_type
                ]
                finite = [v for v in vals if math.isfinite(v)]
                lo, hi = _bootstrap_interval(finite, 1000 + len(model_summary))
                model_summary.append({
                    "analysis": analysis,
                    "model": model,
                    "motif_type": motif_type,
                    "n_seeds": len(finite),
                    "mean_D": sum(finite) / len(finite) if finite else float("nan"),
                    "sd_D": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
                    "ci95_low": lo,
                    "ci95_high": hi,
                })
            seed_values: dict[int, list[float]] = {}
            for row in primary_rows:
                if row["analysis"] == analysis and row["model"] == model:
                    value = _as_float(row["D"])
                    if math.isfinite(value):
                        seed_values.setdefault(int(row["seed"]), []).append(value)
            macro = [sum(values) / len(values) for values in seed_values.values() if values]
            lo, hi = _bootstrap_interval(macro, 5000 + len(model_summary))
            model_summary.append({
                "analysis": analysis,
                "model": model,
                "motif_type": "macro_mean",
                "n_seeds": len(macro),
                "mean_D": sum(macro) / len(macro) if macro else float("nan"),
                "sd_D": float(np.std(macro, ddof=1)) if len(macro) > 1 else float("nan"),
                "ci95_low": lo,
                "ci95_high": hi,
            })

    comparison_rows: list[dict[str, Any]] = []
    for analysis in by_analysis:
        macro: dict[str, dict[int, float]] = {}
        for model in MODEL_ORDER:
            per_seed: dict[int, list[float]] = {}
            for row in primary_rows:
                if row["analysis"] == analysis and row["model"] == model:
                    value = _as_float(row["D"])
                    if math.isfinite(value):
                        per_seed.setdefault(int(row["seed"]), []).append(value)
            macro[model] = {seed: sum(vals) / len(vals) for seed, vals in per_seed.items() if vals}
        comparison_temp: list[dict[str, Any]] = []
        superiority_p: list[float] = []
        equivalence_p_all: list[float] = []
        for idx, model in enumerate(m for m in MODEL_ORDER if m != "full_model"):
            common_seeds = sorted(set(macro["full_model"]) & set(macro[model]))
            diffs = [macro["full_model"][seed] - macro[model][seed] for seed in common_seeds]
            if diffs:
                lo, hi = _bootstrap_interval(diffs, 9000 + idx + (0 if analysis == "baseline_matched" else 100))
                p_sup = _randomization_pvalue([d - MIN_EFFECT for d in diffs], alternative="greater", rng_seed=MULTIPLICITY_SEED + idx)
                p_eq_lo = _randomization_pvalue([d + MIN_EFFECT for d in diffs], alternative="greater", rng_seed=MULTIPLICITY_SEED + 100 + idx)
                p_eq_hi = _randomization_pvalue([d - MIN_EFFECT for d in diffs], alternative="less", rng_seed=MULTIPLICITY_SEED + 200 + idx)
            else:
                lo = hi = p_sup = p_eq_lo = p_eq_hi = float("nan")
            superiority_p.append(p_sup)
            equivalence_p_all.extend((p_eq_lo, p_eq_hi))
            comparison_temp.append({
                "analysis": analysis,
                "model": model,
                "n_paired_seeds": len(diffs),
                "mean_D_full_minus_model": sum(diffs) / len(diffs) if diffs else float("nan"),
                "bootstrap_ci95_low": lo,
                "bootstrap_ci95_high": hi,
                "p_superiority_raw": p_sup,
                "p_equiv_lower_raw": p_eq_lo,
                "p_equiv_upper_raw": p_eq_hi,
                "_diffs": diffs,
            })
        superiority_adj = _holm(superiority_p)
        equivalence_adj = _holm(equivalence_p_all)
        for idx, row in enumerate(comparison_temp):
            row["p_superiority_holm"] = superiority_adj[idx]
            row["p_equiv_lower_holm"] = equivalence_adj[2 * idx]
            row["p_equiv_upper_holm"] = equivalence_adj[2 * idx + 1]
            full_macro = next(
                item for item in model_summary
                if item["analysis"] == analysis and item["model"] == "full_model" and item["motif_type"] == "macro_mean"
            )
            full_positive = _as_float(full_macro["ci95_low"]) > 0.0
            superiority = full_positive and _as_float(row["p_superiority_holm"]) < 0.05
            equivalent = (
                _as_float(row["p_equiv_lower_holm"]) < 0.05
                and _as_float(row["p_equiv_upper_holm"]) < 0.05
            )
            row["decision"] = "supports_discrimination" if superiority else ("practical_equivalence" if equivalent else "inconclusive")
            row.pop("_diffs", None)
            comparison_rows.append(row)
    return model_summary, comparison_rows


def _write_plot(model_summary: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, analysis in zip(axes, ("baseline_matched", "published_parameters")):
        rows = [r for r in model_summary if r["analysis"] == analysis and r["motif_type"] == "macro_mean"]
        rows_by_name = {r["model"]: r for r in rows}
        x = np.arange(len(MODEL_ORDER))
        means = np.array([_as_float(rows_by_name[name]["mean_D"]) for name in MODEL_ORDER])
        low = np.array([_as_float(rows_by_name[name]["ci95_low"]) for name in MODEL_ORDER])
        high = np.array([_as_float(rows_by_name[name]["ci95_high"]) for name in MODEL_ORDER])
        yerr = np.vstack((means - low, high - means))
        ax.errorbar(x, means, yerr=yerr, fmt="o", capsize=3, color="#225ea8", ecolor="#7aa6d8")
        ax.axhline(0.0, color="#555555", linewidth=0.8)
        ax.axhline(MIN_EFFECT, color="#b2182b", linestyle="--", linewidth=0.9)
        ax.set_xticks(x, [MODEL_LABELS[name] for name in MODEL_ORDER], rotation=55, ha="right")
        ax.set_title("Baseline-matchad" if analysis == "baseline_matched" else "Publicerade parametrar")
        ax.set_ylabel("Primärt D, 0–1 readoutskala")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("E024 — gemensam probe: delad gren minus källprivat kontroll, post−pre")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "Fig_e024_01_primary_effects.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True).splitlines()
    except Exception as exc:  # pragma: no cover - records environmental issue
        return {"commit": None, "status_error": str(exc)}
    return {"commit": commit, "working_tree_status": status}


def _save_freeze_artifacts(
    selected_gains: dict[str, float],
    failed: dict[str, bool],
    probe_manifest: list[dict[str, Any]],
) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    protocol_copy = META_DIR / "frozen_model_comparison_protocol.md"
    protocol_copy.write_bytes(PROTOCOL_PATH.read_bytes())
    source_paths = [
        Path(__file__),
        REPO_ROOT / "experiments" / "exp018_comparator_trace_matrix.py",
        REPO_ROOT / "experiments" / "exp022_hard_comparators.py",
        REPO_ROOT / "src" / "cytodend_accessmodel" / "simulator.py",
        REPO_ROOT / "src" / "cytodend_accessmodel" / "contracts.py",
        REPO_ROOT / "src" / "cytodend_accessmodel" / "motifs.py",
    ]
    source_hashes = {p.relative_to(REPO_ROOT).as_posix(): _sha256(p) for p in source_paths}
    registry: dict[str, Any] = {}
    for name in MODEL_ORDER:
        params = _parameters_for(name, selected_gains[name])
        registry[name] = {
            **asdict(MODEL_DEFINITIONS[name]),
            "selected_fast_gain": selected_gains[name],
            "baseline_match_failed": failed[name],
            "parameters": asdict(params),
            "calibration_seeds": list(CALIBRATION_SEEDS),
            "eval_seeds": list(EVALUATION_SEEDS),
        }
    _write_json(SUMMARY_DIR / "model_registry.json", registry)
    _write_csv(SUMMARY_DIR / "probe_manifest.csv", probe_manifest)
    config = {
        "experiment": "E024 common-probe model comparison",
        "protocol_version": "1.1",
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "protocol_snapshot_sha256": _sha256(protocol_copy),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "models": MODEL_ORDER,
        "fast_gain_grid": FAST_GAIN_GRID,
        "selected_fast_gain": selected_gains,
        "baseline_match_failed": failed,
        "calibration_seeds": CALIBRATION_SEEDS,
        "eval_seeds": EVALUATION_SEEDS,
        "motifs": ["canonical", "strong_overlap", "chain_overlap", "sparse_random"],
        "boundary_motif": "hub_overlap",
        "consolidation_passes": CONSOLIDATION_PASSES,
        "readout_gain": 5.0,
        "readout_threshold": 0.3,
        "primary_min_effect": MIN_EFFECT,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "randomization_replicates": RANDOMIZATION_REPLICATES,
        "multiple_comparison_method": "Holm",
        "randomization_seed": MULTIPLICITY_SEED,
        "source_sha256": source_hashes,
    }
    _write_json(META_DIR / "run_config.json", config)
    (OUT_ROOT / "README.md").write_text(
        "# E024 common-probe comparison\n\n"
        "Run generated by `experiments/exp024_common_probe_comparison.py`. "
        "The frozen protocol snapshot, model registry, calibration grid, and probe manifest are in `metadata/` and `summary/`. "
        "E017–E023 results were not modified. The readout values are simulator output scores on a 0–1 scale, not empirically calibrated probabilities.\n",
        encoding="utf-8",
    )


def _load_frozen_artifacts() -> tuple[dict[str, float], dict[str, bool]]:
    config_path = META_DIR / "run_config.json"
    registry_path = SUMMARY_DIR / "model_registry.json"
    required = [
        config_path,
        registry_path,
        SUMMARY_DIR / "calibration_grid.csv",
        SUMMARY_DIR / "probe_manifest.csv",
        META_DIR / "frozen_model_comparison_protocol.md",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Incomplete E024 preflight freeze; missing: {missing}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_protocol = _sha256(PROTOCOL_PATH)
    expected_script = _sha256(Path(__file__))
    if config.get("protocol_sha256") != expected_protocol:
        raise RuntimeError("Frozen E024 protocol hash differs from the current protocol; use a new output folder/version")
    recorded_script = config.get("source_sha256", {}).get("experiments/exp024_common_probe_comparison.py")
    if recorded_script != expected_script:
        raise RuntimeError("E024 implementation changed after freeze; do not reuse this preflight record")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    selected = {name: float(registry[name]["selected_fast_gain"]) for name in MODEL_ORDER}
    failed = {name: bool(registry[name]["baseline_match_failed"]) for name in MODEL_ORDER}
    return selected, failed


def _run_evaluation(selected_gains: dict[str, float], failed: dict[str, bool], *, include_secondary: bool = True) -> None:
    manifests_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    motifs_by_key: dict[tuple[str, int], MotifSpec] = {}
    all_manifest: list[dict[str, Any]] = []
    for seed in EVALUATION_SEEDS:
        for motif in _build_motifs(seed):
            key = (motif.motif_type, seed)
            motifs_by_key[key] = motif
            manifest = build_probe_manifest(motif, seed, "evaluation")
            manifests_by_key[key] = manifest
            all_manifest.extend(manifest)
    _write_csv(SUMMARY_DIR / "probe_manifest.csv", all_manifest)

    primary_rows: list[dict[str, Any]] = []
    raw_probe_rows: list[dict[str, Any]] = []
    secondary_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    requested_motifs = {"canonical", "strong_overlap", "chain_overlap", "sparse_random", "hub_overlap"}
    total = len(EVALUATION_SEEDS) * len(requested_motifs) * len(MODEL_ORDER) * 2
    done = 0
    for analysis, gain_by_model in (
        ("baseline_matched", selected_gains),
        ("published_parameters", {name: 2.0 for name in MODEL_ORDER}),
    ):
        for name in MODEL_ORDER:
            for motif_type, seed in sorted(motifs_by_key, key=lambda item: (EVALUATION_SEEDS.index(item[1]), item[0])):
                motif = motifs_by_key[(motif_type, seed)]
                if motif_type not in requested_motifs:
                    continue
                pair_rows = manifests_by_key[(motif_type, seed)]
                fast_gain = gain_by_model[name]
                pre = train_model(build_model(name, motif, fast_gain=fast_gain, seed=seed, calibration_failed=failed[name]), consolidate=False)
                post = deepcopy(pre)
                for pass_index in range(CONSOLIDATION_PASSES):
                    _consolidate_one(post, pass_index)
                effect, raw = primary_effect(pre, post, pair_rows, collect_raw=True)
                status = "defined" if math.isfinite(effect) else (
                    post.primary_exclusion or "no_eligible_pair_in_motif"
                )
                primary_rows.append({
                    "analysis": analysis,
                    "model": name,
                    "motif_type": motif_type,
                    "motif_id": motif.motif_id,
                    "seed": seed,
                    "fast_gain": fast_gain,
                    "baseline_match_failed": failed[name],
                    "D": effect,
                    "primary_status": status,
                    "n_defined_pairs": sum(1 for row in pair_rows if row["primary_status"] == "defined"),
                })
                for raw_row in raw:
                    raw_probe_rows.append({
                        "analysis": analysis,
                        "model": name,
                        "motif_type": motif_type,
                        "motif_id": motif.motif_id,
                        "seed": seed,
                        **raw_row,
                    })
                if include_secondary and analysis == "baseline_matched":
                    context_score = run_context_separation(name, motif, seed, fast_gain)
                    branch_score = run_branch_perturbation(post, motif, pair_rows)
                    secondary_rows.extend([
                        {"analysis": analysis, "model": name, "motif_type": motif_type, "seed": seed,
                         "outcome": "context_separation", "value": context_score,
                         "status": "defined" if math.isfinite(context_score) else "N/A"},
                        {"analysis": analysis, "model": name, "motif_type": motif_type, "seed": seed,
                         "outcome": "shared_vs_private_mask_effect", "value": branch_score,
                         "status": "defined" if math.isfinite(branch_score) else "N/A"},
                    ])
                    if motif_type != "hub_overlap":
                        for arm in run_recovery_arms(pre, post, pair_rows, seed):
                            secondary_rows.append({
                                "analysis": analysis,
                                "model": name,
                                "motif_type": motif_type,
                                "seed": seed,
                                "outcome": f"recovery_{arm['arm']}",
                                **arm,
                                "value": arm.get("recovery_change_under_perturbation", float("nan")),
                                "status": "defined" if math.isfinite(_as_float(arm.get("recovery_change_under_perturbation"))) else "N/A",
                            })
                    for branch_id, branch in post.simulator.branches.items():
                        branch_rows.append({
                            "analysis": analysis,
                            "model": name,
                            "motif_type": motif_type,
                            "seed": seed,
                            "branch_id": branch_id,
                            "M_b": branch.structural.accessibility,
                            "E_b": branch.eligibility.value,
                            "P_b": branch.translation_readiness.value,
                        })
                done += 1
                if done % 50 == 0:
                    print(f"E024 progress: {done}/{total} model×motif×seed×analysis cells", flush=True)

    _write_csv(SUMMARY_DIR / "primary_by_seed.csv", primary_rows)
    _write_csv(TRACE_DIR / "primary_probe_values.csv", raw_probe_rows)
    if secondary_rows:
        _write_csv(SUMMARY_DIR / "secondary_by_seed.csv", secondary_rows)
    if branch_rows:
        _write_csv(TRACE_DIR / "mechanism_diagnostics.csv", branch_rows)
    model_summary, comparison_rows = _write_primary_summaries(primary_rows)
    _write_csv(SUMMARY_DIR / "primary_model_summary.csv", model_summary)
    _write_csv(SUMMARY_DIR / "paired_comparator_summary.csv", comparison_rows)
    _write_plot(model_summary)
    _write_claim_ledger(model_summary, comparison_rows)


def _write_claim_ledger(model_summary: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    lines = [
        "# E024 claim ledger",
        "",
        "## Primary model-independent readout",
        "",
        "The outcome is the post-minus-pre change in target-trace readout for a one-hot cue to a shared branch relative to a matched source-private branch. Values are simulator readout scores, not empirical probabilities.",
        "",
        "## Comparator decisions",
        "",
    ]
    for analysis in ("baseline_matched", "published_parameters"):
        lines.extend([f"### {analysis.replace('_', ' ').title()}", ""])
        for row in comparisons:
            if row["analysis"] == analysis:
                lines.append(
                    f"- `{row['model']}`: {row['decision']}; mean full-minus-comparator "
                    f"`{_fmt(row['mean_D_full_minus_model'])}`; paired seeds {row['n_paired_seeds']}."
                )
        lines.append("")
    full = [r for r in model_summary if r["analysis"] == "baseline_matched" and r["model"] == "full_model" and r["motif_type"] == "macro_mean"]
    lines.extend([
        "## Interpretation limits",
        "",
        "- `N/A` motifs or adapters are not failures; see `probe_manifest.csv` and `primary_by_seed.csv`.",
        "- Internal `M_b`, `E_b`, and `P_b` summaries are mechanism diagnostics and do not establish biological identity or validation.",
        "- A comparator classified as practically equivalent prevents a claim that the full model is required for this readout.",
        "- Results apply only to the motifs, parameters, adapter equations, and common task recorded in `metadata/run_config.json`.",
    ])
    if full:
        lines.insert(6, f"Full-model baseline-matched macro mean D: `{_fmt(full[0]['mean_D'])}` (95% bootstrap CI `{_fmt(full[0]['ci95_low'])}` to `{_fmt(full[0]['ci95_high'])}`).")
    (SUMMARY_DIR / "claim_ledger.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    number = _as_float(value)
    return f"{number:.6g}" if math.isfinite(number) else "N/A"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true", help="freeze metadata, calibrate models, and run technical checks; do not run the evaluation matrix")
    parser.add_argument("--overwrite", action="store_true", help="replace only the E024 common-probe output folder")
    args = parser.parse_args()
    has_frozen = all(path.exists() for path in (
        META_DIR / "run_config.json",
        SUMMARY_DIR / "model_registry.json",
        SUMMARY_DIR / "calibration_grid.csv",
        SUMMARY_DIR / "probe_manifest.csv",
    ))
    if OUT_ROOT.exists() and any(OUT_ROOT.iterdir()) and not args.overwrite and not has_frozen:
        raise FileExistsError(f"Refusing to overwrite incomplete E024 output: {OUT_ROOT}; inspect it or pass --overwrite intentionally")
    if has_frozen and not args.overwrite:
        if (SUMMARY_DIR / "primary_by_seed.csv").exists():
            raise FileExistsError(f"E024 evaluation results already exist: {SUMMARY_DIR / 'primary_by_seed.csv'}; pass --overwrite only for an intentional rerun")
        selected_gains, failed = _load_frozen_artifacts()
        checks = _preflight_checks()
        _write_csv(META_DIR / "preflight_checks.csv", checks)
        if args.preflight_only:
            print(f"E024 preflight freeze is ready. Frozen artifacts: {OUT_ROOT}")
            return
        _run_evaluation(selected_gains, failed)
        print(f"E024 complete. Results: {OUT_ROOT}")
        return
    if OUT_ROOT.exists() and args.overwrite:
        # Remove only the fixed, explicitly checked E024 destination.
        resolved = OUT_ROOT.resolve()
        expected = (REPO_ROOT / "results" / "e024_common_probe_comparison").resolve()
        if resolved != expected or REPO_ROOT not in resolved.parents:
            raise RuntimeError(f"Unsafe E024 output path: {resolved}")
        import shutil
        shutil.rmtree(resolved)

    checks = _preflight_checks()
    _write_csv(OUT_ROOT / "metadata" / "preflight_checks.csv", checks)
    selected_gains, calibration_rows, failed = calibrate_models()
    probe_manifest: list[dict[str, Any]] = []
    for seed in EVALUATION_SEEDS:
        for motif in _build_motifs(seed):
            probe_manifest.extend(build_probe_manifest(motif, seed, "evaluation"))
    _save_freeze_artifacts(selected_gains, failed, probe_manifest)
    _write_csv(SUMMARY_DIR / "calibration_grid.csv", calibration_rows)
    if args.preflight_only:
        print(f"E024 preflight complete. Frozen artifacts: {OUT_ROOT}")
        return
    _run_evaluation(selected_gains, failed)
    print(f"E024 complete. Results: {OUT_ROOT}")


if __name__ == "__main__":
    main()
