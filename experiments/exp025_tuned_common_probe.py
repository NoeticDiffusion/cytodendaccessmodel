"""E025: equal-budget tuning and locked common-probe comparison.

The registry and topology/probe manifests are generated before this script is
frozen. ``--preflight`` creates a source-hash lock; ``--run`` refuses to run if
any locked input changed. E024 is imported for its frozen model definitions and
probe semantics, but E024 files and results are never written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import random
import statistics
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NEURAL_DIR = ROOT / "project" / "neural_computation"
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cytodend_accessmodel.contracts import DynamicsParameters  # noqa: E402
from cytodend_accessmodel.motifs import MotifSpec  # noqa: E402

E024_PATH = ROOT / "experiments" / "exp024_common_probe_comparison.py"
E024_SPEC = importlib.util.spec_from_file_location("e024_frozen", E024_PATH)
if E024_SPEC is None or E024_SPEC.loader is None:
    raise RuntimeError(f"Cannot load E024 implementation at {E024_PATH}")
e024 = importlib.util.module_from_spec(E024_SPEC)
sys.modules[E024_SPEC.name] = e024
E024_SPEC.loader.exec_module(e024)

PARAM_GEN_PATH = NEURAL_DIR / "generate_e025_parameter_registry.py"
PARAM_GEN_SPEC = importlib.util.spec_from_file_location("e025_param_registry", PARAM_GEN_PATH)
if PARAM_GEN_SPEC is None or PARAM_GEN_SPEC.loader is None:
    raise RuntimeError(f"Cannot load parameter registry specification at {PARAM_GEN_PATH}")
param_gen = importlib.util.module_from_spec(PARAM_GEN_SPEC)
sys.modules[PARAM_GEN_SPEC.name] = param_gen
PARAM_GEN_SPEC.loader.exec_module(param_gen)

OUT_ROOT = ROOT / "results" / "e025_common_probe_tuning"
META_DIR = OUT_ROOT / "metadata"
SCORE_DIR = OUT_ROOT / "scores"
SUMMARY_DIR = OUT_ROOT / "summary"
PLAN_PATH = NEURAL_DIR / "e025_preregistration_plan.md"
REGISTRY_PATH = NEURAL_DIR / "e025_parameter_registry.csv"
TOPOLOGY_PATH = NEURAL_DIR / "e025_topology_manifest.csv"
PROBE_PATH = NEURAL_DIR / "e025_probe_manifest.csv"
REGISTRY_HASH_PATH = REGISTRY_PATH.with_suffix(REGISTRY_PATH.suffix + ".sha256")
TOPOLOGY_HASH_PATH = TOPOLOGY_PATH.with_suffix(TOPOLOGY_PATH.suffix + ".sha256")
PROBE_HASH_PATH = PROBE_PATH.with_suffix(PROBE_PATH.suffix + ".sha256")
PARAM_GENERATOR_PATH = NEURAL_DIR / "generate_e025_parameter_registry.py"
TOPOLOGY_GENERATOR_PATH = NEURAL_DIR / "generate_e025_topology_manifests.py"
FREEZE_PATH = META_DIR / "pre_run_freeze.json"
RUNNING_PATH = OUT_ROOT / "run_in_progress.json"

MODEL_ORDER = tuple(e024.MODEL_ORDER)
PRIMARY_STRATA = ("canonical", "strong_overlap", "chain_overlap", "sparse_random")
ALL_STRATA = (*PRIMARY_STRATA, "hub_overlap")
SPLIT_COUNTS = {"development": 40, "validation": 40, "holdout": 100}
CONSOLIDATION_PASSES = 9
WASHOUT_PASSES = 9
MIN_EFFECT = 0.02
BOOTSTRAP_REPLICATES = 10_000
RANDOMIZATION_REPLICATES = 100_000
INFERENCE_SEED = 250925
NO_REPLAY_SENTINEL = "__E025_EXPLICIT_NO_REPLAY__"
PARAMETER_FIELDS = tuple(DynamicsParameters.__dataclass_fields__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _verify_sidecar(path: Path) -> str:
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() or not digest_path.exists():
        raise RuntimeError(f"Missing generated E025 artifact or SHA-256 sidecar: {path}")
    expected = digest_path.read_text(encoding="ascii").split()[0]
    actual = _sha256(path)
    if expected != actual:
        raise RuntimeError(f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}")
    return actual


def _source_paths() -> list[Path]:
    return [
        Path(__file__),
        PLAN_PATH,
        PARAM_GENERATOR_PATH,
        TOPOLOGY_GENERATOR_PATH,
        REGISTRY_PATH,
        REGISTRY_HASH_PATH,
        TOPOLOGY_PATH,
        TOPOLOGY_HASH_PATH,
        PROBE_PATH,
        PROBE_HASH_PATH,
        E024_PATH,
        SRC_DIR / "cytodend_accessmodel" / "simulator.py",
        SRC_DIR / "cytodend_accessmodel" / "contracts.py",
        SRC_DIR / "cytodend_accessmodel" / "motifs.py",
    ]


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines()
        return {"commit": commit, "working_tree_status": status}
    except Exception as exc:
        return {"commit": None, "error": str(exc)}


def _load_registry() -> list[dict[str, Any]]:
    raw = _read_csv(REGISTRY_PATH)
    expected_models = set(MODEL_ORDER)
    if len(raw) != 320:
        raise RuntimeError(f"Expected 320 candidate rows, got {len(raw)}")
    rows: list[dict[str, Any]] = []
    gains: dict[str, list[float]] = {name: [] for name in MODEL_ORDER}
    ids: dict[str, set[str]] = {name: set() for name in MODEL_ORDER}
    for item in raw:
        model = item["model"]
        if model not in expected_models:
            raise RuntimeError(f"Unknown candidate family {model}")
        row: dict[str, Any] = dict(item)
        for field in (*PARAMETER_FIELDS, "hebbian_lr", "drift_sd", "e024_selected_fast_gain"):
            row[field] = float(item[field])
        row["active_tuning_parameters"] = tuple(item["active_tuning_parameters"].split(";"))
        gains[model].append(row["fast_gain"])
        if row["candidate_id"] in ids[model]:
            raise RuntimeError(f"Duplicate candidate ID: {row['candidate_id']}")
        ids[model].add(row["candidate_id"])
        rows.append(row)
    if any(len(ids[name]) != 32 for name in MODEL_ORDER):
        raise RuntimeError("Each model family must have exactly 32 unique candidate IDs")
    reference = gains[MODEL_ORDER[0]]
    if any(values != reference for values in gains.values()):
        raise RuntimeError("Candidate registry does not use a common fast_gain list")
    if len(set(reference)) != 32:
        raise RuntimeError("fast_gain values must be unique within the common candidate list")
    for row in rows:
        for parameter in row["active_tuning_parameters"]:
            lo, hi = param_gen.RANGES[parameter]
            if not (lo - 1e-12 <= float(row[parameter]) <= hi + 1e-12):
                raise RuntimeError(f"Candidate out of range: {row['candidate_id']} {parameter}")
    return rows


def _load_topologies() -> tuple[dict[str, dict[str, dict[str, Any]]], dict[tuple[str, str], list[dict[str, str]]]]:
    topo_raw = _read_csv(TOPOLOGY_PATH)
    probe_raw = _read_csv(PROBE_PATH)
    by_draw: dict[str, dict[str, dict[str, Any]]] = {}
    by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in topo_raw:
        partition, draw_id, stratum = row["partition"], row["draw_id"], row["stratum"]
        branches = json.loads(row["branch_ids_json"])
        traces = json.loads(row["trace_ids_json"])
        allocations = json.loads(row["allocation_json"])
        if len(branches) != int(row["n_branches"]) or len(traces) != int(row["n_traces"]):
            raise RuntimeError(f"Manifest dimension mismatch at {row['motif_id']}")
        overlap: dict[str, list[str]] = {}
        linked: list[tuple[str, str]] = []
        unlinked: list[tuple[str, str]] = []
        for i, first in enumerate(traces):
            for second in traces[i + 1:]:
                common = [b for b in branches if allocations[first][b] > 0.40 and allocations[second][b] > 0.40]
                key = ":".join(sorted((first, second)))
                overlap[key] = common
                (linked if common else unlinked).append((first, second))
        shared_any = [b for values in overlap.values() for b in values]
        motif = MotifSpec(
            motif_id=row["motif_id"],
            motif_type=stratum,
            n_branches=len(branches),
            n_traces=len(traces),
            branch_ids=branches,
            trace_ids=traces,
            allocations=allocations,
            overlap_branches_per_pair=overlap,
            expected_linked_pairs=linked,
            expected_unlinked_pairs=unlinked,
            damage_target_branches=shared_any[:1] or branches[:1],
            rescue_target_branches=shared_any[:1] or branches[:1],
            metadata={"partition": partition, "split_seed": int(row["split_seed"])},
        )
        by_draw.setdefault(draw_id, {})[stratum] = {
            "partition": partition,
            "split_seed": int(row["split_seed"]),
            "motif": motif,
        }
    for row in probe_raw:
        key = (row["draw_id"], row["stratum"])
        by_key.setdefault(key, []).append(row)
    for draw_id, strata in by_draw.items():
        if set(strata) != set(ALL_STRATA):
            raise RuntimeError(f"Topology draw {draw_id} does not contain the five frozen strata")
        for stratum in PRIMARY_STRATA:
            if not any(row["primary_status"] == "defined" for row in by_key[(draw_id, stratum)]):
                raise RuntimeError(f"Primary topology has no eligible pair: {draw_id}/{stratum}")
    counts: dict[str, int] = {}
    for strata in by_draw.values():
        partition = next(iter(strata.values()))["partition"]
        counts[partition] = counts.get(partition, 0) + 1
    if counts != SPLIT_COUNTS:
        raise RuntimeError(f"Unexpected topology split sizes: {counts}")
    return by_draw, by_key


def _freeze_inputs() -> dict[str, Any]:
    for path in (REGISTRY_PATH, TOPOLOGY_PATH, PROBE_PATH):
        _verify_sidecar(path)
    registry = _load_registry()
    topology, _ = _load_topologies()
    if len(registry) != 320 or len(topology) != sum(SPLIT_COUNTS.values()):
        raise RuntimeError("E025 manifest dimensions do not match the preregistration")
    missing = [path for path in _source_paths() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing files needed for freeze: {missing}")
    hashes = {path.relative_to(ROOT).as_posix(): _sha256(path) for path in _source_paths()}
    return {
        "experiment": "E025 tuned common-probe comparison",
        "protocol_version": "1.0-preregistered",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": hashes,
        "generated_file_sha256": {
            "parameter_registry": _sha256(REGISTRY_PATH),
            "topology_manifest": _sha256(TOPOLOGY_PATH),
            "probe_manifest": _sha256(PROBE_PATH),
        },
        "candidate_count_per_model": 32,
        "models": list(MODEL_ORDER),
        "split_draw_counts": SPLIT_COUNTS,
        "primary_strata": list(PRIMARY_STRATA),
        "sensitivity_stratum": "hub_overlap",
        "consolidation_passes": CONSOLIDATION_PASSES,
        "washout_passes": WASHOUT_PASSES,
        "minimum_effect": MIN_EFFECT,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "randomization_replicates": RANDOMIZATION_REPLICATES,
        "inference_seed": INFERENCE_SEED,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git": _git_state(),
    }


def _preflight() -> None:
    freeze = _freeze_inputs()
    if FREEZE_PATH.exists():
        existing = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if existing.get("input_sha256") != freeze["input_sha256"]:
            raise RuntimeError("A different pre-run freeze already exists; refusing to replace it")
        print(f"E025 inputs already frozen: {FREEZE_PATH}")
        return
    META_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(FREEZE_PATH, freeze)
    print(f"E025 preflight locked 320 candidates and 180 topology draws: {FREEZE_PATH}")


def _check_frozen_run() -> None:
    if not FREEZE_PATH.exists():
        raise RuntimeError("No pre-run hash lock; run --preflight first")
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    current = {path.relative_to(ROOT).as_posix(): _sha256(path) for path in _source_paths()}
    if current != frozen.get("input_sha256"):
        raise RuntimeError("A frozen input changed after preflight; create a new E025 version")
    if RUNNING_PATH.exists():
        raise RuntimeError("A previous E025 run left a run_in_progress marker")
    existing_results = [p for p in OUT_ROOT.rglob("*") if p.is_file() and p != FREEZE_PATH]
    if existing_results:
        raise RuntimeError(f"Output folder is not empty; refusing to overwrite: {existing_results[:3]}")


def _dynamics_seed(partition: str, draw_id: str, stratum: str) -> int:
    text = f"e025_dynamics_v1|{partition}|{draw_id}|{stratum}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _model_for(candidate: dict[str, Any], topology: dict[str, Any]) -> Any:
    motif: MotifSpec = topology["motif"]
    # Use the full draw ID, which disambiguates split seeds.
    seed = _dynamics_seed(topology["partition"], topology["draw_id"], motif.motif_type)
    model = e024.build_model(candidate["model"], motif, fast_gain=candidate["fast_gain"], seed=seed)
    dynamics = DynamicsParameters(**{name: float(candidate[name]) for name in PARAMETER_FIELDS})
    model.params = dynamics
    model.simulator.parameters = dynamics
    for branch in model.simulator.branches.values():
        branch.structural.decay_rate = dynamics.structural_decay
        branch.structural.max_accessibility = dynamics.structural_max
        branch.structural.noise_scale = dynamics.structural_noise
    e024.HEBBIAN_LR = float(candidate["hebbian_lr"])
    e024.SHUFFLE_SIGMA = float(candidate["drift_sd"])
    return e024.train_model(model, consolidate=False)


def _private_metrics(model: Any) -> tuple[float, float, int]:
    correct_values: list[float] = []
    margins: list[float] = []
    for trace_id in model.motif.trace_ids:
        for branch_id in e024.private_branches(model.motif, trace_id):
            scores = e024.probe_readout(model, e024._one_hot(branch_id))
            correct = scores[trace_id]["readout"]
            wrong = max((score["readout"] for tid, score in scores.items() if tid != trace_id), default=float("nan"))
            if math.isfinite(correct) and math.isfinite(wrong):
                correct_values.append(correct)
                margins.append(correct - wrong)
    return (
        statistics.mean(correct_values) if correct_values else float("nan"),
        statistics.mean(margins) if margins else float("nan"),
        len(correct_values),
    )


def _effect(pre: Any, post: Any, pair_rows: list[dict[str, str]]) -> tuple[float, dict[str, float], int, int]:
    valid = [row for row in pair_rows if row["primary_status"] == "defined"]
    direction_scores: list[float] = []
    raw: dict[str, list[float]] = {key: [] for key in ("shared_pre", "control_pre", "shared_post", "control_post")}
    n_combinations = 0
    for row in valid:
        shared = [b for b in row["shared_branches"].split(";") if b]
        controls = [b for b in row["source_private_branches"].split(";") if b]
        pair_values: list[float] = []
        for shared_branch in shared:
            for control_branch in controls:
                vals: dict[str, float] = {}
                for stage, model in (("pre", pre), ("post", post)):
                    vals[f"shared_{stage}"] = e024.probe_readout(model, e024._one_hot(shared_branch))[row["target_trace"]]["readout"]
                    vals[f"control_{stage}"] = e024.probe_readout(model, e024._one_hot(control_branch))[row["target_trace"]]["readout"]
                delta = (vals["shared_post"] - vals["control_post"]) - (vals["shared_pre"] - vals["control_pre"])
                pair_values.append(delta)
                for key in raw:
                    raw[key].append(vals[key])
                n_combinations += 1
        if pair_values:
            direction_scores.append(statistics.mean(pair_values))
    return (
        statistics.mean(direction_scores) if direction_scores else float("nan"),
        {key: statistics.mean(values) if values else float("nan") for key, values in raw.items()},
        len(valid),
        n_combinations,
    )


def _passive_washout(model: Any) -> Any:
    late = deepcopy(model)
    params = late.params
    for _ in range(WASHOUT_PASSES):
        for branch in late.simulator.branches.values():
            branch.eligibility.value = max(0.0, min(1.0, (1.0 - params.eligibility_decay) * branch.eligibility.value))
            branch.translation_readiness.value = max(0.0, min(1.0, (1.0 - params.translation_decay) * branch.translation_readiness.value))
            state = branch.structural
            state.accessibility = max(0.0, min(state.max_accessibility, state.accessibility - state.decay_rate * state.accessibility))
            branch.slow_access = 1.0 / (1.0 + math.exp(-params.structural_gain * state.accessibility))
    return late


def _evaluate_topology(
    candidate: dict[str, Any],
    topology: dict[str, Any],
    pair_rows: list[dict[str, str]],
    reference_model: Any,
    *,
    include_late: bool,
) -> dict[str, Any]:
    if candidate["model"] == "full_model" and candidate["candidate_id"] == "full_model_c02":
        pre = deepcopy(reference_model)
    else:
        pre = _model_for(candidate, topology)
    pre_private, pre_margin, n_private = _private_metrics(pre)
    post = deepcopy(pre)
    for pass_index in range(CONSOLIDATION_PASSES):
        e024._consolidate_one(post, pass_index)
    post_private, post_margin, _ = _private_metrics(post)
    immediate, levels, n_pairs, n_combos = _effect(pre, post, pair_rows)
    late_d = float("nan")
    late_levels: dict[str, float] = {}
    if include_late:
        late = _passive_washout(post)
        late_d, late_levels, _, _ = _effect(pre, late, pair_rows)
    reference_private, _, _ = _private_metrics(reference_model)
    return {
        "D": immediate,
        "D_late": late_d,
        "D_late_minus_D": late_d - immediate if math.isfinite(late_d) else float("nan"),
        "baseline_difference": pre_private - reference_private,
        "private_pre": pre_private,
        "private_post": post_private,
        "private_post_minus_pre": post_private - pre_private,
        "private_margin_pre": pre_margin,
        "private_margin_post": post_margin,
        "n_private_cues": n_private,
        "n_defined_pairs": n_pairs,
        "n_probe_combinations": n_combos,
        **levels,
        **{f"late_{key}": value for key, value in late_levels.items()},
    }


def _references_for_partition(
    partition: str,
    draws: dict[str, dict[str, dict[str, Any]]],
    pair_index: dict[tuple[str, str], list[dict[str, str]]],
    nominal_full: dict[str, Any],
) -> dict[tuple[str, str], Any]:
    references: dict[tuple[str, str], Any] = {}
    for draw_id, strata in draws.items():
        if next(iter(strata.values()))["partition"] != partition:
            continue
        for stratum in PRIMARY_STRATA:
            topology = dict(strata[stratum])
            topology["draw_id"] = draw_id
            candidate = dict(nominal_full)
            model = _model_for(candidate, topology)
            references[(draw_id, stratum)] = model
    return references


def _evaluate_partition(
    partition: str,
    candidates: list[dict[str, Any]],
    draws: dict[str, dict[str, dict[str, Any]]],
    pair_index: dict[tuple[str, str], list[dict[str, str]]],
    references: dict[tuple[str, str], Any],
    *,
    include_late: bool = False,
    include_hub: bool = False,
    role: str = "candidate",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_draws = [
        draw_id for draw_id, strata in draws.items()
        if next(iter(strata.values()))["partition"] == partition
    ]
    total = len(candidates) * len(selected_draws)
    completed = 0
    for candidate in candidates:
        for draw_id in selected_draws:
            strata_metrics: dict[str, dict[str, float]] = {}
            for stratum in (*PRIMARY_STRATA, *(('hub_overlap',) if include_hub else ())):
                topology = dict(draws[draw_id][stratum])
                topology["draw_id"] = draw_id
                # The hub is a defined sensitivity endpoint, but the primary
                # full-model nominal reference is still used for its baseline.
                if stratum == "hub_overlap":
                    ref_candidate = dict(NOMINAL_BY_MODEL["full_model"])
                    reference_model = _model_for(ref_candidate, topology)
                else:
                    reference_model = references[(draw_id, stratum)]
                strata_metrics[stratum] = _evaluate_topology(
                    candidate,
                    topology,
                    pair_index[(draw_id, stratum)],
                    reference_model,
                    include_late=include_late,
                )
            primary = [strata_metrics[stratum] for stratum in PRIMARY_STRATA]
            row = {
                "partition": partition,
                "role": role,
                "draw_id": draw_id,
                "model": candidate["model"],
                "candidate_id": candidate["candidate_id"],
                "fast_gain": candidate["fast_gain"],
                "D_macro": statistics.mean(item["D"] for item in primary),
                "D_late_macro": statistics.mean(item["D_late"] for item in primary) if include_late else float("nan"),
                "D_late_minus_D_macro": statistics.mean(item["D_late_minus_D"] for item in primary) if include_late else float("nan"),
                "stratum_metrics_json": json.dumps(strata_metrics, sort_keys=True, separators=(",", ":"), allow_nan=True),
            }
            rows.append(row)
            completed += 1
        print(f"{partition}: {candidate['model']} {candidate['candidate_id']} complete ({completed}/{total})", flush=True)
    return rows


def nominal_candidate_for(candidate: dict[str, Any]) -> dict[str, Any]:
    # Filled from the frozen registry by _nominal_map before evaluation.
    return NOMINAL_BY_MODEL[candidate["model"]]


NOMINAL_BY_MODEL: dict[str, dict[str, Any]] = {}


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return statistics.mean(values) if values else float("nan")


def _candidate_summary(rows: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(row["candidate_id"], []).append(row)
    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        group = by_id.get(candidate["candidate_id"], [])
        summary: dict[str, Any] = {
            "partition": group[0]["partition"] if group else "",
            "model": candidate["model"],
            "candidate_id": candidate["candidate_id"],
            "n_draws": len(group),
            "mean_D": _mean(group, "D_macro"),
            "functional_pass": False,
        }
        parsed = [json.loads(row["stratum_metrics_json"]) for row in group]
        pass_components: list[bool] = []
        nominal = NOMINAL_BY_MODEL[candidate["model"]]
        distance_sq = 0.0
        for parameter in candidate["active_tuning_parameters"]:
            lo, hi = param_gen.RANGES[parameter]
            distance_sq += ((float(candidate[parameter]) - float(nominal[parameter])) / (hi - lo)) ** 2
        summary["nominal_distance"] = math.sqrt(distance_sq)
        for stratum in PRIMARY_STRATA:
            vals = [row[stratum] for row in parsed]
            baseline_error = statistics.mean(item["baseline_difference"] for item in vals)
            retention = statistics.mean(item["private_post_minus_pre"] for item in vals)
            margin = statistics.mean(item["private_margin_post"] for item in vals)
            summary[f"baseline_difference_{stratum}"] = baseline_error
            summary[f"private_post_minus_pre_{stratum}"] = retention
            summary[f"private_margin_post_{stratum}"] = margin
            pass_components.extend((abs(baseline_error) <= MIN_EFFECT, retention >= -MIN_EFFECT, margin >= 0.0))
            summary[f"mean_D_{stratum}"] = statistics.mean(item["D"] for item in vals)
        summary["functional_pass"] = bool(group) and all(pass_components)
        summary["baseline_pass"] = bool(group) and all(
            abs(float(summary[f"baseline_difference_{stratum}"])) <= MIN_EFFECT for stratum in PRIMARY_STRATA
        )
        summary["retention_pass"] = bool(group) and all(
            float(summary[f"private_post_minus_pre_{stratum}"]) >= -MIN_EFFECT for stratum in PRIMARY_STRATA
        )
        summary["discrimination_pass"] = bool(group) and all(
            float(summary[f"private_margin_post_{stratum}"]) >= 0.0 for stratum in PRIMARY_STRATA
        )
        summaries.append(summary)
    return summaries


def _ordered(candidates: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_by_id = {row["candidate_id"]: row for row in summaries}
    return sorted(
        candidates,
        key=lambda candidate: (
            -float(summary_by_id[candidate["candidate_id"]]["mean_D"]),
            float(summary_by_id[candidate["candidate_id"]]["nominal_distance"]),
            candidate["candidate_id"],
        ),
    )


def _select_candidates(
    dev_summaries: list[dict[str, Any]],
    val_summaries: list[dict[str, Any]],
    registry: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    selected: dict[str, dict[str, Any]] = {}
    report: list[dict[str, Any]] = []
    by_model = {name: [row for row in registry if row["model"] == name] for name in MODEL_ORDER}
    dev_by = {name: [row for row in dev_summaries if row["model"] == name] for name in MODEL_ORDER}
    val_by = {name: [row for row in val_summaries if row["model"] == name] for name in MODEL_ORDER}
    for model in MODEL_ORDER:
        ds = dev_by[model]
        dev_pass = {row["candidate_id"] for row in ds if row["functional_pass"]}
        ordered_dev = [item for item in _ordered(by_model[model], ds) if item["candidate_id"] in dev_pass]
        shortlist = ordered_dev[:8]
        val_good = {row["candidate_id"] for row in val_by[model] if row["functional_pass"]}
        selected_candidates = [item for item in shortlist if item["candidate_id"] in val_good]
        if selected_candidates:
            ordered_val = _ordered(selected_candidates, val_by[model])
            selected[model] = ordered_val[0]
            status = "selected"
        else:
            status = "no_qualifying_tuned_candidate"
        report.append({
            "model": model,
            "n_development_qualified": len(dev_pass),
            "n_shortlisted": len(shortlist),
            "n_validation_qualified": len(selected_candidates),
            "selected_candidate_id": selected.get(model, {}).get("candidate_id", ""),
            "status": status,
            "development_shortlist": ";".join(item["candidate_id"] for item in shortlist),
        })
    return selected, report


def _bootstrap_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    indices = BOOTSTRAP_INDICES
    means = np.mean(array[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _randomization_p(centered: list[float], alternative: str, seed: int) -> float:
    values = np.asarray(centered, dtype=float)
    if values.size == 0:
        return float("nan")
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    exceed = 0
    batch = 5000
    done = 0
    while done < RANDOMIZATION_REPLICATES:
        size = min(batch, RANDOMIZATION_REPLICATES - done)
        signs = rng.integers(0, 2, size=(size, values.size), dtype=np.int8) * 2 - 1
        permuted = np.mean(signs * values[None, :], axis=1)
        if alternative == "greater":
            exceed += int(np.count_nonzero(permuted >= observed - 1e-15))
        else:
            exceed += int(np.count_nonzero(permuted <= observed + 1e-15))
        done += size
    return (exceed + 1) / (RANDOMIZATION_REPLICATES + 1)


def _holm(pvalues: list[float]) -> list[float]:
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: (math.inf if not math.isfinite(pvalues[i]) else pvalues[i], i))
    adjusted = [float("nan")] * n
    running = 0.0
    for rank, idx in enumerate(order):
        if not math.isfinite(pvalues[idx]):
            continue
        running = max(running, min(1.0, (n - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted


def _holdout_analysis(
    holdout_rows: list[dict[str, Any]],
    holdout_summaries: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    result_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    global BOOTSTRAP_INDICES
    bootstrap_rng = np.random.default_rng(INFERENCE_SEED)
    BOOTSTRAP_INDICES = bootstrap_rng.integers(
        0, SPLIT_COUNTS["holdout"], size=(BOOTSTRAP_REPLICATES, SPLIT_COUNTS["holdout"])
    )
    for row in holdout_summaries:
        model, candidate_id = row["model"], row["candidate_id"]
        group = [item for item in holdout_rows if item["model"] == model and item["candidate_id"] == candidate_id]
        macro = [float(item["D_macro"]) for item in group]
        late = [float(item["D_late_macro"]) for item in group]
        d_lo, d_hi = _bootstrap_ci(macro)
        l_lo, l_hi = _bootstrap_ci(late)
        summary_rows.append({
            "model": model,
            "candidate_id": candidate_id,
            "n_topology_draws": len(group),
            "mean_D": statistics.mean(macro) if macro else float("nan"),
            "D_ci95_low": d_lo,
            "D_ci95_high": d_hi,
            "mean_D_late": statistics.mean(late) if late else float("nan"),
            "D_late_ci95_low": l_lo,
            "D_late_ci95_high": l_hi,
            "functional_pass_holdout": row["functional_pass"],
        })

    valid_families = {row["model"] for row in summary_rows if row["functional_pass_holdout"]}
    full_summary = next((row for row in summary_rows if row["model"] == "full_model"), None)
    comparisons: list[dict[str, Any]] = []
    hypotheses: list[dict[str, Any]] = []
    directions = ("full_superiority", "comparator_superiority", "practical_equivalence")
    for endpoint_index, (endpoint, field) in enumerate((("D", "D_macro"), ("D_late", "D_late_macro"))):
        for comparator_index, comparator in enumerate(name for name in MODEL_ORDER if name != "full_model"):
            available = comparator in valid_families and "full_model" in valid_families
            full_rows = {item["draw_id"]: item for item in holdout_rows if item["model"] == "full_model" and item["candidate_id"] == selected.get("full_model", {}).get("candidate_id")}
            comp_rows = {item["draw_id"]: item for item in holdout_rows if item["model"] == comparator and item["candidate_id"] == selected.get(comparator, {}).get("candidate_id")}
            draw_ids = sorted(set(full_rows) & set(comp_rows)) if available else []
            diffs = [float(full_rows[key][field]) - float(comp_rows[key][field]) for key in draw_ids]
            if diffs and all(math.isfinite(value) for value in diffs):
                lo, hi = _bootstrap_ci(diffs)
                test_seed = INFERENCE_SEED + 1000 * endpoint_index + 10 * comparator_index
                h_full = _randomization_p([value - MIN_EFFECT for value in diffs], "greater", test_seed + 1)
                h_comp = _randomization_p([-value - MIN_EFFECT for value in diffs], "greater", test_seed + 2)
                p_eq_lo = _randomization_p([value + MIN_EFFECT for value in diffs], "greater", test_seed + 3)
                p_eq_hi = _randomization_p([value - MIN_EFFECT for value in diffs], "less", test_seed + 4)
                h_eq = max(p_eq_lo, p_eq_hi)
                raw_ps = (h_full, h_comp, h_eq)
            else:
                lo = hi = h_full = h_comp = h_eq = p_eq_lo = p_eq_hi = float("nan")
                raw_ps = (h_full, h_comp, h_eq)
            hypotheses.extend({
                "endpoint": endpoint,
                "comparator": comparator,
                "hypothesis": direction,
                "p_raw": pvalue,
            } for direction, pvalue in zip(directions, raw_ps))
            comparisons.append({
                "endpoint": endpoint,
                "comparator": comparator,
                "n_topology_draws": len(diffs),
                "mean_delta_full_minus_comparator": statistics.mean(diffs) if diffs else float("nan"),
                "delta_ci95_low": lo,
                "delta_ci95_high": hi,
                "p_full_superiority_raw": h_full,
                "p_comparator_superiority_raw": h_comp,
                "p_equivalence_lower_raw": p_eq_lo,
                "p_equivalence_upper_raw": p_eq_hi,
                "p_equivalence_composite_raw": h_eq,
                "status": "testable" if diffs else "unavailable_no_qualifying_tuned_candidate",
            })
    adjusted = _holm([row["p_raw"] for row in hypotheses])
    for row, p_adj in zip(hypotheses, adjusted):
        row["p_holm_54"] = p_adj
    adj_map = {(row["endpoint"], row["comparator"], row["hypothesis"]): row["p_holm_54"] for row in hypotheses}
    for row in comparisons:
        ep, comparator = row["endpoint"], row["comparator"]
        pfs = adj_map[(ep, comparator, "full_superiority")]
        pcs = adj_map[(ep, comparator, "comparator_superiority")]
        peq = adj_map[(ep, comparator, "practical_equivalence")]
        decisions = [name for name, value in (("full_superiority", pfs), ("comparator_superiority", pcs), ("practical_equivalence", peq)) if math.isfinite(value) and value < 0.05]
        row.update({
            "p_full_superiority_holm_54": pfs,
            "p_comparator_superiority_holm_54": pcs,
            "p_equivalence_holm_54": peq,
            "decision": decisions[0] if len(decisions) == 1 else ("inconsistent_multiple_rejections" if decisions else "inconclusive"),
        })
    primary_claim = bool(
        full_summary
        and full_summary["functional_pass_holdout"]
        and math.isfinite(float(full_summary["D_ci95_low"]))
        and float(full_summary["D_ci95_low"]) > 0.0
        and all(
            any(row["endpoint"] == "D" and row["comparator"] == comp and row["decision"] == "full_superiority" for row in comparisons)
            for comp in (name for name in MODEL_ORDER if name != "full_model")
        )
        and all(
            next(row for row in summary_rows if row["model"] == comp)["functional_pass_holdout"]
            for comp in (name for name in MODEL_ORDER if name != "full_model")
        )
    )
    return comparisons, hypotheses, {"unique_immediate_advantage_supported": primary_claim}


def _score_summary(rows: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = _candidate_summary(rows, candidates)
    return result


def _hub_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["candidate_id"], row["role"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (model, candidate_id, role), group in sorted(grouped.items()):
        values = [json.loads(row["stratum_metrics_json"])["hub_overlap"] for row in group]
        summaries.append({
            "model": model,
            "candidate_id": candidate_id,
            "role": role,
            "n_topology_draws": len(group),
            "mean_D_hub": statistics.mean(item["D"] for item in values),
            "mean_D_late_hub": statistics.mean(item["D_late"] for item in values),
            "mean_defined_pairs": statistics.mean(item["n_defined_pairs"] for item in values),
            "status": "descriptive_sensitivity_only",
        })
    return summaries


def _build_nominal_map(registry: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for model in MODEL_ORDER:
        row = next(item for item in registry if item["model"] == model and item["candidate_id"] == f"{model}_c02")
        result[model] = row
    return result


def _annotate_topology(draws: dict[str, dict[str, dict[str, Any]]]) -> None:
    for draw_id, strata in draws.items():
        for item in strata.values():
            item["draw_id"] = draw_id


def _run() -> None:
    _check_frozen_run()
    registry = _load_registry()
    draws, pair_index = _load_topologies()
    _annotate_topology(draws)
    global NOMINAL_BY_MODEL
    NOMINAL_BY_MODEL = _build_nominal_map(registry)
    RUNNING_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_json(RUNNING_PATH, {"started_utc": datetime.now(timezone.utc).isoformat(), "freeze_sha256": _sha256(FREEZE_PATH)})
    try:
        dev_candidates = registry
        dev_refs = _references_for_partition("development", draws, pair_index, NOMINAL_BY_MODEL["full_model"])
        dev_rows = _evaluate_partition("development", dev_candidates, draws, pair_index, dev_refs)
        dev_scores = _score_summary(dev_rows, dev_candidates)
        _write_csv(SCORE_DIR / "development_draw_scores.csv", dev_rows)
        _write_csv(SCORE_DIR / "development_candidate_scores.csv", dev_scores)

        shortlist: list[dict[str, Any]] = []
        for model in MODEL_ORDER:
            scores = [row for row in dev_scores if row["model"] == model]
            qualified = {row["candidate_id"] for row in scores if row["functional_pass"]}
            ordered = [item for item in _ordered([row for row in registry if row["model"] == model], scores) if item["candidate_id"] in qualified]
            shortlist.extend(ordered[:8])
        val_refs = _references_for_partition("validation", draws, pair_index, NOMINAL_BY_MODEL["full_model"])
        val_rows = _evaluate_partition("validation", shortlist, draws, pair_index, val_refs)
        val_scores = _score_summary(val_rows, shortlist)
        _write_csv(SCORE_DIR / "validation_draw_scores.csv", val_rows)
        _write_csv(SCORE_DIR / "validation_candidate_scores.csv", val_scores)

        selected, selection_report = _select_candidates(dev_scores, val_scores, registry)
        _write_csv(SUMMARY_DIR / "candidate_selection.csv", selection_report)

        # Candidate selection is now complete. Open holdout once, and score
        # only each selected candidate plus the E024-nominal candidate.
        holdout_candidates: list[tuple[str, dict[str, Any]]] = []
        for model in MODEL_ORDER:
            nominal = NOMINAL_BY_MODEL[model]
            tuned = selected.get(model)
            if tuned is not None and tuned["candidate_id"] == nominal["candidate_id"]:
                holdout_candidates.append(("tuned_and_nominal", nominal))
            else:
                if tuned is not None:
                    holdout_candidates.append(("tuned", tuned))
                holdout_candidates.append(("nominal", nominal))
        holdout_nominal_refs = _references_for_partition("holdout", draws, pair_index, NOMINAL_BY_MODEL["full_model"])
        holdout_rows: list[dict[str, Any]] = []
        for role, candidate in holdout_candidates:
            rows = _evaluate_partition("holdout", [candidate], draws, pair_index, holdout_nominal_refs, include_late=True, include_hub=True, role=role)
            holdout_rows.extend(rows)
        _write_csv(SCORE_DIR / "holdout_draw_scores.csv", holdout_rows)

        tuned_rows = [row for row in holdout_rows if row["role"] in {"tuned", "tuned_and_nominal"}]
        tuned_candidates = [selected[m] for m in MODEL_ORDER if m in selected]
        holdout_scores = _score_summary(tuned_rows, tuned_candidates)
        # If the selected tuned candidate is also nominal, the role is still
        # included above; _candidate_summary keys on candidate ID.
        _write_csv(SCORE_DIR / "holdout_candidate_scores.csv", holdout_scores)
        nominal_rows = [row for row in holdout_rows if row["role"] in {"nominal", "tuned_and_nominal"}]
        nominal_scores = _score_summary(nominal_rows, list(NOMINAL_BY_MODEL.values()))
        _write_csv(SCORE_DIR / "holdout_nominal_scores.csv", nominal_scores)
        _write_csv(SUMMARY_DIR / "hub_sensitivity.csv", _hub_sensitivity(holdout_rows))
        comparisons, hypotheses, claim = _holdout_analysis(tuned_rows, holdout_scores, selected)
        _write_csv(SUMMARY_DIR / "holdout_comparisons.csv", comparisons)
        _write_csv(SUMMARY_DIR / "holdout_hypotheses.csv", hypotheses)
        _write_json(SUMMARY_DIR / "claim_status.json", claim)
        _write_json(SUMMARY_DIR / "run_summary.json", {
            "selected_candidates": {model: row["candidate_id"] for model, row in selected.items()},
            "selection_status": {row["model"]: row["status"] for row in selection_report},
            "unique_immediate_advantage_supported": claim["unique_immediate_advantage_supported"],
            "holdout_draws": SPLIT_COUNTS["holdout"],
            "holdout_configurations_scored": len(holdout_candidates),
        })

        output_paths = [p for p in OUT_ROOT.rglob("*") if p.is_file() and p not in {FREEZE_PATH, RUNNING_PATH}]
        _write_json(OUT_ROOT / "output_artifact_manifest.json", {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_sha256": _sha256(FREEZE_PATH),
            "outputs_sha256": {p.relative_to(OUT_ROOT).as_posix(): _sha256(p) for p in output_paths},
        })
    finally:
        if RUNNING_PATH.exists():
            RUNNING_PATH.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="Validate and freeze all E025 source/data hashes without running models")
    mode.add_argument("--run", action="store_true", help="Run development, validation, and once-only holdout after hash freeze")
    args = parser.parse_args()
    if args.preflight:
        _preflight()
    else:
        _run()


if __name__ == "__main__":
    main()
