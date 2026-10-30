"""Generate E025's frozen, outcome-blind topology and probe manifests.

This script only creates allocation/probe manifests and their SHA-256 files.
It does not import or run the simulator.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from itertools import combinations
from pathlib import Path


HERE = Path(__file__).resolve().parent
NAMESPACE = "e025_topology_v1"
THRESHOLD = 0.40
SPLITS = (
    ("development", range(200, 240)),
    ("validation", range(240, 280)),
    ("holdout", range(300, 400)),
)
PRIMARY_STRATA = ("canonical", "strong_overlap", "chain_overlap", "sparse_random")
ALL_STRATA = (*PRIMARY_STRATA, "hub_overlap")

TOPOLOGY_PATH = HERE / "e025_topology_manifest.csv"
PROBE_PATH = HERE / "e025_probe_manifest.csv"


def _fmt(value: float) -> str:
    return format(float(value), ".12g")


def _rng_for(partition: str, seed: int, stratum: str) -> tuple[random.Random, str]:
    key = f"{NAMESPACE}|{partition}|{seed}|{stratum}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16)), digest[:16]


def _blank_allocations(n_branches: int, n_traces: int) -> tuple[list[str], list[str], dict[str, dict[str, float]]]:
    branches = [f"b{i}" for i in range(n_branches)]
    traces = [f"t{i}" for i in range(n_traces)]
    allocations = {trace: {branch: 0.05 for branch in branches} for trace in traces}
    return branches, traces, allocations


def _draw(stratum: str, rng: random.Random) -> tuple[list[str], list[str], dict[str, dict[str, float]]]:
    if stratum in {"canonical", "strong_overlap"}:
        n_branches = rng.randint(4, 8)
        branches, traces, allocations = _blank_allocations(n_branches, 2)
        if stratum == "canonical":
            lo, hi = 0.70, 0.90
        else:
            lo, hi = 0.90, 0.98
        for trace, private in zip(traces, (branches[1], branches[2])):
            allocations[trace][branches[0]] = rng.uniform(lo, hi)
            allocations[trace][private] = rng.uniform(0.85, 0.95)
        return branches, traces, allocations

    if stratum == "chain_overlap":
        n_traces = rng.choice((3, 4))
        n_branches = rng.randint(2 * n_traces - 1, 11)
        branches, traces, allocations = _blank_allocations(n_branches, n_traces)
        for i, trace in enumerate(traces):
            allocations[trace][branches[i]] = rng.uniform(0.85, 0.95)
        first_edge = n_traces
        for i in range(n_traces - 1):
            branch = branches[first_edge + i]
            for trace in (traces[i], traces[i + 1]):
                allocations[trace][branch] = rng.uniform(0.70, 0.90)
        return branches, traces, allocations

    if stratum == "sparse_random":
        n_branches = rng.randint(8, 16)
        n_traces = rng.randint(4, 5)
        branches, traces, allocations = _blank_allocations(n_branches, n_traces)
        for i, trace in enumerate(traces):
            allocations[trace][branches[i]] = rng.uniform(0.85, 0.95)
        possible_pairs = list(combinations(traces, 2))
        n_edges = rng.randint(1, min(len(possible_pairs), n_branches - n_traces))
        selected_pairs = rng.sample(possible_pairs, n_edges)
        for i, pair in enumerate(selected_pairs):
            branch = branches[n_traces + i]
            for trace in pair:
                allocations[trace][branch] = rng.uniform(0.55, 0.90)
        return branches, traces, allocations

    if stratum == "hub_overlap":
        n_branches = rng.randint(5, 8)
        branches, traces, allocations = _blank_allocations(n_branches, 4)
        for i, trace in enumerate(traces):
            allocations[trace][branches[0]] = rng.uniform(0.55, 0.85)
            allocations[trace][branches[i + 1]] = rng.uniform(0.85, 0.95)
        return branches, traces, allocations

    raise ValueError(f"Unknown topology stratum: {stratum}")


def _probe_rows(
    partition: str,
    draw_id: str,
    split_seed: int,
    stratum: str,
    motif_id: str,
    branches: list[str],
    traces: list[str],
    allocations: dict[str, dict[str, float]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in traces:
        for target in traces:
            if source == target:
                continue
            shared = [
                branch for branch in branches
                if allocations[source][branch] > THRESHOLD
                and allocations[target][branch] > THRESHOLD
            ]
            controls = [
                branch for branch in branches
                if allocations[source][branch] > THRESHOLD
                and allocations[target][branch] <= THRESHOLD
            ]
            eligible = bool(shared and controls)
            rows.append({
                "partition": partition,
                "draw_id": draw_id,
                "split_seed": str(split_seed),
                "stratum": stratum,
                "motif_id": motif_id,
                "source_trace": source,
                "target_trace": target,
                "shared_branches": ";".join(shared),
                "source_private_branches": ";".join(controls),
                "primary_status": "defined" if eligible else "N/A",
                "reason": "" if eligible else ("no_shared_branch" if not shared else "no_source_private_control"),
            })
    return rows


def build_manifests() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    topology_rows: list[dict[str, str]] = []
    probe_rows: list[dict[str, str]] = []
    for partition, seeds in SPLITS:
        for seed in seeds:
            draw_id = f"{partition}_{seed:03d}"
            per_draw: dict[str, int] = {}
            for stratum in ALL_STRATA:
                rng, rng_key = _rng_for(partition, seed, stratum)
                branches, traces, allocations = _draw(stratum, rng)
                motif_id = f"{stratum}_{partition}_{seed:03d}"
                pairs = _probe_rows(
                    partition, draw_id, seed, stratum, motif_id,
                    branches, traces, allocations,
                )
                eligible_count = sum(row["primary_status"] == "defined" for row in pairs)
                if stratum in PRIMARY_STRATA and eligible_count < 1:
                    raise RuntimeError(f"Nonqualifying primary topology: {motif_id}")
                if stratum == "hub_overlap" and eligible_count != len(pairs):
                    raise RuntimeError(f"Hub D must be defined for every directed pair: {motif_id}")
                per_draw[stratum] = eligible_count
                allocation_json = json.dumps(
                    {trace: {branch: float(_fmt(weight)) for branch, weight in weights.items()}
                     for trace, weights in allocations.items()},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                topology_rows.append({
                    "partition": partition,
                    "draw_id": draw_id,
                    "split_seed": str(seed),
                    "stratum": stratum,
                    "motif_id": motif_id,
                    "generator_namespace": NAMESPACE,
                    "rng_key_sha256_prefix": rng_key,
                    "n_branches": str(len(branches)),
                    "n_traces": str(len(traces)),
                    "branch_ids_json": json.dumps(branches, separators=(",", ":")),
                    "trace_ids_json": json.dumps(traces, separators=(",", ":")),
                    "threshold": _fmt(THRESHOLD),
                    "n_eligible_directed_pairs": str(eligible_count),
                    "qualification": "eligible",
                    "allocation_json": allocation_json,
                })
                probe_rows.extend(pairs)
            if not all(per_draw.get(stratum, 0) >= 1 for stratum in PRIMARY_STRATA):
                raise RuntimeError(f"Draw lacks a required primary stratum: {draw_id}")
    return topology_rows, probe_rows


def _write_csv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> str:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def main() -> None:
    topology_rows, probe_rows = build_manifests()
    topology_fields = (
        "partition", "draw_id", "split_seed", "stratum", "motif_id",
        "generator_namespace", "rng_key_sha256_prefix", "n_branches", "n_traces",
        "branch_ids_json", "trace_ids_json", "threshold", "n_eligible_directed_pairs",
        "qualification", "allocation_json",
    )
    probe_fields = (
        "partition", "draw_id", "split_seed", "stratum", "motif_id",
        "source_trace", "target_trace", "shared_branches", "source_private_branches",
        "primary_status", "reason",
    )
    topology_hash = _write_csv(TOPOLOGY_PATH, topology_rows, topology_fields)
    probe_hash = _write_csv(PROBE_PATH, probe_rows, probe_fields)
    print(f"topologies={len(topology_rows)} sha256={topology_hash}")
    print(f"directed_pairs={len(probe_rows)} sha256={probe_hash}")


if __name__ == "__main__":
    main()
