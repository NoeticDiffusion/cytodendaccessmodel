"""Generalized motif and allocation generator for the cytodend accessibility model.

A ``MotifSpec`` describes a complete simulation topology:

- branch IDs and trace IDs
- allocation matrix (how each trace writes to each branch)
- expected linked / unlinked trace pairs (based on realized branch overlap)
- damage and rescue target branches for generalized signature computation

Six motif types
---------------
canonical         Two traces sharing one overlap branch (reproduces E017–E020).
weak_overlap      Two traces, overlap weight below the E020 threshold (~0.30).
strong_overlap    Two traces, high overlap weight (~0.95).
chain_overlap     Three traces in a chain (t0–t1 overlap, t1–t2 overlap, t0–t2 no overlap).
hub_overlap       Four traces, all sharing one hub branch.
sparse_random     Randomly sparse allocation with controlled density.

Usage
-----
    from cytodend_accessmodel.motifs import build_motif
    motif = build_motif("canonical", n_branches=4)
    motif = build_motif("chain_overlap", n_branches=8, n_traces=3)
    motif = build_motif("sparse_random", n_branches=16, n_traces=4, seed=42)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MotifSpec:
    """Complete topology descriptor for one simulation instance.

    Attributes
    ----------
    motif_id:
        Human-readable identifier, e.g. ``canonical_n4``.
    motif_type:
        One of ``canonical``, ``weak_overlap``, ``strong_overlap``,
        ``chain_overlap``, ``hub_overlap``, ``sparse_random``.
    n_branches:
        Total number of dendritic branches.
    n_traces:
        Number of engram traces.
    branch_ids:
        Ordered list of branch identifiers.
    trace_ids:
        Ordered list of trace identifiers.
    allocations:
        Mapping ``trace_id -> {branch_id: weight}``.  Weights are in
        ``[0, 1]``; branches absent from a trace's dict are implicitly 0.
    overlap_branches_per_pair:
        Mapping ``"t_i:t_j"`` (lexicographically sorted) to the list of
        branch IDs that carry substantial weight (> 0.40) for *both* traces.
    expected_linked_pairs:
        Trace pairs for which structural linking is expected after
        consolidation (non-empty overlap_branches).
    expected_unlinked_pairs:
        Trace pairs with no realized shared branch (overlap_branches empty).
    damage_target_branches:
        Branches to subject to accelerated structural decay for gSIG-D/E.
    rescue_target_branches:
        Branches to target with cue-priming for targeted rescue.
    metadata:
        Arbitrary scalar properties (overlap_weight, density, seed, …).
    """
    motif_id: str
    motif_type: str
    n_branches: int
    n_traces: int
    branch_ids: list[str]
    trace_ids: list[str]
    allocations: dict[str, dict[str, float]]
    overlap_branches_per_pair: dict[str, list[str]]
    expected_linked_pairs: list[tuple[str, str]]
    expected_unlinked_pairs: list[tuple[str, str]]
    damage_target_branches: list[str]
    rescue_target_branches: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pair_key(t1: str, t2: str) -> str:
    return ":".join(sorted([t1, t2]))


def _compute_overlap_map(
    allocations: dict[str, dict[str, float]],
    trace_ids: list[str],
    threshold: float = 0.40,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for i, ti in enumerate(trace_ids):
        for j in range(i + 1, len(trace_ids)):
            tj = trace_ids[j]
            w_i = allocations[ti]
            w_j = allocations[tj]
            shared = [b for b in w_i if w_i.get(b, 0.0) > threshold and w_j.get(b, 0.0) > threshold]
            result[_pair_key(ti, tj)] = shared
    return result


def _split_linked_unlinked(
    overlap_map: dict[str, list[str]],
    trace_ids: list[str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    linked, unlinked = [], []
    for i, ti in enumerate(trace_ids):
        for j in range(i + 1, len(trace_ids)):
            tj = trace_ids[j]
            key = _pair_key(ti, tj)
            if overlap_map.get(key):
                linked.append((ti, tj))
            else:
                unlinked.append((ti, tj))
    return linked, unlinked


def _neutral_weight(n_neutral: int) -> float:
    return 0.05


# ---------------------------------------------------------------------------
# Motif factories
# ---------------------------------------------------------------------------

def _build_canonical(n_branches: int, overlap_weight: float = 0.85) -> MotifSpec:
    """Two traces sharing one overlap branch — reproduces E017–E020."""
    if n_branches < 3:
        raise ValueError(f"canonical requires n_branches >= 3, got {n_branches}")
    bids = [f"b{i}" for i in range(n_branches)]
    tids = ["mu1", "mu2"]
    b_ovlp, b_t1, b_t2 = bids[1], bids[0], bids[2]
    rest = bids[3:]
    alloc_mu1 = {b: 0.05 for b in bids}
    alloc_mu1[b_t1]  = 0.90
    alloc_mu1[b_ovlp] = overlap_weight
    alloc_mu2 = {b: 0.05 for b in bids}
    alloc_mu2[b_t2]  = 0.90
    alloc_mu2[b_ovlp] = overlap_weight
    allocations = {"mu1": alloc_mu1, "mu2": alloc_mu2}
    ovlp_map = _compute_overlap_map(allocations, tids)
    linked, unlinked = _split_linked_unlinked(ovlp_map, tids)
    motif_id = f"canonical_n{n_branches}"
    return MotifSpec(
        motif_id=motif_id, motif_type="canonical",
        n_branches=n_branches, n_traces=2,
        branch_ids=bids, trace_ids=tids,
        allocations=allocations,
        overlap_branches_per_pair=ovlp_map,
        expected_linked_pairs=linked,
        expected_unlinked_pairs=unlinked,
        damage_target_branches=[b_ovlp],
        rescue_target_branches=[b_ovlp],
        metadata={"overlap_weight": overlap_weight},
    )


def _build_weak_overlap(n_branches: int) -> MotifSpec:
    spec = _build_canonical(n_branches, overlap_weight=0.30)
    spec = MotifSpec(**{**vars(spec), "motif_type": "weak_overlap",
                       "motif_id": f"weak_overlap_n{n_branches}"})
    return spec


def _build_strong_overlap(n_branches: int) -> MotifSpec:
    spec = _build_canonical(n_branches, overlap_weight=0.95)
    spec = MotifSpec(**{**vars(spec), "motif_type": "strong_overlap",
                       "motif_id": f"strong_overlap_n{n_branches}"})
    return spec


def _build_chain_overlap(n_branches: int, n_chain: int = 3,
                         overlap_weight: float = 0.85) -> MotifSpec:
    """n_chain traces in a chain: t0–t1 overlap, t1–t2 overlap, t0–t2 no overlap.

    Requires n_branches >= 2*n_chain - 1.
    """
    min_branches = 2 * n_chain - 1
    if n_branches < min_branches:
        raise ValueError(
            f"chain_overlap with n_chain={n_chain} requires n_branches >= {min_branches}, got {n_branches}"
        )
    bids = [f"b{i}" for i in range(n_branches)]
    tids = [f"tc{i}" for i in range(n_chain)]
    allocations: dict[str, dict[str, float]] = {}
    for i, tid in enumerate(tids):
        alloc = {b: 0.05 for b in bids}
        b_private = bids[2 * i]          # private branch for trace i
        alloc[b_private] = 0.90
        if i > 0:
            b_left_ovlp = bids[2 * i - 1]   # overlap with trace i-1
            alloc[b_left_ovlp] = overlap_weight
        if i < n_chain - 1:
            b_right_ovlp = bids[2 * i + 1]  # overlap with trace i+1
            alloc[b_right_ovlp] = overlap_weight
        allocations[tid] = alloc

    ovlp_map = _compute_overlap_map(allocations, tids)
    linked, unlinked = _split_linked_unlinked(ovlp_map, tids)
    first_ovlp = bids[1]   # overlap between t0 and t1
    return MotifSpec(
        motif_id=f"chain_overlap_n{n_branches}",
        motif_type="chain_overlap",
        n_branches=n_branches, n_traces=n_chain,
        branch_ids=bids, trace_ids=tids,
        allocations=allocations,
        overlap_branches_per_pair=ovlp_map,
        expected_linked_pairs=linked,
        expected_unlinked_pairs=unlinked,
        damage_target_branches=[first_ovlp],
        rescue_target_branches=[first_ovlp],
        metadata={"overlap_weight": overlap_weight, "n_chain": n_chain},
    )


def _build_hub_overlap(n_branches: int, n_traces: int = 4,
                       hub_weight: float = 0.85) -> MotifSpec:
    """n_traces traces all sharing one hub branch.

    Hub motif highlights false-linking risk: all traces share b0, so
    consolidation of any trace reinforces b0 for all others.
    expected_linked_pairs = all pairs sharing the hub
    expected_unlinked_pairs = [] (empty — all pairs are linked through hub)
    """
    if n_branches < n_traces + 1:
        raise ValueError(
            f"hub_overlap with n_traces={n_traces} requires n_branches >= {n_traces+1}, got {n_branches}"
        )
    bids = [f"b{i}" for i in range(n_branches)]
    tids = [f"th{i}" for i in range(n_traces)]
    b_hub = bids[0]
    allocations: dict[str, dict[str, float]] = {}
    for i, tid in enumerate(tids):
        alloc = {b: 0.05 for b in bids}
        alloc[b_hub] = hub_weight
        alloc[bids[i + 1]] = 0.90   # private branch
        allocations[tid] = alloc

    ovlp_map = _compute_overlap_map(allocations, tids)
    linked, unlinked = _split_linked_unlinked(ovlp_map, tids)
    return MotifSpec(
        motif_id=f"hub_overlap_n{n_branches}",
        motif_type="hub_overlap",
        n_branches=n_branches, n_traces=n_traces,
        branch_ids=bids, trace_ids=tids,
        allocations=allocations,
        overlap_branches_per_pair=ovlp_map,
        expected_linked_pairs=linked,
        expected_unlinked_pairs=unlinked,
        damage_target_branches=[b_hub],
        rescue_target_branches=[b_hub],
        metadata={"hub_weight": hub_weight, "n_traces": n_traces},
    )


def _build_sparse_random(
    n_branches: int,
    n_traces: int = 4,
    density: float = 0.30,
    strong_weight: float = 0.80,
    seed: int = 42,
) -> MotifSpec:
    """Random sparse allocation with controlled density.

    Each trace randomly activates ~density × n_branches branches strongly.
    Expected linked pairs = pairs with at least one shared strong branch.
    """
    bids = [f"b{i}" for i in range(n_branches)]
    tids = [f"ts{i}" for i in range(n_traces)]
    rng = random.Random(seed)
    n_active = max(1, round(density * n_branches))
    allocations: dict[str, dict[str, float]] = {}
    for tid in tids:
        alloc = {b: 0.05 for b in bids}
        active = rng.sample(bids, k=n_active)
        for b in active:
            alloc[b] = strong_weight
        allocations[tid] = alloc

    ovlp_map = _compute_overlap_map(allocations, tids, threshold=0.40)
    linked, unlinked = _split_linked_unlinked(ovlp_map, tids)

    # Pick a damage / rescue target: first shared branch among expected linked pairs, or b0
    damage_branch = bids[0]
    if linked:
        t_i, t_j = linked[0]
        pair_shared = ovlp_map[_pair_key(t_i, t_j)]
        if pair_shared:
            damage_branch = pair_shared[0]

    return MotifSpec(
        motif_id=f"sparse_random_n{n_branches}_s{seed}",
        motif_type="sparse_random",
        n_branches=n_branches, n_traces=n_traces,
        branch_ids=bids, trace_ids=tids,
        allocations=allocations,
        overlap_branches_per_pair=ovlp_map,
        expected_linked_pairs=linked,
        expected_unlinked_pairs=unlinked,
        damage_target_branches=[damage_branch],
        rescue_target_branches=[damage_branch],
        metadata={"density": density, "n_active_per_trace": n_active,
                  "strong_weight": strong_weight, "seed": seed},
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_motif(
    motif_type: str,
    n_branches: int = 4,
    n_traces: int | None = None,
    overlap_weight: float = 0.85,
    density: float = 0.30,
    seed: int = 42,
) -> MotifSpec:
    """Build a MotifSpec for the given type and scale.

    Parameters
    ----------
    motif_type:
        One of ``canonical``, ``weak_overlap``, ``strong_overlap``,
        ``chain_overlap``, ``hub_overlap``, ``sparse_random``.
    n_branches:
        Total number of branches.
    n_traces:
        Trace count; defaults to the motif's natural default (2 for two-trace
        motifs, 3 for chain, 4 for hub and sparse).
    overlap_weight:
        Overlap branch allocation weight (ignored by hub and sparse).
    density:
        Active branch fraction for sparse_random.
    seed:
        RNG seed for sparse_random.
    """
    mt = motif_type.lower()
    if mt == "canonical":
        return _build_canonical(n_branches, overlap_weight)
    if mt == "weak_overlap":
        return _build_weak_overlap(n_branches)
    if mt == "strong_overlap":
        return _build_strong_overlap(n_branches)
    if mt == "chain_overlap":
        nc = n_traces if n_traces is not None else 3
        return _build_chain_overlap(n_branches, n_chain=nc, overlap_weight=overlap_weight)
    if mt == "hub_overlap":
        nt = n_traces if n_traces is not None else 4
        return _build_hub_overlap(n_branches, n_traces=nt)
    if mt == "sparse_random":
        nt = n_traces if n_traces is not None else 4
        return _build_sparse_random(n_branches, n_traces=nt, density=density, seed=seed)
    raise ValueError(f"Unknown motif_type: {motif_type!r}")


# ---------------------------------------------------------------------------
# Utilities for experiment scripts
# ---------------------------------------------------------------------------

def alloc_to_cue(alloc: dict[str, float]) -> dict[str, float]:
    """Return a cue derived from a trace's allocation weights.

    The cue equals the raw allocation dict.  All weights stay in ``[0, 1]``
    since the motif factories already produce values in that range.
    """
    return dict(alloc)


def private_cue(
    motif: MotifSpec,
    trace_id: str,
    strength: float = 1.0,
) -> dict[str, float]:
    """Return a cue strongly biased toward trace_id's private branches.

    A "private" branch is one where trace_id has a high allocation weight
    *and* no other trace has weight > 0.40 on the same branch.
    Falls back to the full allocation cue if no private branch is found.
    """
    other_traces = [t for t in motif.trace_ids if t != trace_id]
    own_alloc = motif.allocations[trace_id]
    private = [
        b for b, w in own_alloc.items()
        if w > 0.60 and all(motif.allocations[t].get(b, 0.0) < 0.40 for t in other_traces)
    ]
    if private:
        cue = {b: 0.05 for b in motif.branch_ids}
        for b in private:
            cue[b] = strength
        return cue
    return alloc_to_cue(own_alloc)


def linking_score(
    branch_accessibilities: dict[str, float],
    alloc_i: dict[str, float],
    alloc_j: dict[str, float],
) -> float:
    """Compute L_ij = sum_b( w_i[b] * w_j[b] * M_b[b] )."""
    return sum(
        alloc_i.get(b, 0.0) * alloc_j.get(b, 0.0) * m
        for b, m in branch_accessibilities.items()
    )
