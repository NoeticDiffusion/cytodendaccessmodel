"""Unit-level protocol checks for the E024 common-probe experiment."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import exp024_common_probe_comparison as e024  # noqa: E402
from cytodend_accessmodel.motifs import build_motif  # noqa: E402


def test_common_readout_zero_state_and_probe_does_not_mutate_frozen_state():
    motif = build_motif("canonical", n_branches=4, n_traces=2)
    model = e024.build_model("full_model", motif)
    before_step = model.simulator.step_index
    before_e = {bid: branch.eligibility.value for bid, branch in model.simulator.branches.items()}

    result = e024.probe_readout(model, {})

    assert all(row["support"] == 0.0 for row in result.values())
    assert all(math.isclose(row["readout"], 1 / (1 + math.exp(1.5))) for row in result.values())
    assert model.simulator.step_index == before_step
    assert {bid: branch.eligibility.value for bid, branch in model.simulator.branches.items()} == before_e


def test_explicit_no_replay_only_decays_readiness():
    motif = build_motif("canonical", n_branches=4, n_traces=2)
    model = e024.build_model("eligibility_only", motif)
    model.simulator.branches["b0"].translation_readiness.value = 0.5

    e024._explicit_no_replay(model)

    assert math.isclose(model.simulator.branches["b0"].translation_readiness.value, 0.475)
    assert all(trace.replay_count == 0 for trace in model.simulator.traces.values())


def test_canonical_manifest_has_two_directions_and_no_overlap_is_na():
    canonical = build_motif("canonical", n_branches=4, n_traces=2)
    rows = e024.build_probe_manifest(canonical, 7, "unit")
    assert len(rows) == 2
    assert all(row["primary_status"] == "defined" for row in rows)

    disjoint = copy.deepcopy(canonical)
    first, second = disjoint.trace_ids
    for branch in disjoint.branch_ids:
        disjoint.allocations[first][branch] = 0.05
        disjoint.allocations[second][branch] = 0.05
    disjoint.allocations[first][disjoint.branch_ids[0]] = 0.9
    disjoint.allocations[second][disjoint.branch_ids[2]] = 0.9
    rows = e024.build_probe_manifest(disjoint, 7, "unit")
    assert all(row["primary_status"] == "N/A" for row in rows)
    assert all(row["reason"] == "no_shared_branch" for row in rows)


def test_hub_motif_is_registered_as_boundary_na():
    motif = build_motif("hub_overlap", n_branches=5, n_traces=4)
    rows = e024.build_probe_manifest(motif, 9, "unit")
    assert len(rows) == 1
    assert rows[0]["primary_status"] == "N/A"
    assert rows[0]["reason"] == "hub_boundary_no_unlinked_pairs"


def test_all_registered_models_return_same_target_trace_interface():
    motif = build_motif("canonical", n_branches=4, n_traces=2)
    expected = set(motif.trace_ids)
    for name in e024.MODEL_ORDER:
        model = e024.build_model(name, motif, seed=3)
        scores = e024.probe_readout(model, {"b1": 1.0})
        assert set(scores) == expected
        assert all(0.0 <= row["readout"] <= 1.0 for row in scores.values())


def test_identical_frozen_states_have_zero_primary_difference():
    motif = build_motif("canonical", n_branches=4, n_traces=2)
    pair_rows = e024.build_probe_manifest(motif, 0, "unit")
    pre = e024.train_model(e024.build_model("full_model", motif), consolidate=False)
    post = copy.deepcopy(pre)

    effect, _ = e024.primary_effect(pre, post, pair_rows)

    assert effect == 0.0
