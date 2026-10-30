"""Generate the fixed 32-point E025 candidate registry.

This only writes a parameter CSV and its SHA-256 file. It does not import or
run the simulator. The first five rows reproduce E024's fast-gain calibration
grid; the remaining rows use a deterministic Halton sequence.
"""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().with_name("e025_parameter_registry.csv")
HASH_OUT = OUT.with_suffix(OUT.suffix + ".sha256")

MODELS = (
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
)
E024_GAIN_GRID = (1.0, 1.5, 2.0, 2.5, 3.0)
E024_SELECTED_GAIN = {name: 2.0 for name in MODELS}
E024_SELECTED_GAIN["soma_global_gain_only"] = 1.0
E024_SELECTED_GAIN["eligibility_only"] = 1.0

# All active parameter dimensions use a fixed Halton base so that each
# parameter has the same deterministic sequence wherever it is tuned.
HALTON_BASE = {
    "fast_gain": 2,
    "structural_lr": 3,
    "structural_gain": 5,
    "replay_gain": 7,
    "structural_decay": 11,
    "eligibility_decay": 13,
    "translation_decay": 17,
    "hebbian_lr": 19,
    "drift_sd": 23,
}
RANGES = {
    "fast_gain": (1.0, 3.0),
    "structural_lr": (0.06, 0.30),
    "structural_gain": (3.0, 9.0),
    "replay_gain": (0.40, 1.20),
    "structural_decay": (0.0025, 0.020),
    "eligibility_decay": (0.06, 0.24),
    "translation_decay": (0.025, 0.10),
    "hebbian_lr": (0.06, 0.36),
    "drift_sd": (0.0125, 0.050),
}
FAMILY_TUNABLES = {
    "full_model": ("structural_lr", "replay_gain", "structural_decay", "eligibility_decay", "translation_decay"),
    "fast_context_only": (),
    "replay_no_structure": ("structural_decay",),
    "random_slow_drift": ("drift_sd",),
    "fixed_allocation_only": (),
    "hebbian_weight_only": ("hebbian_lr", "structural_decay", "eligibility_decay"),
    "soma_global_gain_only": ("structural_lr", "replay_gain", "structural_decay", "eligibility_decay", "translation_decay"),
    "shuffled_replay": ("structural_lr", "replay_gain", "structural_decay", "eligibility_decay", "translation_decay"),
    "eligibility_only": ("structural_decay", "eligibility_decay"),
    "resource_only": ("structural_decay", "replay_gain", "translation_decay"),
}

PARAMETER_DEFAULTS = {
    "fast_gain": 2.0,
    "context_gain": 1.0,
    "inhibition_gain": 1.0,
    "spine_gain": 0.5,
    "structural_gain": 6.0,
    "eligibility_decay": 0.12,
    "translation_decay": 0.05,
    "structural_lr": 0.18,
    "structural_decay": 0.005,
    "structural_max": 1.0,
    "readout_gain": 5.0,
    "readout_threshold": 0.3,
    "replay_gain": 0.80,
    "sleep_gain": 0.0,
    "context_mismatch_penalty": 0.25,
    "structural_noise": 0.0,
    "translation_budget": 0.0,
    "spillover_rate": 0.0,
    "hebbian_lr": 0.18,
    "drift_sd": 0.025,
}


def _halton(index: int, base: int) -> float:
    """Return the index-th Halton coordinate (index starts at one)."""
    result = 0.0
    fraction = 1.0 / base
    i = index
    while i:
        result += fraction * (i % base)
        i //= base
        fraction /= base
    return result


def _family_defaults(model: str) -> dict[str, float]:
    values = dict(PARAMETER_DEFAULTS)
    if model == "fast_context_only":
        values.update(structural_lr=0.0, structural_decay=0.0)
    elif model in {"replay_no_structure", "hebbian_weight_only", "resource_only"}:
        values["structural_lr"] = 0.0
    elif model == "eligibility_only":
        values.update(structural_lr=0.0, replay_gain=0.0)
    elif model == "fixed_allocation_only":
        values.update(
            structural_lr=0.0,
            replay_gain=0.0,
            eligibility_decay=0.0,
            structural_decay=0.0,
            translation_decay=0.0,
        )
    return values


def _candidate_points() -> list[tuple[float, int | None, str]]:
    points = [(gain, None, "e024_fast_gain_grid") for gain in E024_GAIN_GRID]
    used = list(E024_GAIN_GRID)
    index = 1
    while len(points) < 32:
        gain = 1.0 + 2.0 * _halton(index, HALTON_BASE["fast_gain"])
        index += 1
        if any(math.isclose(gain, seen, rel_tol=0.0, abs_tol=1e-12) for seen in used):
            continue
        used.append(gain)
        points.append((gain, index - 1, "halton"))
    return points


FIELDS = (
    "model", "candidate_id", "design", "halton_index", "fast_gain",
    "context_gain", "inhibition_gain", "spine_gain", "structural_gain",
    "eligibility_decay", "translation_decay", "structural_lr",
    "structural_decay", "structural_max", "readout_gain", "readout_threshold",
    "replay_gain", "sleep_gain", "context_mismatch_penalty", "structural_noise",
    "translation_budget", "spillover_rate", "hebbian_lr", "drift_sd",
    "active_tuning_parameters", "e024_selected_fast_gain",
)


def _fmt(value: float) -> str:
    return format(float(value), ".12g")


def build_rows() -> list[dict[str, str]]:
    points = _candidate_points()
    rows: list[dict[str, str]] = []
    for model in MODELS:
        active = ("fast_gain", "structural_gain", *FAMILY_TUNABLES[model])
        defaults = _family_defaults(model)
        for idx, (gain, halton_index, design) in enumerate(points):
            values = dict(defaults)
            values["fast_gain"] = gain
            if halton_index is not None:
                for parameter in active:
                    if parameter == "fast_gain":
                        continue
                    low, high = RANGES[parameter]
                    coordinate = _halton(halton_index, HALTON_BASE[parameter])
                    values[parameter] = low + (high - low) * coordinate
            row = {
                "model": model,
                "candidate_id": f"{model}_c{idx:02d}",
                "design": design,
                "halton_index": "" if halton_index is None else str(halton_index),
                "active_tuning_parameters": ";".join(active),
                "e024_selected_fast_gain": _fmt(E024_SELECTED_GAIN[model]),
            }
            row.update({key: _fmt(values[key]) for key in PARAMETER_DEFAULTS})
            rows.append(row)
    return rows


def main() -> None:
    rows = build_rows()
    if len(rows) != len(MODELS) * 32:
        raise RuntimeError(f"Expected 320 candidates; got {len(rows)}")
    gains_by_model = {
        model: [row["fast_gain"] for row in rows if row["model"] == model]
        for model in MODELS
    }
    reference = gains_by_model[MODELS[0]]
    if any(gains != reference for gains in gains_by_model.values()):
        raise RuntimeError("All model families must use the identical fast_gain list")
    if len(set(reference)) != 32:
        raise RuntimeError("Each model must have 32 unique fast_gain values")
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    HASH_OUT.write_text(f"{digest}  {OUT.name}\n", encoding="ascii")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
