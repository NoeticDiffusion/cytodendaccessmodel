# Configs

Use this directory for named experiment and benchmark configurations.

Recommended use:

- one file per experiment family
- explicit names for encoder, gating, memory core, consolidation, and substrate choices
- keep benchmark settings separate from implementation code so pivots are easy to track

Current benchmark config fields typically include:

- `run.name`
- `run.seed`
- `run.logging` with `full`, `partial`, or `none`
- `run.log_file` for optional separate log output
- `system.dim`
- `system.settle_steps`
- `consolidation.retention_trigger_threshold`
- `consolidation.max_refresh_traces`
- `report.modes`
- `report.direct_compare_mode` for preferred ordering in the all-gated direct compare table
- `report.multi_seed.count` for deterministic seed averaging in reports
- benchmark-specific settings such as cue corruption, context counts, competitor overlap, and capacity sweep sizes

The default scientific report config lives at:

- `configs/benchmark/capacity_interference_default.yaml`

The default pseudo-likelihood report config lives at:

- `configs/benchmark/pseudo_likelihood_default.yaml`

The capacity/interference report derives a stable seed sequence from `run.seed`, so a
fixed base seed plus `report.multi_seed.count` always reproduces the same aggregate
mean/std table. You can also override the sweep size from the CLI with
`--multi-seed-count`.

The default report now also:

- prints direct gated-vs-ungated comparisons for all gated modes
- includes an inline nearest-neighbor cosine baseline in those direct comparisons
- runs a cross-capacity context sweep at the configured `gated_vs_ungated.context_counts`
- runs a corruption-fraction sweep for the configured `gated_vs_ungated` benchmark regime
- uses `report.direct_compare_mode` to put the preferred gate first in that comparison

## DANDI configs

Dataset-selection YAML files for the open-data workflow live under `configs/dandi/`.

Primary current configs:

- `configs/dandi/dataset_000718.yaml`
- `configs/dandi/dataset_000871.yaml`

Use them with the generic DANDI CLI:

- `python -m dandi_io.cli list --config configs/dandi/dataset_000718.yaml`
- `python -m dandi_io.cli download --config configs/dandi/dataset_000718.yaml`
- `python -m dandi_io.cli probe --config configs/dandi/dataset_000718.yaml`

The same command pattern applies to `dataset_000871.yaml` and other dataset YAML files in the same directory.
