# Configs

Use this directory for named DANDI dataset configurations.

Recommended use:

- one file per dataset family or reviewer bundle
- explicit dataset IDs, versions, path filters, and storage roots
- keep DANDI selection policy separate from implementation code so reviewer
  downloads and probes remain easy to audit

## DANDI configs

Dataset-selection YAML files for the open-data workflow live under `configs/dandi/`.

Primary current-manuscript configs:

- `configs/dandi/dataset_000336.yaml`
- `configs/dandi/dataset_000718.yaml`
- `configs/dandi/dataset_001710.yaml`
- `configs/dandi/dataset_001710_replication_bundle_01.yaml`
- `configs/dandi/dataset_001710_group_bundle_01.yaml`
- `configs/dandi/dataset_000718_subject_complete_bundle_01.yaml`

Legacy or supplementary config:

- `configs/dandi/dataset_000871.yaml`

Use them with the generic DANDI CLI:

- `python -m dandi_io.cli list --config configs/dandi/dataset_000718.yaml`
- `python -m dandi_io.cli download --config configs/dandi/dataset_000718.yaml`
- `python -m dandi_io.cli probe --config configs/dandi/dataset_000718.yaml`

The same command pattern applies to the other current-manuscript dataset YAML
files in the same directory. `dataset_000871.yaml` is retained for legacy
cross-plane analyses and is not required for the current manuscript claims.
