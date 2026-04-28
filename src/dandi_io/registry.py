from __future__ import annotations

from typing import Sequence

from dandi_io.contracts import AssetRecord, DandiIngestionConfig, ProbeSummary, TriageResult
from dandi_io.datasets import Dataset000336Adapter
from dandi_io.datasets import Dataset000718Adapter
from dandi_io.datasets import Dataset000871Adapter
from dandi_io.datasets import Dataset001710Adapter


class GenericDatasetAdapter:
    """Default adapter for datasets without custom triage logic.

    The generic adapter applies only the filters already encoded in
    `DandiIngestionConfig`. Dataset-specific adapters can override selection,
    triage metadata, and markdown rendering while preserving the same protocol.
    """

    adapter_id = "generic"

    def select_assets(
        self,
        records: Sequence[AssetRecord],
        config: DandiIngestionConfig,
    ) -> list[AssetRecord]:
        """Select assets after config-level filters have been applied.

        Args:
            records: Candidate asset records.
            config: Ingestion config containing the optional asset limit.

        Returns:
            Selected records. If `asset_limit` is unset, all records are
            returned.
        """
        limit = config.selection.asset_limit
        if limit is None:
            return list(records)
        return list(records[: max(0, limit)])

    def build_triage(
        self,
        records: Sequence[AssetRecord],
        config: DandiIngestionConfig,
        *,
        probes: Sequence[ProbeSummary] | None = None,
    ) -> TriageResult:
        """Build a generic triage result for selected records.

        Args:
            records: Selected asset records.
            config: Ingestion config for the run.
            probes: Optional probe summaries, used only for count metadata in
                the generic adapter.

        Returns:
            Triage summary with generic notes and selected-asset metadata.
        """
        notes = (
            "Generic adapter used: no dataset-specific ranking policy applied.",
            "Assets were selected only by config filters and optional asset limit.",
        )
        metadata = {
            "selected_count": len(records),
            "probed_count": len(probes or ()),
        }
        return TriageResult(
            adapter_id=self.adapter_id,
            dandiset_id=config.dataset.dandiset_id,
            selected_assets=tuple(records),
            notes=notes,
            metadata=metadata,
        )

    def render_triage_markdown(self, triage: TriageResult) -> str:
        """Render a generic triage report as Markdown.

        Args:
            triage: Triage result to render.

        Returns:
            Markdown document listing notes and selected asset paths.
        """
        lines = [
            f"# Triage Summary: DANDI {triage.dandiset_id}",
            "",
            "## Notes",
            "",
        ]
        for note in triage.notes:
            lines.append(f"- {note}")
        lines.extend(
            [
                "",
                "## Selected Assets",
                "",
            ]
        )
        for record in triage.selected_assets:
            lines.append(f"- `{record.path}`")
        return "\n".join(lines) + "\n"


_ADAPTERS = {
    "generic": GenericDatasetAdapter(),
    "dataset_000336": Dataset000336Adapter(),
    "dataset_000718": Dataset000718Adapter(),
    "dataset_000871": Dataset000871Adapter(),
    "dataset_001710": Dataset001710Adapter(),
}


def get_dataset_adapter(adapter_id: str):
    """Return a registered dataset adapter by identifier.

    Args:
        adapter_id: Adapter key such as `"generic"` or `"dataset_000718"`.

    Returns:
        Adapter instance implementing the dataset adapter protocol.

    Raises:
        ValueError: If `adapter_id` is not registered.
    """
    if adapter_id not in _ADAPTERS:
        raise ValueError(
            f"Unknown dataset adapter `{adapter_id}`. "
            f"Known adapters: {', '.join(sorted(_ADAPTERS))}."
        )
    return _ADAPTERS[adapter_id]


def known_adapters() -> tuple[str, ...]:
    """List adapter identifiers available in this installation.

    Returns:
        Sorted tuple of registered adapter IDs.
    """
    return tuple(sorted(_ADAPTERS))
