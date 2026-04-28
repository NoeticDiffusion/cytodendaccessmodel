from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from urllib.parse import quote

from dandi_io.contracts import AssetRecord


_SUBJECT_PATTERN = re.compile(r"sub-([^/]+)")
_SESSION_PATTERN = re.compile(r"ses-([^/]+)")


class DandiClient:
    """Small wrapper around the official DANDI API and file download workflow.

    The client is intentionally narrow: it lists DANDI assets as local
    `AssetRecord` contracts and downloads records that already expose a direct
    download URL. Repository experiment scripts use the official `dandi`
    command-line tool for large reviewer downloads, while this class supports
    metadata-driven ingestion and small programmatic workflows.
    """

    def list_assets(self, dandiset_id: str, version: str = "draft") -> list[AssetRecord]:
        """List assets for a DANDI dataset version.

        Args:
            dandiset_id: DANDI identifier without the `DANDI:` prefix, for
                example `"000718"`.
            version: Dandiset version to inspect. Use `"draft"` for draft
                datasets or a published version string.

        Returns:
            Asset records sorted by their DANDI-relative path.

        Raises:
            RuntimeError: If the optional `dandi` package is unavailable.
            ValueError: If the DANDI API returns an asset without a path.
        """
        dandi_api_client = _require_dandi_api_client()
        records: list[AssetRecord] = []
        with dandi_api_client() as client:
            dandiset = client.get_dandiset(dandiset_id, version)
            for asset in dandiset.get_assets():
                records.append(self._asset_to_record(asset, dandiset_id=dandiset_id, version=version))
        records.sort(key=lambda record: record.path)
        return records

    def download_assets(
        self,
        records: list[AssetRecord],
        *,
        output_root: Path,
    ) -> list[Path]:
        """Download assets that are not already present under a raw-data root.

        Args:
            records: Asset records to download or confirm locally.
            output_root: Local root used with `AssetRecord.local_path()`.

        Returns:
            Local paths for all requested assets, including files that already
            existed before the call.

        Raises:
            RuntimeError: If a record does not contain a direct download URL.
            OSError: If the target directory cannot be created or the file
                cannot be written.
        """
        output_root.mkdir(parents=True, exist_ok=True)
        downloaded_paths: list[Path] = []
        for record in records:
            local_path = record.local_path(output_root)
            if local_path.exists():
                downloaded_paths.append(local_path)
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._download_file(record, local_path)
            downloaded_paths.append(local_path)
        return downloaded_paths

    def asset_query_url(self, record: AssetRecord) -> str:
        """Build the DANDI API query URL for a single asset path.

        Args:
            record: Asset record whose path, dataset ID, and version should be
                encoded into the query.

        Returns:
            DANDI API URL using the `assets/?path=...` endpoint.
        """
        quoted_path = quote(record.path, safe="/")
        return (
            "https://api.dandiarchive.org/api/dandisets/"
            f"{record.dandiset_id}/versions/{record.version}/assets/?path={quoted_path}"
        )

    def _asset_to_record(self, asset: Any, *, dandiset_id: str, version: str) -> AssetRecord:
        raw_metadata = self._raw_metadata(asset)
        path = self._first_non_empty(
            getattr(asset, "path", None),
            raw_metadata.get("path"),
            raw_metadata.get("asset_path"),
        )
        if not path:
            raise ValueError("Encountered DANDI asset without a path.")
        metadata = dict(raw_metadata)
        size_value = self._first_non_empty(
            getattr(asset, "size", None),
            raw_metadata.get("size"),
            raw_metadata.get("contentSize"),
        )
        asset_url = self._first_non_empty(
            getattr(asset, "api_url", None),
            raw_metadata.get("api_url"),
            raw_metadata.get("url"),
        )
        download_url = self._extract_download_url(asset, raw_metadata)
        subject_id = self._match_group(_SUBJECT_PATTERN, path)
        session_id = self._match_group(_SESSION_PATTERN, path)
        identifier = self._first_non_empty(
            getattr(asset, "identifier", None),
            raw_metadata.get("identifier"),
            raw_metadata.get("asset_id"),
            path,
        )
        return AssetRecord(
            dandiset_id=dandiset_id,
            version=version,
            identifier=str(identifier),
            path=str(path),
            size=int(size_value) if size_value not in {None, ""} else None,
            asset_url=str(asset_url) if asset_url else None,
            download_url=str(download_url) if download_url else None,
            subject_id=subject_id,
            session_id=session_id,
            metadata=metadata,
        )

    def _raw_metadata(self, asset: Any) -> dict[str, Any]:
        if hasattr(asset, "get_raw_metadata"):
            raw = asset.get_raw_metadata()
            if isinstance(raw, dict):
                return raw
        if hasattr(asset, "json_dict"):
            raw = asset.json_dict()
            if isinstance(raw, dict):
                return raw
        return {}

    def _extract_download_url(self, asset: Any, raw_metadata: dict[str, Any]) -> str | None:
        direct = getattr(asset, "download_url", None)
        if direct:
            return str(direct)
        content_url = raw_metadata.get("contentUrl")
        if isinstance(content_url, list) and content_url:
            return str(content_url[0])
        if isinstance(content_url, str):
            return content_url
        return None

    def _first_non_empty(self, *values: Any) -> Any:
        for value in values:
            if value is not None and value != "":
                return value
        return None

    def _match_group(self, pattern: re.Pattern[str], path: str) -> str | None:
        match = pattern.search(path)
        if match is None:
            return None
        return match.group(1)

    def _download_file(self, record: AssetRecord, local_path: Path) -> None:
        if not record.download_url:
            raise RuntimeError(f"No direct download URL is available for {record.path}.")
        tmp_path = local_path.with_suffix(local_path.suffix + ".part")
        if tmp_path.exists():
            tmp_path.unlink()
        with urlopen(record.download_url) as response, tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        tmp_path.replace(local_path)


def _require_dandi_api_client():
    try:
        from dandi.dandiapi import DandiAPIClient
    except ImportError as exc:
        raise RuntimeError(
            "The `dandi` package is required for DANDI asset listing. "
            "Install it with `pip install dandi`."
        ) from exc
    return DandiAPIClient
