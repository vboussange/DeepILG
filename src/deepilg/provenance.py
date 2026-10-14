"""Non-sensitive workflow manifests used to prevent demo-result publication."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import ExperimentConfig

MANIFEST_NAME = "workflow_manifest.json"


def manifest_payload(config: ExperimentConfig) -> dict[str, Any]:
    """Return provenance that deliberately excludes coordinates and bounds."""
    return {
        "format_version": 1,
        "project_name": config.project.name,
        "geolocation_status": config.data.geolocation_status,
        "scientific_use": config.data.scientific_use,
    }


def write_workflow_manifest(config: ExperimentConfig, directory: str | Path) -> Path:
    """Write or validate the immutable safety manifest in an output directory."""
    destination = Path(directory) / MANIFEST_NAME
    payload = manifest_payload(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        observed = json.loads(destination.read_text(encoding="utf-8"))
        if observed != payload:
            raise ValueError(
                f"Output directory belongs to a different data configuration: {destination.parent}"
            )
        return destination
    # Write then rename, so concurrent commands never read a partial manifest.
    temporary = destination.with_name(f".{MANIFEST_NAME}.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def require_publication_safe_manifest(directory: str | Path) -> dict[str, Any]:
    """Load a manifest and reject artificial or non-scientific outputs."""
    source = Path(directory) / MANIFEST_NAME
    if not source.is_file():
        raise FileNotFoundError(
            f"Publication export requires {source}; rerun evaluation with this release"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        payload.get("geolocation_status") != "empirical"
        or payload.get("scientific_use") is not True
    ):
        raise ValueError(
            f"Publication export is disabled for artificial or non-scientific outputs: {source}"
        )
    return payload
