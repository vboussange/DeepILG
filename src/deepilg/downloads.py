"""Retrieval of the public geospatial products used by the benchmark."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from shapely.geometry import LineString, mapping

from .config import ExperimentConfig
from .data import Coordinates, load_coordinates

PLANETARY_COMPUTER_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
USGS_3DEP_EXPORT = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
)
OVERPASS_API = "https://overpass-api.de/api/interpreter"


@dataclass(frozen=True)
class DownloadedProducts:
    """Cached source paths for one geographic region."""

    region: str
    dem: Path
    worldcover: tuple[Path, ...]
    roads: Path


def coordinate_bounds(
    coordinates: Coordinates, *, margin_degrees: float = 0.1
) -> tuple[float, float, float, float]:
    """Return ``(west, south, east, north)`` around population coordinates."""
    if margin_degrees <= 0:
        raise ValueError("margin_degrees must be positive")
    return (
        float(coordinates.longitude.min() - margin_degrees),
        float(coordinates.latitude.min() - margin_degrees),
        float(coordinates.longitude.max() + margin_degrees),
        float(coordinates.latitude.max() + margin_degrees),
    )


def download_public_products(
    config: ExperimentConfig,
    *,
    regions: Iterable[str] | None = None,
    overwrite: bool = False,
) -> tuple[DownloadedProducts, ...]:
    """Download WorldCover 2021, USGS 3DEP, and OpenStreetMap roads.

    Files are cached below ``[data].cache_dir``.  Existing complete files are
    reused unless ``overwrite`` is true.
    """
    coordinates = load_coordinates(config.data.coordinates)
    requested = set(regions) if regions is not None else set(coordinates.region.tolist())
    unknown = requested - set(coordinates.region.tolist())
    if unknown:
        raise ValueError(f"Coordinates do not contain requested regions: {sorted(unknown)}")
    products = []
    for region in sorted(requested):
        products.append(
            _download_region(
                region,
                coordinates.by_region(region),
                config.data.cache_dir / region,
                resolution_m=config.preprocessing.resolution_m,
                overwrite=overwrite,
            )
        )
    return tuple(products)


def _download_region(
    region: str,
    coordinates: Coordinates,
    directory: Path,
    *,
    resolution_m: float,
    overwrite: bool,
) -> DownloadedProducts:
    directory.mkdir(parents=True, exist_ok=True)
    bounds = coordinate_bounds(coordinates)
    dem = directory / "usgs_3dep.tif"
    roads = directory / "osm_roads.geojson"
    if overwrite or not dem.is_file():
        _download_3dep(bounds, dem, resolution_m=resolution_m)
    if overwrite or not roads.is_file():
        _download_roads(bounds, roads)

    item_file = directory / "worldcover_items.json"
    if overwrite or not item_file.is_file():
        items = _worldcover_items(bounds)
        item_file.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
    else:
        items = json.loads(item_file.read_text(encoding="utf-8"))
    paths: list[Path] = []
    for index, item in enumerate(items):
        destination = directory / f"worldcover_{index:02d}.tif"
        if overwrite or not destination.is_file():
            href = str(item["assets"]["map"]["href"])
            try:
                import planetary_computer

                href = planetary_computer.sign_url(href)
            except ImportError:
                pass
            _stream_download(href, destination)
        paths.append(destination)
    if not paths:
        raise RuntimeError(f"No ESA WorldCover assets found for {region}")
    return DownloadedProducts(region, dem, tuple(paths), roads)


def _worldcover_items(bounds: tuple[float, float, float, float]) -> list[dict[str, Any]]:
    response = requests.post(
        PLANETARY_COMPUTER_SEARCH,
        json={"collections": ["esa-worldcover"], "bbox": list(bounds), "limit": 32},
        timeout=180,
    )
    response.raise_for_status()
    items = response.json().get("features", [])
    selected = [item for item in items if "2021" in str(item.get("id", ""))]
    return selected or items


def _download_3dep(
    bounds: tuple[float, float, float, float],
    destination: Path,
    *,
    resolution_m: float,
) -> None:
    west, south, east, north = bounds
    mean_latitude = 0.5 * (south + north)
    metres_per_degree = 111_320.0
    width = max(
        round(
            (east - west) * metres_per_degree * math.cos(math.radians(mean_latitude)) / resolution_m
        ),
        64,
    )
    height = max(round((north - south) * metres_per_degree / resolution_m), 64)
    response = requests.get(
        USGS_3DEP_EXPORT,
        params={
            "bbox": f"{west},{south},{east},{north}",
            "bboxSR": 4326,
            "imageSR": 4326,
            "size": f"{width},{height}",
            "format": "tiff",
            "pixelType": "F32",
            "interpolation": "+RSP_BilinearInterpolation",
            "f": "image",
        },
        timeout=300,
    )
    response.raise_for_status()
    destination.write_bytes(response.content)


def _download_roads(bounds: tuple[float, float, float, float], destination: Path) -> None:
    west, south, east, north = bounds
    query = f"[out:json][timeout:90];(way['highway']({south},{west},{north},{east}););out geom;"
    response = requests.get(
        OVERPASS_API,
        params={"data": query},
        headers={"User-Agent": "deepilg-research-code/0.1"},
        timeout=180,
    )
    response.raise_for_status()
    excluded = {
        "footway",
        "cycleway",
        "path",
        "pedestrian",
        "steps",
        "bridleway",
        "corridor",
        "construction",
        "proposed",
        "platform",
    }
    features = []
    for element in response.json().get("elements", []):
        tags = element.get("tags", {})
        highway = str(tags.get("highway", ""))
        vertices = element.get("geometry", [])
        if element.get("type") != "way" or highway in excluded or len(vertices) < 2:
            continue
        line = LineString([(float(point["lon"]), float(point["lat"])) for point in vertices])
        if not line.is_empty and line.length > 0:
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(line),
                    "properties": {"osm_id": int(element["id"]), "highway": highway},
                }
            )
    destination.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2) + "\n",
        encoding="utf-8",
    )


def _stream_download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    stream.write(chunk)
    temporary.replace(destination)
