"""Construction of aligned 14-channel neural raster inputs."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import rowcol
from rasterio.warp import calculate_default_transform, reproject
from scipy.ndimage import convolve

from deepilg.config import ExperimentConfig
from deepilg.data import Coordinates, load_coordinates
from deepilg.downloads import DownloadedProducts
from deepilg.provenance import write_workflow_manifest

WORLDCOVER_CODES = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
FEATURE_NAMES = (
    *(f"worldcover_{code}" for code in WORLDCOVER_CODES),
    "elevation",
    "road_density",
)


@dataclass(frozen=True)
class ProcessedRegion:
    """Paths produced for one geographic region."""

    region: str
    features: Path
    covariates: Path
    populations: Path
    metadata: Path


def encode_worldcover(classes: np.ndarray) -> np.ndarray:
    """One-hot encode 11 WorldCover classes plus an explicit unknown class."""
    values = np.asarray(classes)
    finite = np.isfinite(values)
    rounded = np.zeros(values.shape, dtype=np.int16)
    rounded[finite] = np.rint(values[finite]).astype(np.int16)
    known = np.isin(rounded, WORLDCOVER_CODES[1:]) & finite
    canonical = np.where(known, rounded, 0)
    return np.stack([(canonical == code).astype(np.float32) for code in WORLDCOVER_CODES], axis=-1)


def disk_kernel(radius_cells: int) -> np.ndarray:
    """Return a binary circular neighbourhood kernel."""
    if radius_cells < 0:
        raise ValueError("radius_cells must be non-negative")
    coordinate = np.arange(-radius_cells, radius_cells + 1)
    rows, columns = np.meshgrid(coordinate, coordinate, indexing="ij")
    return (rows * rows + columns * columns <= radius_cells * radius_cells).astype(np.float32)


def rasterized_road_density(
    occupancy: np.ndarray,
    *,
    resolution_m: float,
    radius_m: float,
) -> np.ndarray:
    """Approximate road length per km² from rasterized road occupancy.

    Each occupied cell contributes one pixel width of road.  This preserves the
    study's rasterized road-length approximation rather than implying exact
    line-length integration.
    """
    if resolution_m <= 0 or radius_m <= 0:
        raise ValueError("resolution_m and radius_m must be positive")
    radius_cells = max(round(radius_m / resolution_m), 1)
    kernel = disk_kernel(radius_cells)
    local_occupied = convolve(np.asarray(occupancy, dtype=np.float32), kernel, mode="constant")
    road_km = local_occupied * resolution_m / 1_000.0
    neighbourhood_km2 = float(kernel.sum()) * resolution_m * resolution_m / 1_000_000.0
    return np.asarray(road_km / neighbourhood_km2, dtype=np.float32)


def impute_and_scale(features: np.ndarray) -> np.ndarray:
    """Median-impute and independently min-max scale each raster channel."""
    values = np.asarray(features, dtype=np.float32).copy()
    if values.ndim != 3:
        raise ValueError(f"Expected HWC features, got {values.shape}")
    for channel in range(values.shape[-1]):
        band = values[..., channel]
        finite = np.isfinite(band)
        if not finite.any():
            raise ValueError(f"Feature channel {channel} contains no finite values")
        median = float(np.median(band[finite]))
        band[~finite] = median
        minimum, maximum = float(band.min()), float(band.max())
        if maximum > minimum:
            band[:] = (band - minimum) / (maximum - minimum)
        else:
            band[:] = 0.0
    if not np.all(np.isfinite(values)):
        raise AssertionError("Imputation left non-finite feature values")
    return values


def crop_to_patch_grid(array: np.ndarray, patch_size: int) -> np.ndarray:
    """Crop only bottom and right edges to an exact patch grid."""
    if patch_size <= 0 or array.ndim < 2:
        raise ValueError("patch_size must be positive and array at least two-dimensional")
    height = (array.shape[0] // patch_size) * patch_size
    width = (array.shape[1] // patch_size) * patch_size
    if height == 0 or width == 0:
        raise ValueError("Raster is smaller than one patch")
    return array[:height, :width, ...]


def preprocess_downloaded_products(
    config: ExperimentConfig,
    products: Sequence[DownloadedProducts],
    *,
    overwrite: bool = False,
) -> tuple[ProcessedRegion, ...]:
    """Align public products and write model-ready arrays for all regions."""
    coordinates = load_coordinates(config.data.coordinates)
    write_workflow_manifest(config, config.data.processed_dir)
    output = []
    for product in products:
        output.append(
            preprocess_region(
                product,
                coordinates.by_region(product.region),
                config.data.processed_dir / product.region,
                resolution_m=config.preprocessing.resolution_m,
                road_radius_m=config.preprocessing.road_radius_m,
                patch_size=config.preprocessing.patch_size,
                geolocation_status=config.data.geolocation_status,
                scientific_use=config.data.scientific_use,
                overwrite=overwrite,
            )
        )
    return tuple(output)


def preprocess_region(
    products: DownloadedProducts,
    coordinates: Coordinates,
    output_dir: Path,
    *,
    resolution_m: float,
    road_radius_m: float,
    patch_size: int,
    geolocation_status: str,
    scientific_use: bool,
    overwrite: bool = False,
) -> ProcessedRegion:
    """Build aligned raw covariates, neural features, and population nodes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features.npy"
    covariate_path = output_dir / "covariates.tif"
    population_path = output_dir / "populations.csv"
    metadata_path = output_dir / "metadata.json"
    outputs = ProcessedRegion(
        products.region, feature_path, covariate_path, population_path, metadata_path
    )
    if not overwrite and all(
        path.is_file() for path in (feature_path, covariate_path, population_path, metadata_path)
    ):
        validate_processed_region(
            outputs,
            patch_size=patch_size,
            resolution_m=resolution_m,
            geolocation_status=geolocation_status,
            scientific_use=scientific_use,
            expected_population_ids=set(coordinates.population_id.tolist()),
        )
        return outputs

    with rasterio.open(products.dem) as source:
        destination_crs = _utm_crs(
            float(coordinates.longitude.mean()), float(coordinates.latitude.mean())
        )
        transform, width, height = calculate_default_transform(
            source.crs,
            destination_crs,
            source.width,
            source.height,
            *source.bounds,
            resolution=resolution_m,
        )
        elevation = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            rasterio.band(source, 1),
            elevation,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=transform,
            dst_crs=destination_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    sources = [rasterio.open(path) for path in products.worldcover]
    try:
        mosaic, mosaic_transform = merge(sources)
        landcover = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            mosaic[0],
            landcover,
            src_transform=mosaic_transform,
            src_crs=sources[0].crs,
            dst_transform=transform,
            dst_crs=destination_crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
    finally:
        for source in sources:
            source.close()

    roads = gpd.read_file(products.roads).to_crs(destination_crs)
    occupancy = rasterize(
        [
            (geometry, 1.0)
            for geometry in roads.geometry
            if geometry is not None and not geometry.is_empty
        ],
        out_shape=(height, width),
        transform=transform,
        fill=0.0,
        all_touched=True,
        dtype=np.float32,
    )
    road_density = rasterized_road_density(
        occupancy, resolution_m=resolution_m, radius_m=road_radius_m
    )

    rounded_landcover = np.zeros(landcover.shape, dtype=np.float32)
    landcover_finite = np.isfinite(landcover)
    rounded_landcover[landcover_finite] = np.rint(landcover[landcover_finite]).astype(np.float32)
    rounded_landcover[~np.isin(rounded_landcover, WORLDCOVER_CODES[1:])] = 0.0

    elevation_filled = np.asarray(elevation, dtype=np.float32).copy()
    finite_elevation = np.isfinite(elevation_filled)
    if not finite_elevation.any():
        raise ValueError(f"{products.region}: elevation raster contains no finite values")
    elevation_filled[~finite_elevation] = np.median(elevation_filled[finite_elevation])
    road_density_filled = np.asarray(road_density, dtype=np.float32).copy()
    finite_roads = np.isfinite(road_density_filled)
    if not finite_roads.any():
        raise ValueError(f"{products.region}: road-density raster contains no finite values")
    road_density_filled[~finite_roads] = np.median(road_density_filled[finite_roads])

    raw = np.stack([rounded_landcover, elevation_filled, road_density_filled], axis=-1)
    features = np.concatenate(
        [
            encode_worldcover(rounded_landcover),
            elevation_filled[..., None],
            road_density_filled[..., None],
        ],
        axis=-1,
    )
    raw = crop_to_patch_grid(raw, patch_size)
    features = crop_to_patch_grid(features, patch_size)
    features = impute_and_scale(features)

    profile = {
        "driver": "GTiff",
        "height": raw.shape[0],
        "width": raw.shape[1],
        "count": 3,
        "dtype": "float32",
        "crs": destination_crs,
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(covariate_path, "w", **profile) as destination:
        destination.write(np.moveaxis(raw.astype(np.float32), -1, 0))
        destination.descriptions = ("worldcover", "elevation", "road_density")
    np.save(feature_path, features.astype(np.float32))

    points = gpd.GeoDataFrame(
        {"population_id": coordinates.population_id},
        geometry=gpd.points_from_xy(coordinates.longitude, coordinates.latitude),
        crs="EPSG:4326",
    ).to_crs(destination_crs)
    rows, columns = rowcol(transform, points.geometry.x.to_numpy(), points.geometry.y.to_numpy())
    rows, columns = np.asarray(rows), np.asarray(columns)
    valid = (rows >= 0) & (rows < raw.shape[0]) & (columns >= 0) & (columns < raw.shape[1])
    if not np.all(valid):
        missing = coordinates.population_id[~valid].tolist()
        raise ValueError(f"{products.region}: populations cropped outside raster: {missing}")
    patch_rows = rows // patch_size
    patch_columns = columns // patch_size
    patch_grid_columns = raw.shape[1] // patch_size
    with population_path.open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "population_id",
            "pixel_row",
            "pixel_column",
            "pixel_node",
            "patch_row",
            "patch_column",
            "patch_node",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for population, row, column, patch_row, patch_column in zip(
            coordinates.population_id,
            rows,
            columns,
            patch_rows,
            patch_columns,
            strict=True,
        ):
            writer.writerow(
                {
                    "population_id": population,
                    "pixel_row": int(row),
                    "pixel_column": int(column),
                    "pixel_node": int(row) * raw.shape[1] + int(column),
                    "patch_row": int(patch_row),
                    "patch_column": int(patch_column),
                    "patch_node": int(patch_row) * patch_grid_columns + int(patch_column),
                }
            )
    metadata_path.write_text(
        json.dumps(
            {
                "region": products.region,
                "feature_names": FEATURE_NAMES,
                "shape": list(features.shape),
                "crs": str(destination_crs),
                "resolution_m": resolution_m,
                "patch_size": patch_size,
                "geolocation_status": geolocation_status,
                "scientific_use": scientific_use,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    validate_processed_region(
        outputs,
        patch_size=patch_size,
        resolution_m=resolution_m,
        geolocation_status=geolocation_status,
        scientific_use=scientific_use,
        expected_population_ids=set(coordinates.population_id.tolist()),
    )
    return outputs


def validate_processed_region(
    processed: ProcessedRegion,
    *,
    patch_size: int,
    resolution_m: float,
    geolocation_status: str,
    scientific_use: bool,
    expected_population_ids: set[str] | None = None,
) -> None:
    """Validate reusable preprocessing outputs before model or export use."""
    paths = (
        processed.features,
        processed.covariates,
        processed.populations,
        processed.metadata,
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{processed.region}: incomplete processed outputs: {missing}")
    metadata = json.loads(processed.metadata.read_text(encoding="utf-8"))
    if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError(f"{processed.region}: processed feature channel order is invalid")
    if int(metadata.get("patch_size", -1)) != patch_size:
        raise ValueError(f"{processed.region}: processed patch_size does not match configuration")
    if not np.isclose(float(metadata.get("resolution_m", np.nan)), resolution_m):
        raise ValueError(f"{processed.region}: processed resolution does not match configuration")
    if metadata.get("geolocation_status") != geolocation_status:
        raise ValueError(
            f"{processed.region}: processed geolocation status does not match configuration"
        )
    if metadata.get("scientific_use") is not scientific_use:
        raise ValueError(
            f"{processed.region}: processed scientific-use flag does not match configuration"
        )

    features = np.load(processed.features, mmap_mode="r")
    if (
        features.ndim != 3
        or features.shape[-1] != len(FEATURE_NAMES)
        or features.shape[0] % patch_size
        or features.shape[1] % patch_size
        or tuple(metadata.get("shape", ())) != features.shape
        or not np.all(np.isfinite(features))
    ):
        raise ValueError(f"{processed.region}: invalid processed neural feature array")
    with rasterio.open(processed.covariates) as raster:
        if (
            raster.count != 3
            or (raster.height, raster.width) != features.shape[:2]
            or raster.crs is None
            or not raster.crs.is_projected
            or not np.isclose(abs(raster.transform.a), resolution_m)
            or not np.isclose(abs(raster.transform.e), resolution_m)
            or not np.all(np.isfinite(raster.read()))
        ):
            raise ValueError(f"{processed.region}: invalid aligned raw-covariate raster")

    with processed.populations.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fields = {
        "population_id",
        "pixel_row",
        "pixel_column",
        "pixel_node",
        "patch_row",
        "patch_column",
        "patch_node",
    }
    if not rows or set(rows[0]) != fields:
        raise ValueError(f"{processed.region}: invalid population-node columns")
    population_ids = [row["population_id"] for row in rows]
    if len(population_ids) != len(set(population_ids)):
        raise ValueError(f"{processed.region}: duplicate processed population identifier")
    if expected_population_ids is not None and set(population_ids) != expected_population_ids:
        raise ValueError(f"{processed.region}: processed population identifiers changed")
    patch_columns = features.shape[1] // patch_size
    for row in rows:
        try:
            pixel_row = int(row["pixel_row"])
            pixel_column = int(row["pixel_column"])
            pixel_node = int(row["pixel_node"])
            patch_row = int(row["patch_row"])
            patch_column = int(row["patch_column"])
            patch_node = int(row["patch_node"])
        except ValueError as error:
            raise ValueError(f"{processed.region}: population nodes must be integers") from error
        if not (0 <= pixel_row < features.shape[0] and 0 <= pixel_column < features.shape[1]):
            raise ValueError(f"{processed.region}: population pixel lies outside the raster")
        expected = (
            pixel_row * features.shape[1] + pixel_column,
            pixel_row // patch_size,
            pixel_column // patch_size,
        )
        expected_patch_node = expected[1] * patch_columns + expected[2]
        if (
            pixel_node != expected[0]
            or patch_row != expected[1]
            or patch_column != expected[2]
            or patch_node != expected_patch_node
        ):
            raise ValueError(f"{processed.region}: inconsistent pixel-to-patch node mapping")


def _utm_crs(longitude: float, latitude: float) -> str:
    zone = int((longitude + 180.0) // 6.0) + 1
    epsg = 32600 + zone if latitude >= 0 else 32700 + zone
    return f"EPSG:{epsg}"
