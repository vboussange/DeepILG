"""Offline array and local-fixture tests for geospatial preprocessing."""

import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as transform_coordinates
from shapely.geometry import LineString

from deepilg.data import Coordinates
from deepilg.downloads import DownloadedProducts
from deepilg.preprocessing import (
    encode_worldcover,
    impute_and_scale,
    preprocess_region,
    rasterized_road_density,
)
from deepilg.resistancega import _aggregate_covariates


def test_feature_encoding_imputation_and_road_fixture() -> None:
    landcover = np.asarray([[10.0, 20.0], [999.0, np.nan]])
    encoded = encode_worldcover(landcover)
    assert encoded.shape == (2, 2, 12)
    assert encoded[0, 0, 1] == 1.0
    assert encoded[1, 0, 0] == 1.0
    assert encoded[1, 1, 0] == 1.0

    features = np.dstack([landcover, np.asarray([[1.0, 2.0], [np.nan, 4.0]])])
    scaled = impute_and_scale(features)
    assert np.all(np.isfinite(scaled))
    assert scaled.min() >= 0 and scaled.max() <= 1

    occupancy = np.zeros((25, 25), dtype=np.float32)
    occupancy[:, 12] = 1.0
    density = rasterized_road_density(occupancy, resolution_m=30, radius_m=300)
    assert density.shape == occupancy.shape
    assert density[12, 12] > density[12, 0]


def test_local_products_execute_full_preprocessing_without_network(tmp_path: Path) -> None:
    shape = (64, 64)
    transform = from_origin(500_000, 4_002_000, 30, 30)
    profile = {
        "driver": "GTiff",
        "height": shape[0],
        "width": shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32616",
        "transform": transform,
    }
    dem = tmp_path / "dem.tif"
    worldcover = tmp_path / "worldcover.tif"
    with rasterio.open(dem, "w", **profile) as raster:
        raster.write(np.linspace(100, 200, shape[0] * shape[1], dtype=np.float32).reshape(shape), 1)
    with rasterio.open(worldcover, "w", **profile) as raster:
        raster.write(np.full(shape, 10, dtype=np.float32), 1)
    roads = tmp_path / "roads.geojson"
    gpd.GeoDataFrame(
        {"highway": ["primary"]},
        geometry=[LineString([(500_000, 4_001_000), (501_900, 4_001_000)])],
        crs="EPSG:32616",
    ).to_file(roads, driver="GeoJSON")

    # Two distinct population pixels deliberately occupy the same 32x32 patch.
    eastings = [500_000 + 10.5 * 30, 500_000 + 20.5 * 30]
    northings = [4_002_000 - 10.5 * 30, 4_002_000 - 20.5 * 30]
    longitudes, latitudes = transform_coordinates("EPSG:32616", "EPSG:4326", eastings, northings)
    coordinates = Coordinates(
        region=np.asarray(["DEMO", "DEMO"]),
        population_id=np.asarray(["P1", "P2"]),
        latitude=np.asarray(latitudes),
        longitude=np.asarray(longitudes),
    )
    result = preprocess_region(
        DownloadedProducts("DEMO", dem, (worldcover,), roads),
        coordinates,
        tmp_path / "processed",
        resolution_m=30,
        road_radius_m=300,
        patch_size=32,
        geolocation_status="artificial",
        scientific_use=False,
    )
    features = np.load(result.features)
    assert features.shape == (64, 64, 14)
    assert np.all(np.isfinite(features))
    with result.populations.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert set(rows[0]) == {
        "population_id",
        "pixel_row",
        "pixel_column",
        "pixel_node",
        "patch_row",
        "patch_column",
        "patch_node",
    }
    assert rows[0]["patch_node"] == rows[1]["patch_node"] == "0"

    coarse = tmp_path / "surface.tif"
    _aggregate_covariates(result.covariates, coarse, patch_size=32)
    with rasterio.open(coarse) as raster:
        assert (raster.height, raster.width, raster.count) == (2, 2, 3)
        assert abs(raster.transform.a) == 960

    metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
    metadata["feature_names"] = list(reversed(metadata["feature_names"]))
    result.metadata.write_text(json.dumps(metadata), encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "channel order"):
        preprocess_region(
            DownloadedProducts("DEMO", dem, (worldcover,), roads),
            coordinates,
            tmp_path / "processed",
            resolution_m=30,
            road_radius_m=300,
            patch_size=32,
            geolocation_status="artificial",
            scientific_use=False,
            overwrite=False,
        )
