"""Genetic-distance and population-coordinate data contracts."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

REGIONS = ("CWMA", "GSMNP", "ORR", "PCSF", "RRG")
EXPECTED_POPULATIONS = {"CWMA": 14, "GSMNP": 12, "ORR": 16, "PCSF": 10, "RRG": 16}


@dataclass(frozen=True)
class FSTMatrix:
    """A labelled, validated pairwise fixation-index matrix."""

    region: str
    population_ids: tuple[str, ...]
    values: np.ndarray

    @property
    def n_populations(self) -> int:
        """Number of populations represented by the matrix."""
        return len(self.population_ids)


@dataclass(frozen=True)
class Coordinates:
    """WGS84 coordinates keyed by region and population identifier."""

    region: np.ndarray
    population_id: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray

    def by_region(self, region: str) -> Coordinates:
        """Return rows belonging to one geographic region."""
        selected = self.region == region
        return Coordinates(
            region=self.region[selected],
            population_id=self.population_id[selected],
            latitude=self.latitude[selected],
            longitude=self.longitude[selected],
        )

    def as_mapping(self) -> dict[tuple[str, str], tuple[float, float]]:
        """Return a mapping from population keys to latitude/longitude."""
        return {
            (str(region), str(population)): (float(latitude), float(longitude))
            for region, population, latitude, longitude in zip(
                self.region,
                self.population_id,
                self.latitude,
                self.longitude,
                strict=True,
            )
        }


def validate_fst_matrix(
    values: np.ndarray,
    population_ids: Iterable[str],
    *,
    region: str,
    symmetry_tolerance: float = 1e-7,
) -> FSTMatrix:
    """Validate an observed FST matrix and return an immutable value object.

    Slightly negative off-diagonal estimates are allowed because sampling
    variation can produce them.  They are clipped only by :func:`linearize_fst`.
    """
    ids = tuple(str(value).strip() for value in population_ids)
    if not ids or any(not value for value in ids):
        raise ValueError(f"{region}: population identifiers must be non-empty")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{region}: population identifiers must be unique")
    matrix = np.asarray(values, dtype=np.float64)
    expected_shape = (len(ids), len(ids))
    if matrix.shape != expected_shape:
        raise ValueError(f"{region}: expected FST shape {expected_shape}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{region}: FST matrix contains non-finite values")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=symmetry_tolerance):
        maximum = float(np.max(np.abs(matrix - matrix.T)))
        raise ValueError(f"{region}: FST matrix is not symmetric (maximum error {maximum:g})")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=symmetry_tolerance):
        raise ValueError(f"{region}: FST diagonal must be zero")
    off_diagonal = matrix[~np.eye(len(ids), dtype=bool)]
    if np.any(off_diagonal >= 1.0):
        raise ValueError(f"{region}: FST values must be strictly smaller than one")
    matrix = matrix.copy()
    matrix.setflags(write=False)
    return FSTMatrix(region=region, population_ids=ids, values=matrix)


def load_fst_csv(path: str | Path, *, region: str | None = None) -> FSTMatrix:
    """Load a labelled square FST CSV with row identifiers in its first column."""
    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"Malformed FST matrix: {source}")
    column_ids = tuple(cell.strip() for cell in rows[0][1:])
    row_ids = tuple(row[0].strip() for row in rows[1:])
    if row_ids != column_ids:
        raise ValueError(f"{source}: FST row and column identifiers differ or are reordered")
    if any(len(row) != len(column_ids) + 1 for row in rows[1:]):
        raise ValueError(f"{source}: FST rows have inconsistent lengths")

    def parse(cell: str) -> float:
        token = cell.strip()
        if token.lower() in {"", "na", "nan"}:
            return float("nan")
        return float(token)

    try:
        values = np.asarray([[parse(cell) for cell in row[1:]] for row in rows[1:]])
    except ValueError as error:
        raise ValueError(f"{source}: FST matrix contains non-numeric entries") from error
    off_diagonal = ~np.eye(len(row_ids), dtype=bool)
    if np.any(~np.isfinite(values[off_diagonal])):
        raise ValueError(f"{source}: off-diagonal FST values must not be missing")
    diagonal = np.diag_indices(len(row_ids))
    if np.any(np.isfinite(values[diagonal]) & (np.abs(values[diagonal]) > 1e-7)):
        raise ValueError(f"{source}: finite FST diagonal entries must be zero")
    values[diagonal] = 0.0
    return validate_fst_matrix(values, row_ids, region=region or source.stem.split("_")[0])


def load_fst_directory(path: str | Path) -> dict[str, FSTMatrix]:
    """Load the five canonical ``<REGION>_fst_matrix.csv`` files."""
    directory = Path(path)
    matrices: dict[str, FSTMatrix] = {}
    for region in REGIONS:
        candidates = (
            directory / f"{region}_fst_matrix.csv",
            directory / f"{region}_FST.csv",
            directory / f"{region}_fst.csv",
        )
        source = next((candidate for candidate in candidates if candidate.is_file()), None)
        if source is None:
            raise FileNotFoundError(
                f"Missing FST matrix for {region}; expected {candidates[0].name}"
            )
        matrices[region] = load_fst_csv(source, region=region)
    return matrices


def linearize_fst(values: np.ndarray | jnp.ndarray) -> jnp.ndarray:
    """Return the standard distance-like transform ``max(FST, 0)/(1-FST)``."""
    array = jnp.asarray(values)
    clipped = jnp.maximum(array, 0.0)
    return clipped / (1.0 - clipped)


def load_coordinates(path: str | Path) -> Coordinates:
    """Load and validate ``region,population_id,latitude,longitude`` in WGS84."""
    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"region", "population_id", "latitude", "longitude"}
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError(f"{source}: coordinate columns must be exactly {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{source}: coordinate table is empty")
    try:
        latitude = np.asarray([float(row["latitude"]) for row in rows], dtype=np.float64)
        longitude = np.asarray([float(row["longitude"]) for row in rows], dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"{source}: coordinates must be numeric") from error
    region = np.asarray([row["region"].strip() for row in rows], dtype=str)
    population = np.asarray([row["population_id"].strip() for row in rows], dtype=str)
    if not np.all(np.isfinite(latitude)) or not np.all(np.isfinite(longitude)):
        raise ValueError(f"{source}: coordinates must be finite")
    if np.any((latitude < -90.0) | (latitude > 90.0)):
        raise ValueError(f"{source}: latitude must lie in [-90, 90]")
    if np.any((longitude < -180.0) | (longitude > 180.0)):
        raise ValueError(f"{source}: longitude must lie in [-180, 180]")
    if any(not item for item in region) or any(not item for item in population):
        raise ValueError(f"{source}: region and population_id must be non-empty")
    keys = list(zip(region.tolist(), population.tolist(), strict=True))
    if len(keys) != len(set(keys)):
        raise ValueError(f"{source}: region/population_id keys must be unique")
    return Coordinates(region, population, latitude, longitude)


def validate_population_alignment(
    matrices: Mapping[str, FSTMatrix], coordinates: Coordinates
) -> None:
    """Require coordinate and genetic inputs to contain identical population keys."""
    genetic_keys = {
        (region, population)
        for region, matrix in matrices.items()
        for population in matrix.population_ids
    }
    coordinate_keys = set(coordinates.as_mapping())
    missing = sorted(genetic_keys - coordinate_keys)
    extra = sorted(coordinate_keys - genetic_keys)
    if missing or extra:
        raise ValueError(
            "Coordinate/FST population mismatch: "
            f"missing_coordinates={missing}, extra_coordinates={extra}"
        )
