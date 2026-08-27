"""Export the shared folds and covariates to ResistanceGA2's R workflow."""

from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio

from deepilg.config import ExperimentConfig
from deepilg.data import linearize_fst, load_fst_directory
from deepilg.provenance import write_workflow_manifest
from deepilg.splits import load_fold_assignments, pair_mask, upper_triangle

REGIONS = ("CWMA", "GSMNP", "ORR", "PCSF", "RRG")


def run_resistancega(
    config: ExperimentConfig,
    *,
    fold: int,
    region: str,
    mode: Literal["fit", "predict"],
    split: Literal["validation", "test"] | None = None,
    rscript: str = "Rscript",
) -> Path:
    """Run one configured ResistanceGA2 fit or prediction through the R entry point.

    The fit seed is read from ``config.training.seed``. Prediction always reuses
    the existing region-specific fit and never invokes optimization.
    """
    if fold not in range(5):
        raise ValueError("fold must lie in 0..4")
    if region not in REGIONS:
        raise ValueError(f"region must be one of: {', '.join(REGIONS)}")
    if mode == "fit" and split is not None:
        raise ValueError("--split is valid only when --mode predict")
    if mode == "predict" and split is None:
        raise ValueError("--split is required when --mode predict")

    script = config.root / "resistancega" / "run.R"
    if not script.is_file():
        raise FileNotFoundError(f"ResistanceGA2 entry point not found: {script}")
    executable = shutil.which(rscript)
    if executable is None:
        raise FileNotFoundError(
            f"R executable {rscript!r} was not found; restore resistancega/renv.lock "
            "and ensure Rscript is on PATH"
        )

    input_dir = config.project.output_dir / "resistancega" / "inputs" / f"fold_{fold:02d}" / region
    fit_dir = config.project.output_dir / "resistancega" / "fits" / f"fold_{fold:02d}" / region
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"ResistanceGA2 inputs not found at {input_dir}; run "
            f"`deepilg resistancega-export --config {config.source} --fold {fold}` first"
        )

    if mode == "fit":
        arguments = (
            executable,
            str(script),
            "fit",
            str(input_dir),
            str(fit_dir),
            str(config.training.seed),
        )
        expected = fit_dir / "fit.rds"
    else:
        if not fit_dir.is_dir():
            raise FileNotFoundError(f"ResistanceGA2 fit not found at {fit_dir}")
        arguments = (
            executable,
            str(script),
            "predict",
            str(input_dir),
            str(fit_dir),
            str(split),
        )
        expected = fit_dir / f"{split}_predictions.csv"

    subprocess.run(arguments, cwd=config.root, check=True)
    if not expected.is_file():
        raise RuntimeError(f"ResistanceGA2 completed without producing {expected}")
    return expected


def export_resistancega_inputs(config: ExperimentConfig, *, fold: int) -> Path:
    """Export one population-holdout fold without fitting or invoking R."""
    if fold not in range(5):
        raise ValueError("fold must lie in 0..4")
    matrices = load_fst_directory(config.data.fst_dir)
    assignments = load_fold_assignments(config.data.folds)
    root = config.project.output_dir / "resistancega" / "inputs" / f"fold_{fold:02d}"
    write_workflow_manifest(config, config.project.output_dir / "resistancega" / "inputs")
    for region, matrix in matrices.items():
        source_dir = config.data.processed_dir / region
        destination = root / region
        destination.mkdir(parents=True, exist_ok=True)
        surface_path = destination / "surface.tif"
        _aggregate_covariates(
            source_dir / "covariates.tif",
            surface_path,
            patch_size=config.preprocessing.patch_size,
        )
        with (source_dir / "populations.csv").open(newline="", encoding="utf-8") as stream:
            population_rows = list(csv.DictReader(stream))
        by_id = {row["population_id"]: row for row in population_rows}
        if set(by_id) != set(matrix.population_ids):
            raise ValueError(f"{region}: processed and FST population IDs differ")
        with rasterio.open(surface_path) as raster:
            transform = raster.transform
        points = []
        for population in matrix.population_ids:
            row = int(by_id[population]["patch_row"])
            column = int(by_id[population]["patch_column"])
            x, y = transform * (column + 0.5, row + 0.5)
            points.append({"population_id": population, "x": x, "y": y})

        target = np.asarray(linearize_fst(matrix.values))
        for split in ("train", "validation", "test"):
            split_mask = pair_mask(
                matrix.population_ids,
                assignments,
                region=region,
                fold=fold,
                partition=split,
            )
            included_ids = (
                assignments.ids(region, fold, "train")
                if split == "train"
                else tuple(
                    population
                    for population in matrix.population_ids
                    if split == "test" or assignments.partition(region, fold, population) != "test"
                )
            )
            _write_csv(
                destination / f"{split}_points.csv",
                [row for row in points if row["population_id"] in included_ids],
                ("population_id", "x", "y"),
            )
            if split == "train":
                _write_labelled_matrix(
                    destination / "train_target.csv",
                    matrix.population_ids,
                    target,
                    included_ids,
                )
            else:
                pair_rows = _pair_rows(matrix.population_ids, target, split_mask)
                _write_csv(
                    destination / f"{split}_pairs.csv",
                    pair_rows,
                    ("population_i", "population_j", "observed"),
                )
    return root


def _pair_rows(
    population_ids: tuple[str, ...], matrix: np.ndarray, mask: np.ndarray
) -> list[dict[str, object]]:
    row, column = np.triu_indices(len(population_ids), k=1)
    values = upper_triangle(matrix, mask)
    output = []
    for index, pair_index in enumerate(np.flatnonzero(mask)):
        output.append(
            {
                "population_i": population_ids[int(row[pair_index])],
                "population_j": population_ids[int(column[pair_index])],
                "observed": float(values[index]),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_labelled_matrix(
    path: Path,
    population_ids: tuple[str, ...],
    matrix: np.ndarray,
    selected_ids: tuple[str, ...],
) -> None:
    index = {population: position for position, population in enumerate(population_ids)}
    selected_indices = [index[population] for population in selected_ids]
    selected = matrix[np.ix_(selected_indices, selected_indices)]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["", *selected_ids])
        for population, values in zip(selected_ids, selected, strict=True):
            writer.writerow([population, *[float(value) for value in values]])


def _aggregate_covariates(source: Path, destination: Path, *, patch_size: int) -> None:
    """Aggregate native covariates to the shared patch grid used by all models."""
    with rasterio.open(source) as raster:
        values = raster.read().astype(np.float64)
        profile = raster.profile.copy()
        descriptions = raster.descriptions
        transform = raster.transform
    if values.shape[0] != 3:
        raise ValueError(f"{source}: expected WorldCover, elevation, and road-density bands")
    height, width = values.shape[1:]
    if height % patch_size or width % patch_size:
        raise ValueError(f"{source}: covariate dimensions must be divisible by patch_size")
    patch_rows, patch_columns = height // patch_size, width // patch_size
    blocks = values.reshape(3, patch_rows, patch_size, patch_columns, patch_size)
    blocks = blocks.transpose(0, 1, 3, 2, 4)
    aggregated = np.empty((3, patch_rows, patch_columns), dtype=np.float32)
    for row in range(patch_rows):
        for column in range(patch_columns):
            landcover = blocks[0, row, column].reshape(-1)
            landcover = landcover[np.isfinite(landcover)]
            if landcover.size == 0:
                aggregated[0, row, column] = 0.0
            else:
                classes, counts = np.unique(np.rint(landcover).astype(np.int16), return_counts=True)
                aggregated[0, row, column] = float(classes[np.argmax(counts)])
    with np.errstate(invalid="ignore"):
        aggregated[1:] = np.nanmean(blocks[1:], axis=(3, 4)).astype(np.float32)
    if not np.all(np.isfinite(aggregated)):
        raise ValueError(f"{source}: patch aggregation produced non-finite covariates")
    profile.update(
        height=patch_rows,
        width=patch_columns,
        count=3,
        dtype="float32",
        transform=transform * rasterio.Affine.scale(patch_size, patch_size),
        compress="deflate",
    )
    with rasterio.open(destination, "w", **profile) as raster:
        raster.write(aggregated)
        raster.descriptions = descriptions or ("worldcover", "elevation", "road_density")
