"""Small, deterministic helpers for benchmark and surface figures."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def summarize_surfaces(surfaces: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return the pixelwise median and interquartile range across folds."""
    if not surfaces:
        raise ValueError("At least one surface is required")
    stack = np.stack([np.asarray(surface, dtype=np.float64) for surface in surfaces])
    if not np.all(np.isfinite(stack)):
        raise ValueError("Surfaces must be finite")
    return np.median(stack, axis=0), np.percentile(stack, 75, axis=0) - np.percentile(
        stack, 25, axis=0
    )


def benchmark_figure(
    metrics_csv: str | Path,
    output: str | Path,
    *,
    model_order: Sequence[str] = ("reference", "resistancega", "unet", "resnet9"),
) -> Path:
    """Plot fold-level Pearson correlation and RMSE from a tidy CSV."""
    with Path(metrics_csv).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"model", "fold", "pearson_r", "rmse"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Metrics CSV must contain {sorted(required)}")
    present = {row["model"] for row in rows}
    labels = [model for model in model_order if model in present]
    labels += sorted(present - set(model_order))
    if not labels:
        raise ValueError("Metrics CSV contains no model rows")
    values: dict[str, dict[str, list[float]]] = {
        model: {"pearson_r": [], "rmse": []} for model in labels
    }
    for row in rows:
        values[row["model"]]["pearson_r"].append(float(row["pearson_r"]))
        values[row["model"]]["rmse"].append(float(row["rmse"]))

    figure, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("pearson_r", "rmse"),
        ("Equal-region Pearson correlation", "Pooled RMSE"),
        strict=True,
    ):
        means = [np.mean(values[model][metric]) for model in labels]
        errors = [np.std(values[model][metric], ddof=1) for model in labels]
        axis.errorbar(range(len(labels)), means, yerr=errors, fmt="o", capsize=3, color="#1f5a94")
        axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=300)
    plt.close(figure)
    return destination


def resistance_surface_figure(
    evaluation_dir: str | Path,
    processed_dir: str | Path,
    output: str | Path,
    *,
    patch_size: int = 32,
    regions: Sequence[str] | None = None,
) -> Path:
    """Plot context and resistance summaries across population-holdout models.

    ``regions`` defaults to every processed region; the folds summarised are
    discovered from the ResNet-9 surfaces written during evaluation.
    """
    evaluation_dir = Path(evaluation_dir)
    processed_dir = Path(processed_dir)
    if regions is None:
        regions = tuple(sorted(path.name for path in processed_dir.iterdir() if path.is_dir()))
    if not regions:
        raise FileNotFoundError(f"No processed regions found under {processed_dir}")
    figure, axes = plt.subplots(
        len(regions),
        3,
        figsize=(7.2, 1.75 * len(regions)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, region in enumerate(regions):
        paths = sorted(
            (evaluation_dir / "resnet9").glob(f"fold_*/surfaces/{region}_resistance.npy")
        )
        if not paths:
            raise FileNotFoundError(
                f"No fold-wise resistance surfaces found for {region} under {evaluation_dir}"
            )
        median, iqr = summarize_surfaces([np.load(path) for path in paths])
        with rasterio.open(processed_dir / region / "covariates.tif") as raster:
            landcover = raster.read(1)
        context = _modal_blocks(landcover, patch_size)
        if context.shape != median.shape:
            raise ValueError(
                f"{region}: context shape {context.shape} does not match resistance {median.shape}"
            )
        panels = (
            (context, "WorldCover context", "tab20"),
            (np.log10(np.clip(median, 1e-12, None)), "Median log10 resistance", "viridis"),
            (iqr, "Resistance IQR", "magma"),
        )
        for column, (values, title, colourmap) in enumerate(panels):
            axis = axes[row, column]
            axis.imshow(values, cmap=colourmap, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(region)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=300)
    plt.close(figure)
    return destination


def _modal_blocks(values: np.ndarray, patch_size: int) -> np.ndarray:
    height, width = values.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Context raster is not divisible by patch_size")
    rows, columns = height // patch_size, width // patch_size
    blocks = values.reshape(rows, patch_size, columns, patch_size).transpose(0, 2, 1, 3)
    output = np.empty((rows, columns), dtype=np.float32)
    for row in range(rows):
        for column in range(columns):
            block = blocks[row, column].reshape(-1)
            block = block[np.isfinite(block)]
            classes, counts = np.unique(np.rint(block).astype(np.int16), return_counts=True)
            output[row, column] = float(classes[np.argmax(counts)]) if classes.size else 0.0
    return output
