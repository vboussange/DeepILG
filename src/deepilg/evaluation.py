"""Checkpoint evaluation and tidy held-out prediction export."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import equinox as eqx
import jax
import numpy as np

from .config import ExperimentConfig
from .metrics import FoldMetrics, evaluate_fold
from .models import GeographicDistanceModel
from .provenance import write_workflow_manifest
from .splits import upper_triangle
from .training import (
    RegionBatch,
    build_solver_contexts,
    fit_directory,
    initialise_model,
    load_fold_batches,
    predict_matrix,
)


def evaluated_pooling(config: ExperimentConfig, model_name: str) -> str:
    """Return the pooling regime of an evaluated model; ResistanceGA never pools."""
    return "none" if model_name == "resistancega" else config.training.pooling


def evaluation_directory(config: ExperimentConfig, *, model_name: str, fold: int) -> Path:
    """Return ``<output>/evaluation/<model>/<pooling>/fold_XX`` for one evaluated fold."""
    pooling = evaluated_pooling(config, model_name)
    return config.project.output_dir / "evaluation" / model_name / pooling / f"fold_{fold:02d}"


def evaluate_model(
    config: ExperimentConfig,
    *,
    model_name: Literal["reference", "resistancega", "loglinear", "resnet9", "unet"],
    fold: int,
) -> tuple[FoldMetrics, Path]:
    """Score saved fits on the pairs touching held-out test populations.

    Under no pooling (always the case for ResistanceGA), the per-region fits
    are reassembled into one prediction set for the fold before scoring.
    """
    if model_name == "reference" and config.training.pooling != "complete":
        raise ValueError("The IBD model (reference) supports only complete pooling")
    batches = load_fold_batches(config, fold)
    write_workflow_manifest(config, config.project.output_dir / "evaluation")
    if model_name == "resistancega":
        predictions, training_seconds = _resistancega_predictions(config, batches, fold=fold)
    else:
        predictions, training_seconds = _fitted_predictions(
            config, batches, model_name=model_name, fold=fold
        )
    pooling = evaluated_pooling(config, model_name)
    rows: list[dict[str, object]] = []
    predictions_by_region: dict[str, np.ndarray] = {}
    targets_by_region: dict[str, np.ndarray] = {}
    for batch in batches:
        pairs = _test_pairs(batch)
        targets = upper_triangle(np.asarray(batch.targets), np.asarray(batch.masks["test"]))
        predictions_by_region[batch.region] = predictions[batch.region]
        targets_by_region[batch.region] = targets
        for (left, right), target, prediction in zip(
            pairs, targets, predictions[batch.region], strict=True
        ):
            rows.append(
                dict(
                    model=model_name,
                    pooling=pooling,
                    fold=fold,
                    region=batch.region,
                    population_i=left,
                    population_j=right,
                    observed=float(target),
                    prediction=float(prediction),
                )
            )
    metrics = evaluate_fold(predictions_by_region, targets_by_region)
    output = evaluation_directory(config, model_name=model_name, fold=fold)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "test_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "metrics.json").write_text(
        json.dumps({**asdict(metrics), "training_seconds": training_seconds}, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics, prediction_path


def _test_pairs(batch: RegionBatch) -> list[tuple[str, str]]:
    """Return sorted endpoint identifiers of the test pairs, in pair-mask order."""
    rows, columns = np.triu_indices(len(batch.population_ids), k=1)
    return [
        tuple(sorted((batch.population_ids[rows[index]], batch.population_ids[columns[index]])))
        for index in np.flatnonzero(np.asarray(batch.masks["test"]))
    ]


def _fitted_predictions(
    config: ExperimentConfig,
    batches: Sequence[RegionBatch],
    *,
    model_name: str,
    fold: int,
) -> tuple[dict[str, np.ndarray], float]:
    """Predict test pairs with every region's selected fit and sum their training time."""
    directories = {
        batch.region: fit_directory(
            config,
            model_name=model_name,
            fold=fold,
            region=None if config.training.pooling == "complete" else batch.region,
        )
        for batch in batches
    }
    artifact = "calibration.json" if model_name == "reference" else "best.eqx"
    for region, directory in directories.items():
        if not (directory / artifact).is_file():
            raise FileNotFoundError(f"{region}: missing fit {directory / artifact}")
    training_seconds = sum(
        float(
            json.loads((directory / "training_state.json").read_text(encoding="utf-8"))[
                "wall_clock_seconds"
            ]
        )
        for directory in set(directories.values())
    )
    if model_name == "reference":
        payload = json.loads(
            (directories[batches[0].region] / artifact).read_text(encoding="utf-8")
        )
        reference = GeographicDistanceModel(float(payload["slope"]), float(payload["intercept"]))
        matrices = {}
        for batch in batches:
            width = batch.features.shape[1]
            points = np.stack(
                [
                    np.asarray(batch.pixel_nodes // width) * batch.resolution_m,
                    np.asarray(batch.pixel_nodes % width) * batch.resolution_m,
                ],
                axis=1,
            )
            delta = points[:, None, :] - points[None, :, :]
            matrices[batch.region] = reference.predict(np.sqrt(np.sum(delta * delta, axis=-1)))
    else:
        template = initialise_model(
            model_name, patch_size=config.preprocessing.patch_size, seed=config.training.seed
        )
        models = {
            directory: eqx.tree_deserialise_leaves(directory / artifact, template)
            for directory in set(directories.values())
        }
        contexts = build_solver_contexts(template, batches, config)
        matrices = {
            batch.region: np.asarray(
                predict_matrix(
                    models[directories[batch.region]],
                    batch,
                    context=contexts.get(batch.region),
                    inference=True,
                    key=jax.random.PRNGKey(config.training.seed),
                )
            )
            for batch in batches
        }
    predictions = {
        batch.region: upper_triangle(matrices[batch.region], np.asarray(batch.masks["test"]))
        for batch in batches
    }
    return predictions, training_seconds


def _resistancega_predictions(
    config: ExperimentConfig, batches: Sequence[RegionBatch], *, fold: int
) -> tuple[dict[str, np.ndarray], float]:
    """Validate ResistanceGA2's per-region test predictions against the benchmark pairs."""
    fit_root = config.project.output_dir / "resistancega" / "fits" / f"fold_{fold:02d}"
    required = {"population_i", "population_j", "observed", "predicted", "resistance_distance"}
    predictions: dict[str, np.ndarray] = {}
    training_seconds = 0.0
    for batch in batches:
        source = fit_root / batch.region / "test_predictions.csv"
        with source.open(newline="", encoding="utf-8") as stream:
            records = list(csv.DictReader(stream))
        if not records or set(records[0]) != required:
            raise ValueError(f"{source}: expected columns {sorted(required)}")
        observed: dict[tuple[str, str], tuple[float, float]] = {}
        for record in records:
            endpoints = tuple(sorted((record["population_i"], record["population_j"])))
            if endpoints in observed:
                raise ValueError(f"{source}: duplicate prediction for pair {endpoints}")
            try:
                target, prediction, resistance = (
                    float(record[name]) for name in ("observed", "predicted", "resistance_distance")
                )
            except ValueError as error:
                raise ValueError(f"{source}: predictions must be numeric") from error
            if not np.all(np.isfinite((target, prediction, resistance))):
                raise ValueError(f"{source}: predictions must be finite")
            observed[endpoints] = (target, prediction)
        pairs = _test_pairs(batch)
        if set(observed) != set(pairs):
            missing = sorted(set(pairs) - set(observed))
            extra = sorted(set(observed) - set(pairs))
            raise ValueError(
                f"{source}: held-out endpoint mismatch; missing={missing}, extra={extra}"
            )
        targets = upper_triangle(np.asarray(batch.targets), np.asarray(batch.masks["test"]))
        # Targets are float32; R writes them back with 15 significant digits,
        # which still round to exactly the same float32 values.
        file_targets = np.asarray([observed[pair][0] for pair in pairs], dtype=np.float32)
        if not np.array_equal(file_targets, targets):
            raise ValueError(f"{source}: observed targets differ from the benchmark targets")
        predictions[batch.region] = np.asarray([observed[pair][1] for pair in pairs])
        timing = json.loads((source.parent / "fit_timing.json").read_text(encoding="utf-8"))
        training_seconds += float(timing["wall_clock_seconds"])
    return predictions, training_seconds
