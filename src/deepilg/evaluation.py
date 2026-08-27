"""Checkpoint evaluation and tidy held-out prediction export."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import equinox as eqx
import jax
import numpy as np

from deepilg.config import ExperimentConfig
from deepilg.metrics import FoldMetrics, evaluate_fold
from deepilg.models import GeographicDistanceModel, ResNet9Conductance, UNetEmbeddingDistance
from deepilg.provenance import write_workflow_manifest
from deepilg.splits import upper_triangle
from deepilg.training import (
    build_solver_contexts,
    initialise_model,
    load_fold_batches,
    predict_matrix,
    predict_resnet_matrix_and_conductance,
)

EXPECTED_TEST_PAIRS = (170, 158, 159, 167, 167)


def evaluate_model(
    config: ExperimentConfig,
    *,
    model_name: Literal["reference", "resistancega", "resnet9", "unet"],
    fold: int,
) -> tuple[FoldMetrics, Path]:
    """Evaluate a saved fit on pairs touching held-out test populations."""
    batches = load_fold_batches(config, fold)
    write_workflow_manifest(config, config.project.output_dir / "evaluation")
    if model_name == "resistancega":
        return _evaluate_resistancega(config, batches=batches, fold=fold)
    predictions_by_region: dict[str, np.ndarray] = {}
    targets_by_region: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []

    if model_name == "reference":
        source = config.project.output_dir / "reference" / f"fold_{fold}" / "calibration.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        reference = GeographicDistanceModel(float(payload["slope"]), float(payload["intercept"]))
        model = None
        contexts = {}
    else:
        model = initialise_model(
            model_name,
            patch_size=config.preprocessing.patch_size,
            seed=config.training.seed,
        )
        source = config.project.output_dir / model_name / f"fold_{fold}" / "best.eqx"
        model = eqx.tree_deserialise_leaves(source, model)
        contexts = build_solver_contexts(model, batches, config)

    for batch in batches:
        if model_name == "reference":
            width = batch.features.shape[1]
            row_coordinate = np.asarray(batch.pixel_nodes // width) * batch.resolution_m
            column_coordinate = np.asarray(batch.pixel_nodes % width) * batch.resolution_m
            points = np.stack([row_coordinate, column_coordinate], axis=1)
            delta = points[:, None, :] - points[None, :, :]
            prediction_matrix = reference.predict(np.sqrt(np.sum(delta * delta, axis=-1)))
        else:
            assert isinstance(model, (ResNet9Conductance, UNetEmbeddingDistance))
            if isinstance(model, ResNet9Conductance):
                predicted_matrix, conductance_array = predict_resnet_matrix_and_conductance(
                    model,
                    batch,
                    context=contexts.get(batch.region),
                )
                prediction_matrix = np.asarray(predicted_matrix)
                surface_dir = (
                    config.project.output_dir
                    / "evaluation"
                    / "resnet9"
                    / f"fold_{fold}"
                    / "surfaces"
                )
                surface_dir.mkdir(parents=True, exist_ok=True)
                conductance = np.asarray(conductance_array)
                np.save(
                    surface_dir / f"{batch.region}_resistance.npy",
                    np.reciprocal(np.clip(conductance, 1e-12, None)).astype(np.float32),
                )
            else:
                prediction_matrix = np.asarray(
                    predict_matrix(
                        model,
                        batch,
                        context=contexts.get(batch.region),
                        inference=True,
                        key=jax.random.PRNGKey(config.training.seed),
                    )
                )
        mask = np.asarray(batch.masks["test"])
        predictions = upper_triangle(prediction_matrix, mask)
        targets = upper_triangle(np.asarray(batch.targets), mask)
        predictions_by_region[batch.region] = predictions
        targets_by_region[batch.region] = targets
        tri_row, tri_column = np.triu_indices(len(batch.population_ids), k=1)
        selected = np.flatnonzero(mask)
        for output_index, pair_index in enumerate(selected):
            left, right = sorted(
                (
                    batch.population_ids[int(tri_row[pair_index])],
                    batch.population_ids[int(tri_column[pair_index])],
                )
            )
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "region": batch.region,
                    "population_i": left,
                    "population_j": right,
                    "observed": float(targets[output_index]),
                    "prediction": float(predictions[output_index]),
                }
            )

    metrics = evaluate_fold(predictions_by_region, targets_by_region)
    if metrics.n_pairs != EXPECTED_TEST_PAIRS[fold]:
        raise ValueError(
            f"Fold {fold}: expected {EXPECTED_TEST_PAIRS[fold]} held-out test pairs, "
            f"found {metrics.n_pairs}"
        )
    output = config.project.output_dir / "evaluation" / model_name / f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "test_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "metrics.json").write_text(
        json.dumps(asdict(metrics), indent=2) + "\n", encoding="utf-8"
    )
    _refresh_fold_metrics(config)
    return metrics, prediction_path


def _evaluate_resistancega(
    config: ExperimentConfig,
    *,
    batches: tuple,
    fold: int,
) -> tuple[FoldMetrics, Path]:
    """Validate and evaluate ResistanceGA2's held-out prediction files."""
    predictions_by_region: dict[str, np.ndarray] = {}
    targets_by_region: dict[str, np.ndarray] = {}
    normalized_rows: list[dict[str, object]] = []
    fit_root = config.project.output_dir / "resistancega" / "fits" / f"fold_{fold:02d}"
    required_columns = {
        "population_i",
        "population_j",
        "observed",
        "predicted",
        "resistance_distance",
    }
    for batch in batches:
        source = fit_root / batch.region / "test_predictions.csv"
        with source.open(newline="", encoding="utf-8") as stream:
            records = list(csv.DictReader(stream))
        if not records or set(records[0]) != required_columns:
            raise ValueError(f"{source}: expected columns {sorted(required_columns)}")
        tri_row, tri_column = np.triu_indices(len(batch.population_ids), k=1)
        selected = np.flatnonzero(np.asarray(batch.masks["test"]))
        expected = {}
        target_vector = upper_triangle(np.asarray(batch.targets), np.asarray(batch.masks["test"]))
        for output_index, pair_index in enumerate(selected):
            endpoints = tuple(
                sorted(
                    (
                        batch.population_ids[int(tri_row[pair_index])],
                        batch.population_ids[int(tri_column[pair_index])],
                    )
                )
            )
            expected[endpoints] = float(target_vector[output_index])
        observed: dict[tuple[str, str], tuple[float, float]] = {}
        for record in records:
            endpoints = tuple(sorted((record["population_i"], record["population_j"])))
            if endpoints in observed:
                raise ValueError(f"{source}: duplicate prediction for pair {endpoints}")
            try:
                target = float(record["observed"])
                prediction = float(record["predicted"])
                resistance = float(record["resistance_distance"])
            except ValueError as error:
                raise ValueError(f"{source}: predictions must be numeric") from error
            if not np.all(np.isfinite((target, prediction, resistance))):
                raise ValueError(f"{source}: predictions must be finite")
            observed[endpoints] = (target, prediction)
        if set(observed) != set(expected):
            missing = sorted(set(expected) - set(observed))
            extra = sorted(set(observed) - set(expected))
            raise ValueError(
                f"{source}: held-out endpoint mismatch; missing={missing}, extra={extra}"
            )
        ordered_endpoints = sorted(expected)
        targets = np.asarray([expected[pair] for pair in ordered_endpoints])
        file_targets = np.asarray([observed[pair][0] for pair in ordered_endpoints])
        predictions = np.asarray([observed[pair][1] for pair in ordered_endpoints])
        if not np.allclose(file_targets, targets, rtol=1e-6, atol=1e-8):
            raise ValueError(f"{source}: observed targets differ from the shared benchmark targets")
        predictions_by_region[batch.region] = predictions
        targets_by_region[batch.region] = targets
        for pair, target, prediction in zip(ordered_endpoints, targets, predictions, strict=True):
            normalized_rows.append(
                {
                    "model": "resistancega",
                    "fold": fold,
                    "region": batch.region,
                    "population_i": pair[0],
                    "population_j": pair[1],
                    "observed": float(target),
                    "prediction": float(prediction),
                }
            )

    metrics = evaluate_fold(predictions_by_region, targets_by_region)
    if metrics.n_pairs != EXPECTED_TEST_PAIRS[fold]:
        raise ValueError(
            f"Fold {fold}: expected {EXPECTED_TEST_PAIRS[fold]} held-out test pairs, "
            f"found {metrics.n_pairs}"
        )
    output = config.project.output_dir / "evaluation" / "resistancega" / f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "test_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=normalized_rows[0].keys())
        writer.writeheader()
        writer.writerows(normalized_rows)
    (output / "metrics.json").write_text(
        json.dumps(asdict(metrics), indent=2) + "\n", encoding="utf-8"
    )
    _refresh_fold_metrics(config)
    return metrics, prediction_path


def _refresh_fold_metrics(config: ExperimentConfig) -> Path:
    """Rebuild the deterministic tidy index consumed by figure generation."""
    evaluation_root = config.project.output_dir / "evaluation"
    rows = []
    for source in sorted(evaluation_root.glob("*/fold_*/metrics.json")):
        model = source.parent.parent.name
        try:
            fold = int(source.parent.name.removeprefix("fold_"))
        except ValueError:
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": model,
                "fold": fold,
                "pearson_r": float(payload["pearson_r"]),
                "r2": float(payload["r2"]),
                "rmse": float(payload["rmse"]),
                "n_pairs": int(payload["n_pairs"]),
            }
        )
    destination = evaluation_root / "fold_metrics.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("model", "fold", "pearson_r", "r2", "rmse", "n_pairs"),
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (str(row["model"]), int(row["fold"]))))
    return destination
