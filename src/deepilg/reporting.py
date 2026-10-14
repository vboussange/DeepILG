"""Matched benchmark report over every evaluated treatment."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import POOLING, ExperimentConfig
from .data import linearize_fst, load_fst_directory
from .figures import benchmark_figure
from .metrics import aggregate_folds, evaluate_fold
from .splits import load_fold_assignments, pair_mask, validate_fold_assignments
from .training import predicts_through_resistance_layer

# Display names, from simple to complex within each group of treatments.
MODEL_LABELS = {
    "reference": "IBD model",
    "resistancega": "ResistanceGA",
    "unet": "U-Net",
    "loglinear": "Log-linear encoder",
    "resnet9": "ResNet9 encoder",
}
NO_POOLING_MARK = "†"
METRICS = ("pearson_r", "r2", "rmse")


def treatment_label(model: str, pooling: str) -> str:
    """Return the display name of one model under one pooling regime; † marks no pooling."""
    label = MODEL_LABELS.get(model, model)
    return f"{label}{NO_POOLING_MARK}" if pooling == "none" else label


def _treatment_order(treatment: tuple[str, str]) -> tuple[bool, int, str, int]:
    """Baselines first, then effective-resistance-layer models; complete before no pooling."""
    model, pooling = treatment
    models = tuple(MODEL_LABELS)
    rank = models.index(model) if model in models else len(models)
    return predicts_through_resistance_layer(model), rank, model, POOLING.index(pooling)


def _expected_pairs(config: ExperimentConfig) -> dict[int, dict[tuple[str, str, str], float]]:
    """Rebuild every fold's held-out pairs and targets from the folds and genetic data."""
    assignments = load_fold_assignments(config.data.folds)
    matrices = load_fst_directory(config.data.fst_dir)
    validate_fold_assignments(
        assignments, {region: matrix.population_ids for region, matrix in matrices.items()}
    )
    expected: dict[int, dict[tuple[str, str, str], float]] = {}
    for fold in assignments.fold_indices:
        pairs = {}
        for region, matrix in matrices.items():
            rows, columns = np.triu_indices(matrix.n_populations, 1)
            mask = pair_mask(
                matrix.population_ids, assignments, region=region, fold=fold, partition="test"
            )
            targets = linearize_fst(matrix.values)
            for left, right in zip(rows[mask], columns[mask], strict=True):
                endpoints = sorted((matrix.population_ids[left], matrix.population_ids[right]))
                pairs[(region, *endpoints)] = float(targets[left, right])
        expected[fold] = pairs
    return expected


def _read_fold(
    source: Path,
    *,
    model: str,
    pooling: str,
    fold: int,
    expected: dict[tuple[str, str, str], float],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Validate one fold's predictions against the expected pairs and targets."""
    if not source.is_file():
        raise FileNotFoundError(f"{model}/{pooling} is missing fold {fold}: {source}")
    with source.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    observed: dict[tuple[str, str, str], tuple[float, float]] = {}
    for row in records:
        if (row["model"], row["pooling"], int(row["fold"])) != (model, pooling, fold):
            raise ValueError(f"{source}: rows belong to another model, pooling, or fold")
        key = (row["region"], *sorted((row["population_i"], row["population_j"])))
        if key in observed:
            raise ValueError(f"{source}: duplicate pair {key}")
        target, prediction = float(row["observed"]), float(row["prediction"])
        if not np.isfinite(target) or not np.isfinite(prediction):
            raise ValueError(f"{source}: non-finite prediction or target")
        if key not in expected or target != expected[key]:
            raise ValueError(f"{source}: pair {key} or its target differs from the folds")
        observed[key] = (target, prediction)
    if set(observed) != set(expected):
        raise ValueError(f"{source}: held-out pairs are incomplete")
    targets: dict[str, list[float]] = {}
    predictions: dict[str, list[float]] = {}
    for key in sorted(observed):
        targets.setdefault(key[0], []).append(observed[key][0])
        predictions.setdefault(key[0], []).append(observed[key][1])
    return (
        {region: np.asarray(values) for region, values in predictions.items()},
        {region: np.asarray(values) for region, values in targets.items()},
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def generate_benchmark_report(
    config: ExperimentConfig,
    *,
    evaluation_dir: str | Path | None = None,
    figure: str | Path | None = None,
) -> Path:
    """Score every evaluated treatment on identical held-out pairs and draw the fold figure.

    Treatments are the ``<model>/<pooling>`` directories under the evaluation
    directory. Each needs every fold of the fold file, finite predictions, and
    exactly the held-out pairs and targets rebuilt from the folds and genetic
    data. ``fold_metrics.csv`` and ``summary.csv`` are written next to the figure.
    They list the baselines first, then the models that predict through the
    effective-resistance layer, each model under complete pooling before no pooling.
    """
    evaluation = Path(evaluation_dir or config.project.output_dir / "evaluation")
    destination = Path(figure or config.project.output_dir / "figures" / "benchmark.pdf")
    expected = _expected_pairs(config)
    fold_names = {f"fold_{fold:02d}" for fold in expected}
    treatments = {
        (path.parent.parent.name, path.parent.name) for path in evaluation.glob("*/*/fold_*")
    }
    if not treatments:
        raise FileNotFoundError(f"No evaluated treatments under {evaluation}")
    for model, pooling in treatments:
        if pooling not in POOLING:
            raise ValueError(f"{evaluation / model / pooling}: pooling must be one of {POOLING}")
        present = {path.name for path in (evaluation / model / pooling).glob("fold_*")}
        if present - fold_names:
            raise ValueError(f"{model}/{pooling} has folds absent from the fold file")
    fold_rows, summary_rows = [], []
    for model, pooling in sorted(treatments, key=_treatment_order):
        label = treatment_label(model, pooling)
        scores, seconds = [], []
        for fold, pairs in expected.items():
            directory = evaluation / model / pooling / f"fold_{fold:02d}"
            predictions, targets = _read_fold(
                directory / "test_predictions.csv",
                model=model,
                pooling=pooling,
                fold=fold,
                expected=pairs,
            )
            result = evaluate_fold(predictions, targets)
            training_seconds = float(
                json.loads((directory / "metrics.json").read_text(encoding="utf-8"))[
                    "training_seconds"
                ]
            )
            scores.append(result)
            seconds.append(training_seconds)
            fold_rows.append(
                dict(
                    model=model,
                    pooling=pooling,
                    label=label,
                    fold=fold,
                    **{metric: getattr(result, metric) for metric in METRICS},
                    n_pairs=result.n_pairs,
                    training_seconds=training_seconds,
                )
            )
        summary_rows.append(
            dict(
                model=model,
                pooling=pooling,
                label=label,
                **{
                    key: value
                    for key, value in asdict(aggregate_folds(scores)).items()
                    if key != "n_folds"
                },
                n_folds=len(scores),
                n_pairs=sum(result.n_pairs for result in scores),
                training_seconds_mean=float(np.mean(seconds)),
                training_seconds_sd=float(np.std(seconds, ddof=1)),
            )
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(destination.parent / "fold_metrics.csv", fold_rows)
    _write_csv(destination.parent / "summary.csv", summary_rows)
    return benchmark_figure(destination.parent / "fold_metrics.csv", destination)
