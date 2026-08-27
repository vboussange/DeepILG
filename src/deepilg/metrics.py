"""Held-out metrics and five-fold aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FoldMetrics:
    """Metrics reported for one population-holdout fold."""

    pearson_r: float
    r2: float
    rmse: float
    n_pairs: int
    region_pearson_r: Mapping[str, float]


@dataclass(frozen=True)
class AggregateMetrics:
    """Mean and sample standard deviation across population-holdout folds."""

    pearson_r_mean: float
    pearson_r_sd: float
    r2_mean: float
    r2_sd: float
    rmse_mean: float
    rmse_sd: float
    n_folds: int


def pearson_r(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute signed Pearson correlation with explicit degeneracy checks."""
    prediction, target = _paired_finite(prediction, target)
    if prediction.size < 2:
        raise ValueError("Pearson correlation requires at least two finite pairs")
    prediction = prediction - np.mean(prediction)
    target = target - np.mean(target)
    denominator = float(np.linalg.norm(prediction) * np.linalg.norm(target))
    if denominator == 0.0:
        raise ValueError("Pearson correlation is undefined for a constant vector")
    return float(np.dot(prediction, target) / denominator)


def fisher_mean(correlations: Sequence[float]) -> float:
    """Combine region-level correlations equally in Fisher-z space."""
    values = np.asarray(correlations, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Fisher aggregation requires finite correlations")
    if np.any((values < -1.0) | (values > 1.0)):
        raise ValueError("Correlations must lie in [-1, 1]")
    clipped = np.clip(values, -1.0 + 1e-12, 1.0 - 1e-12)
    return float(np.tanh(np.mean(np.arctanh(clipped))))


def r2_score(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute predictive R-squared relative to the held-out target mean."""
    prediction, target = _paired_finite(prediction, target)
    if prediction.size == 0:
        raise ValueError("R-squared requires at least one finite pair")
    total = float(np.sum(np.square(target - np.mean(target))))
    if total == 0.0:
        raise ValueError("R-squared is undefined for a constant target")
    residual = float(np.sum(np.square(target - prediction)))
    return 1.0 - residual / total


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute root mean squared prediction error."""
    prediction, target = _paired_finite(prediction, target)
    if prediction.size == 0:
        raise ValueError("RMSE requires at least one finite pair")
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def evaluate_fold(
    predictions_by_region: Mapping[str, np.ndarray],
    targets_by_region: Mapping[str, np.ndarray],
) -> FoldMetrics:
    """Evaluate one fold using equal-region correlation and pooled errors."""
    if set(predictions_by_region) != set(targets_by_region) or not predictions_by_region:
        raise ValueError("Predictions and targets must contain identical non-empty regions")
    region_scores: dict[str, float] = {}
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for region in sorted(predictions_by_region):
        prediction, target = _paired_finite(
            predictions_by_region[region], targets_by_region[region]
        )
        region_scores[region] = pearson_r(prediction, target)
        predictions.append(prediction)
        targets.append(target)
    pooled_prediction = np.concatenate(predictions)
    pooled_target = np.concatenate(targets)
    return FoldMetrics(
        pearson_r=fisher_mean(list(region_scores.values())),
        r2=r2_score(pooled_prediction, pooled_target),
        rmse=rmse(pooled_prediction, pooled_target),
        n_pairs=int(pooled_prediction.size),
        region_pearson_r=region_scores,
    )


def aggregate_folds(folds: Sequence[FoldMetrics]) -> AggregateMetrics:
    """Return means and sample SDs across at least two population-holdout folds."""
    if len(folds) < 2:
        raise ValueError("Fold aggregation requires at least two folds")
    pearson = np.asarray([fold.pearson_r for fold in folds])
    r2 = np.asarray([fold.r2 for fold in folds])
    errors = np.asarray([fold.rmse for fold in folds])
    return AggregateMetrics(
        pearson_r_mean=float(np.mean(pearson)),
        pearson_r_sd=float(np.std(pearson, ddof=1)),
        r2_mean=float(np.mean(r2)),
        r2_sd=float(np.std(r2, ddof=1)),
        rmse_mean=float(np.mean(errors)),
        rmse_sd=float(np.std(errors, ddof=1)),
        n_folds=len(folds),
    )


def _paired_finite(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction.shape} != {target.shape}"
        )
    finite = np.isfinite(prediction) & np.isfinite(target)
    if not np.all(finite):
        raise ValueError("Predictions and targets must be finite")
    return prediction, target
