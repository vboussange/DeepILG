"""Tests for population holdouts, the objective, and reported metrics."""

import numpy as np
import pytest

from deepilg.loss import region_balanced_log1p_mse
from deepilg.metrics import aggregate_folds, evaluate_fold, fisher_mean
from deepilg.splits import FoldAssignments, pair_mask, validate_fold_assignments


def _folds() -> tuple[FoldAssignments, tuple[str, ...]]:
    populations = tuple(f"P{index}" for index in range(10))
    assignments = {}
    for fold in range(5):
        test = {populations[2 * fold], populations[2 * fold + 1]}
        validation = {populations[(2 * fold + 2) % len(populations)]}
        for population in populations:
            partition = (
                "test"
                if population in test
                else "validation"
                if population in validation
                else "train"
            )
            assignments[("REGION", fold, population)] = partition
    return FoldAssignments(assignments), populations


def test_population_masks_are_endpoint_based_and_disjoint() -> None:
    folds, populations = _folds()
    validate_fold_assignments(folds)
    train = pair_mask(populations, folds, region="REGION", fold=0, partition="train")
    validation = pair_mask(populations, folds, region="REGION", fold=0, partition="validation")
    test = pair_mask(populations, folds, region="REGION", fold=0, partition="test")
    assert train.sum() == 21  # seven choose two
    assert validation.sum() == 7
    assert test.sum() == 17
    assert not np.any(train & validation)
    assert not np.any(train & test)
    assert not np.any(validation & test)


def test_validate_fold_assignments_infers_the_fold_count() -> None:
    populations = tuple(f"P{index}" for index in range(6))
    assignments = {}
    for fold in range(3):  # three folds, not five
        test = {populations[2 * fold], populations[2 * fold + 1]}
        validation = {populations[(2 * fold + 2) % len(populations)]}
        for population in populations:
            partition = (
                "test"
                if population in test
                else "validation"
                if population in validation
                else "train"
            )
            assignments[("REGION", fold, population)] = partition
    folds = FoldAssignments(assignments)
    assert folds.fold_indices == (0, 1, 2)
    validate_fold_assignments(folds)


def test_validate_fold_assignments_rejects_non_contiguous_folds() -> None:
    folds, _ = _folds()
    shifted = {
        (region, fold + 1, population): value
        for (region, fold, population), value in folds.assignments.items()
    }
    with pytest.raises(ValueError, match="contiguous range"):
        validate_fold_assignments(FoldAssignments(shifted))


def test_region_balanced_loss_does_not_weight_regions_by_pair_count() -> None:
    predictions = {"small": np.asarray([0.0]), "large": np.zeros(10)}
    targets = {"small": np.asarray([1.0]), "large": np.zeros(10)}
    expected = 0.5 * np.log(2.0) ** 2
    assert float(region_balanced_log1p_mse(predictions, targets)) == pytest.approx(expected)


def test_fold_metrics_use_fisher_region_mean_and_pooled_errors() -> None:
    targets = {
        "A": np.asarray([0.0, 1.0, 2.0]),
        "B": np.asarray([0.0, 2.0, 1.0]),
    }
    predictions = {
        "A": np.asarray([0.0, 1.0, 2.0]),
        "B": np.asarray([0.0, 1.0, 2.0]),
    }
    result = evaluate_fold(predictions, targets)
    assert result.pearson_r == pytest.approx(fisher_mean([1.0, 0.5]))
    assert result.n_pairs == 6
    assert result.rmse == pytest.approx(np.sqrt(2.0 / 6.0))
    aggregate = aggregate_folds([result, result])
    assert aggregate.pearson_r_sd == 0.0
    assert aggregate.n_folds == 2
