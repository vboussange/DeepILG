"""Reports cover whichever treatments were evaluated, on identical pairs and targets."""

import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from deepilg.config import load_config
from deepilg.reporting import generate_benchmark_report

POPULATIONS = tuple(f"P{index}" for index in range(6))
TEST_POPULATIONS = ({"P0", "P1", "P2"}, {"P3", "P4", "P5"})


def _fst(region_index: int) -> np.ndarray:
    index = np.arange(len(POPULATIONS))
    values = 0.01 + 0.02 * np.abs(index[:, None] - index[None, :]) ** (1 + region_index)
    np.fill_diagonal(values, 0.0)
    return values


def _expected(fold: int) -> dict[tuple[str, str, str], float]:
    """Held-out pairs and float32 linearized targets, built without the package."""
    pairs = {}
    for region_index, region in enumerate(("A", "B")):
        fst = np.maximum(_fst(region_index).astype(np.float32), np.float32(0.0))
        targets = fst / (np.float32(1.0) - fst)
        for i in range(len(POPULATIONS)):
            for j in range(i + 1, len(POPULATIONS)):
                if {POPULATIONS[i], POPULATIONS[j]} & TEST_POPULATIONS[fold]:
                    pairs[(region, POPULATIONS[i], POPULATIONS[j])] = float(targets[i, j])
    return pairs


def _study(tmp_path: Path):
    """Two regions and two folds, with the folds and genetic data the report rebuilds."""
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "demo.toml")
    fst_dir = tmp_path / "fst"
    fst_dir.mkdir()
    folds = []
    for region_index, region in enumerate(("A", "B")):
        with (fst_dir / f"{region}_FST.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["population_id", *POPULATIONS])
            writer.writerows(
                [population, *row]
                for population, row in zip(POPULATIONS, _fst(region_index), strict=True)
            )
        for fold, tested in enumerate(TEST_POPULATIONS):
            validation = "P3" if fold == 0 else "P0"
            for population in POPULATIONS:
                partition = (
                    "test"
                    if population in tested
                    else "validation"
                    if population == validation
                    else "train"
                )
                folds.append(
                    dict(region=region, fold=fold, population_id=population, partition=partition)
                )
    with (tmp_path / "folds.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=folds[0])
        writer.writeheader()
        writer.writerows(folds)
    return replace(
        config,
        project=replace(config.project, output_dir=tmp_path / "outputs"),
        data=replace(config.data, fst_dir=fst_dir, folds=tmp_path / "folds.csv"),
    )


def _evaluate(config, model: str, pooling: str, predict) -> None:
    """Write evaluation outputs as `deepilg evaluate` lays them out."""
    for fold in range(len(TEST_POPULATIONS)):
        directory = config.project.output_dir / "evaluation" / model / pooling / f"fold_{fold:02d}"
        directory.mkdir(parents=True)
        rows = [
            dict(
                model=model,
                pooling=pooling,
                fold=fold,
                region=region,
                population_i=left,
                population_j=right,
                observed=target,
                prediction=predict(target),
            )
            for (region, left, right), target in _expected(fold).items()
        ]
        with (directory / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        (directory / "metrics.json").write_text(
            json.dumps(dict(training_seconds=10.0 * (fold + 1)))
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _fold_scores(predict, fold: int) -> tuple[float, float, float]:
    """Equal-region Fisher-z Pearson r and pooled R-squared and RMSE, computed directly."""
    pairs = _expected(fold)
    correlations, observed, predicted = [], [], []
    for region in ("A", "B"):
        targets = np.asarray([value for key, value in pairs.items() if key[0] == region])
        predictions = np.asarray([predict(value) for value in targets])
        correlations.append(np.corrcoef(predictions, targets)[0, 1])
        observed.extend(targets)
        predicted.extend(predictions)
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    r2 = 1 - np.sum((observed - predicted) ** 2) / np.sum((observed - observed.mean()) ** 2)
    rmse = np.sqrt(np.mean((observed - predicted) ** 2))
    return float(np.tanh(np.mean(np.arctanh(correlations)))), float(r2), float(rmse)


def _skewed(target: float) -> float:
    return 0.5 * target + 0.2 * target**2 + 0.003


def test_report_covers_present_treatments_in_a_fixed_order(tmp_path):
    config = _study(tmp_path)
    _evaluate(config, "resnet9", "none", _skewed)
    _evaluate(config, "loglinear", "complete", lambda target: target + 0.01)
    _evaluate(config, "reference", "complete", lambda target: 0.9 * target + 0.004)

    figure = generate_benchmark_report(config)
    assert figure == config.project.output_dir / "figures" / "benchmark.pdf"
    assert figure.read_bytes().startswith(b"%PDF")
    summary = _read_csv(figure.parent / "summary.csv")
    assert [row["label"] for row in summary] == [
        "IBD model",
        "Log-linear encoder",
        "ResNet9 encoder†",
    ]
    total_pairs = sum(len(_expected(fold)) for fold in range(2))
    assert {(row["n_folds"], int(row["n_pairs"])) for row in summary} == {("2", total_pairs)}
    loglinear = next(row for row in summary if row["model"] == "loglinear")
    assert float(loglinear["rmse_mean"]) == pytest.approx(0.01)
    assert float(loglinear["rmse_sd"]) == pytest.approx(0.0, abs=1e-12)
    assert float(loglinear["training_seconds_mean"]) == 15.0

    fold_metrics = _read_csv(figure.parent / "fold_metrics.csv")
    assert len(fold_metrics) == 3 * 2
    scores = [row for row in fold_metrics if row["pooling"] == "none"]
    for row in scores:
        expected = _fold_scores(_skewed, int(row["fold"]))
        observed = tuple(float(row[name]) for name in ("pearson_r", "r2", "rmse"))
        assert observed == pytest.approx(expected, rel=1e-6)
    resnet9 = next(row for row in summary if row["model"] == "resnet9")
    r2 = [float(row["r2"]) for row in scores]
    assert float(resnet9["r2_mean"]) == pytest.approx(np.mean(r2))
    assert float(resnet9["r2_sd"]) == pytest.approx(np.std(r2, ddof=1))


@pytest.mark.parametrize(
    "defect",
    ["duplicate", "target", "pair", "nonfinite", "region", "missing_fold", "extra_fold", "pooling"],
)
def test_report_rejects_incomplete_or_unmatched_predictions(tmp_path, defect):
    config = _study(tmp_path)
    _evaluate(config, "reference", "complete", lambda target: target)
    _evaluate(config, "loglinear", "none", lambda target: target + 0.01)
    root = config.project.output_dir / "evaluation" / "loglinear" / "none"
    source = root / "fold_01" / "test_predictions.csv"
    records = _read_csv(source)
    if defect == "duplicate":
        records.append(records[0])
    elif defect == "target":
        records[0]["observed"] = float(records[0]["observed"]) + 1e-7
    elif defect == "pair":
        records[0]["population_i"] = "unknown"
    elif defect == "nonfinite":
        records[0]["prediction"] = "nan"
    elif defect == "region":
        records = [row for row in records if row["region"] != "B"]
    elif defect == "missing_fold":
        shutil.rmtree(root / "fold_01")
    elif defect == "extra_fold":
        shutil.copytree(root / "fold_01", root / "fold_02")
    elif defect == "pooling":
        shutil.copytree(root, root.parent / "partial")
    if source.parent.is_dir():
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0])
            writer.writeheader()
            writer.writerows(records)
    with pytest.raises((ValueError, FileNotFoundError)):
        generate_benchmark_report(config)
    assert not (config.project.output_dir / "figures").exists()
