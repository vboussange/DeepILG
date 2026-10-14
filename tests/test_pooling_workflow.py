"""Tiny offline studies run train, evaluate, and report under both pooling regimes."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from deepilg.cli import main
from deepilg.config import TrainingConfig
from deepilg.preprocessing import FEATURE_NAMES

REGIONS = ("A", "B")
POPULATIONS = tuple(f"P{index}" for index in range(6))
# Six populations on distinct patches of a 3x3 patch grid (patch size 4).
PIXELS = ((1, 1), (1, 6), (2, 10), (6, 2), (9, 5), (10, 10))


def _write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _fst(region_index: int) -> np.ndarray:
    index = np.arange(len(POPULATIONS))
    values = 0.02 + 0.03 * np.abs(index[:, None] - index[None, :]) + 0.01 * region_index
    values += 0.005 * ((index[:, None] + index[None, :]) % 3)
    np.fill_diagonal(values, 0.0)
    return values


def _test_populations(fold: int) -> set[str]:
    return {POPULATIONS[2 * fold], POPULATIONS[2 * fold + 1]}


def _tiny_study(root: Path, *, epochs: int = 1) -> Path:
    """Write a two-region, three-fold study with processed rasters and a TOML config."""
    config = root / "configs" / "tiny.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        f"""[project]
name = "tiny-study"
output_dir = "outputs"

[data]
fst_dir = "data/fst"
coordinates = "data/coordinates.csv"
folds = "data/folds.csv"
cache_dir = "cache"
processed_dir = "processed"
geolocation_status = "empirical"
scientific_use = true

[preprocessing]
resolution_m = 30
patch_size = 4
road_radius_m = 300

[training]
seed = 17
epochs = {epochs}
patience = 50

[solver]
rtol = 1e-10
atol = 1e-10
max_steps = 1000
use_amg = false
""",
        encoding="utf-8",
    )
    folds = []
    for fold in range(3):
        validation = POPULATIONS[(2 * fold + 2) % len(POPULATIONS)]
        for region in REGIONS:
            for population in POPULATIONS:
                partition = (
                    "test"
                    if population in _test_populations(fold)
                    else "validation"
                    if population == validation
                    else "train"
                )
                folds.append(
                    dict(region=region, fold=fold, population_id=population, partition=partition)
                )
    _write_csv(root / "data" / "folds.csv", folds[0].keys(), folds)
    (root / "data" / "fst").mkdir(parents=True)
    for region_index, region in enumerate(REGIONS):
        with (root / "data" / "fst" / f"{region}_FST.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["population_id", *POPULATIONS])
            writer.writerows(
                [population, *row]
                for population, row in zip(POPULATIONS, _fst(region_index), strict=True)
            )
        directory = root / "processed" / region
        directory.mkdir(parents=True)
        rows, columns = np.mgrid[0:12, 0:12]
        features = np.zeros((12, 12, len(FEATURE_NAMES)), dtype=np.float32)
        water = columns >= 6 + region_index
        features[..., 1] = ~water  # Tree cover.
        features[..., 8] = water  # Permanent water.
        features[..., 12] = (rows + 2 * columns) / 33
        features[..., 13] = np.where(water, (rows + 1) / 12, 0.0)
        np.save(directory / "features.npy", features)
        with rasterio.open(
            directory / "covariates.tif",
            "w",
            driver="GTiff",
            height=12,
            width=12,
            count=3,
            dtype="float32",
            crs="EPSG:32616",
            transform=from_origin(500_000, 4_000_000, 30, 30),
        ) as raster:
            raster.write(
                np.stack([np.where(water, 80, 10), features[..., 12], features[..., 13]]).astype(
                    np.float32
                )
            )
        _write_csv(
            directory / "populations.csv",
            (
                "population_id",
                "pixel_row",
                "pixel_column",
                "pixel_node",
                "patch_row",
                "patch_column",
                "patch_node",
            ),
            [
                dict(
                    population_id=population,
                    pixel_row=row,
                    pixel_column=column,
                    pixel_node=row * 12 + column,
                    patch_row=row // 4,
                    patch_column=column // 4,
                    patch_node=row // 4 * 3 + column // 4,
                )
                for population, (row, column) in zip(POPULATIONS, PIXELS, strict=True)
            ],
        )
        (directory / "metadata.json").write_text(
            json.dumps(
                dict(
                    feature_names=FEATURE_NAMES,
                    patch_size=4,
                    resolution_m=30,
                    shape=features.shape,
                    geolocation_status="empirical",
                    scientific_use=True,
                )
            ),
            encoding="utf-8",
        )
    return config


def _expected_test_pairs(fold: int) -> dict[tuple[str, str, str], float]:
    """Independently list the held-out pairs and their linearized FST targets."""
    expected = {}
    for region_index, region in enumerate(REGIONS):
        fst = np.maximum(_fst(region_index).astype(np.float32), np.float32(0.0))
        targets = fst / (np.float32(1.0) - fst)
        for i in range(len(POPULATIONS)):
            for j in range(i + 1, len(POPULATIONS)):
                if {POPULATIONS[i], POPULATIONS[j]} & _test_populations(fold):
                    expected[(region, POPULATIONS[i], POPULATIONS[j])] = float(targets[i, j])
    return expected


def _write_resistancega_fits(root: Path) -> None:
    """Write per-region ResistanceGA2 outputs in the layout of its R entry point."""
    for fold in range(3):
        for region in REGIONS:
            directory = root / "outputs" / "resistancega" / "fits" / f"fold_{fold:02d}" / region
            pairs = [
                dict(
                    population_i=i,
                    population_j=j,
                    # R's write.csv keeps 15 significant digits.
                    observed=f"{target:.15g}",
                    predicted=0.8 * target + 0.01,
                    resistance_distance=10 * target,
                )
                for (pair_region, i, j), target in _expected_test_pairs(fold).items()
                if pair_region == region
            ]
            _write_csv(directory / "test_predictions.csv", pairs[0].keys(), pairs)
            (directory / "fit_timing.json").write_text(
                json.dumps(dict(workers=4, wall_clock_seconds=60.0 + fold)), encoding="utf-8"
            )


def _pairs(path: Path) -> set[tuple[str, str, str]]:
    return {(row["region"], row["population_i"], row["population_j"]) for row in _read_csv(path)}


def test_no_pooling_fits_ignore_other_regions_and_are_reassembled_per_fold(tmp_path):
    config = str(_tiny_study(tmp_path, epochs=3))
    fits = tmp_path / "outputs" / "loglinear"
    evaluation = tmp_path / "outputs" / "evaluation" / "loglinear"

    def train(*options: str) -> None:
        main(["train", "--config", config, "--model", "loglinear", "--fold", "0", *options])

    def evaluate(*options: str) -> None:
        main(["evaluate", "--config", config, "--model", "loglinear", "--fold", "0", *options])

    train("--pooling", "none")
    train()
    assert sorted(path.name for path in (fits / "none" / "fold_00").iterdir()) == ["A", "B"]
    region_a = (fits / "none" / "fold_00" / "A" / "best.eqx").read_bytes()
    pooled = (fits / "complete" / "fold_00" / "best.eqx").read_bytes()

    evaluate("--pooling", "none")
    evaluate()
    reassembled = evaluation / "none" / "fold_00" / "test_predictions.csv"
    assert _pairs(reassembled) == _pairs(
        evaluation / "complete" / "fold_00" / "test_predictions.csv"
    )
    assert _pairs(reassembled) == set(_expected_test_pairs(0))
    assert {row["pooling"] for row in _read_csv(reassembled)} == {"none"}

    # Perturb every input of region B, then refit region A alone.
    features = tmp_path / "processed" / "B" / "features.npy"
    np.save(features, np.load(features) * np.float32(0.5))
    fst = tmp_path / "data" / "fst" / "B_FST.csv"
    fst.write_text(fst.read_text().replace("0.06", "0.16"))
    shutil.rmtree(tmp_path / "outputs")
    train("--pooling", "none", "--region", "A")
    assert (fits / "none" / "fold_00" / "A" / "best.eqx").read_bytes() == region_a
    assert not (fits / "none" / "fold_00" / "B").exists()
    # The same perturbation reaches a complete-pooling fit.
    train()
    assert (fits / "complete" / "fold_00" / "best.eqx").read_bytes() != pooled

    with pytest.raises(FileNotFoundError, match="B"):
        evaluate("--pooling", "none")
    assert not (evaluation / "none").exists()


def test_ibd_calibration_reaches_its_training_objective_minimum(tmp_path):
    """Each fold's IBD fit is no worse than a dense grid over its training objective."""
    config = str(_tiny_study(tmp_path))
    slopes = np.linspace(0.0, 2e-3, 801)[:, None, None]
    intercepts = np.linspace(0.0, 0.2, 801)[None, :, None]
    for fold in range(3):
        assert main(["train", "--config", config, "--model", "reference", "--fold", str(fold)]) == 0
        fit = tmp_path / "outputs" / "reference" / "complete" / f"fold_{fold:02d}"
        calibration = json.loads((fit / "calibration.json").read_text(encoding="utf-8"))
        validation = POPULATIONS[(2 * fold + 2) % len(POPULATIONS)]
        training = [
            index
            for index, population in enumerate(POPULATIONS)
            if population not in _test_populations(fold) and population != validation
        ]
        rows, columns = np.asarray(
            [(i, j) for k, i in enumerate(training) for j in training[k + 1 :]]
        ).T
        distances = 30.0 * np.hypot(*(np.asarray(PIXELS)[rows] - np.asarray(PIXELS)[columns]).T)

        def objective(slope, intercept, distances=distances, rows=rows, columns=columns):
            """Equal-region mean of the log1p squared error over training pairs."""
            losses = []
            for region_index in range(len(REGIONS)):
                fst = np.maximum(_fst(region_index).astype(np.float32), np.float32(0.0))
                targets = (fst / (np.float32(1.0) - fst))[rows, columns].astype(np.float64)
                predictions = slope * distances + intercept
                losses.append(np.mean(np.square(np.log1p(predictions) - np.log1p(targets)), -1))
            return np.mean(losses, axis=0)

        fitted = objective(calibration["slope"], calibration["intercept"])
        assert fitted <= objective(slopes, intercepts).min()


def test_pooling_options_reject_unsupported_combinations(tmp_path):
    config = str(_tiny_study(tmp_path))
    with pytest.raises(ValueError, match="complete pooling"):
        main(
            [
                "train",
                "--config",
                config,
                "--model",
                "reference",
                "--fold",
                "0",
                "--pooling",
                "none",
            ]
        )
    with pytest.raises(ValueError, match="no pooling"):
        main(["train", "--config", config, "--model", "unet", "--fold", "0", "--region", "A"])
    with pytest.raises(ValueError, match="Unknown region"):
        main(
            [
                "train",
                "--config",
                config,
                "--model",
                "unet",
                "--fold",
                "0",
                "--pooling",
                "none",
                "--region",
                "C",
            ]
        )
    with pytest.raises(ValueError, match="pooling"):
        TrainingConfig(pooling="partial")


def test_every_treatment_is_trained_evaluated_and_reported_on_identical_pairs(tmp_path):
    config = str(_tiny_study(tmp_path))
    treatments = (
        ("reference", "complete"),
        ("loglinear", "complete"),
        ("loglinear", "none"),
        ("resnet9", "complete"),
        ("resnet9", "none"),
    )
    for fold in ("0", "1", "2"):
        for model, pooling in treatments:
            options = ["--config", config, "--model", model, "--fold", fold, "--pooling", pooling]
            assert main(["train", *options]) == 0
            assert main(["evaluate", *options]) == 0
    _write_resistancega_fits(tmp_path)
    for fold in ("0", "1", "2"):
        assert (
            main(["evaluate", "--config", config, "--model", "resistancega", "--fold", fold]) == 0
        )

    states = list((tmp_path / "outputs").glob("*/*/fold_*/**/training_state.json"))
    assert len(states) == 3 * (1 + 1 + 2 + 1 + 2)
    for state in states:
        assert json.loads(state.read_text())["wall_clock_seconds"] > 0

    evaluation = tmp_path / "outputs" / "evaluation"
    for fold in range(3):
        expected = _expected_test_pairs(fold)
        for model, pooling in (*treatments, ("resistancega", "none")):
            rows = _read_csv(
                evaluation / model / pooling / f"fold_{fold:02d}" / "test_predictions.csv"
            )
            observed = {
                (row["region"], row["population_i"], row["population_j"]): float(row["observed"])
                for row in rows
            }
            assert len(observed) == len(rows)
            assert set(observed) == set(expected)
            for pair, target in observed.items():
                assert target == pytest.approx(expected[pair], rel=1e-6)

    assert main(["figures", "--config", config]) == 0
    figures = tmp_path / "outputs" / "figures"
    assert (figures / "benchmark.pdf").read_bytes().startswith(b"%PDF")
    summary = _read_csv(figures / "summary.csv")
    assert [row["label"] for row in summary] == [
        "IBD model",
        "Log-linear",
        "ResNet9",
        "ResistanceGA (no pooling)",
        "Log-linear (no pooling)",
        "ResNet9 (no pooling)",
    ]
    assert {row["n_folds"] for row in summary} == {"3"}
    assert {int(row["n_pairs"]) for row in summary} == {
        sum(len(_expected_test_pairs(fold)) for fold in range(3))
    }
    fold_metrics = _read_csv(figures / "fold_metrics.csv")
    assert len(fold_metrics) == 6 * 3
    for row in fold_metrics:
        assert np.isfinite([float(row[name]) for name in ("pearson_r", "r2", "rmse")]).all()
        assert float(row["training_seconds"]) > 0
    ga = [row for row in fold_metrics if row["model"] == "resistancega"]
    assert [float(row["training_seconds"]) for row in ga] == [120.0, 122.0, 124.0]

    assert main(["figures", "--config", config, "--kind", "resistance"]) == 0
    assert (figures / "resistance_surfaces.pdf").read_bytes().startswith(b"%PDF")
    landcover = _read_csv(figures / "landcover_associations.csv")
    assert {row["landcover_class"] for row in landcover} == {"Tree cover", "Permanent water"}
    assert {row["region_count"] for row in landcover} == {"2"}
    continuous = _read_csv(figures / "continuous_feature_associations.csv")
    assert [row["feature"] for row in continuous] == ["elevation", "road_density"]
