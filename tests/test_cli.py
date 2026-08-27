"""Offline command-line smoke tests."""

import json
from pathlib import Path

import pytest

from deepilg.cli import build_parser, main


def test_every_documented_subcommand_parses() -> None:
    parser = build_parser()
    config = Path("configs/demo.toml")
    commands = (
        ["validate", "--config", config],
        ["download", "--config", config],
        ["preprocess", "--config", config],
        ["train", "--config", config, "--model", "resnet9", "--fold", "0"],
        ["evaluate", "--config", config, "--model", "resistancega", "--fold", "0"],
        ["figures", "--config", config],
        ["resistancega-export", "--config", config, "--fold", "0"],
        [
            "resistancega",
            "--config",
            config,
            "--fold",
            "0",
            "--region",
            "CWMA",
            "--mode",
            "predict",
            "--split",
            "test",
        ],
    )
    for command in commands:
        assert parser.parse_args([str(value) for value in command]).command == command[0]


def test_resistancega_predict_requires_a_split() -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="--split is required"):
        main(
            [
                "resistancega",
                "--config",
                str(root / "configs" / "demo.toml"),
                "--fold",
                "0",
                "--region",
                "CWMA",
                "--mode",
                "predict",
            ]
        )


def test_demo_validation_is_offline_and_warns(capsys) -> None:
    root = Path(__file__).resolve().parents[1]
    assert main(["validate", "--config", str(root / "configs" / "demo.toml")]) == 0
    output = capsys.readouterr().out
    assert '"population_total": 68' in output
    assert "artificial coordinates" in output


def test_empirical_config_cannot_export_artificial_prediction_directory(
    tmp_path: Path,
) -> None:
    config = tmp_path / "configs" / "benchmark.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        """
[project]
name = "empirical"
output_dir = "outputs/benchmark"
[data]
fst_dir = "data/fst"
coordinates = "data/private/coordinates.csv"
folds = "data/folds.csv"
cache_dir = "cache"
processed_dir = "outputs/processed"
geolocation_status = "empirical"
scientific_use = true
[preprocessing]
resolution_m = 30
patch_size = 32
road_radius_m = 300
[training]
seed = 17
epochs = 10
patience = 2
resnet9_learning_rate = 0.0001
unet_learning_rate = 0.001
weight_decay = 0.0001
patch_batch_size = 8
[solver]
rtol = 1e-6
atol = 1e-6
max_steps = 100
use_amg = true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "outputs" / "demo" / "evaluation"
    predictions.mkdir(parents=True)
    (predictions / "workflow_manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "project_name": "demo",
                "geolocation_status": "artificial",
                "scientific_use": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="artificial or non-scientific"):
        main(
            [
                "figures",
                "--config",
                str(config),
                "--predictions-dir",
                str(predictions),
            ]
        )
