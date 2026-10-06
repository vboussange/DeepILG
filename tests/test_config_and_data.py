"""Tests for strict, human-readable input contracts."""

from pathlib import Path

import numpy as np
import pytest

from deepilg.config import load_config
from deepilg.data import discover_fst_regions, linearize_fst, load_fst_csv, validate_fst_matrix


def _write_config(path: Path, *, artificial: bool = True) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        f"""
[project]
name = "test"
output_dir = "outputs/test"

[data]
fst_dir = "data/fst"
coordinates = "data/coordinates.csv"
folds = "data/folds.csv"
cache_dir = "cache"
processed_dir = "outputs/processed"
geolocation_status = "{"artificial" if artificial else "empirical"}"
scientific_use = {str(not artificial).lower()}

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


def test_config_paths_are_resolved_from_repository_root(tmp_path: Path) -> None:
    source = tmp_path / "configs" / "demo.toml"
    _write_config(source)
    config = load_config(source)
    assert config.root == tmp_path
    assert config.data.fst_dir == tmp_path / "data" / "fst"
    assert config.project.output_dir == tmp_path / "outputs" / "test"
    assert config.is_artificial
    with pytest.raises(ValueError, match="Publication export is disabled"):
        config.assert_publication_safe()


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    source = tmp_path / "configs" / "demo.toml"
    _write_config(source)
    text = source.read_text(encoding="utf-8").replace(
        'name = "test"', 'name = "test"\nunsupported_key = true'
    )
    source.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown keys"):
        load_config(source)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("rtol = nan", "rtol must be finite"),
        ("rtol = true", "rtol must be finite"),
        ("atol = inf", "atol must be finite"),
        ("rtol = 0\natol = 0", "At least one solver tolerance"),
        ("max_steps = 1.5", "max_steps must be a positive integer"),
        ("use_amg = 1", "use_amg must be a boolean"),
    ],
)
def test_config_rejects_invalid_solver_tolerances(
    tmp_path: Path, replacement: str, message: str
) -> None:
    source = tmp_path / "configs" / "demo.toml"
    _write_config(source)
    text = source.read_text(encoding="utf-8")
    if replacement.startswith("rtol = 0"):
        text = text.replace("rtol = 1e-6\natol = 1e-6", replacement)
    elif replacement.startswith("max_steps"):
        text = text.replace("max_steps = 100", replacement)
    elif replacement.startswith("use_amg"):
        text = text.replace("use_amg = true", replacement)
    else:
        key = replacement.split(" =", maxsplit=1)[0]
        text = text.replace(f"{key} = 1e-6", replacement)
    source.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(source)


def test_fst_loader_accepts_only_missing_diagonal(tmp_path: Path) -> None:
    source = tmp_path / "CWMA_FST.csv"
    source.write_text(
        ",A,B,C\nA,NA,0.1,-0.01\nB,0.1,NaN,0.2\nC,-0.01,0.2,\n",
        encoding="utf-8",
    )
    matrix = load_fst_csv(source, region="CWMA")
    np.testing.assert_array_equal(np.diag(matrix.values), 0.0)
    assert matrix.values[0, 2] == -0.01

    source.write_text(
        ",A,B,C\nA,NA,NA,-0.01\nB,NA,0,0.2\nC,-0.01,0.2,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="off-diagonal"):
        load_fst_csv(source, region="CWMA")


def test_discover_fst_regions_reads_an_arbitrary_region_set(tmp_path: Path) -> None:
    (tmp_path / "north_FST.csv").write_text("x", encoding="utf-8")
    (tmp_path / "south_fst_matrix.csv").write_text("x", encoding="utf-8")
    (tmp_path / "notes.csv").write_text("x", encoding="utf-8")
    regions = discover_fst_regions(tmp_path)
    assert list(regions) == ["north", "south"]
    assert regions["north"].name == "north_FST.csv"


def test_discover_fst_regions_rejects_ambiguous_names(tmp_path: Path) -> None:
    (tmp_path / "east_FST.csv").write_text("x", encoding="utf-8")
    (tmp_path / "east_fst_matrix.csv").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="map to region"):
        discover_fst_regions(tmp_path)


def test_fst_validation_and_linearization() -> None:
    with pytest.raises(ValueError, match="not symmetric"):
        validate_fst_matrix(np.asarray([[0.0, 0.1], [0.2, 0.0]]), ("A", "B"), region="X")
    values = linearize_fst(np.asarray([-0.1, 0.0, 0.2]))
    np.testing.assert_allclose(values, np.asarray([0.0, 0.0, 0.25]))
