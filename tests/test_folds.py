"""Population-holdout fold generation through the public command."""

from pathlib import Path

import numpy as np

from deepilg.cli import main
from deepilg.splits import load_fold_assignments

REGION_SIZES = {"north": 8, "south": 11, "east": 13, "west": 10}


def _write_study(root: Path, *, split_seed: int) -> Path:
    fst_dir = root / "data" / "fst"
    fst_dir.mkdir(parents=True)
    generator = np.random.default_rng(0)
    for region, size in REGION_SIZES.items():
        ids = [f"{region[:2].upper()}_{index}" for index in range(size)]
        values = generator.uniform(0.0, 0.2, size=(size, size))
        values = (values + values.T) / 2
        lines = ["," + ",".join(ids)]
        for row, population in enumerate(ids):
            cells = [
                "NA" if row == column else f"{values[row, column]:.6f}" for column in range(size)
            ]
            lines.append(population + "," + ",".join(cells))
        (fst_dir / f"{region}_FST.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    config = root / "configs" / "study.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        f"""
[project]
name = "folds"
output_dir = "outputs"
[data]
fst_dir = "data/fst"
coordinates = "data/coordinates.csv"
folds = "data/folds.csv"
split_seed = {split_seed}
cache_dir = "cache"
processed_dir = "outputs/processed"
geolocation_status = "artificial"
scientific_use = false
[preprocessing]
[training]
[solver]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config


def test_folds_command_writes_balanced_valid_folds(tmp_path: Path) -> None:
    config = _write_study(tmp_path, split_seed=17)
    assert main(["folds", "--config", str(config)]) == 0

    folds = load_fold_assignments(tmp_path / "data" / "folds.csv")
    assert folds.fold_indices == (0, 1, 2, 3, 4)
    assert set(folds.regions) == set(REGION_SIZES)
    totals = np.zeros(len(folds.fold_indices), dtype=int)
    for region in folds.regions:
        sizes = [len(folds.ids(region, fold, "test")) for fold in folds.fold_indices]
        assert max(sizes) - min(sizes) <= 1
        totals += sizes
        for fold in folds.fold_indices:
            assert len(folds.ids(region, fold, "validation")) == 2
            assert not set(folds.ids(region, fold, "validation")) & set(
                folds.ids(region, fold, "test")
            )
    assert totals.sum() == sum(REGION_SIZES.values())
    assert totals.max() - totals.min() <= 1


def test_folds_command_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    outputs = []
    for name, seed in (("first", 17), ("again", 17), ("other", 18)):
        config = _write_study(tmp_path / name, split_seed=seed)
        main(["folds", "--config", str(config)])
        outputs.append((tmp_path / name / "data" / "folds.csv").read_bytes())
    assert outputs[0] == outputs[1]
    assert outputs[0] != outputs[2]
