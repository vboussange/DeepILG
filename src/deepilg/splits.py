"""Population-level cross-validation splits and pair masks."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PARTITIONS = ("train", "validation", "test")


@dataclass(frozen=True)
class FoldAssignments:
    """Partition labels indexed by ``(region, fold, population_id)``."""

    assignments: Mapping[tuple[str, int, str], str]

    @property
    def regions(self) -> tuple[str, ...]:
        """Return region identifiers in lexical order."""
        return tuple(sorted({key[0] for key in self.assignments}))

    @property
    def fold_indices(self) -> tuple[int, ...]:
        """Return the sorted fold indices present in the table."""
        return tuple(sorted({key[1] for key in self.assignments}))

    def populations(self, region: str) -> tuple[str, ...]:
        """Return the populations represented for one region."""
        return tuple(sorted({key[2] for key in self.assignments if key[0] == region}))

    def partition(self, region: str, fold: int, population_id: str) -> str:
        """Return one population's partition, with a helpful missing-key error."""
        key = (region, fold, population_id)
        try:
            return self.assignments[key]
        except KeyError as error:
            raise KeyError(f"No fold assignment for {key}") from error

    def ids(self, region: str, fold: int, partition: str) -> tuple[str, ...]:
        """Return sorted population identifiers for one partition."""
        if partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {PARTITIONS}")
        return tuple(
            sorted(
                population
                for (item_region, item_fold, population), value in self.assignments.items()
                if item_region == region and item_fold == fold and value == partition
            )
        )


def load_fold_assignments(path: str | Path) -> FoldAssignments:
    """Load the canonical flat fold-assignment CSV."""
    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"region", "fold", "population_id", "partition"}
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError(f"{source}: fold columns must be exactly {sorted(required)}")
        rows = list(reader)
    assignments: dict[tuple[str, int, str], str] = {}
    for row in rows:
        try:
            fold = int(row["fold"])
        except ValueError as error:
            raise ValueError(f"{source}: fold values must be integers") from error
        key = (row["region"].strip(), fold, row["population_id"].strip())
        partition = row["partition"].strip().lower()
        if key in assignments:
            raise ValueError(f"{source}: duplicate fold assignment {key}")
        assignments[key] = partition
    result = FoldAssignments(assignments)
    validate_fold_assignments(result)
    return result


def validate_fold_assignments(
    folds: FoldAssignments,
    population_ids_by_region: Mapping[str, Iterable[str]] | None = None,
) -> None:
    """Check exhaustive, non-overlapping k-fold population partitions.

    The number of folds is inferred from the table; any k is accepted as long
    as the fold indices form a contiguous range from zero and every region
    shares that range.
    """
    if not folds.assignments:
        raise ValueError("Fold assignment table is empty")
    fold_indices = folds.fold_indices
    if fold_indices != tuple(range(len(fold_indices))):
        raise ValueError(
            f"Fold indices must be a contiguous range starting at zero, got {list(fold_indices)}"
        )
    for key, partition in folds.assignments.items():
        region, _fold, population = key
        if not region or not population:
            raise ValueError("Fold region and population_id must be non-empty")
        if partition not in PARTITIONS:
            raise ValueError(f"Unknown partition {partition!r} for {key}")

    for region in folds.regions:
        populations = folds.populations(region)
        expected = set(populations)
        test_counts = {population: 0 for population in populations}
        for fold in fold_indices:
            observed = {
                population
                for population in populations
                if (region, fold, population) in folds.assignments
            }
            if observed != expected:
                raise ValueError(f"{region}/fold {fold}: assignments are not exhaustive")
            for population in folds.ids(region, fold, "test"):
                test_counts[population] += 1
            if len(folds.ids(region, fold, "train")) < 2:
                raise ValueError(
                    f"{region}/fold {fold}: at least two training populations required"
                )
            if not folds.ids(region, fold, "validation"):
                raise ValueError(f"{region}/fold {fold}: validation partition is empty")
            if not folds.ids(region, fold, "test"):
                raise ValueError(f"{region}/fold {fold}: test partition is empty")
        if any(count != 1 for count in test_counts.values()):
            raise ValueError(f"{region}: each population must occur in test exactly once")

    if population_ids_by_region is not None:
        if set(population_ids_by_region) != set(folds.regions):
            raise ValueError("Fold and genetic inputs contain different geographic regions")
        for region, ids in population_ids_by_region.items():
            if set(str(value) for value in ids) != set(folds.populations(region)):
                raise ValueError(f"{region}: fold and genetic population identifiers differ")


def generate_fold_assignments(
    population_ids_by_region: Mapping[str, Sequence[str]],
    *,
    seed: int,
    n_folds: int = 5,
    n_validation: int = 2,
) -> FoldAssignments:
    """Draw balanced population-holdout folds from a seed.

    Each region's populations are shuffled and cut into ``n_folds`` test sets
    whose sizes differ by at most one.  The regions' larger test sets go to the
    folds with the fewest held-out populations so far, so fold totals also
    differ by at most one.  Every fold then draws ``n_validation`` validation
    populations per region from its non-test populations.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    generator = np.random.default_rng(seed)
    larger_counts = np.zeros(n_folds, dtype=int)
    assignments: dict[tuple[str, int, str], str] = {}
    for region in sorted(population_ids_by_region):
        ids = sorted({str(value) for value in population_ids_by_region[region]})
        base, remainder = divmod(len(ids), n_folds)
        if len(ids) - (base + (remainder > 0)) - n_validation < 2:
            raise ValueError(f"{region}: too few populations for {n_folds} folds")
        larger = np.lexsort((generator.random(n_folds), larger_counts))[:remainder]
        larger_counts[larger] += 1
        sizes = np.full(n_folds, base)
        sizes[larger] += 1
        shuffled = [ids[index] for index in generator.permutation(len(ids))]
        for fold, held_out in enumerate(np.split(np.asarray(shuffled), np.cumsum(sizes)[:-1])):
            test = set(held_out.tolist())
            candidates = [population for population in ids if population not in test]
            validation = set(
                generator.choice(candidates, size=n_validation, replace=False).tolist()
            )
            for population in ids:
                assignments[(region, fold, population)] = (
                    "test"
                    if population in test
                    else "validation"
                    if population in validation
                    else "train"
                )
    folds = FoldAssignments(assignments)
    validate_fold_assignments(folds, population_ids_by_region)
    return folds


def write_fold_assignments(folds: FoldAssignments, path: str | Path) -> Path:
    """Write the canonical flat fold-assignment CSV."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("region", "fold", "population_id", "partition"))
        for (region, fold, population), partition in sorted(folds.assignments.items()):
            writer.writerow((region, fold, population, partition))
    return destination


def pair_mask(
    population_ids: Sequence[str],
    folds: FoldAssignments,
    *,
    region: str,
    fold: int,
    partition: str,
) -> np.ndarray:
    """Return a strict-upper pair mask for one population-level partition.

    Training uses pairs with two training endpoints.  Validation uses pairs
    touching a validation endpoint after test populations are excluded.  Test
    uses all pairs touching at least one test endpoint.
    """
    ids = tuple(str(value) for value in population_ids)
    labels = np.asarray([folds.partition(region, fold, value) for value in ids])
    row, column = np.triu_indices(len(ids), k=1)
    if partition == "train":
        return (labels[row] == "train") & (labels[column] == "train")
    if partition == "validation":
        touches_validation = (labels[row] == "validation") | (labels[column] == "validation")
        touches_test = (labels[row] == "test") | (labels[column] == "test")
        return touches_validation & ~touches_test
    if partition == "test":
        return (labels[row] == "test") | (labels[column] == "test")
    raise ValueError(f"partition must be one of {PARTITIONS}")


def upper_triangle(matrix: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Extract strict-upper entries, optionally applying a pair mask."""
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Expected a square matrix, got {values.shape}")
    row, column = np.triu_indices(values.shape[0], k=1)
    vector = values[row, column]
    if mask is None:
        return vector
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != vector.shape:
        raise ValueError(f"Pair mask has shape {mask.shape}; expected {vector.shape}")
    return vector[mask]
