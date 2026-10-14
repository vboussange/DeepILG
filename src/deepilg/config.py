"""Typed configuration loading for empirical and demonstration workflows."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class ProjectConfig:
    """Locations used by all commands."""

    name: str
    output_dir: Path


@dataclass(frozen=True)
class DataConfig:
    """Input locations and the geolocation safety declaration."""

    fst_dir: Path
    coordinates: Path
    folds: Path
    cache_dir: Path
    processed_dir: Path
    geolocation_status: str
    scientific_use: bool
    split_seed: int = 17


@dataclass(frozen=True)
class PreprocessingConfig:
    """Spatial preprocessing parameters."""

    resolution_m: float = 30.0
    patch_size: int = 32
    road_radius_m: float = 300.0


# Complete pooling fits one model across all regions; no pooling fits one per region.
POOLING = ("complete", "none")


@dataclass(frozen=True)
class TrainingConfig:
    """A single-fit neural optimisation configuration."""

    seed: int = 17
    epochs: int = 1_000
    patience: int = 100
    pooling: Literal["complete", "none"] = "complete"
    resnet9_learning_rate: float = 1e-4
    loglinear_learning_rate: float = 1e-2
    unet_learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patch_batch_size: int | None = None
    lr_patience: int = 20
    lr_factor: float = 0.5
    min_learning_rate: float = 1e-6

    def __post_init__(self) -> None:
        """Validate the pooling regime and the optional patch-chunk policy."""
        validate_patch_batch_size(self.patch_batch_size)
        if self.pooling not in POOLING:
            raise ValueError("[training].pooling must be 'complete' or 'none'")


def validate_patch_batch_size(value: int | None) -> None:
    """Accept full-region encoding (None) or a positive integer chunk size."""
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
    ):
        raise ValueError("patch_batch_size must be None or a positive integer")


@dataclass(frozen=True)
class SolverConfig:
    """Conjugate-gradient and AMG settings."""

    rtol: float = 1e-6
    atol: float = 1e-6
    max_steps: int = 1_000
    use_amg: bool = True

    def validate(self) -> None:
        """Reject settings that cannot define a convergent solver run."""
        for name, tolerance in (("rtol", self.rtol), ("atol", self.atol)):
            if (
                isinstance(tolerance, bool)
                or not isinstance(tolerance, Real)
                or not math.isfinite(tolerance)
                or tolerance < 0
            ):
                raise ValueError(f"[solver].{name} must be finite and non-negative")
        if self.rtol == 0 and self.atol == 0:
            raise ValueError("At least one solver tolerance must be positive")
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, Integral)
            or self.max_steps <= 0
        ):
            raise ValueError("[solver].max_steps must be a positive integer")
        if not isinstance(self.use_amg, bool):
            raise ValueError("[solver].use_amg must be a boolean")


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete configuration resolved relative to the repository root."""

    project: ProjectConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    solver: SolverConfig
    source: Path
    root: Path

    @property
    def is_artificial(self) -> bool:
        """Whether this configuration uses the non-scientific demo locations."""
        return self.data.geolocation_status == "artificial"

    def assert_publication_safe(self) -> None:
        """Reject publication export from artificial-coordinate products."""
        if self.is_artificial or not self.data.scientific_use:
            raise ValueError(
                "Publication export is disabled for artificial-coordinate data. "
                "Use the empirical configuration with authorised coordinates."
            )


def _require_table(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration requires a [{name}] table")
    return value


def _construct[T](cls: type[T], raw: Mapping[str, Any], section: str) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in [{section}]: {', '.join(unknown)}")
    try:
        return cls(**raw)
    except TypeError as error:
        raise ValueError(f"Invalid [{section}] configuration: {error}") from error


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _validate(config: ExperimentConfig) -> None:
    if not config.project.name.strip():
        raise ValueError("[project].name must not be empty")
    if config.data.geolocation_status not in {"artificial", "empirical"}:
        raise ValueError("[data].geolocation_status must be 'artificial' or 'empirical'")
    if config.data.geolocation_status == "artificial" and config.data.scientific_use:
        raise ValueError("Artificial coordinates must set scientific_use = false")
    if config.data.geolocation_status == "empirical" and not config.data.scientific_use:
        raise ValueError("Empirical coordinates must set scientific_use = true")
    if (
        isinstance(config.data.split_seed, bool)
        or not isinstance(config.data.split_seed, Integral)
        or config.data.split_seed < 0
    ):
        raise ValueError("[data].split_seed must be a non-negative integer")
    if config.preprocessing.resolution_m <= 0:
        raise ValueError("[preprocessing].resolution_m must be positive")
    if config.preprocessing.patch_size <= 0:
        raise ValueError("[preprocessing].patch_size must be positive")
    if config.preprocessing.road_radius_m <= 0:
        raise ValueError("[preprocessing].road_radius_m must be positive")
    if (
        config.training.epochs <= 0
        or config.training.patience <= 0
        or config.training.lr_patience <= 0
    ):
        raise ValueError("epochs, patience, and lr_patience must be positive")
    if (
        config.training.resnet9_learning_rate <= 0
        or config.training.loglinear_learning_rate <= 0
        or config.training.unet_learning_rate <= 0
        or config.training.weight_decay < 0
    ):
        raise ValueError("learning rates must be positive and weight_decay non-negative")
    if not 0 < config.training.lr_factor < 1:
        raise ValueError("lr_factor must lie strictly between zero and one")
    if config.training.min_learning_rate <= 0:
        raise ValueError("min_learning_rate must be positive")
    if config.training.min_learning_rate > min(
        config.training.resnet9_learning_rate,
        config.training.loglinear_learning_rate,
        config.training.unet_learning_rate,
    ):
        raise ValueError("min_learning_rate must not exceed any initial learning rate")
    config.solver.validate()


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a repository-relative TOML configuration.

    Relative paths are interpreted from the repository root, taken to be the
    parent of the directory containing the configuration file.  This makes the
    documented ``configs/*.toml`` files independent of the caller's working
    directory.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {source}")
    with source.open("rb") as stream:
        raw = tomllib.load(stream)

    allowed_sections = {"project", "data", "preprocessing", "training", "solver"}
    unknown_sections = sorted(set(raw) - allowed_sections)
    if unknown_sections:
        raise ValueError(f"Unknown configuration tables: {', '.join(unknown_sections)}")

    root = source.parent.parent.resolve()
    project_raw = dict(_require_table(raw, "project"))
    data_raw = dict(_require_table(raw, "data"))
    for name in ("output_dir",):
        if name in project_raw:
            project_raw[name] = _resolve(root, project_raw[name])
    for name in ("fst_dir", "coordinates", "folds", "cache_dir", "processed_dir"):
        if name in data_raw:
            data_raw[name] = _resolve(root, data_raw[name])

    config = ExperimentConfig(
        project=_construct(ProjectConfig, project_raw, "project"),
        data=_construct(DataConfig, data_raw, "data"),
        preprocessing=_construct(
            PreprocessingConfig,
            dict(_require_table(raw, "preprocessing")),
            "preprocessing",
        ),
        training=_construct(TrainingConfig, dict(_require_table(raw, "training")), "training"),
        solver=_construct(SolverConfig, dict(_require_table(raw, "solver")), "solver"),
        source=source,
        root=root,
    )
    _validate(config)
    return config
