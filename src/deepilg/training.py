"""Single-fit training and checkpointing for the benchmark models."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import minimize_scalar

from .config import ExperimentConfig
from .data import FSTMatrix, discover_fst_regions, linearize_fst, load_fst_directory
from .loss import log1p_mse
from .models import (
    ConductanceModel,
    EmbeddingDistanceModel,
    GeographicDistanceModel,
    LogLinearConductance,
    ResNet9Conductance,
    UNetEmbeddingDistance,
)
from .preprocessing import (
    FEATURE_NAMES,
    ProcessedRegion,
    validate_processed_region,
)
from .provenance import write_workflow_manifest
from .solver import (
    AMGContext,
    build_amg_context,
    build_cg_context,
)
from .splits import (
    FoldAssignments,
    load_fold_assignments,
    pair_mask,
    upper_triangle,
)

ModelName = Literal["reference", "resnet9", "unet", "loglinear"]
LearnedModel = ConductanceModel | EmbeddingDistanceModel
Partition = Literal["train", "validation", "test"]
_MODEL_CLASSES: Mapping[str, type] = {
    "reference": GeographicDistanceModel,
    "resnet9": ResNet9Conductance,
    "unet": UNetEmbeddingDistance,
    "loglinear": LogLinearConductance,
}


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RegionBatch:
    """Model inputs and population-level split masks for one region."""

    region: str = field(metadata={"static": True})
    population_ids: tuple[str, ...] = field(metadata={"static": True})
    features: jax.Array
    pixel_nodes: jax.Array
    targets: jax.Array
    masks: Mapping[str, jax.Array]
    resolution_m: float = field(metadata={"static": True})
    patch_batch_size: int | None = field(default=None, metadata={"static": True})


@dataclass(frozen=True)
class TrainingResult:
    """Summary of one population-holdout training run."""

    model: ModelName
    fold: int
    best_epoch: int
    best_validation_loss: float
    epochs_completed: int
    checkpoint: Path
    region: str | None = None


def load_fold_batches(
    config: ExperimentConfig, fold: int, *, region: str | None = None
) -> tuple[RegionBatch, ...]:
    """Load every processed region for a fold, or only ``region``."""
    matrices = load_fst_directory(config.data.fst_dir)
    if region is not None:
        if region not in matrices:
            raise ValueError(f"Unknown region {region!r}; expected one of {sorted(matrices)}")
        matrices = {region: matrices[region]}
    assignments = load_fold_assignments(config.data.folds)
    if fold not in assignments.fold_indices:
        raise ValueError(f"fold must be one of {list(assignments.fold_indices)}")
    batches = []
    for region, matrix in matrices.items():
        batches.append(
            _load_region_batch(
                config.data.processed_dir / region,
                matrix,
                assignments,
                fold=fold,
                resolution_m=config.preprocessing.resolution_m,
                patch_size=config.preprocessing.patch_size,
                patch_batch_size=config.training.patch_batch_size,
                geolocation_status=config.data.geolocation_status,
                scientific_use=config.data.scientific_use,
            )
        )
    return tuple(batches)


def _load_region_batch(
    directory: Path,
    matrix: FSTMatrix,
    assignments: FoldAssignments,
    *,
    fold: int,
    resolution_m: float,
    patch_size: int,
    patch_batch_size: int | None,
    geolocation_status: str,
    scientific_use: bool,
) -> RegionBatch:
    features_path = directory / "features.npy"
    populations_path = directory / "populations.csv"
    if not features_path.is_file() or not populations_path.is_file():
        raise FileNotFoundError(
            f"{matrix.region}: run preprocessing before training; missing {directory}"
        )
    validate_processed_region(
        ProcessedRegion(
            matrix.region,
            features_path,
            directory / "covariates.tif",
            populations_path,
            directory / "metadata.json",
        ),
        patch_size=patch_size,
        resolution_m=resolution_m,
        geolocation_status=geolocation_status,
        scientific_use=scientific_use,
        expected_population_ids=set(matrix.population_ids),
    )
    features = np.load(features_path).astype(np.float32, copy=False)
    if (
        features.ndim != 3
        or features.shape[-1] != len(FEATURE_NAMES)
        or not np.all(np.isfinite(features))
    ):
        raise ValueError(f"{matrix.region}: expected finite HxWx14 features")
    with populations_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required_columns = {
        "population_id",
        "pixel_row",
        "pixel_column",
        "pixel_node",
        "patch_row",
        "patch_column",
        "patch_node",
    }
    if not rows or set(rows[0]) != required_columns:
        raise ValueError(
            f"{matrix.region}: populations.csv must contain explicit pixel and patch nodes"
        )
    nodes_by_id = {row["population_id"]: int(row["pixel_node"]) for row in rows}
    if set(nodes_by_id) != set(matrix.population_ids):
        raise ValueError(f"{matrix.region}: processed and FST population IDs differ")
    pixel_nodes = np.asarray([nodes_by_id[item] for item in matrix.population_ids], dtype=np.int32)
    if np.any(pixel_nodes < 0) or np.any(pixel_nodes >= features.shape[0] * features.shape[1]):
        raise ValueError(f"{matrix.region}: population node lies outside processed raster")
    patch_columns = features.shape[1] // patch_size
    for row in rows:
        pixel_node = int(row["pixel_node"])
        pixel_row, pixel_column = divmod(pixel_node, features.shape[1])
        expected_patch_node = (pixel_row // patch_size) * patch_columns + pixel_column // patch_size
        if int(row["patch_node"]) != expected_patch_node:
            raise ValueError(f"{matrix.region}: inconsistent stored patch_node")
    masks = {
        partition: jnp.asarray(
            pair_mask(
                matrix.population_ids,
                assignments,
                region=matrix.region,
                fold=fold,
                partition=partition,
            )
        )
        for partition in ("train", "validation", "test")
    }
    return RegionBatch(
        region=matrix.region,
        population_ids=matrix.population_ids,
        features=jnp.asarray(features),
        pixel_nodes=jnp.asarray(pixel_nodes),
        targets=jnp.asarray(linearize_fst(matrix.values)),
        masks=masks,
        resolution_m=resolution_m,
        patch_batch_size=patch_batch_size,
    )


def initialise_model(
    name: ModelName,
    *,
    patch_size: int,
    seed: int,
) -> LearnedModel:
    """Create one neural model with the manuscript architecture."""
    key = jax.random.PRNGKey(seed)
    if name == "resnet9":
        return ResNet9Conductance(patch_size=patch_size, key=key)
    if name == "unet":
        return UNetEmbeddingDistance(patch_size=patch_size, key=key)
    if name == "loglinear":
        return LogLinearConductance(patch_size=patch_size)
    raise ValueError("initialise_model supports only 'resnet9', 'unet', and 'loglinear'")


def predicts_through_resistance_layer(model_name: str) -> bool:
    """Return whether a model predicts through the differentiable effective-resistance layer.

    Those are the conductance models. Every other model, including ResistanceGA2
    and names this package does not define, predicts some other way.
    """
    return issubclass(_MODEL_CLASSES.get(model_name, object), ConductanceModel)


def build_solver_contexts(
    model: ConductanceModel | EmbeddingDistanceModel,
    batches: Sequence[RegionBatch],
    config: ExperimentConfig,
) -> dict[str, AMGContext]:
    """Build one shape-specific resistance-solver context per region."""
    if not isinstance(model, ConductanceModel):
        return {}
    contexts = {}
    for batch in batches:
        shape = (
            batch.features.shape[0] // model.patch_size,
            batch.features.shape[1] // model.patch_size,
        )
        builder = build_amg_context if config.solver.use_amg else build_cg_context
        contexts[batch.region] = builder(
            shape,
            rtol=config.solver.rtol,
            atol=config.solver.atol,
            max_steps=config.solver.max_steps,
        )
    return contexts


def predict_matrix(
    model: ConductanceModel | EmbeddingDistanceModel,
    batch: RegionBatch,
    *,
    context: AMGContext | None = None,
    inference: bool = True,
    key: jax.Array | None = None,
) -> jax.Array:
    """Predict one population-distance matrix, dispatching on the encoder family."""
    if isinstance(model, ConductanceModel):
        return model.predict_distances(
            batch.features,
            batch.pixel_nodes,
            context=context,
            patch_batch_size=batch.patch_batch_size,
        )
    return model.predict_distances(
        batch.features,
        batch.pixel_nodes,
        inference=inference,
        key=key,
        patch_batch_size=batch.patch_batch_size,
    )


def initialise_loglinear_scale(
    model: LogLinearConductance,
    batches: Sequence[RegionBatch],
    *,
    contexts: Mapping[str, AMGContext],
) -> tuple[LogLinearConductance, dict[str, object]]:
    """Calibrate uniform conductance using only equally weighted training regions.

    Uniform effective resistance scales as 1/conductance. Solve each unit grid
    once, then optimize only the scalar scale on the host, without rerunning
    landscape solves or accessing validation/test responses.
    """
    if not batches:
        raise ValueError("At least one geographic region is required")
    uniform = eqx.tree_at(
        lambda encoder: (encoder.intercept, encoder.coefficients),
        model,
        (jnp.asarray(np.log(0.99), dtype=jnp.float32), jnp.zeros_like(model.coefficients)),
    )
    training = []
    for batch in batches:
        mask = np.asarray(batch.masks["train"])
        if not mask.any():
            raise ValueError(f"{batch.region}: scalar initialization requires training pairs")
        distances = upper_triangle(
            np.asarray(predict_matrix(uniform, batch, context=contexts.get(batch.region))), mask
        ).astype(np.float64)
        targets = upper_triangle(np.asarray(batch.targets), mask).astype(np.float64)
        if not np.all(np.isfinite(distances)) or not np.all(np.isfinite(targets)):
            raise FloatingPointError(f"{batch.region}: non-finite scalar initialization inputs")
        training.append((distances, np.log1p(targets)))

    def scalar_loss(log_scale: float) -> float:
        scale = np.exp(log_scale)
        return float(
            np.mean(
                [
                    np.mean(np.square(np.log1p(distances / scale) - log_targets))
                    for distances, log_targets in training
                ]
            )
        )

    bounds = (float(np.log(1e-2)), float(np.log(1e3)))
    result = minimize_scalar(scalar_loss, bounds=bounds, method="bounded", options={"xatol": 1e-10})
    if not result.success:
        raise RuntimeError(f"Uniform conductance calibration failed: {result.message}")
    log_scale = min((bounds[0], result.x, bounds[1]), key=scalar_loss)
    scale = float(np.exp(log_scale))
    intercept = np.log(max(scale - 1e-2, np.finfo(np.float32).tiny))
    initialized = eqx.tree_at(
        lambda encoder: encoder.intercept, uniform, jnp.asarray(intercept, dtype=jnp.float32)
    )
    return initialized, {
        "source": "training_pairs_only",
        "objective": "equal_region_log1p_mse",
        "conductance": scale,
        "intercept": float(initialized.intercept),
        "training_loss": scalar_loss(log_scale),
    }


def conductance_diagnostics(
    model: ConductanceModel, batches: Sequence[RegionBatch]
) -> dict[str, dict[str, float]]:
    """Summarize finite conductance bounds and saturation for training records."""
    diagnostics = {}
    for batch in batches:
        surface = np.asarray(
            model.conductance(batch.features, patch_batch_size=batch.patch_batch_size)
        )
        if not np.all(np.isfinite(surface)):
            raise FloatingPointError(f"{batch.region}: non-finite conductance")
        diagnostics[batch.region] = {
            "minimum": float(surface.min()),
            "maximum": float(surface.max()),
            "lower_saturation_fraction": float(np.mean(surface <= np.float32(1e-2))),
            "upper_saturation_fraction": float(np.mean(surface >= np.float32(1e3))),
        }
    return diagnostics


def _region_objective(
    model: LearnedModel,
    batch: RegionBatch,
    selected_pairs: jax.Array,
    *,
    context: AMGContext | None,
    inference: bool,
    key: jax.Array,
) -> jax.Array:
    """Compute one region's loss using fixed-length integer pair indices."""
    matrix = predict_matrix(model, batch, context=context, inference=inference, key=key)
    row, column = jnp.triu_indices(matrix.shape[0], k=1)
    row, column = row[selected_pairs], column[selected_pairs]
    return log1p_mse(matrix[row, column], batch.targets[row, column])


_region_loss = eqx.filter_jit(_region_objective)
_region_value_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(_region_objective))


def _selected_pairs(batch: RegionBatch, partition: Partition) -> jax.Array:
    """Prepare indices on the host without changing the public Boolean masks."""
    return jnp.asarray(np.flatnonzero(np.asarray(batch.masks[partition])), dtype=jnp.int32)


def objective(
    model: LearnedModel,
    batches: Sequence[RegionBatch],
    *,
    partition: Partition,
    contexts: Mapping[str, AMGContext],
    inference: bool,
    key: jax.Array,
) -> jax.Array:
    """Compute the equally region-weighted objective for a data partition."""
    if not batches:
        raise ValueError("At least one geographic region is required")
    keys = jax.random.split(key, len(batches))
    losses = [
        _region_loss(
            model,
            batch,
            _selected_pairs(batch, partition),
            context=contexts.get(batch.region),
            inference=inference,
            key=region_key,
        )
        for batch, region_key in zip(batches, keys, strict=True)
    ]
    return jnp.mean(jnp.stack(losses))


def objective_value_and_grad(
    model: LearnedModel,
    batches: Sequence[RegionBatch],
    *,
    partition: Partition,
    contexts: Mapping[str, AMGContext],
    inference: bool,
    key: jax.Array,
) -> tuple[jax.Array, LearnedModel]:
    """Average per-region gradients without retaining several regions' activations.

    Compile each region's loss-and-gradient calculation, but keep this loop
    outside JIT so each backward pass finishes before starting the next region.
    The caller applies the averaged gradient in one update.
    """
    if not batches:
        raise ValueError("At least one geographic region is required")
    keys = jax.random.split(key, len(batches))
    losses = []
    gradients = None
    for batch, region_key in zip(batches, keys, strict=True):
        loss, region_gradients = jax.block_until_ready(
            _region_value_and_grad(
                model,
                batch,
                _selected_pairs(batch, partition),
                context=contexts.get(batch.region),
                inference=inference,
                key=region_key,
            )
        )
        losses.append(loss)
        gradients = (
            region_gradients
            if gradients is None
            else jax.tree.map(jnp.add, gradients, region_gradients)
        )
    gradients = jax.tree.map(lambda gradient: gradient / len(batches), gradients)
    return jnp.mean(jnp.stack(losses)), gradients


def fit_directory(
    config: ExperimentConfig,
    *,
    model_name: str,
    fold: int,
    region: str | None = None,
    pooling: str | None = None,
) -> Path:
    """Return ``<output>/<model>/<pooling>/fold_XX``, plus ``/<region>`` under no pooling.

    ``pooling`` defaults to the configured regime.
    """
    pooling = pooling or config.training.pooling
    if pooling == "complete" and region is not None:
        raise ValueError("A single region can be fitted only under no pooling")
    if pooling == "none" and region is None:
        raise ValueError("A no-pooling fit belongs to one region")
    directory = config.project.output_dir / model_name / pooling / f"fold_{fold:02d}"
    return directory / region if region is not None else directory


def train_neural_model(
    config: ExperimentConfig,
    *,
    model_name: Literal["resnet9", "unet", "loglinear"],
    fold: int,
    resume: bool = False,
    region: str | None = None,
) -> tuple[TrainingResult, ...]:
    """Fit one learned model for one fold under the configured pooling regime.

    Complete pooling fits one model across all regions. No pooling fits an
    independent model per region, one after the other, or only ``region``.
    """
    regions: tuple[str | None, ...] = (region,)
    if config.training.pooling == "none" and region is None:
        regions = tuple(discover_fst_regions(config.data.fst_dir))
    return tuple(
        _train_fit(config, model_name=model_name, fold=fold, resume=resume, region=selected)
        for selected in regions
    )


def _train_fit(
    config: ExperimentConfig,
    *,
    model_name: Literal["resnet9", "unet", "loglinear"],
    fold: int,
    resume: bool,
    region: str | None,
) -> TrainingResult:
    """Fit one model with validation early stopping, on all regions or on one.

    A validation plateau reduces the learning rate and restarts AdamW's
    accumulated moments. The training state records the elapsed wall-clock
    seconds, accumulated across resumes.
    """
    started = time.perf_counter()
    output = fit_directory(config, model_name=model_name, fold=fold, region=region)
    batches = load_fold_batches(config, fold, region=region)
    write_workflow_manifest(config, config.project.output_dir)
    model = initialise_model(
        model_name,
        patch_size=config.preprocessing.patch_size,
        seed=config.training.seed,
    )
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest.eqx"
    best = output / "best.eqx"
    state_path = output / "training_state.json"
    start_epoch = 0
    best_epoch = -1
    best_loss = float("inf")
    stale_epochs = 0
    lr_stale_epochs = 0
    initial_learning_rate = {
        "resnet9": config.training.resnet9_learning_rate,
        "unet": config.training.unet_learning_rate,
        "loglinear": config.training.loglinear_learning_rate,
    }[model_name]
    current_learning_rate = initial_learning_rate
    # Unit-rate AdamW lets the plateau scheduler scale complete updates.  Its
    # transform state is deliberately restarted after every actual reduction.
    optimiser = optax.adamw(1.0, weight_decay=config.training.weight_decay)
    optimiser_state = optimiser.init(eqx.filter(model, eqx.is_inexact_array))
    key = jax.random.PRNGKey(config.training.seed)
    initialization = None
    previous_seconds = 0.0
    if resume:
        if not latest.is_file() or not state_path.is_file():
            raise FileNotFoundError(f"No resumable checkpoint found in {output}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            state.get("model") != model_name
            or int(state.get("fold", -1)) != fold
            or int(state.get("seed", -1)) != config.training.seed
            or state.get("region") != region
        ):
            raise ValueError("Resume checkpoint does not match model, fold, seed, or region")
        model, optimiser_state, key = eqx.tree_deserialise_leaves(
            latest, (model, optimiser_state, key)
        )
        start_epoch = int(state["epoch"]) + 1
        best_epoch = int(state["best_epoch"])
        best_loss = float(state["best_validation_loss"])
        stale_epochs = int(state["stale_epochs"])
        lr_stale_epochs = int(state["lr_stale_epochs"])
        current_learning_rate = float(state["learning_rate"])
        initialization = state.get("initialization")
        previous_seconds = float(state["wall_clock_seconds"])
        if stale_epochs >= config.training.patience:
            # An early-stopped fit stays stopped when resumed.
            return TrainingResult(
                model_name, fold, best_epoch, best_loss, start_epoch, best, region
            )

    contexts = build_solver_contexts(model, batches, config)
    if isinstance(model, LogLinearConductance) and not resume:
        model, initialization = initialise_loglinear_scale(model, batches, contexts=contexts)
        optimiser_state = optimiser.init(eqx.filter(model, eqx.is_inexact_array))
    epochs_completed = start_epoch
    for epoch in range(start_epoch, config.training.epochs):
        key, train_key, validation_key = jax.random.split(key, 3)
        loss, gradients = objective_value_and_grad(
            model,
            batches,
            partition="train",
            contexts=contexts,
            inference=False,
            key=train_key,
        )
        updates, optimiser_state = optimiser.update(
            gradients,
            optimiser_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        updates = jax.tree.map(
            lambda update, rate=current_learning_rate: rate * update,
            updates,
        )
        model = eqx.apply_updates(model, updates)
        validation_loss = float(
            objective(
                model,
                batches,
                partition="validation",
                contexts=contexts,
                inference=True,
                key=validation_key,
            )
        )
        if not np.isfinite(float(loss)) or not np.isfinite(validation_loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            lr_stale_epochs = 0
            eqx.tree_serialise_leaves(best, model)
        else:
            stale_epochs += 1
            lr_stale_epochs += 1
            if (
                lr_stale_epochs >= config.training.lr_patience
                and current_learning_rate > config.training.min_learning_rate
            ):
                next_learning_rate = max(
                    config.training.min_learning_rate,
                    current_learning_rate * config.training.lr_factor,
                )
                if next_learning_rate < current_learning_rate:
                    current_learning_rate = next_learning_rate
                    lr_stale_epochs = 0
                    # Restart AdamW at the current model parameters while
                    # retaining learned weights and ``stale_epochs``.  Saving
                    # this fresh state below makes resume cross the scheduler
                    # boundary identically to uninterrupted training.
                    optimiser_state = optimiser.init(eqx.filter(model, eqx.is_inexact_array))
        eqx.tree_serialise_leaves(latest, (model, optimiser_state, key))
        state_path.write_text(
            json.dumps(
                {
                    "model": model_name,
                    "pooling": config.training.pooling,
                    "fold": fold,
                    "region": region,
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "training_loss": float(loss),
                    "seed": config.training.seed,
                    "wall_clock_seconds": previous_seconds + time.perf_counter() - started,
                    "stale_epochs": stale_epochs,
                    "lr_stale_epochs": lr_stale_epochs,
                    "learning_rate": current_learning_rate,
                    "initial_learning_rate": initial_learning_rate,
                    "lr_patience": config.training.lr_patience,
                    "lr_factor": config.training.lr_factor,
                    "min_learning_rate": config.training.min_learning_rate,
                    **(
                        {
                            "initialization": initialization,
                            "conductance_diagnostics": conductance_diagnostics(model, batches),
                        }
                        if isinstance(model, LogLinearConductance)
                        else {}
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        epochs_completed = epoch + 1
        if stale_epochs >= config.training.patience:
            break
    if best_epoch < 0:
        raise RuntimeError("Training completed without a finite validation checkpoint")
    return TrainingResult(model_name, fold, best_epoch, best_loss, epochs_completed, best, region)


def fit_reference_model(
    config: ExperimentConfig, *, fold: int, region: str | None = None
) -> tuple[GeographicDistanceModel, Path]:
    """Fit and save an IBD calibration across all regions, or for one under no pooling."""
    started = time.perf_counter()
    path = (
        fit_directory(config, model_name="reference", fold=fold, region=region) / "calibration.json"
    )
    batches = load_fold_batches(config, fold, region=region)
    write_workflow_manifest(config, config.project.output_dir)
    distances, targets = {}, {}
    for batch in batches:
        width = batch.features.shape[1]
        rows = batch.pixel_nodes // width
        columns = batch.pixel_nodes % width
        coordinates = jnp.stack([rows, columns], axis=1) * batch.resolution_m
        delta = coordinates[:, None, :] - coordinates[None, :, :]
        matrix = jnp.sqrt(jnp.sum(jnp.square(delta), axis=-1))
        distances[batch.region] = upper_triangle(
            np.asarray(matrix), np.asarray(batch.masks["train"])
        )
        targets[batch.region] = upper_triangle(
            np.asarray(batch.targets), np.asarray(batch.masks["train"])
        )
    model = GeographicDistanceModel.fit(distances, targets)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"slope": model.slope, "intercept": model.intercept}, indent=2) + "\n",
        encoding="utf-8",
    )
    state = dict(
        model="reference",
        pooling=config.training.pooling,
        fold=fold,
        wall_clock_seconds=time.perf_counter() - started,
    )
    if region is not None:
        state["region"] = region
    (path.parent / "training_state.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )
    return model, path
