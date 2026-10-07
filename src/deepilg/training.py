"""Single-fit training and checkpointing for the benchmark models."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import ExperimentConfig
from .data import FSTMatrix, linearize_fst, load_fst_directory
from .loss import log1p_mse
from .models import (
    ConductanceModel,
    EmbeddingDistanceModel,
    GeographicDistanceModel,
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

ModelName = Literal["reference", "resnet9", "unet"]
Partition = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class RegionBatch:
    """Model inputs and population-level split masks for one region."""

    region: str
    population_ids: tuple[str, ...]
    features: jax.Array
    pixel_nodes: jax.Array
    targets: jax.Array
    masks: Mapping[str, jax.Array]
    resolution_m: float
    patch_batch_size: int


@dataclass(frozen=True)
class TrainingResult:
    """Summary of one population-holdout training run."""

    model: ModelName
    fold: int
    best_epoch: int
    best_validation_loss: float
    epochs_completed: int
    checkpoint: Path


def load_fold_batches(config: ExperimentConfig, fold: int) -> tuple[RegionBatch, ...]:
    """Load every processed region for one population-holdout fold."""
    matrices = load_fst_directory(config.data.fst_dir)
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
    patch_batch_size: int,
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
        targets=linearize_fst(matrix.values).astype(jnp.float32),
        masks=masks,
        resolution_m=resolution_m,
        patch_batch_size=patch_batch_size,
    )


def initialise_model(
    name: ModelName,
    *,
    patch_size: int,
    seed: int,
) -> ResNet9Conductance | UNetEmbeddingDistance:
    """Create one neural model with the manuscript architecture."""
    key = jax.random.PRNGKey(seed)
    if name == "resnet9":
        return ResNet9Conductance(patch_size=patch_size, key=key)
    if name == "unet":
        return UNetEmbeddingDistance(patch_size=patch_size, key=key)
    raise ValueError("initialise_model supports only 'resnet9' and 'unet'")


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


def _region_objective(
    model: ResNet9Conductance | UNetEmbeddingDistance,
    batch: RegionBatch,
    *,
    partition: Partition,
    context: AMGContext | None,
    inference: bool,
    key: jax.Array,
) -> jax.Array:
    """Compute one region's masked loss using its already-split random key."""
    matrix = predict_matrix(model, batch, context=context, inference=inference, key=key)
    row, column = jnp.triu_indices(matrix.shape[0], k=1)
    selected = batch.masks[partition]
    return log1p_mse(matrix[row, column][selected], batch.targets[row, column][selected])


_region_value_and_grad = eqx.filter_value_and_grad(_region_objective)


def objective(
    model: ResNet9Conductance | UNetEmbeddingDistance,
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
        _region_objective(
            model,
            batch,
            partition=partition,
            context=contexts.get(batch.region),
            inference=inference,
            key=region_key,
        )
        for batch, region_key in zip(batches, keys, strict=True)
    ]
    return jnp.mean(jnp.stack(losses))


def objective_value_and_grad(
    model: ResNet9Conductance | UNetEmbeddingDistance,
    batches: Sequence[RegionBatch],
    *,
    partition: Partition,
    contexts: Mapping[str, AMGContext],
    inference: bool,
    key: jax.Array,
) -> tuple[jax.Array, ResNet9Conductance | UNetEmbeddingDistance]:
    """Average regional gradients without retaining several regions' activations.

    Keep this loop outside JIT so each regional backward pass finishes before
    starting the next. The caller applies the averaged gradient in one update.
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
                partition=partition,
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


def train_neural_model(
    config: ExperimentConfig,
    *,
    model_name: Literal["resnet9", "unet"],
    fold: int,
    resume: bool = False,
) -> TrainingResult:
    """Fit one neural model for one fold with validation early stopping."""
    batches = load_fold_batches(config, fold)
    write_workflow_manifest(config, config.project.output_dir)
    model = initialise_model(
        model_name,
        patch_size=config.preprocessing.patch_size,
        seed=config.training.seed,
    )
    output = config.project.output_dir / model_name / f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest.eqx"
    best = output / "best.eqx"
    state_path = output / "training_state.json"
    start_epoch = 0
    best_epoch = -1
    best_loss = float("inf")
    stale_epochs = 0
    lr_stale_epochs = 0
    initial_learning_rate = (
        config.training.resnet9_learning_rate
        if model_name == "resnet9"
        else config.training.unet_learning_rate
    )
    current_learning_rate = initial_learning_rate
    # Unit-rate AdamW lets the plateau scheduler scale complete updates while
    # preserving moments and decoupled weight decay across LR reductions.
    optimiser = optax.adamw(1.0, weight_decay=config.training.weight_decay)
    optimiser_state = optimiser.init(eqx.filter(model, eqx.is_inexact_array))
    key = jax.random.PRNGKey(config.training.seed)
    if resume:
        if not latest.is_file() or not state_path.is_file():
            raise FileNotFoundError(f"No resumable checkpoint found in {output}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            state.get("model") != model_name
            or int(state.get("fold", -1)) != fold
            or int(state.get("seed", -1)) != config.training.seed
        ):
            raise ValueError("Resume checkpoint does not match model, fold, or seed")
        model, optimiser_state, key = eqx.tree_deserialise_leaves(
            latest, (model, optimiser_state, key)
        )
        start_epoch = int(state["epoch"]) + 1
        best_epoch = int(state["best_epoch"])
        best_loss = float(state["best_validation_loss"])
        stale_epochs = int(state["stale_epochs"])
        lr_stale_epochs = int(state["lr_stale_epochs"])
        current_learning_rate = float(state["learning_rate"])

    contexts = build_solver_contexts(model, batches, config)
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
                current_learning_rate = max(
                    config.training.min_learning_rate,
                    current_learning_rate * config.training.lr_factor,
                )
                lr_stale_epochs = 0
        eqx.tree_serialise_leaves(latest, (model, optimiser_state, key))
        state_path.write_text(
            json.dumps(
                {
                    "model": model_name,
                    "fold": fold,
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "training_loss": float(loss),
                    "seed": config.training.seed,
                    "stale_epochs": stale_epochs,
                    "lr_stale_epochs": lr_stale_epochs,
                    "learning_rate": current_learning_rate,
                    "initial_learning_rate": initial_learning_rate,
                    "lr_patience": config.training.lr_patience,
                    "lr_factor": config.training.lr_factor,
                    "min_learning_rate": config.training.min_learning_rate,
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
    return TrainingResult(model_name, fold, best_epoch, best_loss, epochs_completed, best)


def fit_reference_model(
    config: ExperimentConfig, *, fold: int
) -> tuple[GeographicDistanceModel, Path]:
    """Fit and save one non-negative geographic-distance calibration."""
    batches = load_fold_batches(config, fold)
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
    path = config.project.output_dir / "reference" / f"fold_{fold}" / "calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"slope": model.slope, "intercept": model.intercept}, indent=2) + "\n",
        encoding="utf-8",
    )
    return model, path
