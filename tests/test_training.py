"""Regional accumulation preserves the full objective and optimizer semantics."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from deepilg import training
from deepilg.config import load_config
from deepilg.loss import region_balanced_log1p_mse
from deepilg.models import EmbeddingDistanceModel, UNetEmbeddingDistance
from deepilg.training import (
    RegionBatch,
    objective,
    objective_value_and_grad,
    predict_matrix,
)


def _batches() -> tuple[RegionBatch, ...]:
    batches = []
    for index, (width, population_count) in enumerate(((8, 3), (12, 4))):
        targets = jnp.arange(population_count**2, dtype=jnp.float32).reshape(
            population_count, population_count
        ) / 10
        targets = (targets + targets.T) / 2
        pair_count = population_count * (population_count - 1) // 2
        training_mask = jnp.arange(pair_count) % 2 == 0
        batches.append(
            RegionBatch(
                region=f"region_{index}",
                population_ids=tuple(str(i) for i in range(population_count)),
                features=jax.random.normal(
                    jax.random.PRNGKey(index + 1), (8, width, 2), dtype=jnp.float32
                ),
                pixel_nodes=jnp.linspace(0, 8 * width - 1, population_count, dtype=jnp.int32),
                targets=targets,
                masks={"train": training_mask, "validation": ~training_mask},
                resolution_m=30,
                patch_batch_size=5,
            )
        )
    return tuple(batches)


def _original_objective(model, batches, *, partition, inference, key):
    """Independent reference for the original whole-fold differentiation path."""
    predictions, targets = {}, {}
    for batch, region_key in zip(batches, jax.random.split(key, len(batches)), strict=True):
        matrix = predict_matrix(model, batch, inference=inference, key=region_key)
        row, column = jnp.triu_indices(matrix.shape[0], k=1)
        selected = batch.masks[partition]
        predictions[batch.region] = matrix[row, column][selected]
        targets[batch.region] = batch.targets[row, column][selected]
    return region_balanced_log1p_mse(predictions, targets)


def _assert_trees_close(actual, expected) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        if eqx.is_array(actual_leaf):
            assert actual_leaf.dtype == expected_leaf.dtype
            np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-4, atol=2e-6)
        else:
            assert actual_leaf == expected_leaf


@pytest.mark.parametrize(
    ("region_count", "partition", "inference"),
    [(1, "train", False), (2, "train", False), (2, "validation", True)],
)
def test_accumulation_matches_whole_fold_gradient_and_adamw_update(
    region_count, partition, inference
) -> None:
    model = UNetEmbeddingDistance(
        in_channels=2,
        patch_size=2,
        base_channels=8,
        embedding_dim=2,
        dropout=0.25,
        key=jax.random.PRNGKey(20),
    )
    batches = _batches()[:region_count]
    kwargs = dict(partition=partition, inference=inference, key=jax.random.PRNGKey(21))
    expected_loss, expected_gradients = eqx.filter_value_and_grad(_original_objective)(
        model, batches, **kwargs
    )
    loss, gradients = objective_value_and_grad(model, batches, contexts={}, **kwargs)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-6)
    np.testing.assert_allclose(
        objective(model, batches, contexts={}, **kwargs), expected_loss, rtol=1e-6
    )
    _assert_trees_close(gradients, expected_gradients)

    parameters = eqx.filter(model, eqx.is_inexact_array)
    optimizer = optax.adamw(1e-4, weight_decay=1e-4)
    state = optimizer.init(parameters)
    expected_updates, expected_state = optimizer.update(expected_gradients, state, parameters)
    updates, updated_state = optimizer.update(gradients, state, parameters)
    _assert_trees_close(updated_state, expected_state)
    _assert_trees_close(
        eqx.apply_updates(model, updates), eqx.apply_updates(model, expected_updates)
    )


@pytest.mark.parametrize("function", [objective, objective_value_and_grad])
def test_objectives_reject_empty_region_sequence(function) -> None:
    with pytest.raises(ValueError, match="At least one geographic region"):
        function(
            None,
            (),
            partition="train",
            contexts={},
            inference=False,
            key=jax.random.PRNGKey(0),
        )


class _ScalarEmbedding(EmbeddingDistanceModel):
    """Small encoder for exercising the real training/checkpoint loop on CPU."""

    weight: jax.Array
    patch_size: int = eqx.field(static=True)

    def __init__(self, weight):
        self.weight = weight
        self.patch_size = 1

    def embedding_grid(self, features, *, inference=True, key=None, patch_batch_size=None):
        """Scale one raster channel into a scalar embedding."""
        return self.weight * features[..., :1]


def test_training_applies_one_mean_gradient_update_per_epoch(tmp_path, monkeypatch) -> None:
    model = _ScalarEmbedding(jnp.asarray(0.5, dtype=jnp.float32))
    batches = _batches()
    config = load_config("configs/demo.toml")
    config = replace(
        config,
        project=replace(config.project, output_dir=tmp_path),
        training=replace(config.training, epochs=1),
    )
    monkeypatch.setattr(training, "load_fold_batches", lambda *_: batches)
    monkeypatch.setattr(training, "initialise_model", lambda *_, **__: model)
    calls = []

    def accumulated(*args, **kwargs):
        calls.append(kwargs["key"])
        return objective_value_and_grad(*args, **kwargs)

    monkeypatch.setattr(training, "objective_value_and_grad", accumulated)
    result = training.train_neural_model(config, model_name="unet", fold=0)
    assert result.epochs_completed == 1
    assert len(calls) == 1

    next_key, train_key, _ = jax.random.split(jax.random.PRNGKey(config.training.seed), 3)
    _, gradients = eqx.filter_value_and_grad(_original_objective)(
        model, batches, partition="train", inference=False, key=train_key
    )
    parameters = eqx.filter(model, eqx.is_inexact_array)
    optimizer = optax.adamw(1.0, weight_decay=config.training.weight_decay)
    state = optimizer.init(parameters)
    updates, expected_state = optimizer.update(gradients, state, parameters)
    updates = jax.tree.map(lambda value: config.training.unet_learning_rate * value, updates)
    expected_model = eqx.apply_updates(model, updates)
    actual_model, actual_state, actual_key = eqx.tree_deserialise_leaves(
        tmp_path / "unet" / "fold_0" / "latest.eqx", (model, state, next_key)
    )
    _assert_trees_close(actual_model, expected_model)
    _assert_trees_close(actual_state, expected_state)
    np.testing.assert_array_equal(actual_key, next_key)
