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
from deepilg.models import EmbeddingDistanceModel, ResNet9Conductance, UNetEmbeddingDistance
from deepilg.solver import build_amg_context
from deepilg.training import (
    RegionBatch,
    objective,
    objective_value_and_grad,
    predict_matrix,
)


def _batches() -> tuple[RegionBatch, ...]:
    batches = []
    for index, (width, population_count) in enumerate(((8, 3), (12, 4))):
        targets = (
            jnp.arange(population_count**2, dtype=jnp.float32).reshape(
                population_count, population_count
            )
            / 10
        )
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


def _original_objective(model, batches, *, partition, inference, key, contexts=None):
    """Independent reference for the original whole-fold differentiation path."""
    predictions, targets = {}, {}
    for batch, region_key in zip(batches, jax.random.split(key, len(batches)), strict=True):
        matrix = predict_matrix(
            model,
            batch,
            context=None if contexts is None else contexts.get(batch.region),
            inference=inference,
            key=region_key,
        )
        row, column = jnp.triu_indices(matrix.shape[0], k=1)
        selected = batch.masks[partition]
        predictions[batch.region] = matrix[row, column][selected]
        targets[batch.region] = batch.targets[row, column][selected]
    return region_balanced_log1p_mse(predictions, targets)


def _assert_trees_close(actual, expected, *, atol=2e-6) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        if eqx.is_array(actual_leaf):
            assert actual_leaf.dtype == expected_leaf.dtype
            np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-4, atol=atol)
        else:
            assert actual_leaf == expected_leaf


@pytest.mark.parametrize(
    ("region_count", "partition", "inference"),
    [(1, "train", False), (2, "train", False), (2, "validation", True)],
)
def test_compiled_accumulation_matches_eager_loss_gradients_and_moments(
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
    # Whole-region fusion can change float32 rounding relative to eager ops.
    _assert_trees_close(gradients, expected_gradients, atol=5e-6)
    actual_vector = np.concatenate(
        [np.ravel(leaf) for leaf in jax.tree.leaves(gradients) if eqx.is_array(leaf)]
    )
    expected_vector = np.concatenate(
        [np.ravel(leaf) for leaf in jax.tree.leaves(expected_gradients) if eqx.is_array(leaf)]
    )
    assert np.linalg.norm(actual_vector - expected_vector) <= 1e-4 * np.linalg.norm(expected_vector)

    parameters = eqx.filter(model, eqx.is_inexact_array)
    optimizer = optax.adamw(1e-4, weight_decay=1e-4)
    state = optimizer.init(parameters)
    _, expected_state = optimizer.update(expected_gradients, state, parameters)
    _, updated_state = optimizer.update(gradients, state, parameters)
    _assert_trees_close(updated_state, expected_state)
    # Near-zero GroupNorm shift gradients can change sign under fusion, and Adam
    # amplifies that roundoff in its first update. Check optimizer sequencing
    # strictly below with a scalar model, separately from JIT/eager numerics.


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


@pytest.mark.parametrize("patch_batch_size", [None, 5])
def test_compiled_resnet_amg_loss_and_gradients_match_eager(patch_batch_size) -> None:
    model = ResNet9Conductance(in_channels=2, patch_size=4, key=jax.random.PRNGKey(1))
    batch = replace(
        _batches()[0],
        features=jax.random.normal(jax.random.PRNGKey(10), (16, 16, 2), dtype=jnp.float32),
        pixel_nodes=jnp.asarray([0, 119, 255], dtype=jnp.int32),
        patch_batch_size=patch_batch_size,
    )
    context = build_amg_context((4, 4))
    kwargs = dict(
        partition="train",
        inference=False,
        key=jax.random.PRNGKey(3),
        contexts={batch.region: context},
    )
    expected_loss, expected_gradients = eqx.filter_value_and_grad(_original_objective)(
        model, (batch,), **kwargs
    )
    loss, gradients = objective_value_and_grad(model, (batch,), **kwargs)
    np.testing.assert_allclose(loss, expected_loss, rtol=2e-4, atol=2e-6)
    np.testing.assert_allclose(objective(model, (batch,), **kwargs), expected_loss, rtol=2e-4)
    for actual, expected in zip(
        jax.tree.leaves(gradients), jax.tree.leaves(expected_gradients), strict=True
    ):
        assert np.isfinite(actual).all()
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-5)

    # Neither the batch's changing arrays nor the reusable hierarchy may become
    # opaque static objects captured in the compiled executable.
    context_arrays = [leaf for leaf in jax.tree.leaves(context) if eqx.is_array(leaf)]
    state_arrays = [leaf for leaf in jax.tree.leaves(context.solver_state) if eqx.is_array(leaf)]
    assert state_arrays
    assert all(any(leaf is candidate for candidate in context_arrays) for leaf in state_arrays)
    batch_arrays = jax.tree.leaves(batch)
    assert all(
        any(array is candidate for candidate in batch_arrays)
        for array in (
            batch.features,
            batch.pixel_nodes,
            batch.targets,
            batch.masks["train"],
        )
    )


def test_compiled_region_uses_changed_inputs_without_retracing() -> None:
    model = UNetEmbeddingDistance(
        in_channels=2,
        patch_size=2,
        base_channels=8,
        embedding_dim=2,
        dropout=0.25,
        key=jax.random.PRNGKey(2),
    )
    batch = _batches()[1]
    selected = training._selected_pairs(batch, "train")
    kwargs = dict(context=None, inference=False, key=jax.random.PRNGKey(3))
    traces = []

    def traced(*args, **kwargs):
        traces.append(None)
        return eqx.filter_value_and_grad(training._region_objective)(*args, **kwargs)

    compiled = eqx.filter_jit(traced)
    initial = jax.block_until_ready(compiled(model, batch, selected, **kwargs))
    changed_mask = {**batch.masks, "train": ~batch.masks["train"]}
    changed_batch = replace(batch, masks=changed_mask)
    cases = [
        (model, replace(batch, features=batch.features[::-1]), selected, kwargs),
        (model, replace(batch, targets=batch.targets + 0.3), selected, kwargs),
        (model, replace(batch, pixel_nodes=batch.pixel_nodes.at[0].set(19)), selected, kwargs),
        (model, changed_batch, training._selected_pairs(changed_batch, "train"), kwargs),
        (model, batch, selected, {**kwargs, "key": jax.random.PRNGKey(4)}),
        (
            eqx.tree_at(
                lambda m: m.embedding_head.weight, model, model.embedding_head.weight * 1.2
            ),
            batch,
            selected,
            kwargs,
        ),
    ]
    for changed_model, changed_batch, changed_pairs, changed_kwargs in cases:
        result = jax.block_until_ready(
            compiled(changed_model, changed_batch, changed_pairs, **changed_kwargs)
        )
        assert np.isfinite(float(result[0]))
        assert abs(float(result[0] - initial[0])) > 1e-7
        assert len(traces) == 1
    repeated = jax.block_until_ready(compiled(model, batch, selected, **kwargs))
    _assert_trees_close(repeated, initial)
    assert all(mask.dtype == jnp.bool_ for mask in batch.masks.values())
    assert all(mask.dtype == jnp.bool_ for mask in changed_batch.masks.values())


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
