"""Shape and patch-node invariants for the neural encoders."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deepilg.models import (
    ConductanceModel,
    EmbeddingDistanceModel,
    ResNet9Conductance,
    UNetEmbeddingDistance,
)
from deepilg.models.common import patchify, pixel_to_patch_nodes


def test_patchify_and_shared_patch_nodes() -> None:
    features = jnp.arange(8 * 12 * 2, dtype=jnp.float32).reshape(8, 12, 2)
    patches, shape = patchify(features, 4)
    assert patches.shape == (6, 2, 4, 4)
    assert shape == (2, 3)
    nodes = pixel_to_patch_nodes(
        jnp.asarray([0, 3 * 12 + 3, 4 * 12 + 4]),
        raster_shape=(8, 12),
        patch_size=4,
    )
    np.testing.assert_array_equal(nodes, np.asarray([0, 0, 4]))


def test_encoder_families_are_distinct() -> None:
    resnet = ResNet9Conductance(in_channels=14, patch_size=4, key=jax.random.PRNGKey(0))
    unet = UNetEmbeddingDistance(
        in_channels=14,
        patch_size=4,
        base_channels=8,
        embedding_dim=4,
        dropout=0.0,
        key=jax.random.PRNGKey(1),
    )
    assert isinstance(resnet, ConductanceModel) and not isinstance(resnet, EmbeddingDistanceModel)
    assert isinstance(unet, EmbeddingDistanceModel) and not isinstance(unet, ConductanceModel)
    features = jnp.ones((8, 8, 14), dtype=jnp.float32)
    matrix, conductance = resnet.resistance_and_conductance(features, jnp.asarray([0, 63]))
    assert conductance.shape == (2, 2)
    np.testing.assert_allclose(
        matrix, resnet.predict_distances(features, jnp.asarray([0, 63])), rtol=1e-5
    )


def test_abstract_families_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        ConductanceModel()
    with pytest.raises(TypeError):
        EmbeddingDistanceModel()


def test_resnet9_returns_positive_patch_conductances() -> None:
    model = ResNet9Conductance(in_channels=14, patch_size=4, key=jax.random.PRNGKey(0))
    surface = model(jnp.ones((8, 8, 14), dtype=jnp.float32))
    assert surface.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(surface)))
    assert bool(jnp.all(surface > 0))
    np.testing.assert_allclose(
        surface,
        model(jnp.ones((8, 8, 14), dtype=jnp.float32), patch_batch_size=3),
        rtol=1e-6,
    )


@pytest.mark.parametrize("batch_size", [None, 1, 4, 16])
def test_checkpointed_patch_batches_preserve_outputs_and_gradients(batch_size) -> None:
    model = ResNet9Conductance(in_channels=2, patch_size=4, key=jax.random.PRNGKey(10))
    features = jax.random.normal(jax.random.PRNGKey(11), (8, 12, 2), dtype=jnp.float32)

    def reference(model_and_features):
        encoder, raster = model_and_features
        patches, shape = patchify(raster, encoder.patch_size)
        return jax.vmap(encoder.encode_patch)(patches).reshape(shape)

    def checkpointed(model_and_features):
        encoder, raster = model_and_features
        return encoder.conductance(raster, patch_batch_size=batch_size)

    arguments = (model, features)
    np.testing.assert_allclose(checkpointed(arguments), reference(arguments), rtol=1e-5)
    reference_gradients = eqx.filter_grad(lambda args: jnp.sum(reference(args)))(arguments)
    gradients = eqx.filter_grad(lambda args: jnp.sum(checkpointed(args)))(arguments)
    for actual, expected in zip(
        jax.tree.leaves(gradients), jax.tree.leaves(reference_gradients), strict=True
    ):
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)


@pytest.mark.parametrize("batch_size", [None, 4, 16])
def test_patch_batch_size_selects_checkpointing_even_for_one_chunk(batch_size) -> None:
    model = ResNet9Conductance(in_channels=2, patch_size=4, key=jax.random.PRNGKey(10))
    features = jnp.ones((8, 12, 2), dtype=jnp.float32)
    jaxpr, _, _ = eqx.filter_make_jaxpr(
        lambda encoder: encoder.conductance(features, patch_batch_size=batch_size)
    )(model)
    assert ("remat" in str(jaxpr)) == (batch_size is not None)


def test_checkpointed_chunks_use_a_compiled_loop_instead_of_unrolling() -> None:
    model = ResNet9Conductance(in_channels=2, patch_size=4, key=jax.random.PRNGKey(10))
    equation_counts = []
    for side in (8, 32):
        features = jnp.ones((side, side, 2), dtype=jnp.float32)
        jaxpr, _, _ = eqx.filter_make_jaxpr(
            lambda encoder, raster: encoder.conductance(raster, patch_batch_size=2)
        )(model, features)
        assert "scan[" in str(jaxpr)
        equation_counts.append(len(jaxpr.jaxpr.eqns))
    assert equation_counts[0] == equation_counts[1]


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "32"])
def test_encoders_reject_invalid_patch_batch_sizes(value) -> None:
    key = jax.random.PRNGKey(1)
    resnet_model = ResNet9Conductance(in_channels=2, patch_size=4, key=key)
    unet_model = UNetEmbeddingDistance(in_channels=2, patch_size=4, key=key)
    features = jnp.ones((8, 8, 2), dtype=jnp.float32)
    for function in (resnet_model.conductance, unet_model.embedding_grid):
        with pytest.raises(ValueError, match="patch_batch_size must be None or a positive integer"):
            function(features, patch_batch_size=value)


def test_unet_has_global_grid_output_and_zero_shared_patch_distance() -> None:
    model = UNetEmbeddingDistance(
        in_channels=14,
        patch_size=4,
        base_channels=8,
        embedding_dim=4,
        dropout=0.0,
        key=jax.random.PRNGKey(1),
    )
    features = jnp.ones((16, 20, 14), dtype=jnp.float32)
    grid = model.embedding_grid(features, patch_batch_size=7)
    assert grid.shape == (4, 5, 4)
    distances = model.predict_distances(features, jnp.asarray([0, 3 * 20 + 3, 15 * 20 + 19]))
    assert distances.shape == (3, 3)
    assert float(distances[0, 1]) == 0.0
    np.testing.assert_allclose(distances, distances.T, atol=1e-6)
