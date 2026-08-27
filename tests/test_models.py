"""Shape and patch-node invariants for the neural encoders."""

import jax
import jax.numpy as jnp
import numpy as np

from deepilg.models import ResNet9Conductance, UNetEmbeddingDistance
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
