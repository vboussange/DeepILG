"""Shared raster-to-patch utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def patchify(features: jax.Array, patch_size: int) -> tuple[jax.Array, tuple[int, int]]:
    """Partition an HWC raster into non-overlapping channel-first patches."""
    raster = jnp.asarray(features)
    if raster.ndim != 3:
        raise ValueError(f"Expected features with shape (H, W, C), got {raster.shape}")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    height, width, channels = raster.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"Raster shape {(height, width)} is not divisible by patch_size={patch_size}"
        )
    rows, columns = height // patch_size, width // patch_size
    patches = raster.reshape(rows, patch_size, columns, patch_size, channels)
    patches = patches.transpose(0, 2, 4, 1, 3)
    return patches.reshape(rows * columns, channels, patch_size, patch_size), (rows, columns)


def pixel_to_patch_nodes(
    pixel_nodes: jax.Array,
    *,
    raster_shape: tuple[int, int],
    patch_size: int,
) -> jax.Array:
    """Map row-major native-resolution pixels to row-major patch vertices."""
    height, width = raster_shape
    if height % patch_size or width % patch_size:
        raise ValueError("Raster dimensions must be divisible by patch_size")
    nodes = jnp.asarray(pixel_nodes, dtype=jnp.int32)
    rows = (nodes // width) // patch_size
    columns = (nodes % width) // patch_size
    return rows * (width // patch_size) + columns


def squared_embedding_distances(embeddings: jax.Array) -> jax.Array:
    """Compute a stable matrix of squared Euclidean embedding distances."""
    embeddings = jnp.asarray(embeddings)
    squared_norm = jnp.sum(jnp.square(embeddings), axis=-1)
    distances = squared_norm[:, None] + squared_norm[None, :] - 2.0 * embeddings @ embeddings.T
    distances = jnp.maximum(distances, 0.0)
    diagonal = jnp.diag_indices(distances.shape[0])
    return distances.at[diagonal].set(0.0)
