"""Abstract encoder families for inverse landscape genetics.

The two families differ in how population distances are read off an encoded
raster.  :class:`ConductanceModel` is the object of primary interest: it emits a
positive conductance surface and compares populations by effective resistance on
the induced grid graph, through a differentiable solver.  :class:`EmbeddingDistanceModel`
is a purely data-driven reference: it emits node embeddings and compares
populations by squared Euclidean distance, with no effective-resistance layer.
"""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax

from ..solver import AMGContext, effective_resistance
from .common import pixel_to_patch_nodes, squared_embedding_distances


class ConductanceModel(eqx.Module):
    """Encode a raster into a conductance surface and derive effective resistances."""

    patch_size: eqx.AbstractVar[int]

    @abstractmethod
    def conductance(self, features: jax.Array, *, patch_batch_size: int | None = None) -> jax.Array:
        """Return the positive patch-grid conductance surface for one HWC raster."""

    def resistance_and_conductance(
        self,
        features: jax.Array,
        pixel_nodes: jax.Array,
        *,
        context: AMGContext | None = None,
        patch_batch_size: int | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Return pairwise effective resistances and the conductance grid behind them."""
        conductance = self.conductance(features, patch_batch_size=patch_batch_size)
        nodes = pixel_to_patch_nodes(
            pixel_nodes,
            raster_shape=(features.shape[0], features.shape[1]),
            patch_size=self.patch_size,
        )
        return effective_resistance(conductance, nodes, context=context), conductance

    def predict_distances(
        self,
        features: jax.Array,
        pixel_nodes: jax.Array,
        *,
        context: AMGContext | None = None,
        patch_batch_size: int | None = None,
    ) -> jax.Array:
        """Predict all pairwise population effective-resistance distances."""
        matrix, _ = self.resistance_and_conductance(
            features, pixel_nodes, context=context, patch_batch_size=patch_batch_size
        )
        return matrix


class EmbeddingDistanceModel(eqx.Module):
    """Encode a raster into node embeddings and derive squared Euclidean distances."""

    patch_size: eqx.AbstractVar[int]

    @abstractmethod
    def embedding_grid(
        self,
        features: jax.Array,
        *,
        inference: bool = True,
        key: jax.Array | None = None,
        patch_batch_size: int | None = None,
    ) -> jax.Array:
        """Return a ``(patch rows, patch columns, embedding_dim)`` embedding tensor."""

    def predict_distances(
        self,
        features: jax.Array,
        pixel_nodes: jax.Array,
        *,
        inference: bool = True,
        key: jax.Array | None = None,
        patch_batch_size: int | None = None,
    ) -> jax.Array:
        """Predict all pairwise population squared-embedding distances."""
        grid = self.embedding_grid(
            features, inference=inference, key=key, patch_batch_size=patch_batch_size
        )
        nodes = pixel_to_patch_nodes(
            pixel_nodes,
            raster_shape=(features.shape[0], features.shape[1]),
            patch_size=self.patch_size,
        )
        embeddings = grid.reshape(-1, grid.shape[-1])[nodes]
        return squared_embedding_distances(embeddings)
