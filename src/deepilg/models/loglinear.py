"""Low-capacity conductance encoder of patch-mean environmental features."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from ..config import validate_patch_batch_size
from .base import ConductanceModel
from .common import bounded_exponential, patchify


class LogLinearConductance(ConductanceModel):
    """Log-linear encoder with tree-cover reference and 14 parameters.

    The declared input order is unknown, tree cover, ten other WorldCover
    classes, scaled elevation, and scaled road density. Coefficients follow
    that order with tree cover omitted; one intercept sets the overall scale.
    Aggregation precedes the predictor and the bounded exponential head.
    """

    intercept: jax.Array
    coefficients: jax.Array
    patch_size: int = eqx.field(static=True)

    def __init__(self, *, patch_size: int = 32):
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.intercept = jnp.asarray(0.0, dtype=jnp.float32)
        self.coefficients = jnp.zeros(13, dtype=jnp.float32)
        self.patch_size = patch_size

    def conductance(self, features: jax.Array, *, patch_batch_size: int | None = None) -> jax.Array:
        """Return bounded patch conductances from means of the existing channels."""
        validate_patch_batch_size(patch_batch_size)
        raster = jnp.asarray(features, dtype=jnp.float32)
        if raster.ndim != 3 or raster.shape[-1] != 14:
            raise ValueError("LogLinearConductance requires HxWx14 features")
        patches, shape = patchify(raster, self.patch_size)
        means = jnp.mean(patches, axis=(2, 3))
        # Channel 1 is WorldCover 10 (tree cover), the reference category.
        contrasts = jnp.concatenate([means[:, :1], means[:, 2:]], axis=1)
        predictor = self.intercept + contrasts @ self.coefficients
        return bounded_exponential(predictor).reshape(shape)

    def __call__(self, features: jax.Array, *, patch_batch_size: int | None = None) -> jax.Array:
        """Alias for :meth:`conductance`."""
        return self.conductance(features, patch_batch_size=patch_batch_size)
