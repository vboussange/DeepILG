"""ResNet-9 patch encoder for positive graph conductances."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from ..config import validate_patch_batch_size
from .base import ConductanceModel
from .common import bounded_exponential, patchify


@eqx.filter_checkpoint
def _encode_patch(model: ResNet9Conductance, patch: jax.Array) -> jax.Array:
    """Recompute CNN activations in the backward pass instead of retaining them."""
    return model.encode_patch(patch)


class ResidualBlock(eqx.Module):
    """Two convolutions with GroupNorm and an identity skip connection."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm

    def __init__(self, channels: int, *, key: jax.Array):
        first, second = jax.random.split(key)
        self.conv1 = eqx.nn.Conv2d(channels, channels, 3, padding=1, dtype=jnp.float32, key=first)
        self.conv2 = eqx.nn.Conv2d(channels, channels, 3, padding=1, dtype=jnp.float32, key=second)
        self.norm1 = eqx.nn.GroupNorm(min(8, channels), channels, dtype=jnp.float32)
        self.norm2 = eqx.nn.GroupNorm(min(8, channels), channels, dtype=jnp.float32)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Encode one channel-first feature map."""
        residual = inputs
        outputs = jax.nn.relu(self.norm1(self.conv1(inputs)))
        outputs = self.norm2(self.conv2(outputs))
        return jax.nn.relu(outputs + residual)


class ResNet9Conductance(ConductanceModel):
    """Map each native-resolution raster patch to one positive conductance."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    norm3: eqx.nn.GroupNorm
    residual1: ResidualBlock
    residual2: ResidualBlock
    output: eqx.nn.Linear
    patch_size: int = eqx.field(static=True)

    def __init__(self, in_channels: int = 14, *, patch_size: int = 32, key: jax.Array):
        if patch_size < 4:
            raise ValueError("ResNet9Conductance requires patch_size >= 4")
        keys = jax.random.split(key, 8)
        self.conv1 = eqx.nn.Conv2d(in_channels, 64, 3, padding=1, dtype=jnp.float32, key=keys[0])
        self.norm1 = eqx.nn.GroupNorm(8, 64, dtype=jnp.float32)
        self.residual1 = ResidualBlock(64, key=keys[1])
        self.conv2 = eqx.nn.Conv2d(64, 128, 3, padding=1, dtype=jnp.float32, key=keys[2])
        self.norm2 = eqx.nn.GroupNorm(8, 128, dtype=jnp.float32)
        self.residual2 = ResidualBlock(128, key=keys[3])
        self.conv3 = eqx.nn.Conv2d(128, 256, 3, padding=1, dtype=jnp.float32, key=keys[4])
        self.norm3 = eqx.nn.GroupNorm(8, 256, dtype=jnp.float32)
        self.output = eqx.nn.Linear(256, 1, dtype=jnp.float32, key=keys[5])
        self.patch_size = patch_size

    def encode_patch(self, patch: jax.Array) -> jax.Array:
        """Return the bounded positive conductance of one patch."""
        patch = jnp.asarray(patch, dtype=self.conv1.weight.dtype)
        values = jax.nn.relu(self.norm1(self.conv1(patch)))
        values = eqx.nn.MaxPool2d(2, stride=2)(values)
        values = self.residual1(values)
        values = jax.nn.relu(self.norm2(self.conv2(values)))
        values = eqx.nn.MaxPool2d(2, stride=2)(values)
        values = self.residual2(values)
        values = jax.nn.relu(self.norm3(self.conv3(values)))
        values = jnp.mean(values, axis=(1, 2))
        return bounded_exponential(self.output(values).squeeze())

    def conductance(self, features: jax.Array, *, patch_batch_size: int | None = None) -> jax.Array:
        """Return the patch-grid conductance surface for one HWC raster.

        None encodes the complete region without checkpointing. A positive
        integer checkpoints every encoder chunk, even if all patches fit in one
        chunk, trading backward recomputation for lower activation memory.
        """
        validate_patch_batch_size(patch_batch_size)
        patches, grid_shape = patchify(features, self.patch_size)
        if patch_batch_size is None:
            return jax.vmap(self.encode_patch)(patches).reshape(grid_shape)
        # lax.map batches with vmap inside a compiled loop, including a ragged
        # final chunk. This avoids unrolling one CNN graph per chunk under JIT.
        conductance = jax.lax.map(
            lambda patch: _encode_patch(self, patch), patches, batch_size=patch_batch_size
        )
        return conductance.reshape(grid_shape)

    def __call__(self, features: jax.Array, *, patch_batch_size: int | None = None) -> jax.Array:
        """Alias for :meth:`conductance`."""
        return self.conductance(features, patch_batch_size=patch_batch_size)
