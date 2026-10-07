"""Fully data-driven U-Net with a global patch-grid receptive field."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from ..config import validate_patch_batch_size
from .base import EmbeddingDistanceModel
from .common import patchify


def _match_shape(values: jax.Array, reference: jax.Array) -> jax.Array:
    """Centre-crop or zero-pad a CHW tensor to a reference spatial shape."""
    target_height, target_width = reference.shape[1:]
    height, width = values.shape[1:]
    if height > target_height:
        start = (height - target_height) // 2
        values = values[:, start : start + target_height, :]
    elif height < target_height:
        before = (target_height - height) // 2
        after = target_height - height - before
        values = jnp.pad(values, ((0, 0), (before, after), (0, 0)))
    if width > target_width:
        start = (width - target_width) // 2
        values = values[:, :, start : start + target_width]
    elif width < target_width:
        before = (target_width - width) // 2
        after = target_width - width - before
        values = jnp.pad(values, ((0, 0), (0, 0), (before, after)))
    return values


class ResidualConvBlock(eqx.Module):
    """Residual convolutional block used throughout the U-Net."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    projection: eqx.nn.Conv2d | None
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    dropout: eqx.nn.Dropout

    def __init__(self, inputs: int, outputs: int, *, dropout: float, key: jax.Array):
        first, second, projection = jax.random.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(inputs, outputs, 3, padding=1, dtype=jnp.float32, key=first)
        self.conv2 = eqx.nn.Conv2d(outputs, outputs, 3, padding=1, dtype=jnp.float32, key=second)
        self.projection = (
            None
            if inputs == outputs
            else eqx.nn.Conv2d(inputs, outputs, 1, dtype=jnp.float32, key=projection)
        )
        self.norm1 = eqx.nn.GroupNorm(min(8, outputs), outputs, dtype=jnp.float32)
        self.norm2 = eqx.nn.GroupNorm(min(8, outputs), outputs, dtype=jnp.float32)
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self,
        values: jax.Array,
        *,
        inference: bool,
        key: jax.Array | None,
    ) -> jax.Array:
        """Apply the block to one channel-first feature map."""
        residual = values if self.projection is None else self.projection(values)
        values = jax.nn.gelu(self.norm1(self.conv1(values)))
        values = self.norm2(self.conv2(values))
        values = self.dropout(values, inference=inference, key=key)
        return jax.nn.gelu(values + residual)


class UNetEmbeddingDistance(EmbeddingDistanceModel):
    """Predict patch embeddings and squared distances between population nodes."""

    patch_embedding: eqx.nn.Linear
    patch_norm: eqx.nn.GroupNorm
    encoder1: ResidualConvBlock
    down1: eqx.nn.Conv2d
    down1_norm: eqx.nn.GroupNorm
    encoder2: ResidualConvBlock
    down2: eqx.nn.Conv2d
    down2_norm: eqx.nn.GroupNorm
    bottleneck: ResidualConvBlock
    up2: eqx.nn.ConvTranspose2d
    merge2: eqx.nn.Conv2d
    merge2_norm: eqx.nn.GroupNorm
    decoder2: ResidualConvBlock
    up1: eqx.nn.ConvTranspose2d
    merge1: eqx.nn.Conv2d
    merge1_norm: eqx.nn.GroupNorm
    decoder1: ResidualConvBlock
    embedding_head: eqx.nn.Conv2d
    patch_size: int = eqx.field(static=True)
    base_channels: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int = 14,
        *,
        patch_size: int = 32,
        base_channels: int = 32,
        embedding_dim: int = 16,
        dropout: float = 0.1,
        key: jax.Array,
    ):
        if patch_size <= 0 or base_channels <= 0 or embedding_dim <= 0:
            raise ValueError("patch_size, base_channels, and embedding_dim must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        keys = jax.random.split(key, 13)
        stage2, bottleneck = 2 * base_channels, 4 * base_channels
        self.patch_embedding = eqx.nn.Linear(
            in_channels * patch_size * patch_size,
            base_channels,
            dtype=jnp.float32,
            key=keys[0],
        )
        self.patch_norm = eqx.nn.GroupNorm(min(8, base_channels), base_channels, dtype=jnp.float32)
        self.encoder1 = ResidualConvBlock(
            base_channels, base_channels, dropout=dropout, key=keys[1]
        )
        self.down1 = eqx.nn.Conv2d(
            base_channels,
            stage2,
            3,
            stride=2,
            padding=1,
            dtype=jnp.float32,
            key=keys[2],
        )
        self.down1_norm = eqx.nn.GroupNorm(min(8, stage2), stage2, dtype=jnp.float32)
        self.encoder2 = ResidualConvBlock(stage2, stage2, dropout=dropout, key=keys[3])
        self.down2 = eqx.nn.Conv2d(
            stage2,
            bottleneck,
            3,
            stride=2,
            padding=1,
            dtype=jnp.float32,
            key=keys[4],
        )
        self.down2_norm = eqx.nn.GroupNorm(min(8, bottleneck), bottleneck, dtype=jnp.float32)
        self.bottleneck = ResidualConvBlock(bottleneck, bottleneck, dropout=dropout, key=keys[5])
        self.up2 = eqx.nn.ConvTranspose2d(
            bottleneck, stage2, 2, stride=2, dtype=jnp.float32, key=keys[6]
        )
        self.merge2 = eqx.nn.Conv2d(
            2 * stage2, stage2, 3, padding=1, dtype=jnp.float32, key=keys[7]
        )
        self.merge2_norm = eqx.nn.GroupNorm(min(8, stage2), stage2, dtype=jnp.float32)
        self.decoder2 = ResidualConvBlock(stage2, stage2, dropout=dropout, key=keys[8])
        self.up1 = eqx.nn.ConvTranspose2d(
            stage2, base_channels, 2, stride=2, dtype=jnp.float32, key=keys[9]
        )
        self.merge1 = eqx.nn.Conv2d(
            2 * base_channels,
            base_channels,
            3,
            padding=1,
            dtype=jnp.float32,
            key=keys[10],
        )
        self.merge1_norm = eqx.nn.GroupNorm(min(8, base_channels), base_channels, dtype=jnp.float32)
        self.decoder1 = ResidualConvBlock(
            base_channels, base_channels, dropout=dropout, key=keys[11]
        )
        self.embedding_head = eqx.nn.Conv2d(
            base_channels, embedding_dim, 1, dtype=jnp.float32, key=keys[12]
        )
        self.patch_size = patch_size
        self.base_channels = base_channels
        self.embedding_dim = embedding_dim

    def embedding_grid(
        self,
        features: jax.Array,
        *,
        inference: bool = True,
        key: jax.Array | None = None,
        patch_batch_size: int | None = None,
    ) -> jax.Array:
        """Return an ``(patch rows, patch columns, embedding_dim)`` tensor.

        patch_batch_size only chunks the linear patch embedding; it does not
        checkpoint or split the subsequent spatial U-Net computation.
        """
        validate_patch_batch_size(patch_batch_size)
        if not inference and key is None:
            raise ValueError("Training-mode U-Net evaluation requires a PRNG key")
        patches, grid_shape = patchify(features, self.patch_size)
        flattened = jnp.asarray(
            patches.reshape(patches.shape[0], -1),
            dtype=self.patch_embedding.weight.dtype,
        )
        batch_size = flattened.shape[0] if patch_batch_size is None else patch_batch_size
        embedded = jnp.concatenate(
            [
                jax.vmap(self.patch_embedding)(flattened[start : start + batch_size])
                for start in range(0, flattened.shape[0], batch_size)
            ]
        )
        values = embedded.reshape(grid_shape[0], grid_shape[1], self.base_channels)
        values = jnp.moveaxis(values, -1, 0)
        values = jax.nn.gelu(self.patch_norm(values))
        keys = [None] * 5 if key is None else list(jax.random.split(key, 5))

        skip1 = self.encoder1(values, inference=inference, key=keys[0])
        values = jax.nn.gelu(self.down1_norm(self.down1(skip1)))
        skip2 = self.encoder2(values, inference=inference, key=keys[1])
        values = jax.nn.gelu(self.down2_norm(self.down2(skip2)))
        values = self.bottleneck(values, inference=inference, key=keys[2])
        values = _match_shape(self.up2(values), skip2)
        values = self.merge2_norm(self.merge2(jnp.concatenate([values, skip2], axis=0)))
        values = self.decoder2(jax.nn.gelu(values), inference=inference, key=keys[3])
        values = _match_shape(self.up1(values), skip1)
        values = self.merge1_norm(self.merge1(jnp.concatenate([values, skip1], axis=0)))
        values = self.decoder1(jax.nn.gelu(values), inference=inference, key=keys[4])
        return jnp.moveaxis(self.embedding_head(values), 0, -1)
