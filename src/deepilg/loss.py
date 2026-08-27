"""Region-balanced predictive objective used by the neural models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax.numpy as jnp


def log1p_mse(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error after applying ``log(1 + x)`` to both distances."""
    prediction = jnp.asarray(prediction)
    target = jnp.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction.shape} != {target.shape}"
        )
    return jnp.mean(
        jnp.square(jnp.log1p(jnp.maximum(prediction, 0.0)) - jnp.log1p(jnp.maximum(target, 0.0)))
    )


def region_balanced_log1p_mse(
    predictions: Mapping[str, jnp.ndarray] | Sequence[jnp.ndarray],
    targets: Mapping[str, jnp.ndarray] | Sequence[jnp.ndarray],
) -> jnp.ndarray:
    """Average log-distance MSE equally across geographic regions."""
    if isinstance(predictions, Mapping):
        if not isinstance(targets, Mapping) or set(predictions) != set(targets):
            raise ValueError("Prediction and target mappings must use identical regions")
        keys = sorted(predictions)
        losses = [log1p_mse(predictions[key], targets[key]) for key in keys]
    else:
        if isinstance(targets, Mapping) or len(predictions) != len(targets):
            raise ValueError("Prediction and target sequences must have equal length")
        losses = [
            log1p_mse(prediction, target)
            for prediction, target in zip(predictions, targets, strict=True)
        ]
    if not losses:
        raise ValueError("At least one geographic region is required")
    return jnp.mean(jnp.stack(losses))
