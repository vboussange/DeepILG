"""Numerical and differentiation tests for grounded Laplacian solves."""

import jax
import jax.numpy as jnp
import numpy as np

from deepilg.solver import build_amg_context, dense_effective_resistance, effective_resistance


def test_lineax_resistance_matches_dense_grounded_solve_and_differentiates() -> None:
    surface = jnp.asarray([[1.0, 1.4], [0.8, 2.0]], dtype=jnp.float64)
    nodes = jnp.asarray([0, 1, 3])
    observed = effective_resistance(surface, nodes, rtol=1e-10, atol=1e-10)
    expected = dense_effective_resistance(np.asarray(surface), np.asarray(nodes))
    np.testing.assert_allclose(observed, expected, rtol=1e-8, atol=1e-8)

    derivative = jax.grad(lambda values: effective_resistance(values, nodes)[0, 2])(surface)
    assert derivative.shape == surface.shape
    assert bool(jnp.all(jnp.isfinite(derivative)))
    assert float(jnp.linalg.norm(derivative)) > 0.0


def test_amjax_hierarchy_is_reused_for_a_changed_operator() -> None:
    context = build_amg_context((4, 4), rtol=1e-8, atol=1e-8, max_steps=500, dtype=jnp.float64)
    surface = jnp.linspace(0.8, 1.5, 16, dtype=jnp.float64).reshape(4, 4)
    nodes = jnp.asarray([0, 5, 10, 15])
    observed = effective_resistance(surface, nodes, context=context)
    expected = dense_effective_resistance(np.asarray(surface), np.asarray(nodes))
    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-6)
    derivative = jax.grad(
        lambda values: effective_resistance(values, nodes, context=context).sum()
    )(surface)
    assert bool(jnp.all(jnp.isfinite(derivative)))
