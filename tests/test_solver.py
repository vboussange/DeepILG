"""Numerical and differentiation tests for grounded Laplacian solves."""

from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from jaxscape.solvers import AMJaxCGSolver
from jaxscape.utils import graph_laplacian

from deepilg.solver import (
    build_amg_context,
    build_cg_context,
    conductance_grid_graph,
    effective_resistance,
)
from deepilg.training import build_solver_contexts, initialise_model


def grounded_laplacian(conductance: jax.Array) -> BCOO:
    """Construct the small-graph reference Laplacian."""
    adjacency = conductance_grid_graph(conductance).get_adjacency_matrix()
    return graph_laplacian(adjacency)[:-1, :-1]


def dense_effective_resistance(conductance: np.ndarray, terminal_nodes: np.ndarray) -> np.ndarray:
    """Evaluate a dense reference independently of JAXScape's distance solve."""
    surface = jnp.asarray(conductance, dtype=jnp.float64)
    laplacian = np.asarray(grounded_laplacian(surface).todense())
    basis = np.asarray(jax.nn.one_hot(terminal_nodes, laplacian.shape[0], dtype=surface.dtype)).T
    potentials = np.linalg.solve(laplacian, basis)
    gram = basis.T @ potentials
    diagonal = np.diag(gram)
    result = diagonal[:, None] + diagonal[None, :] - gram - gram.T
    np.fill_diagonal(result, 0.0)
    return result


def test_graph_edge_function_is_stable_for_jit_reuse() -> None:
    first = conductance_grid_graph(jnp.ones((2, 2)))
    second = conductance_grid_graph(jnp.full((2, 2), 2.0))
    assert first.fun is second.fun


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
    context = build_amg_context((6, 6), rtol=1e-8, atol=1e-8, max_steps=500)
    assert context.solver_state is not None
    assert isinstance(context.solver, AMJaxCGSolver)
    assert context.solver_state.cg_state is None
    assert context.solver_state.preconditioner.in_structure().dtype == jnp.float64
    surface = jnp.linspace(0.8, 1.5, 36, dtype=jnp.float32).reshape(6, 6)
    nodes = jnp.asarray([0, 7, 7, 35])
    observed = effective_resistance(surface, nodes, context=context)
    expected = dense_effective_resistance(np.asarray(surface), np.asarray(nodes))
    assert observed.dtype == jnp.float32
    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(jnp.diag(observed), 0.0)
    assert observed[1, 2] == 0.0
    derivative = jax.grad(
        lambda values: effective_resistance(values, nodes, context=context).sum()
    )(surface)
    assert derivative.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(derivative)))

    surface64 = surface.astype(jnp.float64)

    def dense_loss(values: jax.Array) -> jax.Array:
        laplacian = grounded_laplacian(values).todense()
        right_hand_sides = jax.nn.one_hot(nodes, laplacian.shape[0], dtype=laplacian.dtype).T
        potentials = jnp.linalg.solve(laplacian, right_hand_sides)
        gram = right_hand_sides.T @ potentials
        diagonal = jnp.diag(gram)
        return jnp.sum(diagonal[:, None] + diagonal[None, :] - gram - gram.T)

    implicit_gradient = jax.grad(
        lambda values: effective_resistance(values, nodes, context=context).sum()
    )(surface64)
    dense_gradient = jax.grad(dense_loss)(surface64)
    np.testing.assert_allclose(implicit_gradient, dense_gradient, rtol=2e-6, atol=2e-6)


def test_tiny_amg_graph_uses_library_fallback() -> None:
    context = build_amg_context((2, 3), rtol=1e-8, atol=1e-8, max_steps=100)
    surface = jnp.linspace(0.8, 1.3, 6, dtype=jnp.float32).reshape(2, 3)
    nodes = jnp.asarray([0, 2, 5])

    observed = effective_resistance(surface, nodes, context=context)
    expected = dense_effective_resistance(np.asarray(surface), np.asarray(nodes))

    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-6)


def test_production_scale_float32_surface_uses_stable_mixed_precision() -> None:
    surface = jnp.linspace(0.658, 0.808, 41 * 43, dtype=jnp.float32).reshape(41, 43)
    nodes = jnp.asarray([0, 924])
    context = build_cg_context(surface.shape, rtol=1e-6, atol=1e-6, max_steps=1_000)

    loss_and_grad = eqx.filter_jit(
        jax.value_and_grad(
            lambda values: effective_resistance(values, nodes, context=context).sum()
        )
    )
    value, derivative = loss_and_grad(surface)

    assert value.dtype == jnp.float32
    assert derivative.dtype == jnp.float32
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(derivative)))
    assert float(jnp.linalg.norm(derivative)) > 0.0


def test_production_scale_amg_forward_and_adjoint_are_finite() -> None:
    surface = jnp.linspace(0.658, 0.808, 41 * 43, dtype=jnp.float32).reshape(41, 43)
    nodes = jnp.asarray([972, 740, 785, 698, 703, 1049, 791, 969, 845, 807, 1010, 1095, 1095, 970])
    context = build_amg_context(surface.shape)

    loss_and_grad = eqx.filter_jit(
        jax.value_and_grad(
            lambda values: effective_resistance(values, nodes, context=context).sum()
        )
    )
    value, derivative = loss_and_grad(surface)

    assert value.dtype == jnp.float32
    assert derivative.dtype == jnp.float32
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(derivative)))
    assert float(jnp.linalg.norm(derivative)) > 0.0


@pytest.mark.parametrize("use_amg", [False, True])
def test_training_context_preserves_configured_solver_options(use_amg: bool) -> None:
    model = initialise_model("resnet9", patch_size=32, seed=17)
    batch = SimpleNamespace(
        region="example",
        features=jnp.ones((64, 96, 14), dtype=jnp.float32),
    )
    config = SimpleNamespace(
        solver=SimpleNamespace(
            use_amg=use_amg,
            rtol=2e-5,
            atol=3e-7,
            max_steps=12_345,
        )
    )

    context = build_solver_contexts(model, [batch], config)["example"]

    assert context.graph_shape == (2, 3)
    assert isinstance(context.solver, AMJaxCGSolver if use_amg else lx.CG)
    assert context.solver.rtol == 2e-5
    assert context.solver.atol == 3e-7
    assert context.solver.max_steps == 12_345
    assert (context.solver_state is not None) == use_amg


def test_mixed_precision_solver_rejects_disabled_x64() -> None:
    with jax.enable_x64(False):
        surface = jnp.ones((2, 2), dtype=jnp.float32)
        with pytest.raises(RuntimeError, match="JAX_ENABLE_X64"):
            effective_resistance(surface, jnp.asarray([0, 1]))


def test_solver_rejects_fractional_iteration_limit() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_cg_context((2, 2), max_steps=1.5)


def test_solver_rejects_boolean_tolerance() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_cg_context((2, 2), rtol=True)
