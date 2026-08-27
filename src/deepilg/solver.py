"""Differentiable grounded effective resistance on conductance grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.experimental.sparse import BCOO
from jaxscape import GridGraph
from jaxscape.solvers import BCOOLinearOperator, batched_linear_solve
from jaxscape.utils import graph_laplacian


@dataclass(frozen=True)
class AMGContext:
    """Reusable AMJax algebraic-multigrid preconditioner.

    The hierarchy is constructed from the unit-conductance graph.  Each solve
    retains the current learned Laplacian as its operator; only the hierarchy
    is reused.
    """

    graph_shape: tuple[int, int]
    preconditioner: lx.AbstractLinearOperator
    rtol: float
    atol: float
    max_steps: int


def conductance_grid_graph(conductance: jax.Array) -> GridGraph:
    """Construct a four-neighbour graph with mean endpoint conductances."""
    surface = jnp.asarray(conductance)
    if surface.ndim != 2:
        raise ValueError(f"Conductance surface must be two-dimensional, got {surface.shape}")
    return GridGraph(surface, fun=lambda left, right: 0.5 * (left + right))


def grounded_laplacian(conductance: jax.Array) -> BCOO:
    """Return the Laplacian grounded at the last row-major graph vertex."""
    surface = jnp.asarray(conductance)
    if surface.size < 2:
        raise ValueError("Effective resistance requires at least two graph vertices")
    adjacency = conductance_grid_graph(surface).get_adjacency_matrix()
    return graph_laplacian(adjacency)[:-1, :-1]


def build_amg_context(
    graph_shape: tuple[int, int],
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    max_steps: int = 1_000,
    dtype: Any = jnp.float32,
) -> AMGContext:
    """Build an AMJax preconditioner from a unit-conductance graph.

    Hierarchy setup runs on the host and must therefore be called outside JIT
    and gradient transformations.  AMJax performs the V-cycle on JAX arrays;
    Lineax supplies CG and its implicit-differentiation rule.
    """
    if len(graph_shape) != 2 or any(int(size) <= 0 for size in graph_shape):
        raise ValueError(f"Invalid graph shape: {graph_shape}")
    if int(np.prod(graph_shape)) < 2:
        raise ValueError("AMG requires at least two graph vertices")
    if rtol < 0 or atol < 0 or max_steps <= 0:
        raise ValueError("CG tolerances must be non-negative and max_steps positive")

    try:
        import pyamg
        import scipy.sparse
        from amjax import MultilevelSolver
    except ImportError as error:
        raise ImportError(
            "AMG preconditioning requires the 'amjax' and 'pyamg' packages"
        ) from error

    template = grounded_laplacian(jnp.ones(graph_shape, dtype=dtype)).sum_duplicates()
    indices = np.asarray(template.indices)
    host_matrix = scipy.sparse.coo_matrix(
        (np.asarray(template.data), (indices[:, 0], indices[:, 1])),
        shape=template.shape,
    ).tocsr()
    host_hierarchy = pyamg.smoothed_aggregation_solver(host_matrix)
    hierarchy = MultilevelSolver.from_pyamg(
        host_hierarchy,
        presmoother=("jacobi", {"iterations": 1, "omega": 1.0}),
        postsmoother=("jacobi", {"iterations": 1, "omega": 1.0}),
        coarse_solver="pinv",
        dtype=dtype,
    )
    apply_v_cycle = hierarchy.aspreconditioner(cycle="V")
    input_structure = BCOOLinearOperator(template).in_structure()
    tags = (lx.positive_semidefinite_tag,)
    if bool(hierarchy.symmetric_smoothing):
        tags += (lx.symmetric_tag,)
    operator = lx.FunctionLinearOperator(
        lambda right_hand_side: apply_v_cycle(right_hand_side),
        input_structure,
        tags=tags,
    )
    return AMGContext(
        graph_shape=(int(graph_shape[0]), int(graph_shape[1])),
        preconditioner=operator,
        rtol=float(rtol),
        atol=float(atol),
        max_steps=int(max_steps),
    )


def effective_resistance(
    conductance: jax.Array,
    terminal_nodes: jax.Array,
    *,
    context: AMGContext | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    max_steps: int = 1_000,
) -> jax.Array:
    """Compute all pairwise terminal effective resistances.

    One grounded Laplacian system is solved per terminal.  The final graph
    vertex is the reference vertex; its right-hand side is represented by the
    zero vector, so terminals are allowed to coincide with it.
    """
    surface = jnp.asarray(conductance)
    nodes = jnp.asarray(terminal_nodes, dtype=jnp.int32)
    if nodes.ndim != 1:
        raise ValueError(f"terminal_nodes must be one-dimensional, got {nodes.shape}")
    if nodes.shape[0] == 0:
        raise ValueError("At least one terminal node is required")
    if context is not None and tuple(surface.shape) != context.graph_shape:
        raise ValueError(f"AMG context shape {context.graph_shape} does not match {surface.shape}")

    laplacian = grounded_laplacian(surface)
    right_hand_sides = jax.nn.one_hot(nodes, laplacian.shape[0], dtype=laplacian.dtype).T
    if context is None:
        solver = lx.CG(rtol=rtol, atol=atol, max_steps=max_steps)
        potentials = batched_linear_solve(laplacian, right_hand_sides, solver)
    else:
        solver = lx.CG(
            rtol=context.rtol,
            atol=context.atol,
            max_steps=context.max_steps,
        )
        potentials = batched_linear_solve(
            laplacian,
            right_hand_sides,
            solver,
            options={"preconditioner": context.preconditioner},
        )

    gram = right_hand_sides.T @ potentials
    diagonal = jnp.diag(gram)
    resistance = diagonal[:, None] + diagonal[None, :] - gram - gram.T
    resistance = 0.5 * (resistance + resistance.T)
    diagonal_indices = jnp.diag_indices(resistance.shape[0])
    return resistance.at[diagonal_indices].set(0.0)


def dense_effective_resistance(conductance: np.ndarray, terminal_nodes: np.ndarray) -> np.ndarray:
    """Small-graph reference implementation used for numerical verification."""
    surface = jnp.asarray(conductance, dtype=jnp.float64)
    laplacian = np.asarray(grounded_laplacian(surface).todense())
    nodes = np.asarray(terminal_nodes, dtype=np.int64)
    basis = np.asarray(jax.nn.one_hot(nodes, laplacian.shape[0], dtype=surface.dtype)).T
    potentials = np.linalg.solve(laplacian, basis)
    gram = basis.T @ potentials
    diagonal = np.diag(gram)
    result = diagonal[:, None] + diagonal[None, :] - gram - gram.T
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)
    return result
