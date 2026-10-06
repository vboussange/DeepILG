"""Differentiable grounded effective resistance on conductance grids."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
import lineax as lx
from jaxscape import GridGraph, ResistanceDistance
from jaxscape.solvers import AMJaxCGSolver, AMJaxCGSolverState

from .config import SolverConfig

SOLVE_DTYPE = jnp.float64


@dataclass(frozen=True)
class AMGContext:
    """Shape-specific solver and optional reusable JAXScape preconditioner state."""

    graph_shape: tuple[int, int]
    solver: lx.AbstractLinearSolver
    solver_state: AMJaxCGSolverState | None = None


def _mean_conductance(left: jax.Array, right: jax.Array) -> jax.Array:
    """Use a stable edge callable so JAXScape can reuse compiled graph solves."""
    return 0.5 * (left + right)


def conductance_grid_graph(conductance: jax.Array) -> GridGraph:
    """Construct a four-neighbour graph with mean endpoint conductances."""
    surface = jnp.asarray(conductance)
    if surface.ndim != 2:
        raise ValueError(f"Conductance surface must be two-dimensional, got {surface.shape}")
    if surface.size < 2:
        raise ValueError("Effective resistance requires at least two graph vertices")
    return GridGraph(surface, fun=_mean_conductance)


def build_amg_context(
    graph_shape: tuple[int, int],
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    max_steps: int = 1_000,
) -> AMGContext:
    """Build a float64 AMJax-preconditioned CG context for a graph shape.

    Hierarchy setup runs on the host and must therefore be
    called outside JIT and gradient transformations. The hierarchy is reused
    only as a preconditioner; every solve retains the learned Laplacian as its
    differentiable operator.
    """
    _require_x64()
    shape = _validate_graph_shape(graph_shape)
    SolverConfig(rtol=rtol, atol=atol, max_steps=max_steps).validate()
    template = conductance_grid_graph(jnp.ones(shape, dtype=SOLVE_DTYPE))
    solver = AMJaxCGSolver(rtol=rtol, atol=atol, max_steps=max_steps, coarse_solver="pinv")
    solver_state = ResistanceDistance(solver=solver).init_preconditioner(template)
    return AMGContext(
        graph_shape=shape,
        solver=solver,
        solver_state=solver_state,
    )


def build_cg_context(
    graph_shape: tuple[int, int],
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    max_steps: int = 1_000,
) -> AMGContext:
    """Carry configured tolerances to the unpreconditioned float64 CG path."""
    _require_x64()
    shape = _validate_graph_shape(graph_shape)
    SolverConfig(rtol=rtol, atol=atol, max_steps=max_steps, use_amg=False).validate()
    return AMGContext(
        graph_shape=shape,
        solver=lx.CG(rtol=rtol, atol=atol, max_steps=max_steps),
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
    """Compute all pairwise terminal effective resistances in float64.

    One grounded Laplacian system is solved per terminal.  The final graph
    vertex is the reference vertex; its right-hand side is represented by the
    zero vector, so terminals are allowed to coincide with it. The conductance
    model may remain float32: its surface is promoted at the solve boundary and
    the resulting distance matrix is cast back to the input dtype.
    """
    _require_x64()
    input_surface = jnp.asarray(conductance)
    if not jnp.issubdtype(input_surface.dtype, jnp.floating):
        raise TypeError("conductance must have a real floating dtype")
    output_dtype = input_surface.dtype
    surface = jnp.asarray(input_surface, dtype=SOLVE_DTYPE)
    nodes = jnp.asarray(terminal_nodes, dtype=jnp.int32)
    if nodes.ndim != 1:
        raise ValueError(f"terminal_nodes must be one-dimensional, got {nodes.shape}")
    if nodes.shape[0] == 0:
        raise ValueError("At least one terminal node is required")
    if context is not None and tuple(surface.shape) != context.graph_shape:
        raise ValueError(
            f"Solver context shape {context.graph_shape} does not match {surface.shape}"
        )

    if context is None:
        context = build_cg_context(surface.shape, rtol=rtol, atol=atol, max_steps=max_steps)
    graph = conductance_grid_graph(surface)
    resistance = ResistanceDistance(solver=context.solver).nodes_to_nodes_distance(
        graph, nodes, state=context.solver_state
    )
    diagonal_indices = jnp.diag_indices(resistance.shape[0])
    resistance = resistance.at[diagonal_indices].set(0.0)
    return jnp.asarray(resistance, dtype=output_dtype)


def _require_x64() -> None:
    """Reject silent float64 truncation before constructing solver arrays."""
    if not bool(jax.config.x64_enabled):
        raise RuntimeError(
            "Effective-resistance solves require float64. Set JAX_ENABLE_X64=true "
            "before starting Python."
        )


def _validate_graph_shape(graph_shape: tuple[int, int]) -> tuple[int, int]:
    if len(graph_shape) != 2 or any(
        isinstance(size, bool) or not isinstance(size, Integral) or size <= 0
        for size in graph_shape
    ):
        raise ValueError(f"Invalid graph shape: {graph_shape}")
    shape = (int(graph_shape[0]), int(graph_shape[1]))
    if shape[0] * shape[1] < 2:
        raise ValueError("Effective resistance requires at least two graph vertices")
    return shape
