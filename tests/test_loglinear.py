"""Independent numerical and offline workflow checks for the additive encoder."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from deepilg.models import ConductanceModel, LogLinearConductance
from deepilg.solver import build_cg_context


def test_loglinear_uses_patch_means_and_tree_reference_with_fourteen_parameters() -> None:
    # Tree is channel 1; unknown is channel 0 and remains estimable.
    features = np.zeros((2, 4, 14), dtype=np.float32)
    features[:, :2, 1] = 1
    features[:, 2:, 1] = 1
    features[0, 2:, 1] = 0
    features[0, 2:, 0] = 1
    features[:, :2, 12] = [[0, 1], [0, 1]]
    features[:, 2:, 12] = 1
    features[:, 2:, 13] = 0.25
    model = LogLinearConductance(patch_size=2)
    assert isinstance(model, ConductanceModel)
    assert sum(leaf.size for leaf in jax.tree.leaves(model)) == 14
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(model))
    np.testing.assert_allclose(model.conductance(features), [[1.01, 1.01]])
    coefficients = np.zeros(13, dtype=np.float32)
    coefficients[0] = 2  # Unknown class relative to tree cover.
    coefficients[-2:] = [0.4, -0.8]
    model = eqx.tree_at(lambda encoder: encoder.coefficients, model, jnp.asarray(coefficients))
    # Independent worked predictors: 0.4*0.5 and 2*0.5+0.4*1-0.8*0.25.
    expected = np.array([[1.231402758, 3.330116923]], dtype=np.float32)
    np.testing.assert_allclose(model.conductance(features), expected, rtol=2e-6)
    np.testing.assert_allclose(model.conductance(features, patch_batch_size=1), expected, rtol=2e-6)
    # Permuting pixels within a patch does not change the additive encoder.
    np.testing.assert_array_equal(model.conductance(features[::-1]), model.conductance(features))


def _dense_square_distances(conductances):
    """Independent four-vertex Laplacian pseudoinverse reference."""
    laplacian = np.zeros((4, 4), dtype=np.float64)
    for left, right in ((0, 1), (0, 2), (1, 3), (2, 3)):
        weight = (conductances[left] + conductances[right]) / 2
        laplacian[left, left] += weight
        laplacian[right, right] += weight
        laplacian[left, right] -= weight
        laplacian[right, left] -= weight
    inverse = np.linalg.pinv(laplacian)
    return np.diag(inverse)[:, None] + np.diag(inverse)[None, :] - 2 * inverse


def test_loglinear_resistance_and_gradients_match_independent_dense_calculation() -> None:
    features = np.zeros((2, 2, 14), dtype=np.float32)
    features[0, 0, 1] = features[0, 1, 0] = features[1, 0, 2] = features[1, 1, 8] = 1
    coefficients = np.zeros(13, dtype=np.float32)
    coefficients[[0, 1, 7]] = np.log([2, 3, 4])
    model = LogLinearConductance(patch_size=1)
    model = eqx.tree_at(
        lambda encoder: (encoder.intercept, encoder.coefficients),
        model,
        (jnp.asarray(np.log(0.99), dtype=jnp.float32), jnp.asarray(coefficients)),
    )
    context = build_cg_context((2, 2), rtol=1e-12, atol=1e-12)
    nodes = jnp.arange(4, dtype=jnp.int32)
    expected = _dense_square_distances([1.0, 1.99, 2.98, 3.97])
    actual = eqx.filter_jit(
        lambda encoder: encoder.predict_distances(features, nodes, context=context)
    )(model)
    assert actual.dtype == jnp.float32
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
    gradients = eqx.filter_grad(
        lambda encoder: encoder.predict_distances(features, nodes, context=context)[0, 3]
    )(model)
    for coefficient in (0, 1, 7):
        perturbed = coefficients.astype(np.float64)
        perturbed[coefficient] += 1e-4
        plus = 0.99 * np.exp([0, perturbed[0], perturbed[1], perturbed[7]]) + 0.01
        perturbed[coefficient] -= 2e-4
        minus = 0.99 * np.exp([0, perturbed[0], perturbed[1], perturbed[7]]) + 0.01
        expected_gradient = (
            _dense_square_distances(plus)[0, 3] - _dense_square_distances(minus)[0, 3]
        ) / 2e-4
        np.testing.assert_allclose(
            gradients.coefficients[coefficient], expected_gradient, rtol=2e-5
        )


def test_loglinear_bounds_have_finite_saturated_gradients() -> None:
    features = jnp.zeros((2, 2, 14), dtype=jnp.float32)
    model = LogLinearConductance(patch_size=1)
    for predictor, expected in ((1000, 1000), (-1000, 0.01)):
        saturated = eqx.tree_at(
            lambda encoder: encoder.intercept, model, jnp.asarray(predictor, dtype=jnp.float32)
        )
        np.testing.assert_allclose(saturated.conductance(features), expected, rtol=1e-6)
        gradient = eqx.filter_grad(lambda encoder: jnp.sum(encoder.conductance(features)))(
            saturated
        )
        assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradient))
