"""Figures summarize resistance in log10 space from interpolated conductance."""

import numpy as np

from deepilg.figures import (
    fold_axis_limits,
    summarize_log10_surfaces,
    upsample_resistance_surface,
)


def test_surface_ensemble_reports_median_and_iqr_in_log10_space():
    median, iqr = summarize_log10_surfaces(
        [np.full((2, 2), resistance) for resistance in (1, 10, 100, 1_000, 10_000)]
    )
    np.testing.assert_allclose(median, 2.0)
    np.testing.assert_allclose(iqr, 2.0)


def test_native_surface_interpolates_conductance_before_inversion():
    resistance = upsample_resistance_surface(np.asarray([[1.0, 3.0]]), shape=(1, 4))
    np.testing.assert_allclose(resistance, [[1.0, 2.0 / 3.0, 0.4, 1.0 / 3.0]], rtol=1e-6)


def test_fold_axis_limits_leave_only_far_outliers_off_scale():
    values = [0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.65, 0.7, -48.5, -803.1]
    low, high = fold_axis_limits(values)
    assert low > -48.5 and low < 0.2 and high > 0.7
    assert fold_axis_limits([0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.65, 0.7, -0.2]) is None
