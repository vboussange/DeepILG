"""Figures summarize resistance in log10 space from interpolated conductance."""

import numpy as np

from deepilg.figures import (
    fold_axis,
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


BULK = [0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.65, 0.7]


def test_far_outliers_move_beyond_breaks_onto_log_segments_without_dropping_values():
    values = [*BULK, -48.5, -803.1, 2.7, 7.2]
    axis = fold_axis(values)
    assert not axis.bulk.log and axis.below.log and axis.above.log
    assert [value for value in values if axis.bulk.contains(value)] == BULK
    assert [value for value in values if axis.below.contains(value)] == [-48.5, -803.1]
    assert [value for value in values if axis.above.contains(value)] == [2.7, 7.2]
    assert axis.below.high < axis.bulk.low < axis.bulk.high < axis.above.low


def test_without_far_outliers_one_linear_segment_shows_every_value():
    values = [*BULK, -0.2]
    axis = fold_axis(values)
    assert axis.below is None and axis.above is None and not axis.bulk.log
    assert all(axis.bulk.contains(value) for value in values)
