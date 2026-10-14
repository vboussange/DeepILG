"""Figures summarize resistance in log10 space and show every fold at its value."""

import csv
import itertools

import matplotlib.pyplot as plt
import numpy as np

from deepilg.figures import (
    draw_benchmark_figure,
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


def test_a_quarter_of_far_outliers_still_leaves_the_bulk_on_its_own_segment():
    bulk = [0.2 + 0.03 * index for index in range(18)]
    extremes = [-500.0, -800.0, -1200.0, -6000.0, -9000.0, -80000.0]
    axis = fold_axis([*bulk, *extremes])
    assert [value for value in [*bulk, *extremes] if axis.bulk.contains(value)] == bulk
    assert [value for value in [*bulk, *extremes] if axis.below.contains(value)] == extremes


def test_moderately_low_folds_stay_on_the_bulk_beside_extremes():
    # The benchmark's R² fold values, rounded: only the four failed U-Net folds are extreme.
    values = [
        *(0.28, 0.23, 0.32, 0.32, -0.24, -0.27, 0.49, 0.51, 0.75, 0.66),
        *(0.45, -935.23, -1.05, 0.2, -48.44, 0.32, -6548.92, 0.1, -1.13, -819.45),
        *(-0.01, 0.32, 0.54, 0.56, 0.45, 0.6, 0.61, 0.77, 0.78, 0.81),
        *(0.81, 0.47, 0.84, 0.42, 0.82, 0.78, 0.53, 0.66, 0.48, 0.83),
    ]
    axis = fold_axis(values)
    assert [value for value in values if not axis.bulk.contains(value)] == [
        -935.23,
        -48.44,
        -6548.92,
        -819.45,
    ]


def test_a_small_deviation_from_equal_values_breaks_no_axis():
    values = [0.5] * 39 + [0.6]
    axis = fold_axis(values)
    assert axis.below is None and axis.above is None
    assert all(axis.bulk.contains(value) for value in values)


def test_fold_figure_shows_extremes_on_readable_log_segments_without_value_labels(tmp_path):
    scores = {
        "reference": ([0.6, 0.65, 0.7, 0.72, 0.75], [0.2, 0.25, 0.3, 0.28, 0.22], [0.05] * 5),
        "unet": (
            [0.5, 0.45, 0.4, 0.55, 0.5],
            [0.3, -900.0, 0.1, 0.2, -50.0],
            [0.04, 2.7] * 2 + [0.36],
        ),
        "loglinear": ([0.7, 0.8, 0.75, 0.78, 0.82], [0.5, 0.6, 0.55, 0.45, 0.4], [0.03] * 5),
    }
    path = tmp_path / "fold_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "label", "fold", "pearson_r", "r2", "rmse"])
        for model, columns in scores.items():
            writer.writerows(
                [model, model, fold, *row] for fold, row in enumerate(zip(*columns, strict=True))
            )

    figure = draw_benchmark_figure(path)
    figure.canvas.draw()
    panels = [
        sorted(axes, key=lambda axis: -axis.get_position().y0)
        for _, axes in itertools.groupby(
            sorted(figure.axes, key=lambda axis: axis.get_position().x0),
            key=lambda axis: round(axis.get_position().x0, 3),
        )
    ]
    (correlation,), (r2_bulk, r2_low), (rmse_high, rmse_bulk) = panels
    assert [axis.get_yscale() for axis in (correlation, r2_bulk, rmse_bulk)] == ["linear"] * 3
    for axis, extremes in ((r2_low, (-900.0, -50.0)), (rmse_high, (2.7, 0.36))):
        assert axis.get_yscale() == "symlog"
        low, high = axis.get_ylim()
        assert all(low < value < high for value in extremes)
        assert axis.yaxis.get_minorticklocs().size > 0
        labels = [label for label in axis.get_yticklabels() if label.get_text()]
        assert len(labels) >= 2
        boxes = [label.get_window_extent() for label in labels]
        assert not any(a.overlaps(b) for a, b in itertools.combinations(boxes, 2))
    assert not [text for axis in figure.axes for text in axis.texts]
    plt.close(figure)
