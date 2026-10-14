"""Benchmark and resistance-surface figures."""

from __future__ import annotations

import csv
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from rasterio.plot import plotting_extent
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from scipy.stats import rankdata

from .config import ExperimentConfig
from .splits import load_fold_assignments
from .training import fit_directory, initialise_model, predicts_through_resistance_layer

# Displayed native-pixel windows (row start, row stop, column start, column stop).
# Regions without an entry are shown in full.
DISPLAY_WINDOWS = {
    "CWMA": (265, 1065, 297, 1097),
    "GSMNP": (280, 1080, 273, 1073),
    "ORR": (176, 976, 28, 828),
    "PCSF": (278, 1078, 59, 859),
    "RRG": (445, 1245, 102, 902),
}
LANDCOVER_CODES = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
LANDCOVER_LABELS = (
    "Unknown",
    "Tree cover",
    "Shrubland",
    "Grassland",
    "Cropland",
    "Built-up",
    "Bare / sparse",
    "Snow / ice",
    "Permanent water",
    "Herbaceous wetland",
    "Mangroves",
    "Moss / lichen",
)
LANDCOVER_COLORS = (
    "#D9D9D9",
    "#005A32",
    "#A6761D",
    "#F0E442",
    "#AA4499",
    "#D55E00",
    "#999999",
    "#FFFFFF",
    "#88CCEE",
    "#332288",
    "#44AA99",
    "#DDCC77",
)


def summarize_log10_surfaces(surfaces: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return the median and IQR of positive resistance in log10 space."""
    if not surfaces or any(np.any(np.asarray(surface) <= 0) for surface in surfaces):
        raise ValueError("Positive resistance surfaces are required")
    return summarize_surfaces(
        [np.log10(np.asarray(surface, dtype=np.float64)) for surface in surfaces]
    )


def upsample_resistance_surface(conductance: np.ndarray, *, shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly interpolate conductance to native pixels, then take its inverse."""
    values = np.asarray(conductance, dtype=np.float32)
    if values.ndim != 2 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Conductance must be a finite positive two-dimensional surface")
    native = jax.image.resize(jnp.asarray(values), shape, method="linear", antialias=False)
    return np.reciprocal(np.asarray(native, dtype=np.float64))


def summarize_surfaces(surfaces: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return the pixelwise median and interquartile range across folds."""
    if not surfaces:
        raise ValueError("At least one surface is required")
    stack = np.stack([np.asarray(surface, dtype=np.float64) for surface in surfaces])
    if not np.all(np.isfinite(stack)):
        raise ValueError("Surfaces must be finite")
    return np.median(stack, axis=0), np.percentile(stack, 75, axis=0) - np.percentile(
        stack, 25, axis=0
    )


FOLD_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")
FOLD_MARKERS = ("o", "s", "^", "D", "P")
# A fold value is far out beyond this many median absolute deviations (MADs) from
# its panel's median: for normal data, Tukey's far-out fences (three interquartile
# ranges beyond the quartiles, 4.7 standard deviations), but robust to up to half
# the values being extreme instead of a quarter.
FAR_OUT_MADS = 7.0
# The MAD is floored at this fraction of the panel's median magnitude, so that
# deviations tiny relative to the values themselves never break the axis.
MIN_MAD_FRACTION = 0.1
# Height of a segment beyond an axis break, relative to the bulk segment.
OUTLIER_SEGMENT_HEIGHT = 0.42
# Figure width in inches: the text width of the manuscript, so text prints at size.
BENCHMARK_FIGURE_WIDTH = 5.5


@dataclass(frozen=True)
class AxisSegment:
    """One y-range of a fold-metric panel, from ``low`` to ``high``."""

    low: float
    high: float
    log: bool = False

    def contains(self, value: float) -> bool:
        """Return whether the segment shows ``value``."""
        return self.low <= value <= self.high


@dataclass(frozen=True)
class FoldAxis:
    """A panel's y-axis: the bulk of the values, and far outliers beyond breaks."""

    bulk: AxisSegment
    below: AxisSegment | None = None
    above: AxisSegment | None = None

    @property
    def segments(self) -> tuple[AxisSegment, ...]:
        """Return the segments from top to bottom."""
        return tuple(
            segment for segment in (self.above, self.bulk, self.below) if segment is not None
        )


def fold_axis(values: Sequence[float]) -> FoldAxis:
    """Split a panel's y-axis so that far outliers neither flatten nor hide other values.

    The bulk segment spans, on a linear scale, the values within seven median
    absolute deviations (MADs) of the median, the robust counterpart of Tukey's
    far-out fences. The MAD is floored at a tenth of the median magnitude, so a
    panel of (nearly) equal values breaks only for a deviation of at least 70%
    of that magnitude. Values beyond the fences move past an axis break onto a
    segment of their own, logarithmic in magnitude when they share a sign and
    linear otherwise. Every value lies in exactly one segment, and segments do
    not overlap.
    """
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    spread = float(np.median(np.abs(array - median)))
    fence = FAR_OUT_MADS * max(spread, MIN_MAD_FRACTION * float(np.median(np.abs(array))))
    inside = array[np.abs(array - median) <= fence]
    margin = 0.08 * (float(np.ptp(inside)) or max(abs(float(inside[0])), 1.0))
    low, high = float(inside.min()) - margin, float(inside.max()) + margin
    below = above = None
    if (outliers := array[array < inside.min()]).size:
        # The break lies halfway between the outliers and the bulk.
        cut = (float(outliers.max()) + float(inside.min())) / 2
        low = max(low, cut)
        below = _outlier_segment(outliers, margin, high=cut)
    if (outliers := array[array > inside.max()]).size:
        cut = (float(outliers.min()) + float(inside.max())) / 2
        high = min(high, cut)
        above = _outlier_segment(outliers, margin, low=cut)
    return FoldAxis(AxisSegment(low, high), below, above)


def _outlier_segment(
    values: np.ndarray, margin: float, *, low: float = -np.inf, high: float = np.inf
) -> AxisSegment:
    """Span far outliers within ``[low, high]``: log magnitudes when they share a sign."""
    log = bool(np.all(values > 0) or np.all(values < 0))
    if log:
        magnitudes = np.abs(values)
        decades = float(np.log10(magnitudes.max() / magnitudes.min()))
        pad = 10.0 ** (0.12 * max(decades, 1.0))
        ends = np.sign(values[0]) * np.asarray([magnitudes.min() / pad, magnitudes.max() * pad])
    else:
        margin = max(0.08 * float(np.ptp(values)), margin)
        ends = np.asarray([values.min() - margin, values.max() + margin])
    return AxisSegment(max(float(ends.min()), low), min(float(ends.max()), high), log=log)


def _log_segment_ticks(segment: AxisSegment, min_gap: float) -> tuple[list[float], list[float]]:
    """Return the labelled and the minor ticks of a log segment.

    Labelled ticks use the finest of 1-9, 1-2-5, 1-3 and 1 multiples of powers of
    ten whose neighbours lie at least ``min_gap`` decades apart, thinning the
    decades when even those would crowd. Minor ticks mark the other 1-9 multiples.
    """
    sign = 1.0 if segment.low > 0 else -1.0
    smallest, largest = sorted((abs(segment.low), abs(segment.high)))
    exponents = range(int(np.floor(np.log10(smallest))), int(np.ceil(np.log10(largest))) + 1)

    def magnitudes(steps: Sequence[int], stride: int = 1) -> list[float]:
        return [
            step * 10.0**exponent
            for exponent in exponents
            if exponent % stride == 0
            for step in steps
            if smallest <= step * 10.0**exponent <= largest
        ]

    for steps in ((1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 5), (1, 3), (1,)):
        labelled = magnitudes(steps)
        if len(labelled) >= 2 and np.diff(np.log10(labelled)).min() >= min_gap:
            break
    else:
        labelled = magnitudes((1,), stride=max(1, int(np.ceil(min_gap))))
        # A segment narrower than one decade step: label its geometric centre.
        labelled = labelled or [float(f"{np.sqrt(smallest * largest):.2g}")]
    minor = [value for value in magnitudes(range(1, 10)) if value not in labelled]
    return [sign * value for value in labelled], [sign * value for value in minor]


def _log_tick_label(value: float, plain: bool) -> str:
    """Label a log-segment tick plainly, or as a power of ten when ``plain`` is false."""
    if plain:
        return f"{value:g}".replace("-", "\N{MINUS SIGN}")
    exponent = int(np.floor(np.log10(abs(value)) + 1e-9))
    mantissa = abs(value) / 10.0**exponent
    sign = "-" if value < 0 else ""
    coefficient = "" if np.isclose(mantissa, 1.0) else f"{mantissa:g}\\times"
    return f"${sign}{coefficient}10^{{{exponent}}}$"


def _style_log_segment(axis, segment: AxisSegment) -> None:
    """Give a segment beyond a break a logarithmic scale with readable magnitudes."""
    smallest, largest = sorted((abs(segment.low), abs(segment.high)))
    axis.set_yscale("symlog", linthresh=10.0 ** np.floor(np.log10(smallest)))
    height = axis.get_position().height * axis.figure.get_figheight() * 72
    points_per_decade = height / np.log10(largest / smallest)
    label_points = FontProperties(size=plt.rcParams["ytick.labelsize"]).get_size_in_points()
    # Labels stay at least one and a half label heights apart.
    labelled, minor = _log_segment_ticks(segment, 1.5 * label_points / points_per_decade)
    plain = all(1e-2 <= abs(tick) <= 1e3 for tick in labelled)
    axis.set_yticks(labelled, [_log_tick_label(tick, plain) for tick in labelled])
    # Minor ticks only where a decade is tall enough to tell them apart.
    axis.set_yticks(minor if points_per_decade >= 6 else [], minor=True)
    axis.tick_params(axis="y", which="minor", length=1.5, width=0.5)


def _tick_label(label: str) -> str:
    """Raise a trailing no-pooling dagger, which reads as a "t" at tick-label size."""
    return f"{label[:-1]}$^\\dagger$" if label.endswith("\N{DAGGER}") else label


BENCHMARK_FIGURE_STYLE = {
    "font.family": "sans-serif",
    "font.size": 6.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.3,
    "ytick.labelsize": 6.3,
    "legend.fontsize": 6.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def benchmark_figure(fold_metrics_csv: str | Path, output: str | Path) -> Path:
    """Save :func:`draw_benchmark_figure` for a fold-metrics CSV to ``output``."""
    with plt.rc_context(BENCHMARK_FIGURE_STYLE):
        figure = draw_benchmark_figure(fold_metrics_csv)
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, bbox_inches="tight", pad_inches=0.02)
        plt.close(figure)
    return destination


def draw_benchmark_figure(fold_metrics_csv: str | Path) -> Figure:
    """Draw one boxplot per treatment and metric, with one colored marker per fold.

    Treatments keep their order of appearance in the CSV, which needs ``model``,
    ``label``, ``fold``, ``pearson_r``, ``r2``, and ``rmse`` columns. A light
    vertical line separates adjacent treatments of which only one predicts through
    the effective-resistance layer. Far outliers of a panel appear at their value
    beyond an axis break (see :func:`fold_axis`).
    """
    with Path(fold_metrics_csv).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"model", "label", "fold", "pearson_r", "r2", "rmse"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Fold metrics must contain {sorted(required)}")
    models = {row["label"]: row["model"] for row in rows}
    labels = list(models)
    through_layer = [predicts_through_resistance_layer(models[label]) for label in labels]
    separators = [
        index + 0.5
        for index, (left, right) in enumerate(itertools.pairwise(through_layer))
        if left != right
    ]
    folds = sorted({int(row["fold"]) for row in rows})
    values = {(row["label"], int(row["fold"])): row for row in rows}
    if len(values) != len(rows) or len(values) != len(labels) * len(folds):
        raise ValueError("Fold metrics need exactly one row per treatment and fold")
    metrics = (("pearson_r", "Pearson's $r$"), ("r2", "$R^2$"), ("rmse", "RMSE"))
    with plt.rc_context(BENCHMARK_FIGURE_STYLE):
        positions = np.arange(len(labels), dtype=float)
        offsets = np.linspace(-0.34, 0.34, len(folds)) if len(folds) > 1 else np.zeros(1)
        figure = plt.figure(figsize=(BENCHMARK_FIGURE_WIDTH, 2.9))
        figure.patch.set_facecolor("white")
        panels = figure.add_gridspec(
            1, len(metrics), left=0.08, right=0.995, bottom=0.31, top=0.87, wspace=0.45
        )
        for column, (metric, ylabel) in enumerate(metrics):
            scores = {
                (label, fold): float(values[label, fold][metric])
                for label in labels
                for fold in folds
            }
            layout = fold_axis(list(scores.values()))
            segments = layout.segments
            grid = panels[0, column].subgridspec(
                len(segments),
                1,
                height_ratios=[
                    1.0 if segment is layout.bulk else OUTLIER_SEGMENT_HEIGHT
                    for segment in segments
                ],
                hspace=0.12,
            )
            for row, segment in enumerate(segments):
                axis = figure.add_subplot(grid[row, 0])
                if segment.log:
                    _style_log_segment(axis, segment)
                boxplot = axis.boxplot(
                    [[scores[label, fold] for fold in folds] for label in labels],
                    positions=positions,
                    widths=0.76,
                    patch_artist=True,
                    showfliers=False,
                    whis=(0, 100),
                    medianprops={"color": "#1F2933", "linewidth": 1.1},
                    whiskerprops={"color": "#64717D", "linewidth": 0.7},
                    capprops={"color": "#64717D", "linewidth": 0.7},
                    boxprops={"edgecolor": "#64717D", "linewidth": 0.7},
                )
                for box in boxplot["boxes"]:
                    box.set_facecolor("#EEF1F3")
                    box.set_alpha(0.78)
                for index, fold in enumerate(folds):
                    shown = [
                        (position + offsets[index], scores[label, fold])
                        for position, label in zip(positions, labels, strict=True)
                        if segment.contains(scores[label, fold])
                    ]
                    if shown:
                        axis.scatter(
                            *zip(*shown, strict=True),
                            marker=FOLD_MARKERS[index % len(FOLD_MARKERS)],
                            s=11,
                            facecolor=FOLD_COLORS[index % len(FOLD_COLORS)],
                            edgecolor="white",
                            linewidth=0.35,
                            zorder=3,
                            clip_on=False,
                        )
                if metric == "r2" and segment.contains(0.0):
                    axis.axhline(0.0, color="#7C8792", linewidth=0.7, linestyle="--", zorder=0)
                for separator in separators:
                    axis.axvline(separator, color="#C3CAD1", linewidth=0.7, zorder=0)
                axis.set_xlim(-0.5, len(labels) - 0.5)
                axis.set_ylim(segment.low, segment.high)
                axis.grid(axis="y", color="#D8DDE2", linewidth=0.5, alpha=0.9, zorder=0)
                axis.spines[["top", "right"]].set_visible(False)
                axis.spines[["left", "bottom"]].set_color("#64717D")
                axis.spines[["left", "bottom"]].set_linewidth(0.7)
                axis.tick_params(axis="y", width=0.7, length=2.5, pad=2)
                if row == len(segments) - 1:
                    axis.set_xticks(
                        positions,
                        [_tick_label(label) for label in labels],
                        rotation=48,
                        ha="right",
                        rotation_mode="anchor",
                    )
                    axis.tick_params(axis="x", length=0, pad=3)
                else:
                    axis.spines["bottom"].set_visible(False)
                    axis.set_xticks(positions)
                    axis.tick_params(axis="x", bottom=False, labelbottom=False)
                # Diagonal marks on the left spine show each break.
                for edge, present in ((1.0, row > 0), (0.0, row < len(segments) - 1)):
                    if present:
                        axis.plot(
                            [0.0],
                            [edge],
                            transform=axis.transAxes,
                            marker=[(-1.0, -0.6), (1.0, 0.6)],
                            markersize=5,
                            linestyle="none",
                            color="#64717D",
                            markeredgecolor="#64717D",
                            markeredgewidth=0.7,
                            clip_on=False,
                        )
                if segment is layout.bulk:
                    axis.set_ylabel(ylabel)
        figure.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=FOLD_COLORS[index % len(FOLD_COLORS)],
                    marker=FOLD_MARKERS[index % len(FOLD_MARKERS)],
                    markersize=4,
                    linestyle="None",
                    label=f"Fold {fold + 1}",
                )
                for index, fold in enumerate(folds)
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(folds),
            frameon=False,
            handletextpad=0.3,
            columnspacing=1.1,
        )
    return figure


@eqx.filter_jit
def _conductance_grid(model, features, patch_batch_size):
    return model.conductance(features, patch_batch_size=patch_batch_size)


def resistance_surface_figure(config: ExperimentConfig, output: str | Path) -> Path:
    """Map ResNet9 resistance across folds and export descriptive feature associations.

    Every fold's complete-pooling best-validation checkpoint predicts a patch
    conductance surface, which is bilinearly interpolated to native pixels and
    then inverted. The median and IQR across folds summarize log10 resistance.
    The maps outline one patch and show no sampling locations. Association CSVs
    are written next to the figure.
    """
    assignments = load_fold_assignments(config.data.folds)
    models = []
    for fold in assignments.fold_indices:
        source = (
            fit_directory(config, model_name="resnet9", fold=fold, pooling="complete") / "best.eqx"
        )
        if not source.is_file():
            raise FileNotFoundError(f"Missing ResNet9 checkpoint {source}")
        template = initialise_model(
            "resnet9", patch_size=config.preprocessing.patch_size, seed=config.training.seed
        )
        models.append(eqx.tree_deserialise_leaves(source, template))
    prepared = {}
    for region in assignments.regions:
        directory = config.data.processed_dir / region
        features = np.load(directory / "features.npy", mmap_mode="r")
        with rasterio.open(directory / "covariates.tif") as raster:
            transform = raster.transform
        surfaces = [
            upsample_resistance_surface(
                np.asarray(
                    _conductance_grid(
                        model, jnp.asarray(features), config.training.patch_batch_size
                    )
                ),
                shape=features.shape[:2],
            )
            for model in models
        ]
        median, iqr = summarize_log10_surfaces(surfaces)
        window = DISPLAY_WINDOWS.get(region, (0, features.shape[0], 0, features.shape[1]))
        r0, r1, c0, c1 = window
        if not (0 <= r0 < r1 <= features.shape[0] and 0 <= c0 < c1 <= features.shape[1]):
            raise ValueError(f"{region}: display window lies outside the native raster")
        prepared[region] = dict(
            features=features,
            landcover=np.argmax(features[..., : len(LANDCOVER_CODES)], axis=-1),
            transform=transform,
            window=window,
            median=median,
            iqr=iqr,
        )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _export_associations(prepared, destination.parent)
    _plot_maps(prepared, destination, patch_size=config.preprocessing.patch_size)
    return destination


def _sample_sd(values: Sequence[float]) -> float | str:
    return float(np.std(values, ddof=1)) if len(values) > 1 else ""


def _export_associations(prepared: dict, directory: Path) -> tuple[Path, Path]:
    """Summarize displayed median log10 resistance by land cover and continuous input.

    Each region contributes one contrast per class or input; summaries weight
    regions equally and do not treat raster pixels as replicates.
    """
    landcover, continuous = [], []
    for region, data in prepared.items():
        r0, r1, c0, c1 = data["window"]
        crop = (slice(r0, r1), slice(c0, c1))
        values = np.asarray(data["median"][crop], dtype=np.float64)
        classes = data["landcover"][crop]
        known = classes != 0
        if not known.any():
            raise ValueError(f"{region}: the displayed window has no known land cover")
        reference = float(np.median(values[known]))
        for index in np.unique(classes[known]):
            selected = classes == index
            landcover.append(
                dict(
                    class_index=int(index),
                    pixel_count=int(selected.sum()),
                    delta=float(np.median(values[selected])) - reference,
                )
            )
        for feature, channel in (("elevation", 12), ("road_density", 13)):
            predictor = np.asarray(data["features"][..., channel][crop], dtype=np.float64)
            if feature == "elevation":
                lower, upper = np.quantile(predictor, [0.25, 0.75])
                reference_mask, comparison_mask = predictor <= lower, predictor >= upper
            else:
                positive = predictor[predictor > 0]
                if positive.size == 0:
                    raise ValueError(f"{region}: road density has no positive pixels")
                upper = np.quantile(positive, 0.75)
                reference_mask, comparison_mask = predictor == 0, predictor >= upper
            if not reference_mask.any() or np.any(reference_mask & comparison_mask):
                raise ValueError(f"{region}: {feature} needs disjoint non-empty contrast groups")
            rho = np.corrcoef(rankdata(values.ravel()), rankdata(predictor.ravel()))[0, 1]
            continuous.append(
                dict(
                    feature=feature,
                    spearman_rho=float(rho),
                    delta=float(np.median(values[comparison_mask]))
                    - float(np.median(values[reference_mask])),
                )
            )
    total_pixels = sum(row["pixel_count"] for row in landcover)
    landcover_summary = []
    for index in sorted({row["class_index"] for row in landcover}):
        rows = [row for row in landcover if row["class_index"] == index]
        deltas = [row["delta"] for row in rows]
        pixels = sum(row["pixel_count"] for row in rows)
        landcover_summary.append(
            dict(
                worldcover_code=LANDCOVER_CODES[index],
                landcover_class=LANDCOVER_LABELS[index],
                pixel_count=pixels,
                area_percent=100 * pixels / total_pixels,
                region_count=len(rows),
                regions_higher_resistance=int(np.count_nonzero(np.asarray(deltas) > 0)),
                mean_delta_log10_resistance=float(np.mean(deltas)),
                sd_delta_log10_resistance=_sample_sd(deltas),
                relative_resistance=float(10.0 ** np.mean(deltas)),
            )
        )
    contrasts = dict(
        elevation="upper quartile minus lower quartile",
        road_density="upper quartile of positive density minus road-free",
    )
    continuous_summary = []
    for feature, contrast in contrasts.items():
        rows = [row for row in continuous if row["feature"] == feature]
        deltas = [row["delta"] for row in rows]
        rhos = [row["spearman_rho"] for row in rows]
        continuous_summary.append(
            dict(
                feature=feature,
                contrast=contrast,
                region_count=len(rows),
                mean_spearman_rho=float(np.mean(rhos)),
                sd_spearman_rho=_sample_sd(rhos),
                mean_delta_log10_resistance=float(np.mean(deltas)),
                sd_delta_log10_resistance=_sample_sd(deltas),
                relative_resistance=float(10.0 ** np.mean(deltas)),
                regions_higher_resistance=int(np.count_nonzero(np.asarray(deltas) > 0)),
            )
        )
    paths = (
        directory / "landcover_associations.csv",
        directory / "continuous_feature_associations.csv",
    )
    for path, rows in zip(paths, (landcover_summary, continuous_summary), strict=True):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return paths


def _plot_maps(prepared, destination, *, patch_size):
    regions = tuple(prepared)
    figure = plt.figure(figsize=(1.4 * len(regions) + 2.2, 4.3))
    grid = figure.add_gridspec(
        3,
        len(regions) + 3,
        width_ratios=(0.055, 0.26, *([1.0] * len(regions)), 1.15),
        left=0.025,
        right=0.99,
        bottom=0.025,
        top=0.985,
        hspace=0.06,
        wspace=0.04,
    )
    cmap = ListedColormap(LANDCOVER_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 12.5), cmap.N)
    limits = {}
    for key in ("median", "iqr"):
        pixels = np.concatenate(
            [
                data[key][
                    slice(data["window"][0], data["window"][1]),
                    slice(data["window"][2], data["window"][3]),
                ].ravel()
                for data in prepared.values()
            ]
        )
        lower, upper = np.quantile(pixels, [0.02, 0.98])
        if upper <= lower:
            lower, upper = lower - 0.01, upper + 0.01
        limits[key] = (lower, upper)
    images = {}
    present_classes = set()
    for column, region in enumerate(regions):
        data = prepared[region]
        r0, r1, c0, c1 = data["window"]
        crop = (slice(r0, r1), slice(c0, c1))
        transform = window_transform(Window(c0, r0, c1 - c0, r1 - r0), data["transform"])
        extent = plotting_extent(data["median"][crop], transform)
        present_classes.update(np.unique(data["landcover"][crop]).tolist())
        for row, key in enumerate(("landcover", "median", "iqr")):
            axis = figure.add_subplot(grid[row, column + 2])
            kwargs = (
                dict(cmap=cmap, norm=norm)
                if row == 0
                else dict(
                    cmap="magma" if row == 1 else "cividis",
                    vmin=limits[key][0],
                    vmax=limits[key][1],
                )
            )
            images[key] = axis.imshow(
                data[key][crop], extent=extent, origin="upper", interpolation="nearest", **kwargs
            )
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal")
            if row == 0:
                axis.text(
                    0.03,
                    0.97,
                    region,
                    transform=axis.transAxes,
                    va="top",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.5),
                )
                if column == 0:
                    pr = (r1 - patch_size) // patch_size * patch_size
                    pc = (c0 + patch_size - 1) // patch_size * patch_size
                    if pr >= r0 and pc + patch_size <= c1:
                        x, y = data["transform"] * (pc, pr + patch_size)
                        axis.add_patch(
                            Rectangle(
                                (x, y),
                                patch_size * data["transform"].a,
                                -patch_size * data["transform"].e,
                                fill=False,
                                edgecolor="white",
                                linewidth=0.8,
                            )
                        )
    for row, label in enumerate(("A", "B", "C")):
        axis = figure.add_subplot(grid[row, 1])
        axis.set_axis_off()
        axis.text(0.02, 0.98, label, va="top", fontsize=12, fontweight="bold")
        if row == 0:
            axis.text(0.75, 0.5, "Land cover", rotation=90, va="center", ha="center", fontsize=9.5)
        else:
            colorbar = figure.colorbar(
                images["median" if row == 1 else "iqr"],
                cax=figure.add_subplot(grid[row, 0]),
                label="Median $\\log_{10} Q$" if row == 1 else "IQR of $\\log_{10} Q$",
            )
            colorbar.ax.yaxis.set_ticks_position("left")
            colorbar.ax.yaxis.set_label_position("left")
            colorbar.ax.tick_params(labelsize=8.5)
            colorbar.ax.yaxis.label.set_size(9.5)
    legend = figure.add_subplot(grid[:, -1])
    legend.set_axis_off()
    legend.legend(
        handles=[
            Patch(facecolor=LANDCOVER_COLORS[i], label=LANDCOVER_LABELS[i])
            for i in sorted(present_classes)
        ],
        loc="center left",
        frameon=False,
        fontsize=9,
        title="Land cover",
        title_fontsize=9.5,
    )
    figure.savefig(
        destination, dpi=300, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(figure)
