"""Human-friendly command-line interface for the complete workflow."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from . import __version__
from .config import POOLING, ExperimentConfig, load_config
from .data import (
    load_coordinates,
    load_fst_directory,
    validate_population_alignment,
)
from .downloads import DownloadedProducts, download_public_products
from .evaluation import evaluate_model
from .figures import resistance_surface_figure
from .preprocessing import preprocess_downloaded_products
from .provenance import require_publication_safe_manifest
from .reporting import generate_benchmark_report
from .resistancega import export_resistancega_inputs, run_resistancega
from .splits import (
    generate_fold_assignments,
    load_fold_assignments,
    validate_fold_assignments,
    write_fold_assignments,
)
from .training import fit_reference_model, train_neural_model


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser without touching data or accelerators."""
    parser = argparse.ArgumentParser(
        prog="deepilg",
        description="Differentiable inverse landscape genetics research pipeline.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate", help="Validate configuration and light inputs.")
    _config_argument(validate)

    folds = subcommands.add_parser(
        "folds", help="Draw population-holdout folds from the configured split seed."
    )
    _config_argument(folds)

    download = subcommands.add_parser("download", help="Download public geospatial products.")
    _config_argument(download)
    download.add_argument("--region", action="append", help="Download one region (repeatable).")
    download.add_argument("--overwrite", action="store_true", help="Replace cached source files.")

    preprocess = subcommands.add_parser(
        "preprocess", help="Build aligned 14-channel raster inputs."
    )
    _config_argument(preprocess)
    preprocess.add_argument("--region", action="append", help="Process one region (repeatable).")
    preprocess.add_argument("--overwrite", action="store_true", help="Replace processed files.")

    train = subcommands.add_parser("train", help="Fit one model for one population-holdout fold.")
    _config_argument(train)
    train.add_argument(
        "--model", choices=("reference", "loglinear", "resnet9", "unet"), required=True
    )
    train.add_argument("--fold", type=_fold, required=True, help="Population-holdout fold index.")
    _pooling_argument(train)
    train.add_argument(
        "--region", help="Fit only this region (no pooling; default: every region in turn)."
    )
    train.add_argument("--resume", action="store_true", help="Resume a neural latest checkpoint.")

    evaluate = subcommands.add_parser(
        "evaluate", help="Score the saved fits of one model and fold on held-out test pairs."
    )
    _config_argument(evaluate)
    evaluate.add_argument(
        "--model",
        choices=("reference", "resistancega", "loglinear", "resnet9", "unet"),
        required=True,
    )
    evaluate.add_argument(
        "--fold", type=_fold, required=True, help="Population-holdout fold index."
    )
    _pooling_argument(evaluate)

    figures = subcommands.add_parser(
        "figures", help="Report every evaluated treatment, or map resistance surfaces."
    )
    _config_argument(figures)
    figures.add_argument(
        "--predictions-dir",
        type=Path,
        help="Evaluation directory to report (defaults to <output>/evaluation).",
    )
    figures.add_argument(
        "--output", type=Path, help="Figure path; its CSV tables are written alongside."
    )
    figures.add_argument(
        "--kind",
        choices=("benchmark", "resistance"),
        default="benchmark",
        help="Figure to generate (default: benchmark).",
    )

    resistancega_export = subcommands.add_parser(
        "resistancega-export", help="Export one fold for the accompanying R workflow."
    )
    _config_argument(resistancega_export)
    resistancega_export.add_argument(
        "--fold", type=_fold, required=True, help="Population-holdout fold index."
    )

    resistancega = subcommands.add_parser(
        "resistancega", help="Run one configured ResistanceGA2 fit or prediction."
    )
    _config_argument(resistancega)
    resistancega.add_argument(
        "--fold", type=_fold, required=True, help="Population-holdout fold index."
    )
    resistancega.add_argument("--region", required=True, help="One region present in the dataset.")
    resistancega.add_argument("--mode", choices=("fit", "predict"), required=True)
    resistancega.add_argument("--split", choices=("validation", "test"))
    resistancega.add_argument(
        "--rscript", default="Rscript", help="Rscript executable (default: Rscript)."
    )
    resistancega.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="CPU workers for the genetic algorithm's fitness evaluations (default: 1).",
    )
    return parser


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="Path to a TOML configuration.")


def _pooling_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pooling",
        choices=POOLING,
        help="Override [training].pooling: one model across regions or one per region.",
    )


def _fold(value: str) -> int:
    """Parse a non-negative fold index; existence is checked against the folds file."""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("fold must be an integer") from error
    if number < 0:
        raise argparse.ArgumentTypeError("fold must be a non-negative integer")
    return number


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def validate_inputs(config: ExperimentConfig) -> dict[str, object]:
    """Validate all lightweight inputs and return a printable summary."""
    matrices = load_fst_directory(config.data.fst_dir)
    coordinates = load_coordinates(config.data.coordinates)
    validate_population_alignment(matrices, coordinates)
    folds = load_fold_assignments(config.data.folds)
    validate_fold_assignments(
        folds,
        {region: matrix.population_ids for region, matrix in matrices.items()},
    )
    return {
        "configuration": str(config.source),
        "geolocation_status": config.data.geolocation_status,
        "scientific_use": config.data.scientific_use,
        "regions": {region: matrix.n_populations for region, matrix in matrices.items()},
        "population_total": sum(matrix.n_populations for matrix in matrices.values()),
        "status": "valid",
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run one CLI command and return a process exit status."""
    arguments = build_parser().parse_args(argv)
    config = load_config(arguments.config)
    pooling = getattr(arguments, "pooling", None)
    if pooling is not None:
        if arguments.model == "resistancega" and pooling != "none":
            raise ValueError("ResistanceGA fits each region separately (no pooling)")
        config = replace(config, training=replace(config.training, pooling=pooling))
    if arguments.command == "validate":
        summary = validate_inputs(config)
        print(json.dumps(summary, indent=2))
        if config.is_artificial:
            print(
                "\nWARNING: artificial coordinates are for execution testing only; "
                "derived results have no ecological interpretation."
            )
        return 0

    if arguments.command == "folds":
        matrices = load_fst_directory(config.data.fst_dir)
        folds = generate_fold_assignments(
            {region: matrix.population_ids for region, matrix in matrices.items()},
            seed=config.data.split_seed,
        )
        destination = write_fold_assignments(folds, config.data.folds)
        print(
            f"wrote {len(folds.fold_indices)} folds for {len(folds.regions)} regions "
            f"(split_seed={config.data.split_seed}) to {destination}"
        )
        return 0

    if arguments.command == "download":
        products = download_public_products(
            config, regions=arguments.region, overwrite=arguments.overwrite
        )
        for product in products:
            print(f"{product.region}: cached public products in {product.dem.parent}")
        return 0

    if arguments.command == "preprocess":
        products = _cached_products(config, arguments.region)
        outputs = preprocess_downloaded_products(config, products, overwrite=arguments.overwrite)
        for output in outputs:
            print(f"{output.region}: wrote {output.features}")
        return 0

    if arguments.command == "train":
        if arguments.model == "reference":
            if arguments.resume:
                raise ValueError("--resume applies only to neural models")
            if arguments.region is not None and config.training.pooling == "complete":
                raise ValueError("A single region can be fitted only under no pooling")
            model, path = fit_reference_model(config, fold=arguments.fold)
            print(
                f"reference fold {arguments.fold}: slope={model.slope:.6g}, "
                f"intercept={model.intercept:.6g}; wrote {path}"
            )
            return 0
        results = train_neural_model(
            config,
            model_name=arguments.model,
            fold=arguments.fold,
            resume=arguments.resume,
            region=arguments.region,
        )
        for result in results:
            suffix = f" {result.region}" if result.region else ""
            print(
                f"{result.model} fold {result.fold}{suffix}: best validation loss "
                f"{result.best_validation_loss:.6g} at epoch {result.best_epoch}; "
                f"wrote {result.checkpoint}"
            )
        return 0

    if arguments.command == "evaluate":
        metrics, predictions = evaluate_model(
            config, model_name=arguments.model, fold=arguments.fold
        )
        print(
            f"{arguments.model} fold {arguments.fold}: Pearson r={metrics.pearson_r:.3f}, "
            f"R2={metrics.r2:.3f}, RMSE={metrics.rmse:.4f}, n={metrics.n_pairs}; "
            f"wrote {predictions}"
        )
        return 0

    if arguments.command == "figures":
        config.assert_publication_safe()
        predictions_dir = arguments.predictions_dir or config.project.output_dir / "evaluation"
        if arguments.kind == "benchmark":
            require_publication_safe_manifest(predictions_dir)
            result = generate_benchmark_report(
                config, evaluation_dir=predictions_dir, figure=arguments.output
            )
        else:
            destination = (
                arguments.output
                or config.project.output_dir / "figures" / "resistance_surfaces.pdf"
            )
            result = resistance_surface_figure(config, destination)
        print(f"wrote {result}")
        return 0

    if arguments.command == "resistancega-export":
        destination = export_resistancega_inputs(config, fold=arguments.fold)
        print(f"wrote ResistanceGA2 inputs to {destination}")
        return 0
    if arguments.command == "resistancega":
        destination = run_resistancega(
            config,
            fold=arguments.fold,
            region=arguments.region,
            mode=arguments.mode,
            split=arguments.split,
            rscript=arguments.rscript,
            workers=arguments.workers,
        )
        print(f"wrote {destination}")
        return 0
    raise AssertionError(f"Unhandled command: {arguments.command}")


def _cached_products(
    config: ExperimentConfig, regions: Sequence[str] | None
) -> tuple[DownloadedProducts, ...]:
    if regions:
        requested = set(regions)
    elif config.data.cache_dir.is_dir():
        requested = {path.name for path in config.data.cache_dir.iterdir() if path.is_dir()}
    else:
        requested = set()
    if not requested:
        raise FileNotFoundError(
            f"No cached regions found under {config.data.cache_dir}; run `deepilg download` first"
        )
    products = []
    for region in sorted(requested):
        directory = config.data.cache_dir / region
        dem = directory / "usgs_3dep.tif"
        roads = directory / "osm_roads.geojson"
        worldcover = tuple(sorted(directory.glob("worldcover_[0-9][0-9].tif")))
        missing = [path for path in (dem, roads) if not path.is_file()]
        if missing or not worldcover:
            raise FileNotFoundError(
                f"{region}: cached products are incomplete; run `deepilg download` first"
            )
        products.append(DownloadedProducts(region, dem, worldcover, roads))
    return tuple(products)
