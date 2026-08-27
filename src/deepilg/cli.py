"""Human-friendly command-line interface for the complete workflow."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from deepilg import __version__
from deepilg.config import ExperimentConfig, load_config
from deepilg.data import (
    load_coordinates,
    load_fst_directory,
    validate_population_alignment,
)
from deepilg.downloads import DownloadedProducts, download_public_products
from deepilg.evaluation import evaluate_model
from deepilg.figures import benchmark_figure, resistance_surface_figure
from deepilg.preprocessing import preprocess_downloaded_products
from deepilg.provenance import require_publication_safe_manifest
from deepilg.resistancega import REGIONS, export_resistancega_inputs, run_resistancega
from deepilg.splits import load_fold_assignments, validate_fold_assignments
from deepilg.training import fit_reference_model, train_neural_model


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
    train.add_argument("--model", choices=("reference", "resnet9", "unet"), required=True)
    train.add_argument("--fold", type=int, choices=range(5), required=True)
    train.add_argument("--resume", action="store_true", help="Resume a neural latest checkpoint.")

    evaluate = subcommands.add_parser(
        "evaluate", help="Evaluate one saved fit on held-out test pairs."
    )
    _config_argument(evaluate)
    evaluate.add_argument(
        "--model", choices=("reference", "resistancega", "resnet9", "unet"), required=True
    )
    evaluate.add_argument("--fold", type=int, choices=range(5), required=True)

    figures = subcommands.add_parser("figures", help="Generate the benchmark performance figure.")
    _config_argument(figures)
    figures.add_argument(
        "--predictions-dir",
        type=Path,
        help="Directory containing fold_metrics.csv (defaults to <output>/evaluation).",
    )
    figures.add_argument("--output", type=Path, help="Output image path.")
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
    resistancega_export.add_argument("--fold", type=int, choices=range(5), required=True)

    resistancega = subcommands.add_parser(
        "resistancega", help="Run one configured ResistanceGA2 fit or prediction."
    )
    _config_argument(resistancega)
    resistancega.add_argument("--fold", type=int, choices=range(5), required=True)
    resistancega.add_argument("--region", choices=REGIONS, required=True)
    resistancega.add_argument("--mode", choices=("fit", "predict"), required=True)
    resistancega.add_argument("--split", choices=("validation", "test"))
    resistancega.add_argument(
        "--rscript", default="Rscript", help="Rscript executable (default: Rscript)."
    )
    return parser


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="Path to a TOML configuration.")


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
    if arguments.command == "validate":
        summary = validate_inputs(config)
        print(json.dumps(summary, indent=2))
        if config.is_artificial:
            print(
                "\nWARNING: artificial coordinates are for execution testing only; "
                "derived results have no ecological interpretation."
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
            model, path = fit_reference_model(config, fold=arguments.fold)
            print(
                f"reference fold {arguments.fold}: slope={model.slope:.6g}, "
                f"intercept={model.intercept:.6g}; wrote {path}"
            )
        else:
            result = train_neural_model(
                config,
                model_name=arguments.model,
                fold=arguments.fold,
                resume=arguments.resume,
            )
            print(
                f"{result.model} fold {result.fold}: best validation loss "
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
        require_publication_safe_manifest(predictions_dir)
        if arguments.kind == "benchmark":
            source = predictions_dir / "fold_metrics.csv"
            destination = (
                arguments.output or config.project.output_dir / "figures" / "benchmark.pdf"
            )
            result = benchmark_figure(source, destination)
        else:
            destination = (
                arguments.output
                or config.project.output_dir / "figures" / "resistance_surfaces.pdf"
            )
            result = resistance_surface_figure(
                predictions_dir,
                config.data.processed_dir,
                destination,
                patch_size=config.preprocessing.patch_size,
            )
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
        )
        print(f"wrote {destination}")
        return 0
    raise AssertionError(f"Unhandled command: {arguments.command}")


def _cached_products(
    config: ExperimentConfig, regions: Sequence[str] | None
) -> tuple[DownloadedProducts, ...]:
    requested = set(regions) if regions else {"CWMA", "GSMNP", "ORR", "PCSF", "RRG"}
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
