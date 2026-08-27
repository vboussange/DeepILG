# DeepILG

This repository contains the code accompanying the paper

> V. Boussange, P. Thiran, F. Missillier, and G. Dalle. *Learning Spatial
> Ecological Networks from Geolocated Molecular and Earth Observation Data*.
> 2026.

It provides the data-processing and modelling pipeline used to compare a
reference geographic distance-based model, ResistanceGA2, a deep patch encoder
with effective-resistance predictions, and a U-Net embedding-distance model.

## Content

- `src/deepilg/` contains the Python package and command-line interface.
- `configs/` contains the empirical benchmark and artificial-coordinate demo
  configurations.
- `data/wade2025/` contains the public genetic inputs and fixed
  population-holdout assignments.
- `data/templates/coordinates.csv` specifies the required format for the exact
  sampling coordinates, which are not redistributed.
- `resistancega/` contains the R entry point and locked R environment for the
  ResistanceGA2 comparison.
- `tests/` contains lightweight tests that do not download data or fit models.

Downloaded covariates, processed rasters, model checkpoints, predictions, and
figures are generated locally and are not included in the repository.

## Quick start

The demo combines the public genetic data with deterministic artificial
coordinates for the same five geographic regions. It exercises the pipeline,
but its outputs have no ecological meaning and cannot be exported as manuscript
figures.

Run the following commands from the repository root:

```bash
uv sync --locked
uv run deepilg validate --config configs/demo.toml
uv run deepilg download --config configs/demo.toml
uv run deepilg preprocess --config configs/demo.toml
uv run deepilg train --config configs/demo.toml --model reference --fold 0
uv run deepilg evaluate --config configs/demo.toml --model reference --fold 0
```

Downloads are cached under `cache/demo/`, and all other generated files are
written under `outputs/demo/`. Both directories are ignored by Git.

## Installation

DeepILG requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Install the
locked CPU environment with

```bash
uv sync --locked
```

On a supported Linux/x86-64 system with CUDA, install the GPU-enabled JAX
dependencies with

```bash
uv sync --locked --extra gpu
```

All Python commands are run as `uv run deepilg ...`; activating the virtual
environment is unnecessary. To check the installation, run

```bash
uv run deepilg --help
uv run pytest
```

Neural training is intended for a GPU. The reference model and unit tests run
on CPU. ResistanceGA2 runs separately in R and is described below.

## Data

The repository includes pairwise $F_{\mathrm{ST}}$ matrices for 68 wetland
salamander populations across five geographic regions, together with the five
population-holdout assignments used in the benchmark. These inputs come from
the [Wade et al. Dryad dataset](https://doi.org/10.5061/dryad.9kd51c5v6) and
are distributed under CC0 1.0. See `data/wade2025/README.md` for the dataset
citation.

Exact sampling coordinates are required for the empirical benchmark but are
not redistributed. After obtaining them from the authors of Wade et al., copy
the template and fill all 68 rows with WGS84 coordinates:

```bash
mkdir -p data/private/wade2025
cp data/templates/coordinates.csv data/private/wade2025/coordinates.csv
```

The required columns are `region,population_id,latitude,longitude`. The file is
ignored by Git, and `configs/benchmark.toml` does not fall back to the demo
coordinates when it is missing.

Given either the empirical or artificial coordinates, the pipeline downloads
[ESA WorldCover 2021](https://esa-worldcover.org/en/data-access),
[USGS 3DEP elevation](https://www.usgs.gov/3d-elevation-program), and
[OpenStreetMap roads](https://www.openstreetmap.org/copyright). It aligns these
products on a 30 m grid and constructs 14 input channels: 11 WorldCover
classes, an unknown-class indicator, elevation, and road density. Downloads
can take several minutes and require several gigabytes of local storage.

The Apache License 2.0 covers the software only. The bundled Wade et al. data
retain their CC0 status; downloaded WorldCover, 3DEP, and OpenStreetMap data
remain subject to their respective terms. OpenStreetMap data are ©
OpenStreetMap contributors and available under the
[ODbL](https://opendatacommons.org/licenses/odbl/).

## Benchmark

The four models are:

- `reference`: a calibrated geographic distance-based model;
- `resistancega`: the classical ILG pipeline implemented with ResistanceGA2;
- `resnet9`: the deep patch encoder, which predicts conductances and uses
  effective resistance as the population distance; and
- `unet`: the fully data-driven model, which predicts patch embeddings and
  compares populations by squared Euclidean distance.

First validate the private inputs, then download and preprocess the geospatial
covariates:

```bash
uv run deepilg validate --config configs/benchmark.toml
uv run deepilg download --config configs/benchmark.toml
uv run deepilg preprocess --config configs/benchmark.toml
```

Each command below handles one model and one population-holdout fold. For
example, to fit and evaluate fold 0:

```bash
uv run deepilg train --config configs/benchmark.toml --model reference --fold 0
uv run deepilg train --config configs/benchmark.toml --model resnet9 --fold 0
uv run deepilg train --config configs/benchmark.toml --model unet --fold 0

uv run deepilg evaluate --config configs/benchmark.toml --model reference --fold 0
uv run deepilg evaluate --config configs/benchmark.toml --model resnet9 --fold 0
uv run deepilg evaluate --config configs/benchmark.toml --model unet --fold 0
```

Repeat these commands for folds 1--4. Interrupted neural runs can be continued
by adding `--resume` to the corresponding `train` command.

ResistanceGA2 uses R 4.4 and a separate `renv` lockfile. Restore its packages,
export one fold, then fit and predict each region independently:

```bash
R -q -e 'install.packages("renv", repos="https://cloud.r-project.org")'
R -q -e 'renv::restore(project="resistancega", prompt=FALSE)'

uv run deepilg resistancega-export --config configs/benchmark.toml --fold 0
uv run deepilg resistancega --config configs/benchmark.toml \
  --fold 0 --region CWMA --mode fit
uv run deepilg resistancega --config configs/benchmark.toml \
  --fold 0 --region CWMA --mode predict --split test
```

Repeat the fit and prediction for all five regions, then evaluate the fold:

```bash
uv run deepilg evaluate --config configs/benchmark.toml \
  --model resistancega --fold 0
```

The ResistanceGA2 fits are CPU-intensive; further details are in
`resistancega/README.md`. Once all four models and five folds have been
evaluated, generate the benchmark figure with

```bash
uv run deepilg figures --config configs/benchmark.toml
```

Results are written under `outputs/benchmark/`. Held-out metrics are computed
on pairs containing test populations and summarized across the five folds.

## Citation and license

Please cite the accompanying paper and the Wade et al. dataset. Machine-readable
citation metadata are provided in `CITATION.cff`.

The source code is released under the [Apache License 2.0](LICENSE).
