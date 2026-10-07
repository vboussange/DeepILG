# Deep inverse landscape genetics

<img align="right" width="380" src="deep_patch_encoder.png" alt="Deep patch encoder: a ResNet maps a landscape feature patch to a conductance; the resulting grid graph defines effective resistance between populations.">

This repository exposes a [JAX](https://docs.jax.dev/en/latest/)-based pipeline for *inverse landscape genetics*. It heavily relies on [JAXScape](https://github.com/vboussange/jaxscape), which implements a GPU-accelerated, differentiable resistance layer. This allows learning deep geospatial encoders mapping Earth-observation data to a *conductance* surface characterising how the landscape influences the gene flow of spatially structured ecological populations from genetic data. The pipeline in this repo trains a deep encoder by matching the effective resistance on the induced grid graph to an empirical genetic dissimilarity matrix.

This is a companion repository for the paper

> V. Boussange, P. Thiran, F. Missillier, and G. Dalle. *Learning Spatial Ecological Networks from Geolocated Molecular and Earth Observation Data*. 2026.

## Installation

This directory is a self-contained Python project. Run the commands below from
this directory; no module or `PYTHONPATH` entry from a parent repository is
required.

```bash
uv sync --locked            # append --extra gpu on Linux/x86-64 with CUDA
uv run deepilg --help
JAX_ENABLE_X64=true uv run pytest
```

The locked GPU extra pins cuDNN 9.8.0.87 for compatibility with the manuscript
V100/driver-535 cluster. ResNet9 keeps its neural network in float32 but solves
effective resistance in float64 for numerical stability. Enable JAX float64 on
each ResNet9 training or evaluation command:

```bash
JAX_ENABLE_X64=true uv run deepilg train --config configs/demo.toml --model resnet9 --fold 0
```

AMG preconditioning and differentiable linear solves are provided by
JAXScape 0.0.10 with its AMJax extra; DeepILG does not maintain a separate solver
implementation. The configured tolerances and iteration limit apply with or
without AMG.

## Quick start

```bash
uv run deepilg validate   --config configs/demo.toml
uv run deepilg download   --config configs/demo.toml
uv run deepilg preprocess --config configs/demo.toml
JAX_ENABLE_X64=true uv run deepilg train    --config configs/demo.toml --model resnet9 --fold 0
JAX_ENABLE_X64=true uv run deepilg evaluate --config configs/demo.toml --model resnet9 --fold 0
```

`configs/demo.toml` contains default configuration settings to run the whole pipeline on a synthetic dataset. Its outputs carry no ecological meaning; files land under `cache/demo/` and `outputs/demo/`.

## The inverse landscape genetics pipeline in brief

A dataset is organised into geographic *regions* — spatially distinct landscapes, each with its own sampled populations. The pipeline consumes two inputs per region: *labels* — one pairwise genetic-dissimilarity matrix (e.g., Fixation index $F_{ST}$) between the sampled populations — and *features* — geospatial covariates over the landscape together with the population coordinates that locate them on the raster. The commands below turn these into a fitted encoder and its held-out scores.

| Command | Role |
| --- | --- |
| `validate` | Check that the genetic, coordinate, and fold inputs align. |
| `download` | Fetch the public geospatial covariates per region. |
| `preprocess` | Align covariates on a common grid and build the raster inputs. |
| `train` | Fit one model for one population-holdout fold; add `--resume` to continue. |
| `evaluate` | Score a saved fit on held-out test pairs and export tidy predictions. |
| `figures` | Render the benchmark and resistance-surface figures. |

Training minimises the objective `region_balanced_log1p_mse` (`loss.py`) across all regions jointly, under population-holdout cross-validation. Models are selectable with `--model`, all evaluated on identical folds, pairs, and metrics:

- **`resnet9`** — the *conductance model* and the object of interest: predicts a positive conductance per landscape patch, then effective resistances between populations.
- **`unet`** — an *embedding-distance* reference: predicts a patch embedding per node and compares populations by squared Euclidean distance, with no resistance layer.
- **`reference`** — a calibrated non-negative affine model of geographic distance.

The [benchmark](#benchmark) additionally compares these against ResistanceGA2, an external classical baseline.

## Adapting to your own dataset

The region set and fold count are read from the data, so a new study needs three tables and a configuration; copy `configs/benchmark.toml` and redirect the `[data]` paths. Genetic inputs are one labelled square distance matrix per region, discovered from any `<REGION>_FST.csv` under `fst_dir` (any number of regions). Sampling locations go in one `coordinates` table (`region,population_id,latitude,longitude`, WGS84). Cross-validation is a `folds` table (`region,fold,population_id,partition`) with any number of folds indexed contiguously from zero, each population held out for testing once. `deepilg validate` reports any mismatch.

`download` and `preprocess` build a 14-channel raster per region (one-hot [WorldCover 2021](https://esa-worldcover.org/en/data-access) classes, elevation, road density) from WorldCover, [USGS 3DEP](https://www.usgs.gov/3d-elevation-program), and [OpenStreetMap](https://www.openstreetmap.org/copyright). WorldCover and OpenStreetMap are global; USGS 3DEP is US-only, so studies elsewhere should point the elevation source in `downloads.py` at a suitable DEM or supply their own aligned rasters and run `train` directly. The channel set lives in `preprocessing.py` (`FEATURE_NAMES`); changing it means updating the encoders' `in_channels`.

## Benchmark

The paper benchmarks the conductance model against the reference models and ResistanceGA2 on the [Wade et al. Dryad dataset](https://doi.org/10.5061/dryad.9kd51c5v6), over the population-holdout folds fixed in `data/wade2025/folds.csv`. Exact coordinates are not redistributed: obtain them from the authors, fill the template, and validate.

```bash
mkdir -p data/private/wade2025
cp data/templates/coordinates.csv data/private/wade2025/coordinates.csv
uv run deepilg validate --config configs/benchmark.toml
```

Fit and score the pipeline models on every fold:

Neural training compiles each region's loss and gradient computation together,
finishes that region's backward pass before starting the next, averages regional
gradients equally, and takes one optimizer step per epoch. Validation losses are
also compiled per region. Compilation happens on first use for each input shape
and execution mode, then the compiled programs are reused as weights and random
keys change. AMG hierarchy setup, logging, and checkpoint-file writes stay outside
the compiled calculations.

#### Patch encoding and memory

The optional `[training]` setting `patch_batch_size` controls **ResNet9 encoder
chunking and activation checkpointing together**:

- **Omitted** (Python API: `None`, the default): encode all patches of one region
  in one call, without activation checkpointing. Start here when the region fits
  in GPU memory.
- **Positive integer**, for example `patch_batch_size = 32`: encode at most that
  many patches per chunk and checkpoint every chunk, recomputing CNN activations
  during backpropagation to save memory. A compiled loop handles the chunks and
  any smaller final chunk without unrolling a separate CNN graph for each.
  Checkpointing remains enabled even when the value exceeds the region's patch count.

The supplied demo and benchmark configurations omit this setting. To opt into
memory-saving execution, uncomment their `patch_batch_size = 32` example. TOML
has no `None`/`null` value: remove or comment out the setting to disable chunking
and checkpointing again. Zero, negative values, booleans, and non-integers are
rejected. Existing configurations with an explicit positive value continue to
use checkpointed chunks.

This setting does **not** change the spatial `[preprocessing].patch_size` (32×32
pixels by default), split the resistance graph, or introduce extra optimizer
updates. The complete regional graph is still needed for its loss. Smaller
checkpointed chunks can reduce peak activation memory, but cost recomputation
and may reduce throughput; measure on your own GPU and region sizes. Sequential
regional gradient accumulation is retained in both modes. Activation
checkpointing is unrelated to saving model checkpoints for resume/evaluation.

For **U-Net**, `patch_batch_size` only chunks its initial linear patch embedding;
it does not enable activation checkpointing or divide the spatial U-Net pass,
which still processes the complete region's patch grid.

```bash
for fold in 0 1 2 3 4; do
  for model in reference unet; do
    uv run deepilg train    --config configs/benchmark.toml --model "$model" --fold "$fold"
    uv run deepilg evaluate --config configs/benchmark.toml --model "$model" --fold "$fold"
  done
  JAX_ENABLE_X64=true uv run deepilg train    --config configs/benchmark.toml --model resnet9 --fold "$fold"
  JAX_ENABLE_X64=true uv run deepilg evaluate --config configs/benchmark.toml --model resnet9 --fold "$fold"
done
```

### ResistanceGA2 baseline

The `resistancega` model wraps [ResistanceGA2](https://github.com/wpeterman/ResistanceGA2) on the same folds and metrics, but its training differs in kind. Instead of gradient descent on one encoder shared across regions, it fits each region independently with a genetic algorithm over a few parametric transforms of the covariate layers, calibrates with a maximum-likelihood population-effects (MLPE) mixed model, and predicts held-out pairs from fixed effects only, on covariates aggregated to the patch grid. It uses its own `renv` lockfile.

```bash
R -q -e 'install.packages("renv", repos="https://cloud.r-project.org")'
R -q -e 'renv::restore(project="resistancega", prompt=FALSE)'

uv run deepilg resistancega-export --config configs/benchmark.toml --fold 0
uv run deepilg resistancega        --config configs/benchmark.toml --fold 0 --region CWMA --mode fit
uv run deepilg resistancega        --config configs/benchmark.toml --fold 0 --region CWMA --mode predict --split test
```

Repeat the fit and prediction for every region and fold, then `evaluate --model resistancega`. The fits are CPU-intensive; see `resistancega/README.md`.

Once every model is scored, render the figures:

```bash
uv run deepilg figures --config configs/benchmark.toml
```

`figures` runs only on authorised empirical data, so unpublished coordinates never reach published outputs. Generated files are never committed.

## Citation and license

Please cite the accompanying paper and the Wade et al. dataset. The source code is released under the [Apache License 2.0](LICENSE).
