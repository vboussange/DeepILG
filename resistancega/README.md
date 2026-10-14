# ResistanceGA2 comparator

This directory contains the complete R-side interface for the classical ILG
comparator. Python exports one directory per population-holdout fold and geographic region;
`run.R` fits that region or predicts a held-out split from an existing fit.

## Environment

Python dependencies remain managed exclusively by `uv`. R packages are the
one exception: the supported R environment is the `renv.lock` lockfile in this
directory, restored with renv.

```bash
R -q -e 'install.packages("renv", repos="https://cloud.r-project.org")'
R -q -e 'renv::restore(project="resistancega", prompt=FALSE)'
```

Use R 4.4. Compiled geospatial packages may require GDAL, GEOS, PROJ, and
system development libraries. The lock selects ResistanceGA2 1.1.20 from
upstream commit `d0a0aab68cf4ee09c61bd1c6aff8df62051ffbfe`.
`run.R` loads the project-local `renv` library automatically when available.

## Export contract

Create the inputs with:

```bash
uv run deepilg resistancega-export --config configs/benchmark.toml --fold 0
```

Each exported region contains:

- `surface.tif`: modal WorldCover class, mean elevation, and mean road density
  at the 32×32-pixel patch scale;
- `train_points.csv`: `population_id,x,y` in the surface's projected CRS;
- `train_target.csv`: square labelled linearized \(F_{\mathrm{ST}}\) matrix;
- `{validation,test}_points.csv`: projected populations needed by that split;
- `{validation,test}_pairs.csv`: `population_i,population_j,observed`.

## Fit and predict

Fit exactly one region and fold with the configured scalar seed:

```bash
uv run deepilg resistancega --config configs/benchmark.toml \
  --fold 0 --region CWMA --mode fit
```

This performs the expensive genetic-algorithm optimization. It refuses to
overwrite an existing fit. Add `--workers N` (default 1) to evaluate each
generation's fitness on N CPU processes. Each process is limited to one BLAS and
OpenMP thread. The seed fixes both the starting suggestions and the genetic
algorithm. Selection, crossover, and mutation stay in the main R process and
fitness is deterministic, so with the lockfile's packages (which exclude
`doRNG`) a parallel fit follows the same trajectory as a single-core fit. The
wrapper records the R process's wall-clock duration and worker count in
`fit_timing.json`. Predict validation or test pairs without refitting:

```bash
uv run deepilg resistancega --config configs/benchmark.toml \
  --fold 0 --region CWMA --mode predict --split test
```

The fit uses exact commute distances on an eight-neighbour graph, a maximum of
1,000 genetic-algorithm generations, and early termination after 25 generations
without improvement. The MLPE calibration is fitted on training pairs.
Held-out predictions use fixed effects only; random effects are not imputed for
unseen populations.

After predicting all five regions for a fold, normalize their rows and compute
the common held-out metrics with:

```bash
uv run deepilg evaluate --config configs/benchmark.toml --model resistancega --fold 0
```

Outputs are generated data and must not be committed. A fit directory contains
the serialized fit, combined resistance surface, compact numeric summary,
feature contributions, optimizer files, and R session information. Prediction
adds only the requested prediction CSV, leaving fit artifacts unchanged.
