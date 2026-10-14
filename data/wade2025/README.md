# Wade et al. (2025) public inputs

This directory contains five public pairwise \(F_{\mathrm{ST}}\) matrices and
the population-level five-fold benchmark definition used by DeepILG.

Source dataset: <https://doi.org/10.5061/dryad.9kd51c5v6>.

Please cite:

> Wade, B. S., Pierson, T. W., Fitzpatrick, B. M., & Carter, E. T. (2025).
> Spatial Replication Is Important for Developing Landscape Genetic Inferences
> for a Wetland Salamander. *Molecular Ecology*, 34(13), e17808.
> <https://doi.org/10.1111/mec.17808>

Dryad releases these source data under CC0 1.0. The fold assignment is a
lightweight derived input distributed with this repository under CC0 1.0;
`deepilg folds --config configs/benchmark.toml` regenerates it from the
matrices' population identifiers with `split_seed = 17`.
Exact coordinates are not included. The root `README.md` explains how to place
authorized coordinates using `data/templates/coordinates.csv`.
