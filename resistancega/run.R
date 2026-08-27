#!/usr/bin/env Rscript

# Focused ResistanceGA2 fit/predict entry point for one fold and region.

abort <- function(message) stop(message, call. = FALSE)

load_project_library <- function() {
  file_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_argument) != 1L || !requireNamespace("renv", quietly = TRUE)) return()
  script <- normalizePath(sub("^--file=", "", file_argument[[1]]), mustWork = TRUE)
  suppressMessages(renv::load(project = dirname(script), quiet = TRUE))
}

require_packages <- function() {
  packages <- c("ResistanceGA2", "terra", "lme4")
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0L) {
    abort(paste0(
      "Missing R packages: ", paste(missing, collapse = ", "),
      ". Restore resistancega/renv.lock before running this command."
    ))
  }
}

read_points <- function(path) {
  points <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c("population_id", "x", "y")
  if (!identical(names(points), required)) {
    abort(paste0(path, " must have columns: ", paste(required, collapse = ", ")))
  }
  if (anyDuplicated(points$population_id) || any(!is.finite(points$x)) ||
      any(!is.finite(points$y))) {
    abort(paste0(path, " contains duplicate IDs or invalid projected coordinates"))
  }
  points
}

read_target <- function(path, population_ids) {
  target <- as.matrix(read.csv(
    path,
    row.names = 1,
    check.names = FALSE,
    stringsAsFactors = FALSE
  ))
  storage.mode(target) <- "double"
  if (!setequal(rownames(target), population_ids) ||
      !setequal(colnames(target), population_ids)) {
    abort("Training target labels do not match train_points.csv")
  }
  target <- target[population_ids, population_ids, drop = FALSE]
  diag(target) <- 0
  if (!isTRUE(all.equal(target, t(target), tolerance = 1e-10)) ||
      any(!is.finite(target))) {
    abort("Training target matrix must be finite and symmetric")
  }
  target
}

read_surface <- function(path) {
  surface <- terra::rast(path)
  if (terra::nlyr(surface) != 3L) {
    abort("surface.tif must have three bands: WorldCover, elevation, road density")
  }
  names(surface) <- c("worldcover", "elevation", "road_density")
  if (is.na(terra::crs(surface, proj = TRUE)) || terra::crs(surface) == "") {
    abort("surface.tif must define a projected coordinate reference system")
  }
  surface
}

fit_region <- function(input_dir, output_dir, seed) {
  if (is.na(seed) || seed < 0L) abort("SEED must be a non-negative integer")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  protected <- file.path(output_dir, c("fit.rds", "best_surface.tif", "fit_summary.csv"))
  if (any(file.exists(protected))) {
    abort("Fit outputs already exist; choose a new output directory")
  }

  surface <- read_surface(file.path(input_dir, "surface.tif"))
  points <- read_points(file.path(input_dir, "train_points.csv"))
  target <- read_target(file.path(input_dir, "train_target.csv"), points$population_id)
  samples <- terra::vect(points, geom = c("x", "y"), crs = terra::crs(surface))

  gdist_inputs <- ResistanceGA2::gdist.prep(
    n.Pops = nrow(points),
    response = ResistanceGA2::lower(target),
    samples = samples,
    directions = 8,
    longlat = FALSE,
    method = "commuteDistance",
    commute.approx = "none"
  )
  ga_inputs <- ResistanceGA2::GA.prep(
    raster = surface,
    Results.dir = file.path(output_dir, "optimizer"),
    max.cat = 500,
    max.cont = 500,
    select.trans = list(NA, "M", "M"),
    cat.levels = 15,
    method = "LL",
    k.value = 2,
    pcrossover = 0.85,
    pmutation = 0.125,
    maxiter = 1000,
    run = 25,
    parallel = FALSE,
    gaisl = FALSE,
    optim = FALSE,
    seed = seed,
    monitor = FALSE,
    quiet = TRUE
  )

  started <- Sys.time()
  fit <- ResistanceGA2::MS_optim(
    gdist.inputs = gdist_inputs,
    GA.inputs = ga_inputs,
    diagnostic_plots = FALSE
  )
  elapsed_seconds <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  saveRDS(fit, file.path(output_dir, "fit.rds"))

  best_surface <- ResistanceGA2::Combine_Surfaces(
    PARM = fit$GA.summary@solution,
    gdist.inputs = gdist_inputs,
    GA.inputs = ga_inputs,
    rescale = TRUE,
    p.contribution = FALSE
  )
  names(best_surface) <- "resistance"
  terra::writeRaster(
    best_surface,
    file.path(output_dir, "best_surface.tif"),
    overwrite = FALSE
  )

  commute <- ResistanceGA2::lower(as.matrix(fit$cd[[1]]))
  commute_center <- mean(commute)
  commute_scale <- stats::sd(commute)
  if (!is.finite(commute_scale) || commute_scale <= 0) {
    abort("Fitted training commute distances are constant")
  }
  fixed <- lme4::fixef(fit$MLPE.model)
  if (!all(c("(Intercept)", "cd") %in% names(fixed))) {
    abort("The fitted MLPE model has incomplete fixed effects")
  }
  if (!is.finite(fixed[["cd"]]) || fixed[["cd"]] < 0) {
    abort("The fitted MLPE distance slope is invalid")
  }

  summary <- data.frame(
    seed = seed,
    max_generations = 1000L,
    stagnation_generations = 25L,
    ga_iterations = fit$GA.summary@iter,
    ga_fitness = fit$GA.summary@fitnessValue,
    elapsed_seconds = elapsed_seconds,
    commute_center = commute_center,
    commute_scale = commute_scale,
    fixed_intercept = unname(fixed[["(Intercept)"]]),
    fixed_commute = unname(fixed[["cd"]]),
    stringsAsFactors = FALSE
  )
  write.csv(summary, file.path(output_dir, "fit_summary.csv"), row.names = FALSE)
  write.csv(
    fit$percent.contribution,
    file.path(output_dir, "percent_contribution.csv"),
    row.names = FALSE
  )
  writeLines(capture.output(sessionInfo()), file.path(output_dir, "r_session_info.txt"))
}

predict_region <- function(input_dir, output_dir, split) {
  if (!split %in% c("validation", "test")) {
    abort("SPLIT must be validation or test")
  }
  surface <- read_surface(file.path(input_dir, "surface.tif"))
  best_surface <- terra::rast(file.path(output_dir, "best_surface.tif"))
  summary <- read.csv(file.path(output_dir, "fit_summary.csv"), stringsAsFactors = FALSE)
  if (nrow(summary) != 1L) abort("fit_summary.csv must contain exactly one row")

  points <- read_points(file.path(input_dir, paste0(split, "_points.csv")))
  pairs_path <- file.path(input_dir, paste0(split, "_pairs.csv"))
  pairs <- read.csv(pairs_path, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c("population_i", "population_j", "observed")
  if (!identical(names(pairs), required)) {
    abort(paste0(pairs_path, " must have columns: ", paste(required, collapse = ", ")))
  }
  if (!all(pairs$population_i %in% points$population_id) ||
      !all(pairs$population_j %in% points$population_id) ||
      any(!is.finite(pairs$observed))) {
    abort("Prediction pairs contain unknown populations or invalid targets")
  }

  samples <- terra::vect(points, geom = c("x", "y"), crs = terra::crs(surface))
  prediction_inputs <- ResistanceGA2::gdist.prep(
    n.Pops = nrow(points),
    samples = samples,
    directions = 8,
    longlat = FALSE,
    method = "commuteDistance",
    commute.approx = "none"
  )
  commute <- as.matrix(ResistanceGA2::Run_gdistance(prediction_inputs, best_surface))
  rownames(commute) <- points$population_id
  colnames(commute) <- points$population_id
  distance <- mapply(
    function(i, j) commute[i, j],
    pairs$population_i,
    pairs$population_j,
    USE.NAMES = FALSE
  )
  standardized <- (distance - summary$commute_center[[1]]) /
    summary$commute_scale[[1]]
  pairs$predicted <- summary$fixed_intercept[[1]] +
    summary$fixed_commute[[1]] * standardized
  pairs$resistance_distance <- distance
  if (any(!is.finite(pairs$predicted))) abort("Predictions contain non-finite values")
  write.csv(
    pairs,
    file.path(output_dir, paste0(split, "_predictions.csv")),
    row.names = FALSE
  )
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 1L || !args[[1]] %in% c("fit", "predict")) {
    abort(paste(
      "Usage:",
      "  run.R fit INPUT_DIR OUTPUT_DIR SEED",
      "  run.R predict INPUT_DIR OUTPUT_DIR {validation|test}",
      sep = "\n"
    ))
  }
  load_project_library()
  require_packages()
  if (length(args) != 4L) abort("Expected exactly four command-line arguments")
  mode <- args[[1]]
  input_dir <- normalizePath(args[[2]], mustWork = TRUE)
  output_dir <- normalizePath(args[[3]], mustWork = mode == "predict")
  if (mode == "fit") {
    fit_region(input_dir, output_dir, as.integer(args[[4]]))
  } else {
    predict_region(input_dir, output_dir, args[[4]])
  }
}

main()
