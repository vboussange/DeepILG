"""Differentiable inverse landscape genetics.

The package exposes a small set of scientific building blocks.  Importing it
does not access the network, inspect local data, or initialise an accelerator.
"""

from deepilg.config import ExperimentConfig, load_config
from deepilg.data import Coordinates, FSTMatrix, linearize_fst
from deepilg.metrics import FoldMetrics, evaluate_fold
from deepilg.solver import AMGContext, effective_resistance

__all__ = [
    "AMGContext",
    "Coordinates",
    "ExperimentConfig",
    "FSTMatrix",
    "FoldMetrics",
    "effective_resistance",
    "evaluate_fold",
    "linearize_fst",
    "load_config",
]

__version__ = "0.1.0"
