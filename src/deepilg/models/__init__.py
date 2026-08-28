"""Encoder families and reference models compared in the benchmark."""

from deepilg.models.base import ConductanceModel, EmbeddingDistanceModel
from deepilg.models.reference import GeographicDistanceModel
from deepilg.models.resnet import ResNet9Conductance
from deepilg.models.unet import UNetEmbeddingDistance

__all__ = [
    "ConductanceModel",
    "EmbeddingDistanceModel",
    "GeographicDistanceModel",
    "ResNet9Conductance",
    "UNetEmbeddingDistance",
]
