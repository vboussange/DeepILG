"""Non-negative affine calibration of geographic distance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class GeographicDistanceModel:
    """Reference model ``a * geographic_distance + b``, with ``a,b >= 0``."""

    slope: float
    intercept: float

    def __post_init__(self) -> None:
        if self.slope < 0 or self.intercept < 0:
            raise ValueError("Reference-model coefficients must be non-negative")

    def predict(self, distances: np.ndarray) -> np.ndarray:
        """Apply the calibration to a distance vector or matrix."""
        values = self.slope * np.asarray(distances, dtype=np.float64) + self.intercept
        if values.ndim == 2 and values.shape[0] == values.shape[1]:
            values = values.copy()
            np.fill_diagonal(values, 0.0)
        return values

    @classmethod
    def fit(
        cls,
        distances_by_region: Mapping[str, np.ndarray],
        targets_by_region: Mapping[str, np.ndarray],
    ) -> GeographicDistanceModel:
        """Fit coefficients using equal-region log-distance MSE."""
        if set(distances_by_region) != set(targets_by_region) or not distances_by_region:
            raise ValueError("Distances and targets must contain identical non-empty regions")
        vectors: list[tuple[np.ndarray, np.ndarray]] = []
        for region in sorted(distances_by_region):
            distance = np.asarray(distances_by_region[region], dtype=np.float64).reshape(-1)
            target = np.asarray(targets_by_region[region], dtype=np.float64).reshape(-1)
            if distance.shape != target.shape or distance.size == 0:
                raise ValueError(f"{region}: distance/target vectors must be aligned and non-empty")
            if (
                np.any(distance < 0)
                or np.any(target < 0)
                or not np.all(np.isfinite(distance + target))
            ):
                raise ValueError(f"{region}: distances and targets must be finite and non-negative")
            vectors.append((distance, target))
        # Optimize the slope per largest training distance: per-metre slopes leave the
        # problem so ill-conditioned that L-BFGS-B stops before moving the intercept.
        scale = max(float(np.max(distance)) for distance, _ in vectors) or 1.0
        vectors = [(distance / scale, target) for distance, target in vectors]

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            slope, intercept = parameters
            losses: list[float] = []
            gradients: list[np.ndarray] = []
            for distance, target in vectors:
                prediction = slope * distance + intercept
                residual = np.log1p(prediction) - np.log1p(target)
                denominator = 1.0 + prediction
                losses.append(float(np.mean(np.square(residual))))
                gradients.append(
                    np.asarray(
                        [
                            np.mean(2.0 * residual * distance / denominator),
                            np.mean(2.0 * residual / denominator),
                        ]
                    )
                )
            return float(np.mean(losses)), np.mean(np.stack(gradients), axis=0)

        candidates = []
        for slope, intercept in ((0.0, 0.0), (1e-4, 0.0), (1e-3, 0.0), (0.0, 1e-2)):
            result = minimize(
                objective,
                np.asarray((slope * scale, intercept)),
                method="L-BFGS-B",
                jac=True,
                bounds=((0.0, None), (0.0, None)),
            )
            if result.success and np.all(np.isfinite(result.x)):
                candidates.append(result)
        if not candidates:
            raise RuntimeError("Reference geographic-distance calibration did not converge")
        best = min(candidates, key=lambda result: objective(result.x)[0])
        return cls(slope=float(best.x[0] / scale), intercept=float(best.x[1]))
