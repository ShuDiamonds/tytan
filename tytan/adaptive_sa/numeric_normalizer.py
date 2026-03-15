"""Normalize QUBO coefficients and report statistics."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class NumericNormalizer:
    """Rescale QUBO coefficients and prune tiny terms for stability."""

    def __init__(
        self,
        small_coeff_threshold: float = 1e-6,
        penalty_threshold: float = 100.0,
    ) -> None:
        self.small_coeff_threshold = small_coeff_threshold
        self.penalty_threshold = penalty_threshold

    def _snapshot(self, matrix: np.ndarray) -> Dict[str, float]:
        flattened = matrix.flatten()
        if flattened.size == 0:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "absmax": 0.0,
            }
        return {
            "min": float(np.min(flattened)),
            "max": float(np.max(flattened)),
            "mean": float(np.mean(flattened)),
            "absmax": float(np.max(np.abs(flattened))),
        }

    def analyze(self, matrix: np.ndarray) -> Dict[str, object]:
        return {
            "before": self._snapshot(matrix),
            "scale_factor": 1.0,
            "pruned_coefficients": 0,
            "penalty_count": int(np.sum(np.abs(matrix) >= self.penalty_threshold))
            if matrix.size
            else 0,
            "penalty_alert": int(np.sum(np.abs(matrix) >= self.penalty_threshold)) > 0
            if matrix.size
            else False,
        }

    def normalize(self, matrix: np.ndarray, apply_scaling: bool = True) -> Tuple[np.ndarray, Dict[str, object]]:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.size == 0:
            info = self.analyze(matrix)
            info["after"] = info["before"].copy()
            return matrix.copy(), info

        snapshot = self._snapshot(matrix)
        scale = 1.0
        if apply_scaling and snapshot["absmax"] > 0:
            scale = 1.0 / max(snapshot["absmax"], 1.0)
        normalized = matrix * scale
        before_prune = normalized.copy()
        mask = (np.abs(before_prune) < self.small_coeff_threshold) & (np.abs(before_prune) > 0.0)
        pruned = before_prune.copy()
        pruned[mask] = 0.0
        after = self._snapshot(pruned)
        pruned_count = int(np.sum(mask))
        penalty_count = int(np.sum(np.abs(pruned) >= self.penalty_threshold))
        info = {
            "before": snapshot,
            "after": after,
            "scale_factor": scale,
            "pruned_coefficients": pruned_count,
            "penalty_count": penalty_count,
            "penalty_alert": penalty_count > 0,
        }
        return pruned, info
