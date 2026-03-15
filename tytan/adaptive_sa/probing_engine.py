"""Simple probing engine that scores fixed-value candidates."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


class ProbingEngine:
    """Derive probing scores from diagonal entries as a proxy for certainty."""

    def __init__(self, strength: str = "medium") -> None:
        self.strength = strength
        self.scores: Dict[str, float] = {}

    def evaluate(self, matrix: np.ndarray, names: List[str]) -> Dict[str, object]:
        diag = np.diag(matrix) if matrix.size else np.zeros(len(names))
        self.scores = {
            name: float(np.abs(diag[idx])) if idx < len(diag) else 0.0
            for idx, name in enumerate(names)
        }
        return {
            "strength": self.strength,
            "scores": self.scores,
        }
