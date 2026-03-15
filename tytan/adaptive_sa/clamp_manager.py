"""Manage soft and hard clamps for bit indices."""
from __future__ import annotations

from typing import Dict

import numpy as np


class ClampManager:
    """Keeps track of soft/hard clamp annotations for variable indices."""

    def __init__(self, clamp_mode: str = "none", threshold: float = 0.75) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.clamp_mode = clamp_mode
        self.threshold = threshold
        self.soft_candidates: Dict[int, float] = {}
        self.hard_assignments: Dict[int, int] = {}

    def update_scores(self, scores: Dict[int, float]) -> None:
        """Identify soft clamp candidates from provided scores."""
        if self.clamp_mode not in {"soft", "hard"}:
            self.soft_candidates = {}
            return
        self.soft_candidates = {
            idx: value for idx, value in scores.items() if value >= self.threshold
        }

    def lock(self, index: int, value: int) -> None:
        """Apply a hard clamp regardless of the mode."""
        if not (0 <= value <= 1):
            raise ValueError("clamp value must be binary")
        self.hard_assignments[index] = int(value)

    def apply(self, state: np.ndarray) -> np.ndarray:
        """Enforce hard clamps on a state copy."""
        state = np.asarray(state, dtype=float).copy()
        for idx, value in self.hard_assignments.items():
            if 0 <= idx < state.shape[-1]:
                state[idx] = float(value)
        return state
