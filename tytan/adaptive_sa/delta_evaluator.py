"""Utilities to support fast QUBO delta evaluations."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .. import _rust_backend


class DeltaEvaluator:
    """Compute energies and incremental deltas for QUBO states."""

    def __init__(self, qmatrix: np.ndarray) -> None:
        self.qmatrix = np.asarray(qmatrix, dtype=float)
        if self.qmatrix.ndim != 2 or self.qmatrix.shape[0] != self.qmatrix.shape[1]:
            raise ValueError("QUBO matrix must be square")
        self.size = self.qmatrix.shape[0]

    def evaluate(self, state: np.ndarray) -> float:
        state = np.asarray(state, dtype=float)
        if state.shape[-1] != self.size:
            raise ValueError("State dimension mismatch")
        return float(state @ self.qmatrix @ state)

    def delta(self, state: np.ndarray, index: int, energy: Optional[float] = None) -> float:
        """Return change in energy when bit at index is flipped."""
        energy = self.evaluate(state) if energy is None else float(energy)
        rust_delta = _rust_backend.try_delta_energy(state, self.qmatrix, index, energy)
        if rust_delta is not None:
            return float(rust_delta)

        flipped = state.copy().astype(float)
        if not (0 <= index < self.size):
            raise IndexError("Index out of range")
        flipped[index] = 1.0 - flipped[index]
        return float(flipped @ self.qmatrix @ flipped) - energy

    def local_field(self, state: np.ndarray, index: int) -> float:
        """Return the weighted sum that contributes to the delta evaluation."""
        state = np.asarray(state, dtype=float)
        if index < 0 or index >= self.size:
            raise IndexError("Index out of range")
        contributions = self.qmatrix[index] * state
        return float(contributions.sum())
