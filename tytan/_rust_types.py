"""Typed adapters for Rust backend I/O."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def as_state_vector(state: Sequence[float]) -> np.ndarray:
    return np.asarray(state, dtype=float)


def as_state_matrix(states: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(states, dtype=float)


def as_energy_vector(energies: Sequence[float]) -> np.ndarray:
    return np.asarray(energies, dtype=float)


def as_index_vector(indices: Sequence[int]) -> np.ndarray:
    return np.asarray(indices, dtype=np.int64)


def normalize_hobomix(hobomix: Tuple[np.ndarray, dict]) -> Tuple[np.ndarray, dict]:
    qmatrix, index_map = hobomix
    return np.asarray(qmatrix, dtype=float), index_map
