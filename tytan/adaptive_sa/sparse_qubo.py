"""Sparse neighbor representation for fast delta-cache SA updates.

This module intentionally keeps the representation simple: a CSR-like layout
for the symmetric coupling matrix (Q + Q.T) excluding the diagonal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SparseNeighbors:
    """CSR-like neighbor list for column-wise access.

    For a flipped variable i, neighbors are all j where (Q + Q.T)[j, i] != 0.
    The stored weight is (Q + Q.T)[j, i].
    """

    offsets: np.ndarray  # int64, shape (n + 1,)
    neighbors: np.ndarray  # int64, shape (nnz,)
    weights: np.ndarray  # float64, shape (nnz,)

    @property
    def n(self) -> int:
        return int(self.offsets.shape[0] - 1)


def build_sparse_neighbors(qmatrix: np.ndarray, threshold: float = 0.0) -> SparseNeighbors:
    """Build SparseNeighbors for (Q + Q.T) excluding diagonal.

    Args:
        qmatrix: Dense square QUBO matrix.
        threshold: Keep entries with abs(value) > threshold.
    """

    q = np.asarray(qmatrix, dtype=float)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("QUBO matrix must be square")

    n = int(q.shape[0])
    qsym = q + q.T
    # Exclude diagonal: delta-cache formula handles diagonal separately.
    np.fill_diagonal(qsym, 0.0)

    offsets = np.zeros(n + 1, dtype=np.int64)
    neighbors_list: list[np.ndarray] = []
    weights_list: list[np.ndarray] = []
    nnz = 0
    for i in range(n):
        col = qsym[:, i]
        if threshold > 0.0:
            mask = np.abs(col) > threshold
        else:
            mask = col != 0.0
        idx = np.flatnonzero(mask).astype(np.int64)
        w = col[idx].astype(float)
        neighbors_list.append(idx)
        weights_list.append(w)
        nnz += int(idx.size)
        offsets[i + 1] = nnz

    neighbors = np.concatenate(neighbors_list) if nnz else np.zeros(0, dtype=np.int64)
    weights = np.concatenate(weights_list) if nnz else np.zeros(0, dtype=float)
    return SparseNeighbors(offsets=offsets, neighbors=neighbors, weights=weights)

