"""Extract fixing candidates and build the reduced QUBO."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .reduced_qubo_mapper import ReducedQuboMapper


class PresolveReducer:
    """Perform light presolve reduction before solving."""

    STRENGTH_THRESHOLDS = {"low": 0.5, "medium": 1.0, "high": 2.0}

    def __init__(self, enable_soft_fix: bool = True, enable_hard_fix: bool = False) -> None:
        self.enable_soft_fix = enable_soft_fix
        self.enable_hard_fix = enable_hard_fix

    def reduce(
        self,
        matrix: np.ndarray,
        mapper: ReducedQuboMapper,
        strength: str = "medium",
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        size = matrix.shape[0]
        threshold = self.STRENGTH_THRESHOLDS.get(strength, self.STRENGTH_THRESHOLDS["medium"])

        diag = np.diag(matrix) if size > 0 else np.array([])
        hard_indices: List[int] = []
        soft_indices: List[int] = []
        for idx, value in enumerate(np.abs(diag)):
            if self.enable_hard_fix and value >= threshold:
                hard_indices.append(idx)
            elif self.enable_soft_fix and value >= 0.5 * threshold:
                soft_indices.append(idx)
        for idx in hard_indices:
            name = mapper.index_to_name[idx]
            mapper.register_fixed(name, 0)
        active_indices = [idx for idx in range(size) if idx not in hard_indices]
        mapper.update_active_indices(active_indices)
        if not active_indices:
            reduced = np.zeros((0, 0))
        else:
            reduced = matrix[np.ix_(active_indices, active_indices)].copy()
        stats = {
            "original_size": size,
            "reduced_size": len(active_indices),
            "hard_fix_count": len(hard_indices),
            "soft_fix_count": len(soft_indices),
            "aggregation_count": 0,
            "fixed_variables": dict(mapper.fixed_variables),
            "soft_fix_candidates": [mapper.index_to_name[idx] for idx in soft_indices],
        }
        return reduced, stats
