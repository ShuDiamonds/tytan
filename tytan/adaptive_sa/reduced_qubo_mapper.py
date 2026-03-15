"""Keep mapping between original QUBO and the reduced problem."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


class ReducedQuboMapper:
    """Track fixed variables and reduced index maps for restoration."""

    def __init__(self, original_index_map: Dict[str, int]) -> None:
        self.original_index_map = dict(original_index_map)
        self.index_to_name = {idx: name for name, idx in original_index_map.items()}
        self.original_size = len(original_index_map)
        self.fixed_variables: Dict[str, int] = {}
        self.linked_variables: Dict[str, List[str]] = {}
        self.active_indices: List[int] = list(range(self.original_size))
        self.reduced_index_map = dict(original_index_map)

    def register_fixed(self, name: str, value: int) -> None:
        self.fixed_variables[name] = int(value)
        idx = self.original_index_map[name]
        if idx in self.active_indices:
            self.active_indices.remove(idx)

    def update_active_indices(self, indices: Iterable[int]) -> None:
        normalized = sorted(set(indices))
        self.active_indices = [idx for idx in normalized if 0 <= idx < self.original_size]
        names = [self.index_to_name[idx] for idx in self.active_indices]
        self.reduced_index_map = {name: i for i, name in enumerate(names)}

    def reduced_index_names(self) -> List[str]:
        return list(self.reduced_index_map.keys())

    def restored_state(self, reduced_state: Dict[str, object]) -> Dict[str, int]:
        state: Dict[str, int] = {}
        for name, value in reduced_state.items():
            if name in self.original_index_map:
                state[name] = int(float(value))
        for name, value in self.fixed_variables.items():
            state[name] = int(value)
        return state

    def restore_results(self, results: List[List[object]]) -> List[List[object]]:
        restored = []
        for entry in results:
            mapping, energy, count = entry
            restored_state = self.restored_state(mapping)
            restored.append([restored_state, float(energy), int(count)])
        return restored

    def full_state_array(self, state_dict: Dict[str, int]) -> np.ndarray:
        array = np.zeros(self.original_size, dtype=float)
        for name, value in state_dict.items():
            if name in self.original_index_map:
                array[self.original_index_map[name]] = float(value)
        return array
