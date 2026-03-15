"""Maintain best and diverse solution pools for multi-start SA."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


class SolutionPool:
    """Tracks unique solutions and exposes best/diverse subsets."""

    def __init__(self, best_k: int = 4, diverse_k: int = 2) -> None:
        if best_k < 1 or diverse_k < 0:
            raise ValueError("best_k must be positive and diverse_k cannot be negative")
        self.best_k = best_k
        self.diverse_k = diverse_k
        self._entries: Dict[tuple, Dict[str, object]] = {}
        self.best: List[Dict[str, object]] = []
        self.diverse: List[Dict[str, object]] = []

    def _key(self, state: Sequence[float]) -> tuple:
        return tuple(int(v) for v in state)

    def _mean_hamming(self, state: np.ndarray) -> float:
        if not self._entries:
            return 0.0
        distances = [self._hamming(state, entry["state"]) for entry in self._entries.values()]
        return float(np.mean(distances))

    def _hamming(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(np.abs(a - b)))

    def _refresh(self) -> None:
        entries = sorted(self._entries.values(), key=lambda entry: float(entry["energy"]))
        self.best = entries[: self.best_k]
        if self.diverse_k > 0:
            diversity = sorted(
                entries,
                key=lambda entry: -self._mean_hamming(entry["state"]),
            )
            self.diverse = diversity[: self.diverse_k]
        else:
            self.diverse = []

    def offer(self, state: Sequence[float], energy: float) -> None:
        key = self._key(state)
        if key not in self._entries:
            self._entries[key] = {
                "state": np.asarray(state, dtype=float).copy(),
                "energy": float(energy),
                "count": 0,
            }
        entry = self._entries[key]
        entry["count"] += 1
        entry["energy"] = min(float(energy), entry["energy"])
        self._refresh()

    def _format(self, entry: Dict[str, object], index_map: Dict[str, int]) -> List[object]:
        state = entry["state"]
        mapping = {name: int(state[idx]) for name, idx in index_map.items()}
        return [mapping, float(entry["energy"]), int(entry["count"])]

    def to_results(self, index_map: Dict[str, int], include_diverse: bool = True) -> List[List[object]]:
        seen = set()
        results: List[List[object]] = []
        for entry in self.best:
            key = self._key(entry["state"])
            if key in seen:
                continue
            seen.add(key)
            results.append(self._format(entry, index_map))
        if include_diverse:
            for entry in self.diverse:
                key = self._key(entry["state"])
                if key in seen:
                    continue
                seen.add(key)
                results.append(self._format(entry, index_map))
        if not results and self._entries:
            entry = next(iter(self._entries.values()))
            results.append(self._format(entry, index_map))
        return results

    def __len__(self) -> int:
        return len(self._entries)
