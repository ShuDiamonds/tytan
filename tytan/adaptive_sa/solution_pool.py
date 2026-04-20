"""Maintain best and diverse solution pools for multi-start SA."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


class SolutionPool:
    """Tracks unique solutions and exposes best/diverse subsets."""

    def __init__(self, best_k: int = 4, diverse_k: int = 2) -> None:
        if best_k < 1 or diverse_k < 0:
            raise ValueError("best_k must be positive and diverse_k cannot be negative")
        self.best_k = best_k
        self.diverse_k = diverse_k
        self._entries: Dict[tuple, Dict[str, object]] = {}
        self._best: List[Dict[str, object]] = []
        self._diverse: List[Dict[str, object]] = []
        self._best_dirty = True
        self._diverse_dirty = True
        self._bit_counts: Optional[np.ndarray] = None
        self._entry_total = 0

    def _key(self, state: Sequence[float]) -> tuple:
        return tuple(int(v) for v in state)

    def _mean_hamming(self, state: np.ndarray) -> float:
        if not self._entries or self._bit_counts is None or self._entry_total == 0:
            return 0.0
        bits = np.asarray(state, dtype=int)
        diffs = np.where(bits > 0, self._entry_total - self._bit_counts, self._bit_counts)
        return float(np.sum(diffs) / self._entry_total)

    def _hamming(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(np.abs(a - b)))

    @property
    def best(self) -> List[Dict[str, object]]:
        self._ensure_best()
        return self._best

    @property
    def diverse(self) -> List[Dict[str, object]]:
        self._ensure_diverse()
        return self._diverse

    def _refresh_best(self) -> None:
        entries = sorted(self._entries.values(), key=lambda entry: float(entry["energy"]))
        self._best = entries[: self.best_k]
        self._best_dirty = False

    def _refresh_diverse(self) -> None:
        self._ensure_best()
        if self.diverse_k > 0:
            best_keys = {self._key(entry["state"]) for entry in self._best}
            candidates = [entry for entry in self._entries.values() if self._key(entry["state"]) not in best_keys]
            diversity = sorted(
                candidates,
                key=lambda entry: -self._mean_hamming(entry["state"]),
            )
            self._diverse = diversity[: self.diverse_k]
        else:
            self._diverse = []
        self._diverse_dirty = False

    def _ensure_best(self) -> None:
        if self._best_dirty:
            self._refresh_best()

    def _ensure_diverse(self) -> None:
        if self._diverse_dirty:
            self._refresh_diverse()

    def refresh(self, include_diverse: bool = True) -> None:
        self._refresh_best()
        if include_diverse:
            self._refresh_diverse()
        else:
            self._diverse_dirty = True

    def offer(self, state: Sequence[float], energy: float) -> None:
        key = self._key(state)
        if key not in self._entries:
            state_array = np.asarray(state, dtype=float).copy()
            self._entries[key] = {
                "state": state_array,
                "energy": float(energy),
                "count": 0,
            }
            state_bits = np.asarray(state_array, dtype=int)
            if self._bit_counts is None:
                self._bit_counts = state_bits.copy()
            else:
                self._bit_counts = self._bit_counts + state_bits
            self._entry_total += 1
        entry = self._entries[key]
        entry["count"] += 1
        entry["energy"] = min(float(energy), entry["energy"])
        self._best_dirty = True
        self._diverse_dirty = True

    def _format(self, entry: Dict[str, object], index_map: Dict[str, int]) -> List[object]:
        state = entry["state"]
        mapping = {name: int(state[idx]) for name, idx in index_map.items()}
        return [mapping, float(entry["energy"]), int(entry["count"])]

    def to_results(self, index_map: Dict[str, int], include_diverse: bool = True) -> List[List[object]]:
        self.refresh(include_diverse=include_diverse)
        seen = set()
        results: List[List[object]] = []
        for entry in self._best:
            key = self._key(entry["state"])
            if key in seen:
                continue
            seen.add(key)
            results.append(self._format(entry, index_map))
        if include_diverse:
            for entry in self._diverse:
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
