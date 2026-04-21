"""Maintain best and diverse solution pools for multi-start SA."""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class SolutionPool:
    """Tracks unique solutions and exposes best/diverse subsets."""

    def __init__(
        self,
        best_k: int = 4,
        diverse_k: int = 2,
        max_entries: int = 128,
        near_dup_hamming: int = 2,
        replace_margin: float = 1e-6,
    ) -> None:
        if best_k < 1 or diverse_k < 0:
            raise ValueError("best_k must be positive and diverse_k cannot be negative")
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        if near_dup_hamming < 0:
            raise ValueError("near_dup_hamming cannot be negative")
        if replace_margin < 0:
            raise ValueError("replace_margin cannot be negative")
        self.best_k = best_k
        self.diverse_k = diverse_k
        self.max_entries = max_entries
        self.near_dup_hamming = near_dup_hamming
        self.replace_margin = replace_margin
        self._entries: Dict[tuple, Dict[str, object]] = {}
        self._best: List[Dict[str, object]] = []
        self._diverse: List[Dict[str, object]] = []
        self._best_dirty = True
        self._diverse_dirty = True

    def _key(self, state: Sequence[float]) -> tuple:
        return tuple(int(v) for v in state)

    def _state_array(self, state: Sequence[float]) -> np.ndarray:
        return np.asarray(state, dtype=float).copy()

    def _hamming(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(np.abs(a - b)))

    def _nearest_entry(self, state: Sequence[float]) -> Tuple[Optional[tuple], float]:
        if not self._entries:
            return None, float("inf")
        state_array = np.asarray(state, dtype=float)
        nearest_key: Optional[tuple] = None
        nearest_distance = float("inf")
        for key, entry in self._entries.items():
            distance = self._hamming(state_array, entry["state"])
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_key = key
        return nearest_key, nearest_distance

    def _trim(self) -> None:
        while len(self._entries) > self.max_entries:
            worst_key, _ = max(
                self._entries.items(),
                key=lambda item: (float(item[1]["energy"]), float(item[1]["count"])),
            )
            del self._entries[worst_key]
            self._best_dirty = True
            self._diverse_dirty = True

    def min_distance_to_pool(self, state: Sequence[float]) -> float:
        if not self._entries:
            return 0.0
        state_array = np.asarray(state, dtype=float)
        return float(
            min(self._hamming(state_array, entry["state"]) for entry in self._entries.values())
        )

    def mean_pairwise_distance(self) -> float:
        entries = list(self._entries.values())
        if len(entries) < 2:
            return 0.0
        distances = [
            self._hamming(a["state"], b["state"])
            for a, b in combinations(entries, 2)
        ]
        return float(np.mean(distances))

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
        if self.diverse_k <= 0:
            self._diverse = []
            self._diverse_dirty = False
            return

        selected = list(self._best)
        selected_keys = {self._key(entry["state"]) for entry in selected}
        candidates = [
            entry for entry in self._entries.values() if self._key(entry["state"]) not in selected_keys
        ]
        diverse: List[Dict[str, object]] = []
        while candidates and len(diverse) < self.diverse_k:
            best_candidate = max(
                candidates,
                key=lambda entry: (
                    min(self._hamming(entry["state"], chosen["state"]) for chosen in selected),
                    -float(entry["energy"]),
                ),
            )
            diverse.append(best_candidate)
            selected.append(best_candidate)
            candidates = [entry for entry in candidates if self._key(entry["state"]) != self._key(best_candidate["state"])]
        self._diverse = diverse
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
        if key in self._entries:
            entry = self._entries[key]
            entry["count"] += 1
            entry["energy"] = min(float(energy), float(entry["energy"]))
            self._best_dirty = True
            self._diverse_dirty = True
            return

        nearest_key, nearest_distance = self._nearest_entry(state)
        if nearest_key is not None and nearest_distance <= self.near_dup_hamming:
            nearest_entry = self._entries[nearest_key]
            if float(energy) < float(nearest_entry["energy"]) - self.replace_margin:
                del self._entries[nearest_key]
                self._entries[key] = {
                    "state": self._state_array(state),
                    "energy": float(energy),
                    "count": 1,
                }
                self._best_dirty = True
                self._diverse_dirty = True
                self._trim()
            return

        self._entries[key] = {
            "state": self._state_array(state),
            "energy": float(energy),
            "count": 1,
        }
        self._best_dirty = True
        self._diverse_dirty = True
        self._trim()

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
