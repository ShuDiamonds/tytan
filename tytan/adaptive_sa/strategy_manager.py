"""Manage multiple strategies and their adaptive weights."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


class StrategyManager:
    """Keeps adaptive weights for a list of named strategies."""

    def __init__(self, strategies: Optional[Sequence[Dict[str, object]]] = None, *, epsilon: float = 0.2, seed: Optional[int] = None) -> None:
        if strategies is None:
            strategies = [
                {"name": "linear", "type": "linear", "weight": 1.0},
                {"name": "exponential", "type": "exponential", "weight": 1.0},
            ]
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")
        self.epsilon = epsilon
        self.rng = np.random.RandomState(seed)
        self._strategies: Dict[str, Dict[str, object]] = {}
        for config in strategies:
            name = str(config["name"])
            self._strategies[name] = {
                "weight": float(config.get("weight", 1.0)),
                "type": str(config.get("type", "linear")),
                "history": [],
            }

    def select(self) -> Dict[str, object]:
        names = list(self._strategies.keys())
        weights = np.array([self._strategies[name]["weight"] for name in names], dtype=float)
        if weights.sum() <= 0:
            probs = np.ones_like(weights) / len(weights)
        else:
            probs = weights / float(weights.sum())
        if self.rng.rand() < self.epsilon:
            index = self.rng.randint(len(names))
        else:
            index = int(np.argmax(probs))
        name = names[index]
        data = self._strategies[name]
        return {"name": name, "type": data["type"], "weight": float(data["weight"])}

    def record(self, name: str, reward: float) -> None:
        if name not in self._strategies:
            return
        entry = self._strategies[name]
        weight = max(0.0, entry["weight"] + float(reward))
        entry["weight"] = max(weight, 1e-3)
        entry["history"].append(float(reward))

    @property
    def weights(self) -> Dict[str, float]:
        return {name: float(info["weight"]) for name, info in self._strategies.items()}
