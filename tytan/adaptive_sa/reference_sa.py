"""CPU baseline SA reference implementation for TYTAN."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .delta_evaluator import DeltaEvaluator
from .solution_pool import SolutionPool


class ReferenceSASampler:
    """Simple SA sampler used as a deterministic baseline."""

    def __init__(
        self,
        seed: Optional[int] = None,
        steps: int = 200,
        init_temp: float = 1.0,
        end_temp: float = 0.01,
        schedule: str = "linear",
        neighborhood: str = "random",
        return_stats: bool = False,
    ) -> None:
        self.seed = seed
        self.steps = max(1, steps)
        self.init_temp = float(init_temp)
        self.end_temp = float(end_temp)
        self.schedule = schedule
        self.neighborhood = neighborhood
        self.return_stats = return_stats

    def _temperature(self, step: int) -> float:
        progress = step / max(1, self.steps - 1)
        if self.schedule == "exponential":
            return self.init_temp * (self.end_temp / max(self.init_temp, 1e-9)) ** progress
        return self.init_temp + (self.end_temp - self.init_temp) * progress

    def _state_dict(self, state: np.ndarray, index_map: Dict[str, int]) -> Dict[str, int]:
        return {name: int(state[idx]) for name, idx in index_map.items()}

    def run(self, qubomix: Tuple[np.ndarray, Dict[str, int]], shots: int = 1, return_stats: Optional[bool] = None) -> List[object]:
        qmatrix, index_map = qubomix
        rng = np.random.RandomState(self.seed)
        steps = self.steps
        shots = max(1, int(shots))
        delta = DeltaEvaluator(qmatrix)
        pool = SolutionPool(best_k=shots, diverse_k=0)
        total_improvements = 0
        for shot in range(shots):
            state = rng.randint(0, 2, qmatrix.shape[0]).astype(float)
            energy = delta.evaluate(state)
            best_energy = energy
            best_state = state.copy()
            for step in range(steps):
                temperature = max(self._temperature(step), 1e-8)
                index = rng.randint(qmatrix.shape[0])
                change = delta.delta(state, index, energy)
                if change <= 0 or rng.rand() < np.exp(-change / temperature):
                    state[index] = 1.0 - state[index]
                    energy += change
                    if energy < best_energy:
                        best_energy = energy
                        best_state = state.copy()
                        total_improvements += 1
            pool.offer(best_state, best_energy)
        result = pool.to_results(index_map, include_diverse=False)
        stats = {
            "improvements": total_improvements,
            "steps": steps,
            "shots": shots,
        }
        if return_stats or self.return_stats:
            return result, stats
        return result
