"""Adaptive bulk SA sampler that orchestrates the modular helpers."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .anneal_logger import AnnealLogger
from .clamp_manager import ClampManager
from .delta_evaluator import DeltaEvaluator
from .solution_pool import SolutionPool
from .strategy_manager import StrategyManager


class AdaptiveBulkSASampler:
    """Runs multiple SA chains with strategy adaptation and optional clamping."""

    def __init__(
        self,
        seed: Optional[int] = None,
        shots: int = 64,
        steps: int = 128,
        batch_size: Optional[int] = None,
        init_temp: float = 5.0,
        end_temp: float = 0.01,
        schedule: str = "linear",
        adaptive: bool = True,
        strategy_configs: Optional[List[Dict[str, object]]] = None,
        epsilon: float = 0.2,
        enable_clamp: bool = False,
        clamp_mode: str = "soft",
        return_stats: bool = False,
        device: str = "cpu",
    ) -> None:
        if shots < 1 or steps < 1:
            raise ValueError("shots and steps must be positive")
        self.seed = seed
        self.shots = shots
        self.steps = steps
        self.batch_size = batch_size or shots
        self.init_temp = init_temp
        self.end_temp = end_temp
        self.schedule = schedule
        self.adaptive = adaptive
        self.strategy_manager = StrategyManager(
            strategies=strategy_configs,
            epsilon=epsilon,
            seed=seed,
        )
        self.clamp_manager = ClampManager(clamp_mode=clamp_mode)
        self.logger = AnnealLogger()
        self.enable_clamp = enable_clamp
        self.return_stats = return_stats
        self.device = device
        self.rng = np.random.RandomState(self.seed)

    def _temperature(self, step: int, strategy_type: str) -> float:
        progress = step / max(1, self.steps - 1)
        if strategy_type == "exponential" or self.schedule == "exponential":
            scale = self.end_temp / max(self.init_temp, 1e-9)
            return max(self.init_temp * (scale ** progress), 1e-8)
        return max(self.init_temp + (self.end_temp - self.init_temp) * progress, 1e-8)

    def run(
        self,
        qubomix: Tuple[np.ndarray, Dict[str, int]],
        shots: Optional[int] = None,
        return_stats: Optional[bool] = None,
    ):  # pragma: no cover - wrapping ensures compatibility
        qmatrix, index_map = qubomix
        if qmatrix.ndim != 2:
            raise ValueError("QUBO matrix must be 2D")
        shots = shots or self.shots
        shots = max(1, int(shots))
        states = self.rng.randint(0, 2, size=(shots, qmatrix.shape[0])).astype(float)
        evaluator = DeltaEvaluator(qmatrix)
        energies = np.array([evaluator.evaluate(state) for state in states])
        pool = SolutionPool(best_k=min(self.batch_size, shots), diverse_k=2)
        best_idx = int(np.argmin(energies))
        best_energy = float(energies[best_idx])
        best_state = states[best_idx].copy()
        for step in range(self.steps):
            strategy = self.strategy_manager.select()
            temperature = self._temperature(step, strategy.get("type", "linear"))
            improvements = 0
            for i in range(shots):
                idx = self.rng.randint(qmatrix.shape[0])
                change = evaluator.delta(states[i], idx, energies[i])
                threshold = (change <= 0) or self.rng.rand() < np.exp(-change / temperature)
                if threshold:
                    states[i][idx] = 1.0 - states[i][idx]
                    energies[i] += change
                    improvements += 1
                    pool.offer(states[i], energies[i])
                    reward = -change
                    if self.adaptive:
                        self.strategy_manager.record(strategy["name"], reward)
                    if energies[i] < best_energy:
                        best_energy = float(energies[i])
                        best_state = states[i].copy()
            self.logger.log(
                strategy=strategy["name"],
                temperature=temperature,
                improvements=improvements,
            )
            if self.enable_clamp and improvements == 0:
                occupancy = {
                    idx: float(np.mean(states[:, idx]))
                    for idx in range(qmatrix.shape[0])
                }
                self.clamp_manager.update_scores(occupancy)
        pool.offer(best_state, best_energy)
        result = pool.to_results(index_map)
        stats = {
            "best_energy": best_energy,
            "strategy_weights": self.strategy_manager.weights,
            "log_entries": self.logger.entries,
            "clamp_mode": self.clamp_manager.clamp_mode,
        }
        if return_stats or self.return_stats:
            return result, stats
        return result
