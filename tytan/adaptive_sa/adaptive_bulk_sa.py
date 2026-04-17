"""Adaptive bulk SA sampler that orchestrates the modular helpers."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import _rust_backend

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

    def _should_use_rust_batch(self, shots: int, size: int) -> bool:
        min_work = _rust_backend.rust_min_work()
        if min_work <= 0:
            return True
        return shots * size >= min_work

    def _should_use_rust_step(self, shots: int, size: int) -> bool:
        min_work = _rust_backend.rust_step_min_work()
        if min_work <= 0:
            return False
        return shots * size >= min_work

    def run(
        self,
        qubomix: Tuple[np.ndarray, Dict[str, int]],
        shots: Optional[int] = None,
        return_stats: Optional[bool] = None,
    ):  # pragma: no cover - wrapping ensures compatibility
        qmatrix, index_map = qubomix
        if qmatrix.ndim != 2:
            raise ValueError("QUBO matrix must be 2D")
        qmatrix_f = np.ascontiguousarray(np.asarray(qmatrix, dtype=float))
        shots = shots or self.shots
        shots = max(1, int(shots))
        states = self.rng.randint(0, 2, size=(shots, qmatrix_f.shape[0])).astype(float)
        states = np.ascontiguousarray(states)
        evaluator = DeltaEvaluator(qmatrix_f)
        energies = np.array([evaluator.evaluate(state) for state in states])
        energies = np.ascontiguousarray(energies, dtype=float)
        rust_rng_state = int(self.rng.randint(1, np.iinfo(np.int64).max))
        pool = SolutionPool(best_k=min(self.batch_size, shots), diverse_k=2)
        best_idx = int(np.argmin(energies))
        best_energy = float(energies[best_idx])
        best_state = states[best_idx].copy()
        for step in range(self.steps):
            strategy = self.strategy_manager.select()
            temperature = self._temperature(step, strategy.get("type", "linear"))
            improvements = 0
            use_rust_step = self.device == "cpu" and _rust_backend.rust_available() and self._should_use_rust_step(shots, qmatrix_f.shape[0])
            if use_rust_step:
                step_result = _rust_backend.try_sa_step_single_flip(
                    states,
                    energies,
                    qmatrix_f,
                    1.0 / temperature,
                    rust_rng_state,
                )
                if step_result is not None:
                    next_states, next_energies, step_stats = step_result
                    changed_mask = np.any(next_states != states, axis=1)
                    accepted_indices = np.flatnonzero(changed_mask)
                    improvements = int(step_stats.get("accepted", int(len(accepted_indices))))
                    if accepted_indices.size > 0:
                        deltas = next_energies[accepted_indices] - energies[accepted_indices]
                        states = np.ascontiguousarray(next_states, dtype=float)
                        energies = np.ascontiguousarray(next_energies, dtype=float)
                        for accepted_idx in accepted_indices.tolist():
                            pool.offer(states[accepted_idx], energies[accepted_idx])
                            if energies[accepted_idx] < best_energy:
                                best_energy = float(energies[accepted_idx])
                                best_state = states[accepted_idx].copy()
                        if self.adaptive:
                            self.strategy_manager.record(strategy["name"], float(np.sum(-deltas)))
                    else:
                        states = np.ascontiguousarray(next_states, dtype=float)
                        energies = np.ascontiguousarray(next_energies, dtype=float)
                    rust_rng_state = int(step_stats.get("rng_state", rust_rng_state))
                    self.logger.log(
                        strategy=strategy["name"],
                        temperature=temperature,
                        improvements=improvements,
                    )
                    if self.enable_clamp and improvements == 0:
                        occupancy = {
                            idx: float(np.mean(states[:, idx]))
                            for idx in range(qmatrix_f.shape[0])
                        }
                        self.clamp_manager.update_scores(occupancy)
                    continue
            indices = self.rng.randint(qmatrix_f.shape[0], size=shots, dtype=np.int64)
            if self._should_use_rust_batch(shots, qmatrix_f.shape[0]):
                batch_delta = _rust_backend.try_batch_delta(states, qmatrix_f, indices, energies)
            else:
                batch_delta = None
            for i in range(shots):
                idx = int(indices[i])
                if batch_delta is None:
                    change = evaluator.delta(states[i], idx, energies[i])
                else:
                    change = float(batch_delta[i])
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
                    for idx in range(qmatrix_f.shape[0])
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
