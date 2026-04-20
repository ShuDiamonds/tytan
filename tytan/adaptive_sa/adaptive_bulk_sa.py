"""Adaptive bulk SA sampler that orchestrates the modular helpers."""
from __future__ import annotations

from itertools import combinations
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
        include_diverse: bool = True,
        pool_max_entries: int = 128,
        near_dup_hamming: int = 2,
        replace_margin: float = 1e-6,
        stall_steps: int = 20,
        restart_ratio: float = 0.25,
        restart_min_flips: int = 4,
        restart_burnin_steps: int = 1,
        restart_diversity_threshold: Optional[float] = None,
        novelty_weight: float = 0.05,
    ) -> None:
        if shots < 1 or steps < 1:
            raise ValueError("shots and steps must be positive")
        if pool_max_entries < 1:
            raise ValueError("pool_max_entries must be positive")
        if near_dup_hamming < 0:
            raise ValueError("near_dup_hamming cannot be negative")
        if replace_margin < 0:
            raise ValueError("replace_margin cannot be negative")
        if stall_steps < 1:
            raise ValueError("stall_steps must be positive")
        if not 0 < restart_ratio <= 1:
            raise ValueError("restart_ratio must be between 0 and 1")
        if restart_min_flips < 1:
            raise ValueError("restart_min_flips must be positive")
        if restart_burnin_steps < 0:
            raise ValueError("restart_burnin_steps cannot be negative")
        if novelty_weight < 0:
            raise ValueError("novelty_weight cannot be negative")
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
        self.strategy_configs = strategy_configs
        self.epsilon = epsilon
        self.clamp_manager = ClampManager(clamp_mode=clamp_mode)
        self.logger = AnnealLogger()
        self.enable_clamp = enable_clamp
        self.return_stats = return_stats
        self.device = device
        self.include_diverse = include_diverse
        self.pool_max_entries = pool_max_entries
        self.near_dup_hamming = near_dup_hamming
        self.replace_margin = replace_margin
        self.stall_steps = stall_steps
        self.restart_ratio = restart_ratio
        self.restart_min_flips = restart_min_flips
        self.restart_burnin_steps = restart_burnin_steps
        self.restart_diversity_threshold = restart_diversity_threshold
        self.novelty_weight = novelty_weight
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

    @staticmethod
    def _state_diversity(states: np.ndarray) -> float:
        if states.shape[0] < 2:
            return 0.0
        distances = [
            float(np.sum(np.abs(a - b)))
            for a, b in combinations(states, 2)
        ]
        return float(np.mean(distances))

    def _restart_states(
        self,
        states: np.ndarray,
        energies: np.ndarray,
        best_state: np.ndarray,
        best_energy: float,
        evaluator: DeltaEvaluator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
        restart_count = max(1, int(np.ceil(states.shape[0] * self.restart_ratio)))
        worst_indices = np.argsort(energies)[-restart_count:]
        dim = int(states.shape[1])
        flip_count = min(max(1, self.restart_min_flips), dim)
        updated_best_state = best_state.copy()
        updated_best_energy = float(best_energy)
        for idx in worst_indices.tolist():
            new_state = best_state.copy()
            flip_indices = self.rng.choice(dim, size=flip_count, replace=False)
            new_state[flip_indices] = 1.0 - new_state[flip_indices]
            new_energy = float(evaluator.evaluate(new_state))
            states[idx] = new_state
            energies[idx] = new_energy
            if new_energy < updated_best_energy:
                updated_best_energy = new_energy
                updated_best_state = new_state.copy()
        return states, energies, updated_best_state, updated_best_energy, worst_indices

    def run(
        self,
        qubomix: Tuple[np.ndarray, Dict[str, int]],
        shots: Optional[int] = None,
        return_stats: Optional[bool] = None,
        include_diverse: Optional[bool] = None,
    ):  # pragma: no cover - wrapping ensures compatibility
        qmatrix, index_map = qubomix
        qmatrix = np.asarray(qmatrix, dtype=float)
        if qmatrix.ndim != 2:
            raise ValueError("QUBO matrix must be 2D")
        qmatrix_f = np.ascontiguousarray(qmatrix)
        shots = shots or self.shots
        shots = max(1, int(shots))
        include_diverse = self.include_diverse if include_diverse is None else include_diverse
        if (
            self.device == "cpu"
            and _rust_backend.adaptive_bulk_sa_available()
            and not self.enable_clamp
            and all(isinstance(name, str) for name in index_map)
        ):
            index_names = [str(name) for name, _ in sorted(index_map.items(), key=lambda item: item[1])]
            rust_result = _rust_backend.try_adaptive_bulk_sa(
                qmatrix_f,
                index_names,
                shots,
                self.steps,
                self.batch_size,
                self.init_temp,
                self.end_temp,
                self.schedule,
                self.adaptive,
                self.strategy_configs,
                self.epsilon if hasattr(self, "epsilon") else 0.2,
                include_diverse,
                self.pool_max_entries,
                self.near_dup_hamming,
                self.replace_margin,
                self.stall_steps,
                self.restart_ratio,
                self.restart_min_flips,
                self.restart_burnin_steps,
                self.restart_diversity_threshold,
                self.novelty_weight,
                self.seed,
            )
            if rust_result is not None:
                result, stats = rust_result
                if return_stats or self.return_stats:
                    return result, stats
                return result
        states = self.rng.randint(0, 2, size=(shots, qmatrix_f.shape[0])).astype(float)
        states = np.ascontiguousarray(states)
        evaluator = DeltaEvaluator(qmatrix_f)
        energies = np.array([evaluator.evaluate(state) for state in states])
        energies = np.ascontiguousarray(energies, dtype=float)
        rust_rng_state = int(self.rng.randint(1, np.iinfo(np.int64).max))
        pool = SolutionPool(
            best_k=min(self.batch_size, shots),
            diverse_k=2,
            max_entries=self.pool_max_entries,
            near_dup_hamming=self.near_dup_hamming,
            replace_margin=self.replace_margin,
        )
        best_idx = int(np.argmin(energies))
        best_energy = float(energies[best_idx])
        best_state = states[best_idx].copy()
        last_best_step = 0
        last_restart_step = -self.restart_burnin_steps
        restart_events = 0
        diversity_threshold = (
            self.restart_diversity_threshold
            if self.restart_diversity_threshold is not None
            else max(1.0, qmatrix_f.shape[0] * 0.25)
        )

        if (
            not self.adaptive
            and self.device == "cpu"
            and _rust_backend.rust_available()
            and self._should_use_rust_step(shots, qmatrix_f.shape[0])
        ):
            step_plan = []
            betas = np.empty(self.steps, dtype=float)
            for step in range(self.steps):
                strategy = self.strategy_manager.select()
                temperature = self._temperature(step, strategy.get("type", "linear"))
                betas[step] = 1.0 / temperature
                step_plan.append((strategy, temperature))

            step_result = _rust_backend.try_sa_step_multi_flip(
                states,
                energies,
                qmatrix_f,
                betas,
                rust_rng_state,
            )
            if step_result is not None:
                history_states, history_energies, step_stats = step_result
                prev_states = states
                prev_energies = energies
                for step, (strategy, temperature) in enumerate(step_plan):
                    next_states = np.ascontiguousarray(history_states[step], dtype=float)
                    next_energies = np.ascontiguousarray(history_energies[step], dtype=float)
                    changed_mask = np.any(next_states != prev_states, axis=1)
                    accepted_indices = np.flatnonzero(changed_mask)
                    improvements = int(len(accepted_indices))
                    if accepted_indices.size > 0:
                        for accepted_idx in accepted_indices.tolist():
                            candidate_state = next_states[accepted_idx]
                            candidate_energy = float(next_energies[accepted_idx])
                            novelty = pool.min_distance_to_pool(candidate_state)
                            pool.offer(candidate_state, candidate_energy)
                            reward = -float(next_energies[accepted_idx] - prev_energies[accepted_idx])
                            if self.adaptive:
                                self.strategy_manager.record(
                                    strategy["name"],
                                    reward + self.novelty_weight * novelty,
                                )
                            if candidate_energy < best_energy:
                                best_energy = candidate_energy
                                best_state = candidate_state.copy()
                                last_best_step = step
                    elif self.enable_clamp:
                        occupancy = {
                            idx: float(np.mean(next_states[:, idx]))
                            for idx in range(qmatrix_f.shape[0])
                        }
                        self.clamp_manager.update_scores(occupancy)
                    self.logger.log(
                        strategy=strategy["name"],
                        temperature=temperature,
                        improvements=improvements,
                    )
                    prev_states = next_states
                    prev_energies = next_energies

                states = np.ascontiguousarray(history_states[-1], dtype=float)
                energies = np.ascontiguousarray(history_energies[-1], dtype=float)
                rust_rng_state = int(step_stats.get("rng_state", rust_rng_state))
                pool.offer(best_state, best_energy)
                result = pool.to_results(index_map, include_diverse=include_diverse)
                stats = {
                    "best_energy": best_energy,
                    "strategy_weights": self.strategy_manager.weights,
                    "log_entries": self.logger.entries,
                    "clamp_mode": self.clamp_manager.clamp_mode,
                    "rust_step_mode": "multi",
                    "restart_count": restart_events,
                    "pool_mean_pairwise_distance": pool.mean_pairwise_distance(),
                    "state_diversity": self._state_diversity(states),
                }
                if return_stats or self.return_stats:
                    return result, stats
                return result

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
                        for pos, accepted_idx in enumerate(accepted_indices.tolist()):
                            candidate_state = states[accepted_idx]
                            candidate_energy = float(energies[accepted_idx])
                            novelty = pool.min_distance_to_pool(candidate_state)
                            pool.offer(candidate_state, candidate_energy)
                            if self.adaptive:
                                reward = float(-deltas[pos])
                                self.strategy_manager.record(
                                    strategy["name"],
                                    reward + self.novelty_weight * novelty,
                                )
                            if candidate_energy < best_energy:
                                best_energy = candidate_energy
                                best_state = candidate_state.copy()
                                last_best_step = step
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
                    candidate_state = states[i]
                    candidate_energy = float(energies[i])
                    novelty = pool.min_distance_to_pool(candidate_state)
                    pool.offer(candidate_state, candidate_energy)
                    reward = -change + self.novelty_weight * novelty
                    if self.adaptive:
                        self.strategy_manager.record(strategy["name"], reward)
                    if candidate_energy < best_energy:
                        best_energy = candidate_energy
                        best_state = candidate_state.copy()
                        last_best_step = step
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
            state_diversity = self._state_diversity(states)
            if (
                step - last_best_step >= self.stall_steps
                and step - last_restart_step >= self.restart_burnin_steps
                and state_diversity <= diversity_threshold
            ):
                previous_best_energy = best_energy
                states, energies, best_state, best_energy, restart_indices = self._restart_states(
                    states,
                    energies,
                    best_state,
                    best_energy,
                    evaluator,
                )
                for idx in restart_indices.tolist():
                    pool.offer(states[idx], energies[idx])
                restart_events += 1
                last_restart_step = step
                if best_energy < previous_best_energy:
                    last_best_step = step
        pool.offer(best_state, best_energy)
        result = pool.to_results(index_map, include_diverse=include_diverse)
        stats = {
            "best_energy": best_energy,
            "strategy_weights": self.strategy_manager.weights,
            "log_entries": self.logger.entries,
            "clamp_mode": self.clamp_manager.clamp_mode,
            "restart_count": restart_events,
            "pool_mean_pairwise_distance": pool.mean_pairwise_distance(),
            "state_diversity": self._state_diversity(states),
        }
        if return_stats or self.return_stats:
            return result, stats
        return result
