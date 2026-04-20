"""Presolve-aware wrapper around AdaptiveBulkSASampler."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .adaptive_bulk_sa import AdaptiveBulkSASampler
from .delta_evaluator import DeltaEvaluator
from .numeric_normalizer import NumericNormalizer
from .presolve_reducer import PresolveReducer
from .probing_engine import ProbingEngine
from .reduced_qubo_mapper import ReducedQuboMapper


class PresolvedAdaptiveBulkSASampler:
    """Apply normalization + presolve before delegating to AdaptiveBulkSASampler."""

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
        presolve: bool = True,
        presolve_strength: str = "medium",
        enable_soft_fix: bool = True,
        enable_hard_fix: bool = False,
        enable_probing: bool = False,
        enable_aggregation: bool = False,
        normalize_coefficients: bool = True,
        small_coeff_threshold: float = 1e-6,
        penalty_threshold: float = 100.0,
        return_stats: bool = False,
    ) -> None:
        self.adaptive_sampler = AdaptiveBulkSASampler(
            seed=seed,
            shots=shots,
            steps=steps,
            batch_size=batch_size,
            init_temp=init_temp,
            end_temp=end_temp,
            schedule=schedule,
            adaptive=adaptive,
            strategy_configs=strategy_configs,
            epsilon=epsilon,
            enable_clamp=enable_clamp,
            clamp_mode=clamp_mode,
            device=device,
            include_diverse=include_diverse,
            pool_max_entries=pool_max_entries,
            near_dup_hamming=near_dup_hamming,
            replace_margin=replace_margin,
            stall_steps=stall_steps,
            restart_ratio=restart_ratio,
            restart_min_flips=restart_min_flips,
            restart_burnin_steps=restart_burnin_steps,
            restart_diversity_threshold=restart_diversity_threshold,
            novelty_weight=novelty_weight,
        )
        self.numeric_normalizer = NumericNormalizer(
            small_coeff_threshold=small_coeff_threshold,
            penalty_threshold=penalty_threshold,
        )
        self.presolve_reducer = PresolveReducer(
            enable_soft_fix=enable_soft_fix,
            enable_hard_fix=enable_hard_fix,
        )
        self.probing_engine = ProbingEngine(strength=presolve_strength)
        self.normalize_coefficients = normalize_coefficients
        self.presolve = presolve
        self.presolve_strength = presolve_strength
        self.enable_probing = enable_probing
        self.enable_aggregation = enable_aggregation
        self.return_stats = return_stats

    def _normalize(self, matrix: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
        if self.normalize_coefficients:
            normalized, info = self.numeric_normalizer.normalize(matrix)
            return normalized, info
        info = self.numeric_normalizer.analyze(matrix)
        info["after"] = info["before"].copy()
        info["scale_factor"] = 1.0
        return np.asarray(matrix, dtype=float).copy(), info

    def _default_presolve_stats(self, size: int) -> Dict[str, object]:
        return {
            "original_size": size,
            "reduced_size": size,
            "hard_fix_count": 0,
            "soft_fix_count": 0,
            "aggregation_count": 0,
            "fixed_variables": {},
            "soft_fix_candidates": [],
        }

    def run(
        self,
        qubomix: Tuple[np.ndarray, Dict[str, int]],
        shots: Optional[int] = None,
        return_stats: Optional[bool] = None,
        include_diverse: Optional[bool] = None,
    ):  # pragma: no cover
        qmatrix, index_map = qubomix
        normalized_matrix, normalization_info = self._normalize(np.asarray(qmatrix, dtype=float))
        mapper = ReducedQuboMapper(index_map)
        mapper.update_active_indices(list(range(len(index_map))))
        if self.presolve:
            reduced_matrix, presolve_stats = self.presolve_reducer.reduce(
                normalized_matrix,
                mapper,
                strength=self.presolve_strength,
            )
        else:
            reduced_matrix = normalized_matrix
            presolve_stats = self._default_presolve_stats(len(index_map))
        reduced_results, reduced_stats = self._run_reduced(
            reduced_matrix, mapper, shots, include_diverse
        )
        restored_results = mapper.restore_results(reduced_results)
        restore_energy_check = self._evaluate_restored(restored_results, qmatrix, mapper)
        probing_summary = (
            self.probing_engine.evaluate(reduced_matrix, mapper.reduced_index_names())
            if self.enable_probing and mapper.reduced_index_map
            else {"enabled": False}
        )
        stats = {
            "normalization_info": normalization_info,
            "presolve_stats": presolve_stats,
            "probing_summary": probing_summary,
            "reduced_problem_size": len(mapper.reduced_index_map),
            "reduced_solve_stats": reduced_stats,
            "restore_energy_check": restore_energy_check,
        }
        should_return_stats = self.return_stats if return_stats is None else return_stats
        if should_return_stats:
            return restored_results, stats
        return restored_results

    def _run_reduced(
        self,
        reduced_matrix: np.ndarray,
        mapper: ReducedQuboMapper,
        shots: Optional[int],
        include_diverse: Optional[bool],
    ) -> Tuple[List[List[object]], Dict[str, object]]:
        if not mapper.reduced_index_map:
            return [[{}, 0.0, 1]], {
                "best_energy": 0.0,
                "strategy_weights": {},
                "log_entries": [],
                "clamp_mode": self.adaptive_sampler.clamp_manager.clamp_mode,
            }
        solver_shots = shots if shots is not None else self.adaptive_sampler.shots
        reduced_result, reduced_stats = self.adaptive_sampler.run(
            (reduced_matrix, mapper.reduced_index_map),
            shots=solver_shots,
            return_stats=True,
            include_diverse=include_diverse,
        )
        return reduced_result, reduced_stats

    def _evaluate_restored(
        self,
        restored_results: List[List[object]],
        original_matrix: np.ndarray,
        mapper: ReducedQuboMapper,
    ) -> List[Dict[str, object]]:
        evaluator = DeltaEvaluator(np.asarray(original_matrix, dtype=float))
        checks: List[Dict[str, object]] = []
        for state, reported_energy, _ in restored_results:
            state_array = mapper.full_state_array(state)
            checks.append(
                {
                    "state": state,
                    "reported_energy": reported_energy,
                    "restored_energy": evaluator.evaluate(state_array),
                }
            )
        return checks
