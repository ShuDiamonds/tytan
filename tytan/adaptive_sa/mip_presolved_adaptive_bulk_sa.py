"""Rust-first MIP-style presolve wrapper around AdaptiveBulkSASampler."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import _rust_backend as rb

from .adaptive_bulk_sa import AdaptiveBulkSASampler
from .delta_evaluator import DeltaEvaluator
from .numeric_normalizer import NumericNormalizer
from .presolve_reducer import PresolveReducer
from .reduced_qubo_mapper import ReducedQuboMapper


class MIPPresolvedAdaptiveBulkSASampler:
    """Apply Rust MIP-style presolve before delegating to the adaptive sampler."""

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
        normalize_coefficients: bool = True,
        small_coeff_threshold: float = 1e-6,
        penalty_threshold: float = 100.0,
        hard_threshold: float = 1.5,
        soft_threshold: float = 1.0,
        coupling_threshold: float = 0.2,
        aggregation_threshold: float = 0.8,
        weak_cut_threshold: float = 0.1,
        probing_budget: int = 64,
        enable_block_parallel: bool = True,
        enable_fallback_presolve: bool = True,
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
        self.presolve_reducer = PresolveReducer(enable_soft_fix=True, enable_hard_fix=True)
        self.normalize_coefficients = normalize_coefficients
        self.presolve = presolve
        self.hard_threshold = hard_threshold
        self.soft_threshold = soft_threshold
        self.coupling_threshold = coupling_threshold
        self.aggregation_threshold = aggregation_threshold
        self.weak_cut_threshold = weak_cut_threshold
        self.probing_budget = probing_budget
        self.enable_block_parallel = enable_block_parallel
        self.enable_fallback_presolve = enable_fallback_presolve
        self.return_stats = return_stats

    def _normalize(self, matrix: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
        if self.normalize_coefficients:
            normalized, info = self.numeric_normalizer.normalize(matrix)
            return normalized, info
        info = self.numeric_normalizer.analyze(matrix)
        info["after"] = info["before"].copy()
        info["scale_factor"] = 1.0
        return np.asarray(matrix, dtype=float).copy(), info

    def _solve_block(
        self,
        reduced_matrix: np.ndarray,
        mapper: ReducedQuboMapper,
        block_indices: List[int],
        shots: Optional[int],
        include_diverse: Optional[bool],
    ) -> Tuple[List[List[object]], Dict[str, object]]:
        if not block_indices:
            return [[{}, 0.0, 1]], {
                "best_energy": 0.0,
                "strategy_weights": {},
                "log_entries": [],
                "clamp_mode": self.adaptive_sampler.clamp_manager.clamp_mode,
            }

        sub_matrix = reduced_matrix[np.ix_(block_indices, block_indices)].copy()
        local_index_map = {
            mapper.index_to_name[mapper.active_indices[idx]]: idx for idx in block_indices
        }
        solver_shots = shots if shots is not None else self.adaptive_sampler.shots
        return self.adaptive_sampler.run(
            (sub_matrix, local_index_map),
            shots=solver_shots,
            return_stats=True,
            include_diverse=include_diverse,
        )

    def _combine_block_results(
        self,
        block_results: List[Tuple[List[List[object]], Dict[str, object]]],
        mapper: ReducedQuboMapper,
    ) -> List[List[object]]:
        merged_state: Dict[str, int] = dict(mapper.fixed_variables)
        total_energy = 0.0
        total_count = 1
        for results, _stats in block_results:
            if not results:
                continue
            state, energy, count = results[0]
            merged_state.update({str(name): int(value) for name, value in state.items()})
            total_energy += float(energy)
            total_count *= max(1, int(count))
        return [[merged_state, total_energy, total_count]]

    def _fallback_plan(
        self,
        normalized_matrix: np.ndarray,
        mapper: ReducedQuboMapper,
        shots: Optional[int],
        include_diverse: Optional[bool],
    ) -> Tuple[List[List[object]], Dict[str, object], Dict[str, object]]:
        if self.enable_fallback_presolve:
            reduced_matrix, presolve_stats = self.presolve_reducer.reduce(
                normalized_matrix,
                mapper,
                strength="medium",
            )
            reduced_result, reduced_stats = self.adaptive_sampler.run(
                (reduced_matrix, mapper.reduced_index_map),
                shots=shots if shots is not None else self.adaptive_sampler.shots,
                return_stats=True,
                include_diverse=include_diverse,
            )
            return reduced_result, reduced_stats, presolve_stats

        mapper.update_active_indices(list(range(mapper.original_size)))
        reduced_result, reduced_stats = self.adaptive_sampler.run(
            (normalized_matrix, mapper.reduced_index_map),
            shots=shots if shots is not None else self.adaptive_sampler.shots,
            return_stats=True,
            include_diverse=include_diverse,
        )
        return reduced_result, reduced_stats, {
            "original_size": mapper.original_size,
            "reduced_size": mapper.original_size,
            "hard_fix_count": 0,
            "soft_fix_count": 0,
            "aggregation_count": 0,
            "fixed_variables": dict(mapper.fixed_variables),
            "soft_fix_candidates": [],
        }

    def run(
        self,
        qubomix: Tuple[np.ndarray, Dict[str, int]],
        shots: Optional[int] = None,
        return_stats: Optional[bool] = None,
        include_diverse: Optional[bool] = None,
        pool_summary: Optional[Dict[str, np.ndarray]] = None,
    ):
        qmatrix, index_map = qubomix
        qmatrix = np.asarray(qmatrix, dtype=float)
        if qmatrix.ndim != 2 or qmatrix.shape[0] != qmatrix.shape[1]:
            raise ValueError("QUBO matrix must be square")
        normalized_matrix, normalization_info = self._normalize(qmatrix)
        mapper = ReducedQuboMapper(index_map)
        mapper.update_active_indices(list(range(len(index_map))))
        include_diverse = self.adaptive_sampler.include_diverse if include_diverse is None else include_diverse
        should_return_stats = self.return_stats if return_stats is None else return_stats

        if not self.presolve:
            reduced_results, reduced_stats, presolve_stats = self._fallback_plan(
                normalized_matrix,
                mapper,
                shots,
                include_diverse,
            )
        elif rb.mip_presolve_available():
            pool_frequency = None
            pair_correlation = None
            if pool_summary is not None:
                pool_frequency = pool_summary.get("variable_frequency")
                pair_correlation = pool_summary.get("pair_correlation")
            plan = rb.try_mip_presolve_plan(
                normalized_matrix,
                hard_threshold=self.hard_threshold,
                soft_threshold=self.soft_threshold,
                coupling_threshold=self.coupling_threshold,
                aggregation_threshold=self.aggregation_threshold,
                weak_cut_threshold=self.weak_cut_threshold,
                probing_budget=self.probing_budget,
                pool_frequency=pool_frequency,
                pair_correlation=pair_correlation,
            )
            if plan is None:
                raise RuntimeError("Rust MIP presolve backend unexpectedly unavailable")

            for idx, value in zip(plan["hard_fixed_indices"], plan["hard_fixed_values"]):
                name = mapper.index_to_name[int(idx)]
                mapper.register_fixed(name, int(value))

            active_indices = [int(i) for i in plan["active_indices"]]
            mapper.update_active_indices(active_indices)
            reduced_matrix = np.asarray(plan["reduced_matrix"], dtype=float)
            if reduced_matrix.size == 0 or not mapper.reduced_index_map:
                reduced_results = [[dict(mapper.fixed_variables), 0.0, 1]]
                reduced_stats = {
                    "best_energy": 0.0,
                    "strategy_weights": {},
                    "log_entries": [],
                    "clamp_mode": self.adaptive_sampler.clamp_manager.clamp_mode,
                }
            else:
                component_ids = np.asarray(plan.get("block_membership", []), dtype=np.int64)
                active_positions = {idx: pos for pos, idx in enumerate(active_indices)}
                block_groups: Dict[int, List[int]] = {}
                for original_idx in active_indices:
                    component = int(component_ids[original_idx]) if component_ids.size > original_idx else 0
                    block_groups.setdefault(component, []).append(original_idx)

                if len(block_groups) > 1:
                    block_results: List[Tuple[List[List[object]], Dict[str, object]]] = []

                    def submit_block(indices: List[int]):
                        positions = [active_positions[i] for i in indices]
                        sub_matrix = reduced_matrix[np.ix_(positions, positions)].copy()
                        local_index_map = {
                            mapper.index_to_name[i]: pos for pos, i in enumerate(indices)
                        }
                        return self.adaptive_sampler.run(
                            (sub_matrix, local_index_map),
                            shots=shots if shots is not None else self.adaptive_sampler.shots,
                            return_stats=True,
                            include_diverse=include_diverse,
                        )

                    if self.enable_block_parallel:
                        with ThreadPoolExecutor(
                            max_workers=min(len(block_groups), os.cpu_count() or len(block_groups))
                        ) as executor:
                            future_map = {
                                executor.submit(submit_block, indices): indices
                                for indices in block_groups.values()
                            }
                            for future in as_completed(future_map):
                                block_results.append(future.result())
                    else:
                        for indices in block_groups.values():
                            block_results.append(submit_block(indices))

                    reduced_results = self._combine_block_results(block_results, mapper)
                    reduced_stats = {
                        "best_energy": reduced_results[0][1],
                        "strategy_weights": {},
                        "log_entries": [],
                        "clamp_mode": self.adaptive_sampler.clamp_manager.clamp_mode,
                        "block_count": len(block_groups),
                        "mip_block_results": [stats for _, stats in block_results],
                    }
                else:
                    reduced_results, reduced_stats = self.adaptive_sampler.run(
                        (reduced_matrix, mapper.reduced_index_map),
                        shots=shots if shots is not None else self.adaptive_sampler.shots,
                        return_stats=True,
                        include_diverse=include_diverse,
                    )

            presolve_stats = {
                "original_size": int(len(index_map)),
                "reduced_size": int(len(active_indices)),
                "hard_fix_count": int(len(plan["hard_fixed_indices"])),
                "soft_fix_count": int(len(plan["soft_fixed_indices"])),
                "aggregation_count": int(len(plan["aggregation_src"])),
                "pool_summary_used": bool(pool_summary is not None),
                "block_count": int(len(np.unique(np.asarray(plan.get("block_membership", []), dtype=np.int64))))
                if len(plan.get("block_membership", []))
                else 0,
                "boundary_count": int(len(plan.get("boundary_indices", []))),
                "frontier_count": int(len(plan.get("frontier_indices", []))),
                "branch_candidate_count": int(len(plan.get("branch_candidate_indices", []))),
                "stats": dict(plan.get("stats", {})),
            }
        else:
            reduced_results, reduced_stats, presolve_stats = self._fallback_plan(
                normalized_matrix,
                mapper,
                shots,
                include_diverse,
            )

        restored_results = mapper.restore_results(reduced_results)
        restore_energy_check = self._evaluate_restored(restored_results, qmatrix, mapper)
        stats = {
            "normalization_info": normalization_info,
            "presolve_stats": presolve_stats,
            "reduced_problem_size": len(mapper.reduced_index_map),
            "reduced_solve_stats": reduced_stats,
            "restore_energy_check": restore_energy_check,
            "presolve_backend": "rust" if rb.mip_presolve_available() and self.presolve else "python",
        }
        if should_return_stats:
            return restored_results, stats
        return restored_results

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
