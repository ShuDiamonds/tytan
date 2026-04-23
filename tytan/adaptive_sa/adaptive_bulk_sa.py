"""Adaptive bulk SA sampler that orchestrates the modular helpers."""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import _rust_backend

from .anneal_logger import AnnealLogger
from .clamp_manager import ClampManager
from .delta_evaluator import DeltaEvaluator
from .sparse_qubo import SparseNeighbors, build_sparse_neighbors
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
        phase2_enabled: bool = False,
        phase2_start_step: Optional[int] = None,
        phase2_top_k: int = 16,
        phase2_sweeps_per_step: int = 1,
        sparse_threshold: float = 0.0,
        pool_offer_mode: str = "per_flip",
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
        if phase2_top_k < 0:
            raise ValueError("phase2_top_k cannot be negative")
        if phase2_sweeps_per_step < 1:
            raise ValueError("phase2_sweeps_per_step must be positive")
        if sparse_threshold < 0:
            raise ValueError("sparse_threshold cannot be negative")
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
        self.phase2_enabled = bool(phase2_enabled)
        self.phase2_start_step = phase2_start_step
        self.phase2_top_k = int(phase2_top_k)
        self.phase2_sweeps_per_step = int(phase2_sweeps_per_step)
        self.sparse_threshold = float(sparse_threshold)
        self.pool_offer_mode = str(pool_offer_mode).strip().lower()
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

    def _phase2_start(self) -> int:
        if not self.phase2_enabled:
            return int(self.steps)
        if self.phase2_start_step is None:
            return max(0, int(self.steps) // 2)
        return int(min(max(int(self.phase2_start_step), 0), int(self.steps)))

    @staticmethod
    def _pool_offer_mode_valid(mode: str) -> bool:
        return mode in {"per_flip", "phase_end", "off"}

    @staticmethod
    def _offer_phase_end(pool: SolutionPool, states: np.ndarray, energies: np.ndarray) -> None:
        for idx in range(states.shape[0]):
            pool.offer(states[idx], float(energies[idx]))

    def _phase2_betas(self, start_step: int) -> np.ndarray:
        betas = np.empty(max(0, int(self.steps) - int(start_step)), dtype=float)
        for pos, step in enumerate(range(int(start_step), int(self.steps))):
            strategy = self.strategy_manager.select()
            temperature = float(max(self._temperature(step, strategy.get("type", "linear")), 1e-8))
            betas[pos] = 1.0 / temperature
        return betas

    @staticmethod
    def _format_rows(
        states_i: np.ndarray,
        energies_f: np.ndarray,
        counts_i: np.ndarray,
        index_map: Dict[str, int],
    ) -> List[List[object]]:
        results: List[List[object]] = []
        for row_idx in range(int(states_i.shape[0])):
            state = states_i[row_idx]
            mapping = {name: int(state[idx]) for name, idx in index_map.items()}
            results.append([mapping, float(energies_f[row_idx]), int(counts_i[row_idx])])
        return results

    def _pool_select_final(
        self,
        states: np.ndarray,
        energies: np.ndarray,
        index_map: Dict[str, int],
        include_diverse: bool,
    ) -> Tuple[List[List[object]], float]:
        rust = _rust_backend.try_pool_select(
            states,
            energies,
            best_k=min(self.batch_size, int(states.shape[0])),
            diverse_k=2 if include_diverse else 0,
            max_entries=self.pool_max_entries,
            near_dup_hamming=self.near_dup_hamming,
            replace_margin=self.replace_margin,
            include_diverse=include_diverse,
        )
        if rust is not None:
            pool_states, pool_energies, pool_counts, mean_dist = rust
            pool_states_i = np.asarray(pool_states, dtype=np.int64)
            pool_energies_f = np.asarray(pool_energies, dtype=float)
            pool_counts_i = np.asarray(pool_counts, dtype=np.int64)
            return self._format_rows(pool_states_i, pool_energies_f, pool_counts_i, index_map), float(mean_dist)

        pool = SolutionPool(
            best_k=min(self.batch_size, int(states.shape[0])),
            diverse_k=2 if include_diverse else 0,
            max_entries=self.pool_max_entries,
            near_dup_hamming=self.near_dup_hamming,
            replace_margin=self.replace_margin,
        )
        self._offer_phase_end(pool, states, energies)
        return pool.to_results(index_map, include_diverse=include_diverse), pool.mean_pairwise_distance()

    @staticmethod
    def _delta_cache_init(states: np.ndarray, qmatrix: np.ndarray) -> np.ndarray:
        """Initialize delta-cache for 0/1 state representation.

        Uses the identity:
          ΔE_i = flip_i * ((Q + Q.T) @ state)_i + Q_ii
        where flip_i = 1 - 2 * state_i.
        """

        q = np.asarray(qmatrix, dtype=float)
        qsym = q + q.T
        diag = np.diag(q)
        flip = 1.0 - 2.0 * states
        cross = states @ qsym.T
        return flip * cross + diag

    def _run_phase2_delta_cache(
        self,
        states: np.ndarray,
        energies: np.ndarray,
        qmatrix: np.ndarray,
        sparse: SparseNeighbors,
        start_step: int,
        pool: SolutionPool,
        best_state: np.ndarray,
        best_energy: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        """Phase2: delta-cache SA on top-k states (CPU)."""

        if start_step >= self.steps:
            return states, energies, best_state, best_energy, 0
        if self.phase2_top_k < 1:
            return states, energies, best_state, best_energy, 0
        top_k = min(int(self.phase2_top_k), int(states.shape[0]))
        selected = np.argsort(energies)[:top_k]
        sel_states = np.ascontiguousarray(states[selected], dtype=float)
        sel_energies = np.ascontiguousarray(energies[selected], dtype=float)
        # Prefer Rust implementation if available; it keeps the delta-cache updates in native code.
        if _rust_backend.phase2_available():
            betas = self._phase2_betas(start_step)
            # Use Rust CSR builder when available, else fall back to Python builder.
            built = _rust_backend.try_build_sparse_neighbors(qmatrix, threshold=self.sparse_threshold)
            if built is None:
                built_py = build_sparse_neighbors(qmatrix, threshold=self.sparse_threshold)
                offsets, neigh, weights = built_py.offsets, built_py.neighbors, built_py.weights
            else:
                offsets, neigh, weights = built
            rust_rng_state = int(self.rng.randint(1, np.iinfo(np.int64).max))
            rust_result = _rust_backend.try_sa_phase2_delta_cache(
                states,
                energies,
                qmatrix,
                offsets,
                neigh,
                weights,
                betas,
                sweeps_per_step=int(self.phase2_sweeps_per_step),
                rng_state=rust_rng_state,
                top_k=int(self.phase2_top_k),
            )
            if rust_result is not None:
                next_states, next_energies, stats = rust_result
                states = np.ascontiguousarray(np.asarray(next_states), dtype=float)
                energies = np.ascontiguousarray(np.asarray(next_energies), dtype=float)
                improvements_total = int(stats.get("accepted", 0))
                best_idx = int(np.argmin(energies))
                candidate_best_energy = float(energies[best_idx])
                if candidate_best_energy < best_energy:
                    best_energy = candidate_best_energy
                    best_state = states[best_idx].copy()
                if self.pool_offer_mode == "phase_end":
                    self._offer_phase_end(pool, states, energies)
                elif self.pool_offer_mode == "per_flip":
                    # Keep behavior similar: offer the selected set at the end of Phase2.
                    top_k = min(int(self.phase2_top_k), int(states.shape[0]))
                    selected = np.argsort(energies)[:top_k]
                    for idx in selected.tolist():
                        pool.offer(states[int(idx)], float(energies[int(idx)]))
                pool.offer(best_state, best_energy)
                return states, energies, best_state, best_energy, improvements_total

        delta_cache = self._delta_cache_init(sel_states, qmatrix)

        n = int(sel_states.shape[1])
        offsets = sparse.offsets
        neigh = sparse.neighbors
        weights = sparse.weights

        improvements_total = 0
        for step in range(int(start_step), int(self.steps)):
            strategy = self.strategy_manager.select()
            temperature = float(max(self._temperature(step, strategy.get("type", "linear")), 1e-8))
            improvements_step = 0
            for _ in range(max(1, int(self.phase2_sweeps_per_step))):
                for r in range(top_k):
                    idx = int(self.rng.randint(n))
                    change = float(delta_cache[r, idx])
                    if change <= 0.0 or self.rng.rand() < np.exp(-change / temperature):
                        flip_i = 1.0 - 2.0 * sel_states[r, idx]
                        sel_states[r, idx] = 1.0 - sel_states[r, idx]
                        sel_energies[r] += change
                        delta_cache[r, idx] = -delta_cache[r, idx]

                        start = int(offsets[idx])
                        end = int(offsets[idx + 1])
                        if start != end:
                            js = neigh[start:end]
                            ws = weights[start:end]
                            flips_j = 1.0 - 2.0 * sel_states[r, js]
                            delta_cache[r, js] += flips_j * ws * flip_i
                        improvements_total += 1
                        improvements_step += 1

            self.logger.log(
                strategy=strategy["name"],
                temperature=temperature,
                improvements=improvements_step,
            )

        states[selected] = sel_states
        energies[selected] = sel_energies

        best_idx = int(np.argmin(energies))
        candidate_best_energy = float(energies[best_idx])
        if candidate_best_energy < best_energy:
            best_energy = candidate_best_energy
            best_state = states[best_idx].copy()

        if self.pool_offer_mode == "phase_end":
            self._offer_phase_end(pool, states, energies)
        elif self.pool_offer_mode == "per_flip":
            for idx in selected.tolist():
                pool.offer(states[int(idx)], float(energies[int(idx)]))

        pool.offer(best_state, best_energy)
        return states, energies, best_state, best_energy, improvements_total

    def _run_phase1_gpu(
        self,
        qmatrix: np.ndarray,
        states: np.ndarray,
        energies: np.ndarray,
        steps: int,
        pool: SolutionPool,
        best_state: np.ndarray,
        best_energy: float,
    ):
        """Phase1: dense batch SA with PyTorch on the configured device.

        This keeps the same 0/1 state representation as the CPU path.
        """

        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "device is not cpu, but PyTorch is not available in the current environment. "
                "If you are using `uv run`, install torch into that environment (e.g. "
                "`/bin/zsh -lc \"UV_CACHE_DIR=.uv-cache uv add torch\"`)."
            ) from exc

        device = torch.device(self.device)
        dev_str = str(device).lower()
        if dev_str.startswith("mps") and not torch.backends.mps.is_available():
            raise ImportError(
                "device is set to mps, but torch.backends.mps.is_available() is False in this environment. "
                "Run with device='cpu' or fix the PyTorch/MPS setup."
            )
        if dev_str.startswith("cuda") and not torch.cuda.is_available():
            raise ImportError(
                "device is set to cuda, but torch.cuda.is_available() is False in this environment. "
                "Run with device='cpu' or fix the PyTorch/CUDA setup."
            )
        q = torch.tensor(np.asarray(qmatrix, dtype=np.float32), device=device)
        qsym = q + q.T
        diag = torch.diag(q)

        torch_states = torch.tensor(np.asarray(states, dtype=np.float32), device=device)
        torch_energies = torch.tensor(np.asarray(energies, dtype=np.float32), device=device)
        shots = int(torch_states.shape[0])
        n = int(torch_states.shape[1])
        rows = torch.arange(shots, device=device, dtype=torch.long)

        improvements_total = 0
        for step in range(int(steps)):
            strategy = self.strategy_manager.select()
            temperature = float(max(self._temperature(step, strategy.get("type", "linear")), 1e-8))
            idx = torch.randint(0, n, (shots,), device=device, dtype=torch.long)
            s_i = torch_states[rows, idx]
            flip = 1.0 - 2.0 * s_i
            cross = (qsym[idx] * torch_states).sum(dim=1)
            delta = flip * cross + diag[idx]

            accept = (delta <= 0.0) | (torch.rand(shots, device=device) < torch.exp(-delta / temperature))
            improvements = 0
            if bool(accept.any()):
                acc_rows = rows[accept]
                acc_idx = idx[accept]
                torch_states[acc_rows, acc_idx] = 1.0 - torch_states[acc_rows, acc_idx]
                torch_energies[acc_rows] += delta[accept]
                improvements = int(acc_rows.numel())
                improvements_total += improvements

                if self.pool_offer_mode == "per_flip":
                    acc_states_cpu = torch_states[acc_rows].detach().to("cpu").numpy()
                    acc_energies_cpu = torch_energies[acc_rows].detach().to("cpu").numpy()
                    for s, e in zip(acc_states_cpu, acc_energies_cpu):
                        pool.offer(s.astype(float), float(e))

            self.logger.log(strategy=strategy["name"], temperature=temperature, improvements=improvements)

            if self.enable_clamp and improvements == 0:
                occ = torch_states.mean(dim=0).detach().to("cpu").numpy()
                occupancy = {i: float(occ[i]) for i in range(n)}
                self.clamp_manager.update_scores(occupancy)

            step_best_idx = int(torch_energies.argmin().item())
            step_best_energy = float(torch_energies[step_best_idx].item())
            if step_best_energy < best_energy:
                best_energy = step_best_energy
                best_state = torch_states[step_best_idx].detach().to("cpu").numpy().astype(float)

        # Return the torch tensors to avoid forcing a full device->host transfer unless needed.
        pool.offer(best_state, best_energy)
        return torch_states, torch_energies, best_state, best_energy, improvements_total

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
        if not self._pool_offer_mode_valid(self.pool_offer_mode):
            raise ValueError("pool_offer_mode must be one of: per_flip, phase_end, off")
        if (
            self.device == "cpu"
            and _rust_backend.adaptive_bulk_sa_available()
            and not self.enable_clamp
            and not self.phase2_enabled
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

        phase2_start = self._phase2_start()
        if self.device != "cpu":
            states = self.rng.randint(0, 2, size=(shots, qmatrix_f.shape[0])).astype(float)
            states = np.ascontiguousarray(states)
            evaluator = DeltaEvaluator(qmatrix_f)
            energies = np.array([evaluator.evaluate(state) for state in states])
            energies = np.ascontiguousarray(energies, dtype=float)

            best_idx = int(np.argmin(energies))
            best_energy = float(energies[best_idx])
            best_state = states[best_idx].copy()

            phase1_steps = min(int(phase2_start), int(self.steps))
            pool = SolutionPool(
                best_k=min(self.batch_size, shots),
                diverse_k=2,
                max_entries=self.pool_max_entries,
                near_dup_hamming=self.near_dup_hamming,
                replace_margin=self.replace_margin,
            )
            torch_states, torch_energies, best_state, best_energy, phase1_improvements = self._run_phase1_gpu(
                qmatrix_f,
                states,
                energies,
                phase1_steps,
                pool,
                best_state,
                best_energy,
            )

            phase2_improvements = 0
            if self.pool_offer_mode == "per_flip":
                states = torch_states.detach().to("cpu").numpy().astype(float)
                energies = torch_energies.detach().to("cpu").numpy().astype(float)
                if self.phase2_enabled and phase2_start < self.steps:
                    sparse = build_sparse_neighbors(qmatrix_f, threshold=self.sparse_threshold)
                    states, energies, best_state, best_energy, phase2_improvements = self._run_phase2_delta_cache(
                        states,
                        energies,
                        qmatrix_f,
                        sparse,
                        int(phase2_start),
                        pool,
                        best_state,
                        best_energy,
                    )
                if self.pool_offer_mode == "off":
                    pool.offer(best_state, best_energy)
                result = pool.to_results(index_map, include_diverse=include_diverse)
                pool_mean_dist = pool.mean_pairwise_distance()
                state_div = self._state_diversity(states)
            else:
                # Phase-end mode: avoid transferring all states. Work only on a limited candidate set.
                try:
                    import torch
                except Exception:  # pragma: no cover
                    torch = None
                if torch is None:
                    # Fallback: transfer everything.
                    cand_states = torch_states.detach().to("cpu").numpy().astype(float)
                    cand_energies = torch_energies.detach().to("cpu").numpy().astype(float)
                else:
                    candidate_count = min(
                        shots,
                        max(int(self.pool_max_entries), int(self.phase2_top_k), min(int(self.batch_size) + 2, shots)),
                    )
                    _, cand_idx = torch.topk(torch_energies, k=int(candidate_count), largest=False)
                    cand_states = torch_states[cand_idx].detach().to("cpu").numpy().astype(float)
                    cand_energies = torch_energies[cand_idx].detach().to("cpu").numpy().astype(float)

                if self.phase2_enabled and phase2_start < self.steps and _rust_backend.phase2_available():
                    betas = self._phase2_betas(int(phase2_start))
                    built = _rust_backend.try_build_sparse_neighbors(qmatrix_f, threshold=self.sparse_threshold)
                    if built is None:
                        built_py = build_sparse_neighbors(qmatrix_f, threshold=self.sparse_threshold)
                        offsets, neigh, weights = built_py.offsets, built_py.neighbors, built_py.weights
                    else:
                        offsets, neigh, weights = built
                    rust_rng_state = int(self.rng.randint(1, np.iinfo(np.int64).max))
                    rust_result = _rust_backend.try_sa_phase2_delta_cache(
                        cand_states,
                        cand_energies,
                        qmatrix_f,
                        offsets,
                        neigh,
                        weights,
                        betas,
                        sweeps_per_step=int(self.phase2_sweeps_per_step),
                        rng_state=rust_rng_state,
                        top_k=min(int(self.phase2_top_k), int(cand_states.shape[0])),
                    )
                    if rust_result is not None:
                        next_states, next_energies, stats = rust_result
                        cand_states = np.ascontiguousarray(np.asarray(next_states), dtype=float)
                        cand_energies = np.ascontiguousarray(np.asarray(next_energies), dtype=float)
                        phase2_improvements = int(stats.get("accepted", 0))

                best_idx = int(np.argmin(cand_energies))
                best_energy = float(cand_energies[best_idx])
                best_state = cand_states[best_idx].copy()
                result, pool_mean_dist = self._pool_select_final(
                    cand_states,
                    cand_energies,
                    index_map,
                    include_diverse=include_diverse,
                )
                state_div = self._state_diversity(cand_states)

            stats = {
                "best_energy": best_energy,
                "strategy_weights": self.strategy_manager.weights,
                "log_entries": self.logger.entries,
                "clamp_mode": self.clamp_manager.clamp_mode,
                "restart_count": 0,
                "pool_mean_pairwise_distance": pool_mean_dist,
                "state_diversity": state_div,
                "phase1_improvements": phase1_improvements,
                "phase2_improvements": phase2_improvements,
                "phase2_enabled": bool(self.phase2_enabled),
                "phase2_start_step": int(phase2_start),
            }
            if return_stats or self.return_stats:
                return result, stats
            return result
        states = self.rng.randint(0, 2, size=(shots, qmatrix_f.shape[0])).astype(float)
        states = np.ascontiguousarray(states)
        evaluator = DeltaEvaluator(qmatrix_f)
        energies = np.array([evaluator.evaluate(state) for state in states])
        energies = np.ascontiguousarray(energies, dtype=float)
        rust_rng_state = int(self.rng.randint(1, np.iinfo(np.int64).max))
        pool_active = self.pool_offer_mode == "per_flip"
        pool = (
            SolutionPool(
                best_k=min(self.batch_size, shots),
                diverse_k=2,
                max_entries=self.pool_max_entries,
                near_dup_hamming=self.near_dup_hamming,
                replace_margin=self.replace_margin,
            )
            if pool_active
            else None
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
            and not self.phase2_enabled
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
                            novelty = pool.min_distance_to_pool(candidate_state) if pool_active else 0.0
                            if pool_active:
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
                if pool_active:
                    pool.offer(best_state, best_energy)
                    result = pool.to_results(index_map, include_diverse=include_diverse)
                    pool_mean_dist = pool.mean_pairwise_distance()
                else:
                    result, pool_mean_dist = self._pool_select_final(
                        states,
                        energies,
                        index_map,
                        include_diverse=include_diverse,
                    )
                stats = {
                    "best_energy": best_energy,
                    "strategy_weights": self.strategy_manager.weights,
                    "log_entries": self.logger.entries,
                    "clamp_mode": self.clamp_manager.clamp_mode,
                    "rust_step_mode": "multi",
                    "restart_count": restart_events,
                    "pool_mean_pairwise_distance": pool_mean_dist,
                    "state_diversity": self._state_diversity(states),
                }
                if return_stats or self.return_stats:
                    return result, stats
                return result

        sparse = None
        if self.phase2_enabled and phase2_start < self.steps:
            sparse = build_sparse_neighbors(qmatrix_f, threshold=self.sparse_threshold)

        for step in range(min(self.steps, phase2_start)):
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
                            novelty = pool.min_distance_to_pool(candidate_state) if pool_active else 0.0
                            if pool_active:
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
                    novelty = pool.min_distance_to_pool(candidate_state) if pool_active else 0.0
                    if pool_active:
                        pool.offer(candidate_state, candidate_energy)
                    reward = -change + (self.novelty_weight * novelty if pool_active else 0.0)
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
                if pool_active:
                    for idx in restart_indices.tolist():
                        pool.offer(states[idx], energies[idx])
                restart_events += 1
                last_restart_step = step
                if best_energy < previous_best_energy:
                    last_best_step = step

        phase2_improvements = 0
        if sparse is not None and phase2_start < self.steps:
            states, energies, best_state, best_energy, phase2_improvements = self._run_phase2_delta_cache(
                states,
                energies,
                qmatrix_f,
                sparse,
                int(phase2_start),
                pool if pool is not None else SolutionPool(best_k=1, diverse_k=0),
                best_state,
                best_energy,
            )
        if pool_active:
            pool.offer(best_state, best_energy)
            result = pool.to_results(index_map, include_diverse=include_diverse)
            pool_mean_dist = pool.mean_pairwise_distance()
        else:
            result, pool_mean_dist = self._pool_select_final(
                states,
                energies,
                index_map,
                include_diverse=include_diverse,
            )
        stats = {
            "best_energy": best_energy,
            "strategy_weights": self.strategy_manager.weights,
            "log_entries": self.logger.entries,
            "clamp_mode": self.clamp_manager.clamp_mode,
            "restart_count": restart_events,
            "pool_mean_pairwise_distance": pool_mean_dist,
            "state_diversity": self._state_diversity(states),
            "phase2_improvements": phase2_improvements,
            "phase2_enabled": bool(self.phase2_enabled),
            "phase2_start_step": int(phase2_start),
        }
        if return_stats or self.return_stats:
            return result, stats
        return result
