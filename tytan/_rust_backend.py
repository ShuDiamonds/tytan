"""Optional Rust backend bridge for TYTAN.

This module never raises on import failure of the Rust extension.
It provides probe-style helpers that return None when unavailable.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

_SYMMETRY_CACHE: dict[tuple[int, tuple[int, ...], tuple[int, ...]], bool] = {}


def _as_float64_c(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype == np.float64 and arr.flags.c_contiguous:
        return arr
    return np.ascontiguousarray(arr, dtype=float)


def _as_int64_c(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype == np.int64 and arr.flags.c_contiguous:
        return arr
    return np.ascontiguousarray(arr, dtype=np.int64)


def _matrix_cache_key(array: np.ndarray) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    return (
        int(array.__array_interface__["data"][0]),
        tuple(int(v) for v in array.shape),
        tuple(int(v) for v in array.strides),
    )


def _is_symmetric_matrix(array: np.ndarray) -> bool:
    arr = _as_float64_c(array)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return False
    key = _matrix_cache_key(arr)
    cached = _SYMMETRY_CACHE.get(key)
    if cached is not None:
        return cached
    symmetric = bool(np.array_equal(arr, arr.T))
    _SYMMETRY_CACHE[key] = symmetric
    return symmetric


def _debug_enabled() -> bool:
    return os.getenv("TYTAN_RUST_DEBUG", "0") == "1"


def _mode() -> str:
    return os.getenv("TYTAN_RUST", "auto").strip().lower()


def rust_min_work() -> int:
    """Return minimum work threshold for enabling Rust hot paths."""
    raw = os.getenv("TYTAN_RUST_MIN_WORK", "0").strip()
    try:
        return max(int(raw), 0)
    except ValueError:
        return 0


def rust_step_min_work() -> int:
    """Return minimum work threshold for enabling whole-step Rust path."""
    raw = os.getenv("TYTAN_RUST_STEP_MIN_WORK", "4096").strip()
    try:
        return max(int(raw), 0)
    except ValueError:
        return 4096


def _load_rust_module():
    mode = _mode()
    if mode == "off":
        return None

    # Prefer packaged namespace, then fallback to top-level module.
    try:
        from tytan import _tytan_rust as mod  # type: ignore
        return mod
    except Exception:
        pass

    try:
        import _tytan_rust as mod  # type: ignore
        return mod
    except Exception:
        if mode == "on":
            raise
        return None


_RUST_MODULE = _load_rust_module()


if _debug_enabled():
    print(f"[TYTAN_RUST] mode={_mode()} available={_RUST_MODULE is not None}")


def rust_available() -> bool:
    return _RUST_MODULE is not None


def try_delta_energy(
    state: np.ndarray,
    qmatrix: np.ndarray,
    index: int,
    current_energy: Optional[float] = None,
) -> Optional[float]:
    if _RUST_MODULE is None:
        return None
    state_f = _as_float64_c(state)
    qmatrix_f = _as_float64_c(qmatrix)
    symmetric = _is_symmetric_matrix(qmatrix_f)
    return float(_RUST_MODULE.delta_energy(state_f, qmatrix_f, int(index), current_energy, symmetric))


def try_delta_energy_fast(
    state_f: np.ndarray,
    qmatrix_f: np.ndarray,
    index: int,
    current_energy: Optional[float] = None,
) -> Optional[float]:
    """Fast path for pre-normalized float arrays.

    Caller must ensure input arrays have compatible dtype/shape.
    """
    if _RUST_MODULE is None:
        return None
    symmetric = _is_symmetric_matrix(qmatrix_f)
    return float(_RUST_MODULE.delta_energy(state_f, qmatrix_f, int(index), current_energy, symmetric))


def try_batch_delta(
    states: np.ndarray,
    qmatrix: np.ndarray,
    indices: np.ndarray,
    energies: np.ndarray,
) -> Optional[np.ndarray]:
    if _RUST_MODULE is None:
        return None
    states_f = _as_float64_c(states)
    qmatrix_f = _as_float64_c(qmatrix)
    indices_i = _as_int64_c(indices)
    energies_f = _as_float64_c(energies)
    symmetric = _is_symmetric_matrix(qmatrix_f)
    return np.asarray(_RUST_MODULE.batch_delta(states_f, qmatrix_f, indices_i, energies_f, symmetric), dtype=float)


def try_batch_delta_fast(
    states_f: np.ndarray,
    qmatrix_f: np.ndarray,
    indices_i: np.ndarray,
    energies_f: np.ndarray,
) -> Optional[np.ndarray]:
    """Fast path for pre-normalized batch delta arrays."""
    if _RUST_MODULE is None:
        return None
    symmetric = _is_symmetric_matrix(qmatrix_f)
    return np.asarray(_RUST_MODULE.batch_delta(states_f, qmatrix_f, indices_i, energies_f, symmetric), dtype=float)


def try_aggregate_results(
    states: np.ndarray,
    energies: np.ndarray,
    variable_names: Sequence[str],
):
    if _RUST_MODULE is None:
        return None
    states_f = _as_float64_c(states)
    energies_f = _as_float64_c(energies)
    return _RUST_MODULE.aggregate_results(states_f, energies_f, list(variable_names))


def try_aggregate_results_fast(
    states_f: np.ndarray,
    energies_f: np.ndarray,
    variable_names: Sequence[str],
):
    """Fast path for pre-normalized result aggregation arrays."""
    if _RUST_MODULE is None:
        return None
    return _RUST_MODULE.aggregate_results(states_f, energies_f, list(variable_names))


def try_sa_step_single_flip(
    states: np.ndarray,
    energies: np.ndarray,
    qmatrix: np.ndarray,
    beta: float,
    rng_state: int,
):
    if _RUST_MODULE is None:
        return None
    states_f = _as_float64_c(states)
    energies_f = _as_float64_c(energies)
    qmatrix_f = _as_float64_c(qmatrix)
    symmetric = _is_symmetric_matrix(qmatrix_f)
    next_states, next_energies, stats = _RUST_MODULE.sa_step_single_flip(
        states_f,
        energies_f,
        qmatrix_f,
        float(beta),
        int(rng_state),
        symmetric,
    )
    return (
        _as_float64_c(next_states),
        _as_float64_c(next_energies),
        dict(stats),
    )


def try_sa_step_multi_flip(
    states: np.ndarray,
    energies: np.ndarray,
    qmatrix: np.ndarray,
    betas: np.ndarray,
    rng_state: int,
):
    if _RUST_MODULE is None:
        return None
    states_f = _as_float64_c(states)
    energies_f = _as_float64_c(energies)
    qmatrix_f = _as_float64_c(qmatrix)
    betas_f = _as_float64_c(betas)
    symmetric = _is_symmetric_matrix(qmatrix_f)
    history_states, history_energies, stats = _RUST_MODULE.sa_step_multi_flip(
        states_f,
        energies_f,
        qmatrix_f,
        betas_f,
        int(rng_state),
        symmetric,
    )
    steps = int(len(betas_f))
    shots = int(states_f.shape[0])
    dims = int(states_f.shape[1])
    history_states = np.asarray(history_states, dtype=float).reshape(steps, shots, dims)
    history_energies = np.asarray(history_energies, dtype=float).reshape(steps, shots)
    return history_states, history_energies, dict(stats)
