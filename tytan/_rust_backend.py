"""Optional Rust backend bridge for TYTAN.

This module never raises on import failure of the Rust extension.
It provides probe-style helpers that return None when unavailable.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np


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
    return float(_RUST_MODULE.delta_energy(state_f, qmatrix_f, int(index), current_energy))


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
    return float(_RUST_MODULE.delta_energy(state_f, qmatrix_f, int(index), current_energy))


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
    return np.asarray(_RUST_MODULE.batch_delta(states_f, qmatrix_f, indices_i, energies_f), dtype=float)


def try_batch_delta_fast(
    states_f: np.ndarray,
    qmatrix_f: np.ndarray,
    indices_i: np.ndarray,
    energies_f: np.ndarray,
) -> Optional[np.ndarray]:
    """Fast path for pre-normalized batch delta arrays."""
    if _RUST_MODULE is None:
        return None
    return np.asarray(_RUST_MODULE.batch_delta(states_f, qmatrix_f, indices_i, energies_f), dtype=float)


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
