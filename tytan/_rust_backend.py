"""Optional Rust backend bridge for TYTAN.

This module never raises on import failure of the Rust extension.
It provides probe-style helpers that return None when unavailable.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np


def _debug_enabled() -> bool:
    return os.getenv("TYTAN_RUST_DEBUG", "0") == "1"


def _mode() -> str:
    return os.getenv("TYTAN_RUST", "auto").strip().lower()


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
    state_f = np.asarray(state, dtype=float)
    qmatrix_f = np.asarray(qmatrix, dtype=float)
    return float(_RUST_MODULE.delta_energy(state_f, qmatrix_f, int(index), current_energy))


def try_batch_delta(
    states: np.ndarray,
    qmatrix: np.ndarray,
    indices: np.ndarray,
    energies: np.ndarray,
) -> Optional[np.ndarray]:
    if _RUST_MODULE is None:
        return None
    states_f = np.asarray(states, dtype=float)
    qmatrix_f = np.asarray(qmatrix, dtype=float)
    indices_i = np.asarray(indices, dtype=np.int64)
    energies_f = np.asarray(energies, dtype=float)
    return np.asarray(_RUST_MODULE.batch_delta(states_f, qmatrix_f, indices_i, energies_f), dtype=float)


def try_aggregate_results(
    states: np.ndarray,
    energies: np.ndarray,
    variable_names: Sequence[str],
):
    if _RUST_MODULE is None:
        return None
    states_f = np.asarray(states, dtype=float)
    energies_f = np.asarray(energies, dtype=float)
    return _RUST_MODULE.aggregate_results(states_f, energies_f, list(variable_names))
