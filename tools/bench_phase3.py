"""
Benchmark: Phase 3 effect only (sa_step_single_flip).

Compares three execution paths for one SA step across all shots:
  - pure_python : Python acceptance loop + numpy delta (no Rust)
  - batch_delta : Rust batch_delta + Python acceptance loop (Phase 2 only)
  - sa_step_rust: Rust sa_step_single_flip (Phase 3 — inner loop in Rust)

Run:
    python tools/bench_phase3.py
or with custom parameters:
    SHOTS=128 DIMS=256 STEPS=5 REPEATS=7 python tools/bench_phase3.py
"""
from __future__ import annotations

import math
import os
import statistics
import time
from typing import List, Tuple

import numpy as np

from tytan import _rust_backend

SHOTS = int(os.getenv("SHOTS", "64"))
DIMS = int(os.getenv("DIMS", "128"))
STEPS = int(os.getenv("STEPS", "200"))   # SA steps per trial
REPEATS = int(os.getenv("REPEATS", "5"))
SEED = int(os.getenv("SEED", "42"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_problem(shots: int, dims: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    q = rng.randn(dims, dims)
    q = np.ascontiguousarray((q + q.T) / 2.0, dtype=np.float64)
    states = np.ascontiguousarray(rng.randint(0, 2, (shots, dims)).astype(np.float64))
    energies = np.array([float(s @ q @ s) for s in states], dtype=np.float64)
    return q, states, energies


def _delta_python(state: np.ndarray, q: np.ndarray, idx: int, energy: float) -> float:
    flipped = state.copy()
    flipped[idx] = 1.0 - flipped[idx]
    return float(flipped @ q @ flipped) - energy


# ---------------------------------------------------------------------------
# Three paths
# ---------------------------------------------------------------------------

def run_pure_python(q: np.ndarray, states0: np.ndarray, energies0: np.ndarray,
                    steps: int, seed: int) -> float:
    """No Rust at all."""
    rng = np.random.RandomState(seed)
    states = states0.copy()
    energies = energies0.copy()
    dims = q.shape[0]
    shots = states.shape[0]

    t0 = time.perf_counter()
    for step in range(steps):
        beta = 0.1 + step * (10.0 / max(steps - 1, 1))
        indices = rng.randint(dims, size=shots)
        for i in range(shots):
            idx = int(indices[i])
            delta = _delta_python(states[i], q, idx, energies[i])
            if delta <= 0.0 or rng.rand() < math.exp(-beta * delta):
                states[i, idx] = 1.0 - states[i, idx]
                energies[i] += delta
    return time.perf_counter() - t0


def run_batch_delta(q: np.ndarray, states0: np.ndarray, energies0: np.ndarray,
                    steps: int, seed: int) -> float:
    """Rust batch_delta + Python acceptance loop (Phase 2 style)."""
    rng = np.random.RandomState(seed)
    states = np.ascontiguousarray(states0.copy(), dtype=np.float64)
    energies = np.ascontiguousarray(energies0.copy(), dtype=np.float64)
    dims = q.shape[0]
    shots = states.shape[0]
    indices_buf = np.empty(shots, dtype=np.int64)

    t0 = time.perf_counter()
    for step in range(steps):
        beta = 0.1 + step * (10.0 / max(steps - 1, 1))
        indices_buf[:] = rng.randint(0, dims, size=shots)
        batch = np.asarray(
            _rust_backend.try_batch_delta(states, q, indices_buf, energies), dtype=np.float64
        )
        for i in range(shots):
            delta = float(batch[i])
            if delta <= 0.0 or rng.rand() < math.exp(-beta * delta):
                states[i, int(indices_buf[i])] = 1.0 - states[i, int(indices_buf[i])]
                energies[i] += delta
    return time.perf_counter() - t0


def run_sa_step_rust(q: np.ndarray, states0: np.ndarray, energies0: np.ndarray,
                     steps: int, rng_seed: int) -> float:
    """Rust sa_step_single_flip (Phase 3) — entire per-step inner loop in Rust."""
    states = np.ascontiguousarray(states0.copy(), dtype=np.float64)
    energies = np.ascontiguousarray(energies0.copy(), dtype=np.float64)
    rng_state = max(1, rng_seed)

    t0 = time.perf_counter()
    for step in range(steps):
        beta = 0.1 + step * (10.0 / max(steps - 1, 1))
        next_states_raw, next_energies_raw, stats = _rust_backend.try_sa_step_single_flip(
            states, energies, q, float(beta), int(rng_state)
        )
        states = np.ascontiguousarray(next_states_raw, dtype=np.float64)
        energies = np.ascontiguousarray(next_energies_raw, dtype=np.float64)
        rng_state = int(stats["rng_state"])
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _median_and_stdev(samples: List[float]) -> Tuple[float, float]:
    med = statistics.median(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return med, std


def main() -> None:
    print(f"=== Phase 3 Benchmark ===")
    print(f"shots={SHOTS}  dims={DIMS}  steps={STEPS}  repeats={REPEATS}  seed={SEED}")
    print()

    if not _rust_backend.rust_available():
        print("ERROR: Rust module not available — cannot benchmark Phase 3.")
        return

    # Pre-build problem once; each trial starts from the same initial state.
    q, states0, energies0 = _make_problem(SHOTS, DIMS, SEED)

    timings: dict[str, List[float]] = {
        "pure_python": [],
        "batch_delta": [],
        "sa_step_rust": [],
    }

    for rep in range(REPEATS):
        rep_seed = SEED + rep
        t_py = run_pure_python(q, states0, energies0, STEPS, rep_seed)
        t_bd = run_batch_delta(q, states0, energies0, STEPS, rep_seed)
        t_rs = run_sa_step_rust(q, states0, energies0, STEPS, rep_seed + 1)
        timings["pure_python"].append(t_py)
        timings["batch_delta"].append(t_bd)
        timings["sa_step_rust"].append(t_rs)
        print(f"  rep {rep+1}/{REPEATS}: pure_python={t_py:.4f}s  batch_delta={t_bd:.4f}s  sa_step_rust={t_rs:.4f}s")

    print()
    print(f"{'Path':<20} {'Median (s)':>12} {'StdDev (s)':>12} {'vs pure_python':>16} {'vs batch_delta':>16}")
    print("-" * 78)

    med_py, std_py = _median_and_stdev(timings["pure_python"])
    med_bd, std_bd = _median_and_stdev(timings["batch_delta"])
    med_rs, std_rs = _median_and_stdev(timings["sa_step_rust"])

    def _ratio(a: float, b: float) -> str:
        if b == 0:
            return "N/A"
        r = a / b
        return f"{r:.2f}x"

    rows = [
        ("pure_python", med_py, std_py),
        ("batch_delta", med_bd, std_bd),
        ("sa_step_rust", med_rs, std_rs),
    ]
    for name, med, std in rows:
        vs_py = _ratio(med, med_py) if name != "pure_python" else "1.00x (baseline)"
        vs_bd = _ratio(med, med_bd) if name != "batch_delta" else "1.00x (baseline)"
        print(f"{name:<20} {med:>12.4f} {std:>12.4f} {vs_py:>16} {vs_bd:>16}")

    print()
    speedup_over_py = med_py / med_rs if med_rs > 0 else float("inf")
    speedup_over_bd = med_bd / med_rs if med_rs > 0 else float("inf")
    if speedup_over_py >= 1.0:
        print(f"Phase 3 speedup over pure_python : {speedup_over_py:.2f}x faster")
    else:
        print(f"Phase 3 slower than pure_python  : {1/speedup_over_py:.2f}x slower")
    if speedup_over_bd >= 1.0:
        print(f"Phase 3 speedup over batch_delta : {speedup_over_bd:.2f}x faster")
    else:
        print(f"Phase 3 slower than batch_delta  : {1/speedup_over_bd:.2f}x slower")


if __name__ == "__main__":
    main()
