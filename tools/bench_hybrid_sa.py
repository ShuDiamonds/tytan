"""
Benchmark: Hybrid SA (GPU Phase1 + delta-cache Phase2).

Compares:
  - CPU baseline (device=cpu, phase2 disabled)
  - Hybrid (device=auto/cuda/mps, phase2 enabled)

This script is intended for quick regression checks. It focuses on end-to-end
time for sampler.run(), plus best_energy, for a fixed dense-in-memory QUBO.

Run:
  /bin/zsh -lc "UV_CACHE_DIR=.uv-cache uv run python tools/bench_hybrid_sa.py"

Env knobs:
  DIMS="256,512,1024"
  DENSITY="0.01"
  SHOTS="64"
  STEPS="128"
  REPEATS="5"
  WARMUP="2"
  SEED="42"
  DEVICE="auto|cpu|cuda:0|mps:0"
  PHASE2_ENABLED="1"
  PHASE2_START_STEP=""  (empty => steps//2)
  PHASE2_TOP_K="16"
  PHASE2_SWEEPS_PER_STEP="1"
  SPARSE_THRESHOLD="0.0"
  POOL_OFFER_MODE="per_flip|phase_end|off"
  DISABLE_RUST="1"      (set TYTAN_RUST=off before importing tytan)
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_dims(raw: str) -> List[int]:
    dims: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        dims.append(int(part))
    return dims or [256, 512, 1024]


def _make_sparse_symmetric_qubo(n: int, density: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    q = np.zeros((n, n), dtype=np.float64)
    if n == 0:
        return q

    # Diagonal
    q[np.arange(n), np.arange(n)] = rng.randn(n).astype(np.float64)

    # Upper triangle sparsity mask
    if density <= 0.0:
        return np.ascontiguousarray(q)
    density = min(max(float(density), 0.0), 1.0)
    mask = rng.rand(n, n) < density
    mask = np.triu(mask, k=1)
    values = rng.randn(n, n).astype(np.float64)
    q[mask] = values[mask]
    q = q + q.T
    return np.ascontiguousarray(q, dtype=np.float64)


def _detect_device(requested: str) -> str:
    req = (requested or "auto").strip().lower()
    if req != "auto":
        return requested
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps:0"
    return "cpu"


def _sync_if_needed(device: str) -> None:
    dev = (device or "cpu").lower()
    if dev.startswith("cuda"):
        import torch

        torch.cuda.synchronize()
        return
    if dev.startswith("mps"):
        try:
            import torch

            if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        except Exception:
            pass


def _require_torch(device: str) -> None:
    dev = (device or "cpu").lower()
    if dev == "cpu":
        return
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for DEVICE != cpu. Install it into the same environment as `uv run`, e.g.\n"
            "  /bin/zsh -lc \"UV_CACHE_DIR=.uv-cache uv add torch\"\n"
            "then re-run this benchmark.\n"
            "If you want to run without PyTorch, set DEVICE=cpu."
        ) from exc

    if dev.startswith("mps") and not torch.backends.mps.is_available():
        raise RuntimeError(
            "DEVICE is set to mps, but torch.backends.mps.is_available() is False in this environment.\n"
            "Check with:\n"
            "  /bin/zsh -lc \"UV_CACHE_DIR=.uv-cache uv run python -c 'import torch; print(torch.__version__); "
            "print(torch.backends.mps.is_built()); print(torch.backends.mps.is_available())'\"\n"
            "If MPS is unavailable on this machine, run with DEVICE=cpu."
        )

    if dev.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "DEVICE is set to cuda, but torch.cuda.is_available() is False in this environment. "
            "If CUDA is unavailable on this machine, run with DEVICE=cpu."
        )


@dataclass(frozen=True)
class Case:
    name: str
    sampler_kwargs: Dict[str, Any]


def _run_case(
    case: Case,
    qubomix: Tuple[np.ndarray, Dict[int, int]],
    shots: int,
    repeats: int,
    warmup: int,
) -> Dict[str, Any]:
    from tytan.adaptive_sa import AdaptiveBulkSASampler

    timings: List[float] = []
    energies: List[float] = []
    last_stats: Optional[Dict[str, Any]] = None

    for _ in range(warmup):
        sampler = AdaptiveBulkSASampler(shots=shots, return_stats=True, **case.sampler_kwargs)
        _sync_if_needed(case.sampler_kwargs.get("device", "cpu"))
        sampler.run(qubomix, return_stats=True)
        _sync_if_needed(case.sampler_kwargs.get("device", "cpu"))

    for rep in range(repeats):
        sampler = AdaptiveBulkSASampler(shots=shots, return_stats=True, **case.sampler_kwargs)
        _sync_if_needed(case.sampler_kwargs.get("device", "cpu"))
        t0 = time.perf_counter()
        _, stats = sampler.run(qubomix, return_stats=True)
        _sync_if_needed(case.sampler_kwargs.get("device", "cpu"))
        dt = time.perf_counter() - t0
        timings.append(dt)
        energies.append(float(stats.get("best_energy", float("nan"))))
        last_stats = stats
        _ = rep

    return {
        "name": case.name,
        "timings_sec": timings,
        "median_sec": statistics.median(timings) if timings else None,
        "mean_sec": statistics.mean(timings) if timings else None,
        "best_energy_median": statistics.median(energies) if energies else None,
        "last_stats": last_stats or {},
    }


def main() -> None:
    if _env_bool("DISABLE_RUST", True):
        os.environ.setdefault("TYTAN_RUST", "off")

    dims_list = _parse_dims(os.getenv("DIMS", "256,512,1024"))
    density = _env_float("DENSITY", 0.01)
    shots = _env_int("SHOTS", 64)
    steps = _env_int("STEPS", 128)
    repeats = _env_int("REPEATS", 5)
    warmup = _env_int("WARMUP", 2)
    seed = _env_int("SEED", 42)
    requested_device = os.getenv("DEVICE", "auto")
    device = _detect_device(requested_device)
    _require_torch(device)
    torch_meta: Dict[str, Any] = {}
    if (device or "cpu").lower() != "cpu":
        import platform

        import torch

        torch_meta = {
            "torch_version": torch.__version__,
            "platform_mac_ver": platform.mac_ver()[0],
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
        }

    phase2_enabled = _env_bool("PHASE2_ENABLED", True)
    phase2_start_raw = os.getenv("PHASE2_START_STEP", "").strip()
    phase2_start_step = None if phase2_start_raw == "" else int(phase2_start_raw)
    phase2_top_k = _env_int("PHASE2_TOP_K", 16)
    phase2_sweeps = _env_int("PHASE2_SWEEPS_PER_STEP", 1)
    sparse_threshold = _env_float("SPARSE_THRESHOLD", 0.0)
    pool_offer_mode = os.getenv("POOL_OFFER_MODE", "phase_end").strip().lower()

    cases = [
        Case(
            name="cpu_baseline",
            sampler_kwargs={
                "seed": seed,
                "steps": steps,
                "device": "cpu",
                "phase2_enabled": False,
                "pool_offer_mode": pool_offer_mode,
            },
        ),
        Case(
            name=f"hybrid_{device}",
            sampler_kwargs={
                "seed": seed,
                "steps": steps,
                "device": device,
                "phase2_enabled": bool(phase2_enabled),
                "phase2_start_step": phase2_start_step,
                "phase2_top_k": phase2_top_k,
                "phase2_sweeps_per_step": phase2_sweeps,
                "sparse_threshold": sparse_threshold,
                "pool_offer_mode": pool_offer_mode,
            },
        ),
    ]

    results: Dict[str, Any] = {
        "meta": {
            "dims": dims_list,
            "density": density,
            "shots": shots,
            "steps": steps,
            "repeats": repeats,
            "warmup": warmup,
            "seed": seed,
            "device_requested": requested_device,
            "device_selected": device,
            "pool_offer_mode": pool_offer_mode,
            "disable_rust": _env_bool("DISABLE_RUST", True),
            **torch_meta,
        },
        "runs": [],
    }

    for dims in dims_list:
        q = _make_sparse_symmetric_qubo(dims, density, seed)
        index_map = {i: i for i in range(dims)}
        qubomix = (q, index_map)

        entry = {"dims": dims, "cases": []}
        for case in cases:
            entry["cases"].append(_run_case(case, qubomix, shots, repeats, warmup))
        results["runs"].append(entry)

    out_dir = os.path.join("output", "tmp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bench_hybrid_sa.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(out_path)
    print(json.dumps(results["meta"], ensure_ascii=False))
    for run in results["runs"]:
        dims = run["dims"]
        cpu = run["cases"][0]
        hyb = run["cases"][1]
        cpu_med = cpu["median_sec"]
        hyb_med = hyb["median_sec"]
        ratio = (cpu_med / hyb_med) if (cpu_med and hyb_med and hyb_med > 0) else None
        print(
            json.dumps(
                {
                    "dims": dims,
                    "cpu_median_sec": cpu_med,
                    "hybrid_median_sec": hyb_med,
                    "speedup": ratio,
                    "cpu_best_energy_med": cpu["best_energy_median"],
                    "hybrid_best_energy_med": hyb["best_energy_median"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
