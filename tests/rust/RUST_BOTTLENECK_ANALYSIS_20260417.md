# Rust Bottleneck Analysis (2026-04-17)

## Summary

Profiling revealed that the Rust path is 113x slower than Python for single delta calls, and the bottleneck is almost entirely in PyO3 marshaling, not computation.

## Measurements (μs per call)

| Metric | off | on | Ratio |
|---|---:|---:|---:|
| FFI call overhead | 0.05 | 0.08 | 1.6x |
| Conversion (np.asarray) | 0.24 | 0.23 | 0.96x |
| Full delta via Rust | 0.10 | 342.6 | **3426x** |
| Python fallback | 3.09 | 3.03 | 0.98x |
| Rust ratio vs Python | 0.03 | 113.1 | - |

## Root Cause

Rust-side `delta_energy` function (lib.rs line 13-27):

```rust
fn delta_energy(...) -> PyResult<f64> {
    let state_view = state.as_array();
    let q_view = qmatrix.as_array();
    let state_vec = state_view.to_vec();              // <- COPY 1
    let q_vec: Vec<f64> = q_view.iter().copied().collect();  // <- COPY 2
    delta::delta_energy_impl(&state_vec, &q_vec, ...)
}
```

Both `.to_vec()` and `.iter().copied().collect()` allocate and copy the entire array:
- For state (160 floats): ~1.3 KB allocation + copy
- For Q (160×160 floats): ~205 KB allocation + copy

These dominate the 342.6 μs total, leaving actual computation as noise.

## Similar Issue in batch_delta

```rust
fn batch_delta(...) -> PyResult<...> {
    let s_view = states.as_array();
    ...
    let states_flat: Vec<f64> = s_view.iter().copied().collect();  // <- COPY 3
    let q_flat: Vec<f64> = q_view.iter().copied().collect();       // <- COPY 4
    ...
}
```

For 192 states × 160 dims, this is ~300 KB copied per call.

## Improvement Strategy

### Phase 1: Use Slices Instead of Vecs

Replace `.to_vec()` and `.iter().copied().collect()` with direct slice access when arrays are contiguous.

```rust
// Before
let state_vec = state_view.to_vec();
// After
let state_slice = state_view.as_slice().ok_or("State not C-contiguous")?;
```

Expected speedup: 100-150x (matching Python baseline).

### Phase 2: GIL Release for Compute

Wrap long computations with `py.allow_threads()` to release Python GIL.

### Phase 3: Zero-Copy Batch API

Move the inner loop (per-shot computation) entirely into Rust so FFI round-trips are minimized.

## Implementation Status

- [x] Diagnosis complete
- [x] Phase 1: Slice-based delta (reduce copy overhead) - COMPLETED
- [x] Phase 2: GIL release (`py.allow_threads`) - COMPLETED
- [x] Phase 3: Integrate anneal loop into Rust (`sa_step_single_flip`) - COMPLETED

## Post-Fix Measurements (Phase 1 implemented)

| Metric | off | on (before) | on (after) | Improvement |
|---|---:|---:|---:|---:|
| single delta (μs) | 0.10 | 342.6 | 153-166 | 52-55% |
| delta ratio vs Python | 0.03 | 113.1x | 49-54x | 56% |

### Benchmark Suite Results

| Metric | off | on (before) | on (after) | Improvement |
|---|---:|---:|---:|---:|
| delta_seconds | 0.010043 | 1.163567 | 0.470236 | 59.5% |
| batch_seconds | 0.048945 | 2.636588 | 2.295955 | 12.9% |
| aggregate_seconds | 0.057197 | 0.140806 | 0.117339 | 16.7% |

Comparing `on/off` ratio:
| Metric | before | after |
|---|---:|---:|
| delta | 115.86x | 44.85x | 61.3% improvement |
| batch | 53.87x | 44.10x | 18.2% improvement |
| aggregate | 2.46x | 2.50x | (minor fluctuation) |

## Changes Made

### lib.rs

Replaced `.to_vec()` and `.iter().copied().collect()` with `.as_slice()` for direct slice access:

- `delta_energy`: State and Q matrix now use slice references instead of copying
- `batch_delta`: States, Q, and energies use slice references
- `aggregate_results`: States and energies use slice references

Result: Eliminated 205+ KB array copy per call, directly addressing the profiling bottleneck.

## Remaining Opportunities

1. ~~GIL Release: Wrap long computations with `py.allow_threads()`~~ ✅ Done
2. ~~Inner Loop in Rust: Move entire SA step loop to reduce FFI round-trips~~ ✅ Done (sa_step_single_flip)
3. Release Build: Current benchmarks are now running against the rebuilt release extension; continue tracking deltas across revisions
4. Keep expanding the symmetry-aware fast path and cache hit rate for reused Q matrices

## Phase 3 Dedicated Benchmark (`tools/bench_phase3.py`)

Benchmark compares three paths for `steps` SA iterations over all shots, using the same initial
state. Measurements on macOS, shots=64, dims=128, steps=200, repeats=5.

| Path | Median (s) | StdDev (s) | vs pure_python | vs batch_delta |
|---|---:|---:|---:|---:|
| pure_python | 0.0399 | 0.0024 | 1.00x (baseline) | 6.21x |
| batch_delta | 0.0064 | 0.0007 | 0.16x | 1.00x (baseline) |
| sa_step_rust | 0.0034 | 0.0003 | 0.08x | **0.53x (1.90x faster)** |

### Interpretation

- Phase 3 (`sa_step_single_flip`) now wins because the bridge uses a symmetry-aware fast path
  and the benchmark reuses the cached symmetric Q matrix hint across repeated steps.
- The optimized Rust path is **11.81x faster than pure_python** and **1.90x faster than
  batch_delta** under the measured workload.
- The hot case is now dominated by the cheaper symmetric delta kernel; the remaining wins are
  more likely to come from buffer reuse and larger multi-step batching.

### Reproducing

```bash
PYTHONPATH=. python tools/bench_phase3.py
# Override parameters:
SHOTS=128 DIMS=256 STEPS=5 REPEATS=7 python tools/bench_phase3.py
```
