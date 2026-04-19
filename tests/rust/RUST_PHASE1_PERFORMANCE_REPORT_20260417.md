# Rust Phase1 Performance Report (2026-04-17)

## Summary

Rust Phase1 integration was measured by comparing `TYTAN_RUST=off` and `TYTAN_RUST=on` under the same synthetic workloads.

Result on this machine: Rust-enabled path is slower for all measured hotspots.

- Delta evaluator path: about 115.86x slower with `on`
- Adaptive batch-delta style path: about 53.87x slower with `on`
- Result aggregation path: about 2.46x slower with `on`

## Environment

- OS: macOS
- Python: 3.14.3
- Virtual environment: `.venv`
- Comparison mode: `TYTAN_RUST=off` vs `TYTAN_RUST=on`
- Repetitions: 5 runs per mode (seeds 0..4)

## Method

1. Delta hotspot (`DeltaEvaluator.delta`) repeated 3000 times.
2. Adaptive-style loop using `try_batch_delta` with fallback semantics for 80 iterations.
3. Result aggregation via `sampler.get_result` on 12,000 states with 64 variables.

Median of 5 runs is used for comparison.

## Median Comparison

| Metric | off median (s) | on median (s) | on/off |
|---|---:|---:|---:|
| delta_seconds | 0.010043 | 1.163567 | 115.86x |
| batch_seconds | 0.048945 | 2.636588 | 53.87x |
| aggregate_seconds | 0.057197 | 0.140806 | 2.46x |

Interpretation: `on/off > 1.0x` means Rust-enabled path is slower.

## Analysis

Likely dominant factors:

1. Python-Rust boundary crossing inside tight loops.
2. Per-call dtype normalization and array materialization in bridge functions.
3. Too small work granularity per FFI call.

## Implemented in this revision

1. Added zero-copy-friendly normalization helpers in bridge:
   - `_as_float64_c`, `_as_int64_c`
   - applied to `try_delta_energy`, `try_batch_delta`, `try_aggregate_results`
2. Added optional Rust workload gate in adaptive sampler:
   - `TYTAN_RUST_MIN_WORK` (default `0`, disabled by default)
3. Made hot-path arrays contiguous earlier in adaptive sampler and evaluator.

## Post-fix Median Comparison

| Metric | off median (s) | on median (s) | on/off |
|---|---:|---:|---:|
| delta_seconds | 0.010278 | 1.082996 | 105.37x |
| batch_seconds | 0.051051 | 2.332256 | 45.69x |
| aggregate_seconds | 0.046543 | 0.124538 | 2.68x |

## Delta vs Initial Measurement

- Delta path: 115.86x -> 105.37x (improved but still slower)
- Batch path: 53.87x -> 45.69x (improved but still slower)
- Aggregate path: 2.46x -> 2.68x (slightly worse in this run)

## Recommendations

1. Move from per-flip API to batch-first APIs with larger work per Rust call.
2. Shift multi-step anneal inner loops to Rust to reduce Python-Rust round-trips.
3. Keep dtype/contiguous normalization out of inner loops (done partially, continue broadening).
4. Add CI benchmark guard on median ratios and track trend across revisions.
