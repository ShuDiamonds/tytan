# Pairwise Test Design (Rust Phase1)

## Scope

This document lists pairwise combinations for Rust integration paths.

## Target 1: sampler.get_result path selection

Factors:

- `key_type`: `str`, `int`
- `rust_result`: `list`, `none`
- `score_pattern`: `ties`, `ordered`

Pairwise case list:

1. `GR-01`: `str`, `list`, `ties`
2. `GR-02`: `str`, `none`, `ordered`
3. `GR-03`: `int`, `list`, `ordered`
4. `GR-04`: `int`, `none`, `ties`

Coverage intent:

- all pairs for (`key_type`, `rust_result`)
- all pairs for (`key_type`, `score_pattern`)
- all pairs for (`rust_result`, `score_pattern`)

## Target 2: AdaptiveBulkSASampler batch/fallback selection

Factors:

- `batch_mode`: `array`, `none`
- `shots`: `1`, `4`
- `steps`: `1`, `3`
- `delta_source`: `negative`, `zero`

Pairwise case list:

1. `AB-01`: `array`, `1`, `1`, `negative`
2. `AB-02`: `array`, `4`, `3`, `zero`
3. `AB-03`: `none`, `1`, `3`, `zero`
4. `AB-04`: `none`, `4`, `1`, `negative`
5. `AB-05`: `array`, `1`, `3`, `negative`
6. `AB-06`: `none`, `4`, `1`, `zero`

Coverage intent:

- verify both execution paths (batch and fallback)
- combine shot/step variations with both delta patterns
- preserve deterministic pass/fail checks on integration behavior
