# Rust Phase1 Test Items

1. Bridge availability
- Fallback returns `None` when Rust module is unavailable.
- `rust_available()` is true when extension is installed.

2. Type conversion helpers
- `as_state_vector` and `as_state_matrix` convert to `float` ndarray.
- `as_energy_vector` converts to `float` ndarray.
- `as_index_vector` converts to `int64` ndarray.
- `normalize_hobomix` converts QUBO matrix dtype to `float`.

3. Delta parity
- `delta_energy` result matches Python full recomputation.
- `batch_delta` result matches Python loop implementation.

4. Result format compatibility
- `aggregate_results` returns list entries of `[state_dict, energy, count]`.
- `state_dict` key/value shape remains compatible with sampler contracts.

5. Sampler integration
- `get_result` uses Rust aggregate when variable names are strings.
- `get_result` falls back to NumPy path for non-string keys.

6. Adaptive integration
- Batch delta path is used when Rust batch delta is available.
- Existing delta path remains the fallback when batch delta is unavailable.

7. Regression targets
- `tests/sampler/test_sa_sampler.py`
- `tests/adaptive_sa/test_adaptive_bulk_sa.py`
- `tests/adaptive_sa/test_delta_evaluator.py`
