import numpy as np
import pytest

from tytan import Compile, sampler, symbols
from tytan.adaptive_sa import AdaptiveBulkSASampler
from tytan.adaptive_sa import delta_evaluator as delta_module


@pytest.mark.parametrize(
    "case_id,key_type,rust_result,score_pattern",
    [
        ("GR-01", "str", "list", "ties"),
        ("GR-02", "str", "none", "ordered"),
        ("GR-03", "int", "list", "ordered"),
        ("GR-04", "int", "none", "ties"),
    ],
)
def test_pairwise_get_result_cases(case_id, key_type, rust_result, score_pattern, monkeypatch):
    del case_id

    if score_pattern == "ties":
        score = np.array([-1.0, -1.0, 0.5], dtype=float)
        pool = np.array([[1, 0], [1, 0], [0, 1]], dtype=int)
    else:
        score = np.array([0.2, -0.3, 1.1], dtype=float)
        pool = np.array([[0, 1], [1, 0], [1, 1]], dtype=int)

    called = {"aggregate": 0}

    def fake_aggregate(_pool, _score, _names):
        called["aggregate"] += 1
        if rust_result == "list":
            return [[{"x": 1, "y": 0}, -1.0, 2]]
        return None

    if key_type == "str":
        index_map = {"x": 0, "y": 1}
        monkeypatch.setattr(sampler._rust_backend, "try_aggregate_results", fake_aggregate)
        result = sampler.get_result(pool, score, index_map)
        assert isinstance(result, list)
        assert result
        if rust_result == "list":
            assert called["aggregate"] == 1
            assert result == [[{"x": 1, "y": 0}, -1.0, 2]]
        else:
            assert called["aggregate"] == 1
            assert isinstance(result[0][0], dict)
            assert "x" in result[0][0]
            assert "y" in result[0][0]
    else:
        # For non-string keys, Rust aggregate must not be called.
        def should_not_be_called(_pool, _score, _names):
            raise AssertionError("Rust aggregate should not be called for non-string keys")

        index_map = {0: 0, 1: 1}
        monkeypatch.setattr(sampler._rust_backend, "try_aggregate_results", should_not_be_called)
        result = sampler.get_result(pool, score, index_map)
        assert isinstance(result, list)
        assert result
        assert isinstance(result[0][0], dict)
        assert 0 in result[0][0]
        assert 1 in result[0][0]


@pytest.mark.parametrize(
    "case_id,batch_mode,shots,steps,delta_source",
    [
        ("AB-01", "array", 1, 1, "negative"),
        ("AB-02", "array", 4, 3, "zero"),
        ("AB-03", "none", 1, 3, "zero"),
        ("AB-04", "none", 4, 1, "negative"),
        ("AB-05", "array", 1, 3, "negative"),
        ("AB-06", "none", 4, 1, "zero"),
    ],
)
def test_pairwise_adaptive_batch_cases(case_id, batch_mode, shots, steps, delta_source, monkeypatch):
    del case_id

    x, y = symbols("x y")
    qubo, _ = Compile((x + y - 1) ** 2).get_qubo()

    call_count = {"batch": 0, "delta": 0}

    delta_value = -0.1 if delta_source == "negative" else 0.0

    def fake_batch_delta(states, qmatrix, indices, energies):
        del states, qmatrix, energies
        call_count["batch"] += 1
        if batch_mode == "none":
            return None
        return np.full(len(indices), delta_value, dtype=float)

    def fake_delta(self, state, index, energy=None):
        del self, state, index, energy
        call_count["delta"] += 1
        return delta_value

    monkeypatch.setattr(delta_module._rust_backend, "try_batch_delta", fake_batch_delta)
    monkeypatch.setattr(delta_module.DeltaEvaluator, "delta", fake_delta)

    solver = AdaptiveBulkSASampler(seed=0, shots=shots, steps=steps)
    result = solver.run(qubo)

    assert call_count["batch"] >= 1
    assert isinstance(result, list)
    assert result

    if batch_mode == "array":
        assert call_count["delta"] == 0
    else:
        assert call_count["delta"] > 0
