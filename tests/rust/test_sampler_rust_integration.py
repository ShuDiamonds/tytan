import numpy as np

from tytan import sampler as sampler_module


def test_get_result_uses_rust_aggregate_when_available(monkeypatch):
    expected = [[{"x": 1, "y": 0}, -1.0, 2]]

    def fake_aggregate(pool, score, names):
        assert list(names) == ["x", "y"]
        return expected

    monkeypatch.setattr(sampler_module._rust_backend, "try_aggregate_results", fake_aggregate)

    pool = np.array([[1, 0], [1, 0]], dtype=int)
    score = np.array([-1.0, -1.0], dtype=float)
    index_map = {"x": 0, "y": 1}
    result = sampler_module.get_result(pool, score, index_map)
    assert result == expected


def test_get_result_falls_back_for_non_string_keys(monkeypatch):
    def should_not_be_called(pool, score, names):
        raise AssertionError("Rust aggregate should not be called for non-string keys")

    monkeypatch.setattr(sampler_module._rust_backend, "try_aggregate_results", should_not_be_called)

    pool = np.array([[1, 0], [1, 0]], dtype=int)
    score = np.array([-1.0, -1.0], dtype=float)
    index_map = {0: 0, 1: 1}
    result = sampler_module.get_result(pool, score, index_map)

    assert isinstance(result, list)
    assert result
    assert isinstance(result[0][0], dict)
    assert 0 in result[0][0]
    assert 1 in result[0][0]
