import numpy as np

from tytan import _rust_backend as rb


def test_try_delta_energy_returns_none_without_module(monkeypatch):
    monkeypatch.setattr(rb, "_RUST_MODULE", None)
    state = np.array([1.0, 0.0], dtype=float)
    q = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=float)
    assert rb.try_delta_energy(state, q, 0, 0.0) is None


def test_try_batch_delta_returns_none_without_module(monkeypatch):
    monkeypatch.setattr(rb, "_RUST_MODULE", None)
    states = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    q = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=float)
    indices = np.array([0, 1], dtype=np.int64)
    energies = np.array([0.0, 0.0], dtype=float)
    assert rb.try_batch_delta(states, q, indices, energies) is None


def test_try_aggregate_returns_none_without_module(monkeypatch):
    monkeypatch.setattr(rb, "_RUST_MODULE", None)
    states = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float)
    energies = np.array([0.0, 0.0], dtype=float)
    assert rb.try_aggregate_results(states, energies, ["x", "y"]) is None
