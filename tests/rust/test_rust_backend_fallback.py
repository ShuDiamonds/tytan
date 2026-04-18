import numpy as np

from tytan import _rust_backend as rb


def test_rust_available_reflects_module_presence(monkeypatch):
    monkeypatch.setattr(rb, "_RUST_MODULE", object())
    assert rb.rust_available() is True

    monkeypatch.setattr(rb, "_RUST_MODULE", None)
    assert rb.rust_available() is False


def test_try_delta_energy_returns_none_without_module(monkeypatch):
    monkeypatch.setattr(rb, "_RUST_MODULE", None)
    state = np.array([1.0, 0.0], dtype=float)
    q = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=float)
    assert rb.try_delta_energy(state, q, 0, 0.0) is None


def test_try_delta_energy_passes_symmetric_hint(monkeypatch):
    class FakeModule:
        def delta_energy(self, state, qmatrix, index, current_energy, symmetric):
            assert symmetric is True
            return 0.0

    monkeypatch.setattr(rb, "_RUST_MODULE", FakeModule())
    rb._SYMMETRY_CACHE.clear()
    state = np.array([1.0, 0.0], dtype=float)
    q = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=float)
    assert rb.try_delta_energy(state, q, 0, 0.0) == 0.0


def test_try_delta_energy_passes_asymmetric_hint(monkeypatch):
    class FakeModule:
        def delta_energy(self, state, qmatrix, index, current_energy, symmetric):
            assert symmetric is False
            return 0.0

    monkeypatch.setattr(rb, "_RUST_MODULE", FakeModule())
    rb._SYMMETRY_CACHE.clear()
    state = np.array([1.0, 0.0], dtype=float)
    q = np.array([[0.0, -1.0], [0.0, 0.0]], dtype=float)
    assert rb.try_delta_energy(state, q, 0, 0.0) == 0.0


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


def test_try_sa_step_single_flip_returns_none_without_module(monkeypatch):
    monkeypatch.setattr(rb, "_RUST_MODULE", None)
    states = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    energies = np.array([0.0, 0.0], dtype=float)
    q = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=float)
    assert rb.try_sa_step_single_flip(states, energies, q, beta=1.0, rng_state=1) is None


def test_try_sa_step_multi_flip_returns_none_without_module(monkeypatch):
    monkeypatch.setattr(rb, "_RUST_MODULE", None)
    states = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    energies = np.array([0.0, 0.0], dtype=float)
    q = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=float)
    betas = np.array([1.0, 0.5], dtype=float)
    assert rb.try_sa_step_multi_flip(states, energies, q, betas, rng_state=1) is None
