import numpy as np

from tytan import Compile, symbols
from tytan.adaptive_sa import AdaptiveBulkSASampler
from tytan.adaptive_sa import delta_evaluator as delta_module


def test_adaptive_bulk_uses_batch_delta_when_rust_available(monkeypatch):
    x, y = symbols("x y")
    qubo, _ = Compile((x + y - 1) ** 2).get_qubo()

    called = {"batch": 0}

    def fake_batch_delta(states, qmatrix, indices, energies):
        called["batch"] += 1
        # Returning zeros guarantees acceptance path without touching evaluator.delta.
        return np.zeros(len(indices), dtype=float)

    def fail_if_called(self, state, index, energy=None):
        raise AssertionError("DeltaEvaluator.delta should not be called when batch delta exists")

    monkeypatch.setattr(delta_module._rust_backend, "try_batch_delta", fake_batch_delta)
    monkeypatch.setattr(delta_module.DeltaEvaluator, "delta", fail_if_called)

    sampler = AdaptiveBulkSASampler(seed=0, shots=4, steps=3)
    result = sampler.run(qubo)

    assert called["batch"] >= 1
    assert isinstance(result, list)
    assert result


def test_adaptive_bulk_falls_back_to_delta_when_batch_unavailable(monkeypatch):
    x, y = symbols("x y")
    qubo, _ = Compile((x + y - 1) ** 2).get_qubo()

    called = {"delta": 0}

    def no_batch_delta(states, qmatrix, indices, energies):
        return None

    def fake_delta(self, state, index, energy=None):
        called["delta"] += 1
        return 0.0

    monkeypatch.setattr(delta_module._rust_backend, "try_batch_delta", no_batch_delta)
    monkeypatch.setattr(delta_module.DeltaEvaluator, "delta", fake_delta)

    sampler = AdaptiveBulkSASampler(seed=0, shots=4, steps=3)
    result = sampler.run(qubo)

    assert called["delta"] > 0
    assert isinstance(result, list)
    assert result
