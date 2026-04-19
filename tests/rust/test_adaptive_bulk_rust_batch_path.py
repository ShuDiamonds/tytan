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


def test_adaptive_bulk_uses_rust_step_when_workload_gate_matches(monkeypatch):
    x, y = symbols("x y")
    qubo, _ = Compile((x + y - 1) ** 2).get_qubo()

    called = {"step": 0}

    def fake_rust_step(states, energies, qmatrix, beta, rng_state):
        del qmatrix, beta
        called["step"] += 1
        next_states = states.copy()
        next_states[:, 0] = 1.0 - next_states[:, 0]
        return next_states, energies.copy(), {"rng_state": rng_state + 1, "accepted": len(states), "proposals": len(states)}

    def fail_batch(*args, **kwargs):
        raise AssertionError("batch delta should not be called when rust step is enabled")

    monkeypatch.setattr(delta_module._rust_backend, "rust_step_min_work", lambda: 1)
    monkeypatch.setattr(delta_module._rust_backend, "try_sa_step_single_flip", fake_rust_step)
    monkeypatch.setattr(delta_module._rust_backend, "try_batch_delta", fail_batch)

    sampler = AdaptiveBulkSASampler(seed=0, shots=4, steps=3)
    result = sampler.run(qubo)

    assert called["step"] >= 1
    assert isinstance(result, list)
    assert result


def test_adaptive_bulk_falls_back_from_rust_step_to_batch(monkeypatch):
    x, y = symbols("x y")
    qubo, _ = Compile((x + y - 1) ** 2).get_qubo()

    called = {"batch": 0}

    def no_step(states, energies, qmatrix, beta, rng_state):
        return None

    def fake_batch_delta(states, qmatrix, indices, energies):
        del states, qmatrix, energies
        called["batch"] += 1
        return np.zeros(len(indices), dtype=float)

    def fail_if_called(self, state, index, energy=None):
        raise AssertionError("DeltaEvaluator.delta should not be called when batch delta exists")

    monkeypatch.setattr(delta_module._rust_backend, "rust_step_min_work", lambda: 1)
    monkeypatch.setattr(delta_module._rust_backend, "try_sa_step_single_flip", no_step)
    monkeypatch.setattr(delta_module._rust_backend, "try_batch_delta", fake_batch_delta)
    monkeypatch.setattr(delta_module.DeltaEvaluator, "delta", fail_if_called)

    sampler = AdaptiveBulkSASampler(seed=0, shots=4, steps=3)
    result = sampler.run(qubo)

    assert called["batch"] >= 1
    assert isinstance(result, list)
    assert result


def test_adaptive_bulk_uses_multi_step_rust_path_when_not_adaptive(monkeypatch):
    x, y = symbols("x y")
    qubo, _ = Compile((x + y - 1) ** 2).get_qubo()

    called = {"multi": 0}

    def fake_multi_step(states, energies, qmatrix, betas, rng_state):
        del qmatrix
        called["multi"] += 1
        history_states = np.stack(
            [
                np.where(np.arange(states.shape[1]) == 0, 1.0 - states, states),
                np.where(np.arange(states.shape[1]) == 1, 1.0 - states, states),
                np.where(np.arange(states.shape[1]) == 0, 1.0 - states, states),
            ],
            axis=0,
        )
        history_energies = np.stack([energies - 1.0, energies - 2.0, energies - 3.0], axis=0)
        return history_states, history_energies, {"rng_state": rng_state + 3, "accepted": len(states) * 3, "proposals": len(states) * 3}

    def fail_single(*args, **kwargs):
        raise AssertionError("single-step Rust path should not be used in multi-step mode")

    def fail_batch(*args, **kwargs):
        raise AssertionError("batch delta should not be used in multi-step mode")

    monkeypatch.setattr(delta_module._rust_backend, "rust_step_min_work", lambda: 1)
    monkeypatch.setattr(delta_module._rust_backend, "try_sa_step_multi_flip", fake_multi_step)
    monkeypatch.setattr(delta_module._rust_backend, "try_sa_step_single_flip", fail_single)
    monkeypatch.setattr(delta_module._rust_backend, "try_batch_delta", fail_batch)

    sampler = AdaptiveBulkSASampler(seed=0, shots=4, steps=3, adaptive=False)
    result = sampler.run(qubo)

    assert called["multi"] == 1
    assert isinstance(result, list)
    assert result
