import numpy as np
import pytest

from tytan import _rust_backend as rb
from tytan.adaptive_sa.delta_evaluator import DeltaEvaluator


@pytest.mark.skipif(not rb.rust_available(), reason="Rust extension not installed")
def test_delta_energy_matches_python():
    q = np.array([[1.0, -2.0, 0.5], [0.0, 0.3, -1.0], [0.0, 0.0, 0.7]], dtype=float)
    state = np.array([1.0, 0.0, 1.0], dtype=float)
    evaluator = DeltaEvaluator(q)
    energy = evaluator.evaluate(state)
    rust_delta = rb.try_delta_energy(state, q, 1, energy)

    flipped = state.copy()
    flipped[1] = 1.0 - flipped[1]
    py_delta = float(flipped @ q @ flipped) - energy

    assert rust_delta is not None
    assert float(rust_delta) == pytest.approx(py_delta, rel=1e-12, abs=1e-12)


@pytest.mark.skipif(not rb.rust_available(), reason="Rust extension not installed")
def test_batch_delta_matches_python_loop():
    q = np.array([[0.0, -1.5], [0.0, 0.2]], dtype=float)
    states = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    energies = np.array([float(s @ q @ s) for s in states], dtype=float)
    indices = np.array([0, 1, 1], dtype=np.int64)

    rust_deltas = rb.try_batch_delta(states, q, indices, energies)
    assert rust_deltas is not None

    py_deltas = []
    for i in range(len(states)):
        flipped = states[i].copy()
        idx = int(indices[i])
        flipped[idx] = 1.0 - flipped[idx]
        py_deltas.append(float(flipped @ q @ flipped) - energies[i])

    np.testing.assert_allclose(rust_deltas, np.array(py_deltas), rtol=1e-12, atol=1e-12)
