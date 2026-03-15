import numpy as np

from tytan.adaptive_sa import DeltaEvaluator


def test_delta_matches_full_recomputation():
    qmatrix = np.array([[2.0, 0.5], [0.5, 3.0]])
    evaluator = DeltaEvaluator(qmatrix)
    state = np.array([0.0, 1.0])
    energy = evaluator.evaluate(state)
    delta = evaluator.delta(state, 0, energy)
    flipped = state.copy()
    flipped[0] = 1.0
    full_delta = evaluator.evaluate(flipped) - energy
    assert np.isclose(delta, full_delta)
    assert not np.isnan(evaluator.local_field(state, 1))


def test_delta_matches_multiple_flips():
    qmatrix = np.array([[1.0, 0.2, 0.1], [0.2, 2.0, 0.3], [0.1, 0.3, 1.5]])
    evaluator = DeltaEvaluator(qmatrix)
    state = np.array([0.0, 1.0, 0.0])
    energy = evaluator.evaluate(state)
    total = energy
    for idx in [0, 2, 1]:
        delta = evaluator.delta(state, idx, total)
        state[idx] = 1.0 - state[idx]
        total += delta
    assert np.isclose(total, evaluator.evaluate(state))


def test_delta_random_qubo_consistency():
    rng = np.random.RandomState(0)
    for _ in range(5):
        matrix = rng.randn(4, 4)
        matrix = matrix + matrix.T
        evaluator = DeltaEvaluator(matrix)
        state = rng.randint(0, 2, size=4).astype(float)
        energy = evaluator.evaluate(state)
        idx = rng.randint(4)
        delta = evaluator.delta(state, idx, energy)
        flipped = state.copy()
        flipped[idx] = 1.0 - flipped[idx]
        assert np.isclose(delta, evaluator.evaluate(flipped) - energy)


def test_delta_evaluator_zero_diagonal_case():
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    evaluator = DeltaEvaluator(matrix)
    state = np.array([1.0, 0.0])
    energy = evaluator.evaluate(state)
    delta = evaluator.delta(state, 1, energy)
    flipped = state.copy()
    flipped[1] = 1.0 - flipped[1]
    assert np.isclose(delta, evaluator.evaluate(flipped) - energy)
