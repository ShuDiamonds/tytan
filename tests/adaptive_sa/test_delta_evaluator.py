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
