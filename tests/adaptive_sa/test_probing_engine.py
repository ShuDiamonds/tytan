import numpy as np

from tytan.adaptive_sa import ProbingEngine


def test_probing_engine_scores_diagonals():
    matrix = np.array([[2.0, 0.0], [0.0, -3.0]])
    engine = ProbingEngine(strength="low")
    summary = engine.evaluate(matrix, ["a", "b"])
    assert summary["strength"] == "low"
    assert summary["scores"]["a"] == 2.0
    assert summary["scores"]["b"] == 3.0
