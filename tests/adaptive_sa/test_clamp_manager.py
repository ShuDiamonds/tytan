import numpy as np

from tytan.adaptive_sa import ClampManager


def test_clamp_manager_applies_hard_assignment():
    clamp = ClampManager(clamp_mode="hard", threshold=0.1)
    clamp.update_scores({0: 0.0, 1: 0.6})
    assert 1 in clamp.soft_candidates
    clamp.lock(2, 1)
    state = np.zeros(3, dtype=float)
    clamped = clamp.apply(state)
    assert clamped[2] == 1
