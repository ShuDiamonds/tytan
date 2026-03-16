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


def test_clamp_manager_soft_clamp_does_not_force_assignment():
    clamp = ClampManager(clamp_mode="soft", threshold=0.5)
    clamp.update_scores({0: 0.1, 1: 0.8})
    base = np.zeros(2, dtype=float)
    result = clamp.apply(base)
    assert np.array_equal(result, base)
    assert 1 in clamp.soft_candidates


def test_clamp_manager_hard_clamp_requires_explicit_lock():
    clamp = ClampManager(clamp_mode="none", threshold=0.1)
    clamp.lock(0, 1)
    state = np.zeros(1, dtype=float)
    assert clamp.apply(state)[0] == 1


def test_clamp_manager_soft_candidates_respect_threshold():
    clamp = ClampManager(clamp_mode="soft", threshold=0.6)
    clamp.update_scores({0: 0.5, 1: 0.7})
    assert 0 not in clamp.soft_candidates
    assert 1 in clamp.soft_candidates
