import numpy as np
import pytest

from tytan import Compile, symbols
from tytan.adaptive_sa import ReferenceSASampler


def test_reference_sa_runs_and_returns_result():
    x, y = symbols("x y")
    qubo, _ = Compile((x + y - 1) ** 2).get_qubo()
    sampler = ReferenceSASampler(seed=0, steps=32)
    result = sampler.run(qubo, shots=2)
    assert isinstance(result, list)
    assert result
    first_solution, energy, count = result[0]
    assert isinstance(first_solution, dict)
    assert "x" in first_solution and "y" in first_solution
    assert isinstance(energy, float)
    assert isinstance(count, int)


def test_reference_sa_can_return_stats():
    x, y = symbols("x y")
    qubo, _ = Compile((x + y - 1) ** 2).get_qubo()
    sampler = ReferenceSASampler(seed=0, steps=16)
    result, stats = sampler.run(qubo, shots=1, return_stats=True)
    assert isinstance(result, list)
    assert stats["shots"] == 1
    assert "improvements" in stats


def test_reference_sa_seed_reproducible():
    x, y = symbols("x y")
    qubo, _ = Compile((x + y - 1) ** 2).get_qubo()
    sampler = ReferenceSASampler(seed=1, steps=8)
    first_result = sampler.run(qubo, shots=1)
    sampler = ReferenceSASampler(seed=1, steps=8)
    second_result = sampler.run(qubo, shots=1)
    assert first_result[0][0] == second_result[0][0]
    assert first_result[0][1] == second_result[0][1]


def test_reference_sa_invalid_qubo_raises():
    sampler = ReferenceSASampler(seed=0, steps=1)
    with pytest.raises(ValueError):
        sampler.run((np.zeros((2, 3)), {}))
