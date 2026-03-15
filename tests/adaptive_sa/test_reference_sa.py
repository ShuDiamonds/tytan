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
