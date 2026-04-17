import numpy as np

from tytan import symbols, Compile
from tytan.sampler import SASampler


def test_sa_sampler_run():
    x, y, z = symbols("x y z")
    expr = (x + y + z - 2)**2
    qubo, offset = Compile(expr).get_qubo()
    sampler = SASampler()
    result = sampler.run(qubo)
    for r in result:
        print(r)
    assert result is not None
    assert result[0][0] is not None
    assert result[0][0]["x"] is not None
    assert result[0][0]["y"] is not None
    assert result[0][0]["z"] is not None
    assert result[0][1] is not None
    assert result[0][2] is not None


def test_sa_sampler_run_with_seed():
    x, y, z = symbols("x y z")
    expr = (x + y + z - 2)**2
    qubo, offset = Compile(expr).get_qubo()

    #1
    print('try 1, ', end='')
    sampler = SASampler(seed=0)
    result = sampler.run(qubo)
    print(result[0][0]["x"], result[0][0]["y"], result[0][0]["z"], result[0][2])
    x = result[0][0]["x"]
    y = result[0][0]["y"]
    z = result[0][0]["z"]
    count = result[0][2]

    #2-
    for i in range(2, 10):
        print(f'try {i}, ', end='')
        sampler = SASampler(seed=0)
        result = sampler.run(qubo)
        print(result[0][0]["x"], result[0][0]["y"], result[0][0]["z"], result[0][2])
        assert result[0][0]["x"] == x
        assert result[0][0]["y"] == y
        assert result[0][0]["z"] == z
        assert result[0][2] == count


def test_polish_passes_stat_tracking():
    qmatrix = np.array([[0.0, -2.0], [-2.0, 0.0]])
    index_map = {0: 0, 1: 1}
    sampler = SASampler(seed=0)
    _, stats_no_polish = sampler.run(
        (qmatrix, index_map),
        shots=1,
        num_sweeps=0,
        initial_states=[[1.0, 0.0]],
        enable_polish=True,
        polish_passes=0,
        return_stats=True,
    )
    assert stats_no_polish["polish_rounds"] == 0
    assert stats_no_polish["best_energy"] == 0.0

    sampler = SASampler(seed=0)
    _, stats_polish = sampler.run(
        (qmatrix, index_map),
        shots=1,
        num_sweeps=0,
        initial_states=[[1.0, 0.0]],
        enable_polish=True,
        polish_passes=2,
        return_stats=True,
    )
    assert stats_polish["polish_rounds"] >= 1
    assert stats_polish["best_energy"] == -4.0
