import numpy as np

from tytan.adaptive_sa import PresolveReducer, ReducedQuboMapper


def test_presolve_reducer_flags_hard_and_soft_fix_candidates():
    index_map = {"x": 0, "y": 1}
    mapper = ReducedQuboMapper(index_map)
    reducer = PresolveReducer(enable_hard_fix=True, enable_soft_fix=True)
    matrix = np.diag([-2.0, -0.6])

    reduced, stats = reducer.reduce(matrix, mapper, strength="medium")

    assert stats["hard_fix_count"] == 1
    assert stats["soft_fix_count"] == 1
    assert stats["reduced_size"] == 1
    assert stats["fixed_variables"].get("x") == 0
    assert reduced.shape == (1, 1)
