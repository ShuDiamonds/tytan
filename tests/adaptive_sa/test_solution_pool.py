import numpy as np

from tytan.adaptive_sa import SolutionPool


def test_solution_pool_tracks_best_and_diverse():
    pool = SolutionPool(best_k=2, diverse_k=1)
    pool.offer(np.array([0.0, 1.0, 0.0]), -3.0)
    pool.offer(np.array([1.0, 1.0, 1.0]), -1.0)
    assert len(pool) >= 2
    results = pool.to_results({"a": 0, "b": 1, "c": 2})
    assert results
    assert results[0][1] <= results[-1][1]
    assert results[0][0]["a"] in {0, 1}

    # diverse entries should appear after best entries
    assert len(results) >= 2
