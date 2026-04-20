import numpy as np

from tytan.adaptive_sa import SolutionPool


def test_solution_pool_tracks_best_and_diverse():
    pool = SolutionPool(best_k=2, diverse_k=1)
    pool.offer(np.array([0.0, 1.0, 0.0]), -3.0)
    pool.offer(np.array([1.0, 0.0, 1.0]), -1.0)
    assert len(pool) >= 2
    results = pool.to_results({"a": 0, "b": 1, "c": 2})
    assert results
    assert results[0][1] <= results[-1][1]
    assert results[0][0]["a"] in {0, 1}

    # diverse entries should appear after best entries
    assert len(results) >= 2


def test_solution_pool_duplicate_suppression():
    pool = SolutionPool(best_k=2, diverse_k=1)
    state = np.array([0.0, 1.0, 0.0])
    pool.offer(state, -1.0)
    pool.offer(state, -2.0)
    results = pool.to_results({"a": 0, "b": 1, "c": 2})
    assert len(results) == 1
    assert results[0][2] >= 2


def test_solution_pool_replaces_near_duplicates():
    pool = SolutionPool(best_k=2, diverse_k=0, near_dup_hamming=1)
    first = np.array([0.0, 0.0, 0.0])
    near = np.array([0.0, 0.0, 1.0])
    pool.offer(first, -1.0)
    pool.offer(near, -2.0)

    results = pool.to_results({"a": 0, "b": 1, "c": 2}, include_diverse=False)
    assert len(results) == 1
    assert results[0][0] == {"a": 0, "b": 0, "c": 1}
    assert results[0][1] == -2.0


def test_solution_pool_greedy_diversity_prefers_farthest_candidate():
    pool = SolutionPool(best_k=1, diverse_k=1, near_dup_hamming=0)
    pool.offer(np.array([0.0, 0.0, 0.0]), -3.0)
    pool.offer(np.array([0.0, 0.0, 1.0]), -2.0)
    pool.offer(np.array([1.0, 1.0, 1.0]), -1.0)

    assert pool.min_distance_to_pool(np.array([1.0, 1.0, 1.0])) == 0.0
    assert pool.mean_pairwise_distance() == 2.0
    results = pool.to_results({"a": 0, "b": 1, "c": 2})
    assert results[1][0] == {"a": 1, "b": 1, "c": 1}


def test_solution_pool_hamming_distance_promotes_diversity():
    pool = SolutionPool(best_k=1, diverse_k=1)
    pool.offer(np.zeros(4), -1.0)
    pool.offer(np.ones(4), 0.0)
    results = pool.to_results({"a": 0, "b": 1, "c": 2, "d": 3}, include_diverse=True)
    assert len(results) >= 2
