from tytan.adaptive_sa import MIPPresolvedAdaptiveBulkSASampler
from tytan import _rust_backend as rb


def test_mip_presolved_sampler_falls_back_without_rust(monkeypatch):
    monkeypatch.setattr(rb, "mip_presolve_available", lambda: False)
    qmatrix = [[0.0, 0.0], [0.0, 0.0]]
    index_map = {"x": 0, "y": 1}
    sampler = MIPPresolvedAdaptiveBulkSASampler(seed=0, shots=4, steps=4, return_stats=True)
    result, stats = sampler.run((qmatrix, index_map), return_stats=True)
    assert result
    assert stats["presolve_backend"] == "python"
    assert stats["reduced_problem_size"] == len(index_map)


def test_mip_presolved_sampler_uses_rust_plan_and_restores(monkeypatch):
    qmatrix = [[-3.0, 0.0], [0.0, -0.2]]
    index_map = {"x": 0, "y": 1}

    plan = {
        "reduced_matrix": [[-0.2]],
        "active_indices": [1],
        "hard_fixed_indices": [0],
        "hard_fixed_values": [0],
        "soft_fixed_indices": [],
        "fix_confidence": [1.0, 0.2],
        "aggregation_src": [],
        "aggregation_dst": [],
        "aggregation_relation": [],
        "aggregation_strength": [],
        "component_ids": [0, 0],
        "block_membership": [0, 0],
        "boundary_indices": [],
        "frontier_indices": [],
        "branch_candidate_indices": [1],
        "branch_candidate_scores": [0.9],
        "stats": {"original_size": 2.0, "reduced_size": 1.0},
    }

    monkeypatch.setattr(rb, "mip_presolve_available", lambda: True)
    monkeypatch.setattr(rb, "try_mip_presolve_plan", lambda *args, **kwargs: plan)

    sampler = MIPPresolvedAdaptiveBulkSASampler(seed=0, shots=4, steps=4, return_stats=True)
    result, stats = sampler.run((qmatrix, index_map), return_stats=True)
    assert result
    assert stats["presolve_backend"] == "rust"
    assert stats["presolve_stats"]["hard_fix_count"] == 1
    assert stats["reduced_problem_size"] == 1
