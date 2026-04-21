import pytest
import numpy as np

from tytan import _rust_backend as rb
from tytan.adaptive_sa import MIPPresolvedAdaptiveBulkSASampler
from tytan.adaptive_sa import mip_presolved_adaptive_bulk_sa as mip_mod


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


def test_mip_presolved_sampler_passes_pool_summary_to_rust(monkeypatch):
    captured = {}

    def fake_plan(*args, **kwargs):
        captured.update(kwargs)
        return {
            "reduced_matrix": [[-0.2]],
            "active_indices": [0],
            "hard_fixed_indices": [],
            "hard_fixed_values": [],
            "soft_fixed_indices": [],
            "fix_confidence": [0.5],
            "aggregation_src": [],
            "aggregation_dst": [],
            "aggregation_relation": [],
            "aggregation_strength": [],
            "component_ids": [0],
            "block_membership": [0],
            "boundary_indices": [],
            "frontier_indices": [],
            "branch_candidate_indices": [],
            "branch_candidate_scores": [],
            "stats": {"original_size": 1.0, "reduced_size": 1.0},
        }

    monkeypatch.setattr(rb, "mip_presolve_available", lambda: True)
    monkeypatch.setattr(rb, "try_mip_presolve_plan", fake_plan)

    pool_summary = {
        "variable_frequency": np.array([0.5], dtype=float),
        "pair_correlation": np.array([[1.0]], dtype=float),
    }
    sampler = MIPPresolvedAdaptiveBulkSASampler(seed=0, shots=4, steps=4, return_stats=True)
    _, stats = sampler.run(([[0.0]], {"x": 0}), return_stats=True, pool_summary=pool_summary)

    assert np.array_equal(captured["pool_frequency"], pool_summary["variable_frequency"])
    assert np.array_equal(captured["pair_correlation"], pool_summary["pair_correlation"])
    assert stats["presolve_stats"]["pool_summary_used"] is True


def test_mip_presolved_sampler_combines_multiple_blocks(monkeypatch):
    plan = {
        "reduced_matrix": [[-0.2, 0.0, 0.0, 0.0], [0.0, -0.2, 0.0, 0.0], [0.0, 0.0, -0.2, 0.0], [0.0, 0.0, 0.0, -0.2]],
        "active_indices": [0, 1, 2, 3],
        "hard_fixed_indices": [],
        "hard_fixed_values": [],
        "soft_fixed_indices": [],
        "fix_confidence": [0.5, 0.5, 0.5, 0.5],
        "aggregation_src": [],
        "aggregation_dst": [],
        "aggregation_relation": [],
        "aggregation_strength": [],
        "component_ids": [0, 0, 1, 1],
        "block_membership": [0, 0, 1, 1],
        "boundary_indices": [],
        "frontier_indices": [],
        "branch_candidate_indices": [],
        "branch_candidate_scores": [],
        "stats": {"original_size": 4.0, "reduced_size": 4.0},
    }

    def fake_plan(*args, **kwargs):
        return plan

    def fake_run(self, qubomix, shots=None, return_stats=None, include_diverse=None):
        _, local_index_map = qubomix
        names = set(local_index_map.keys())
        if names == {"x", "y"}:
            return ([[{"x": 1, "y": 0}, -1.0, 2]], {"best_energy": -1.0, "strategy_weights": {}, "log_entries": [], "clamp_mode": "none"})
        if names == {"z", "w"}:
            return ([[{"z": 0, "w": 1}, -2.0, 3]], {"best_energy": -2.0, "strategy_weights": {}, "log_entries": [], "clamp_mode": "none"})
        raise AssertionError(f"unexpected local index map: {local_index_map}")

    monkeypatch.setattr(rb, "mip_presolve_available", lambda: True)
    monkeypatch.setattr(rb, "try_mip_presolve_plan", fake_plan)
    monkeypatch.setattr(mip_mod.AdaptiveBulkSASampler, "run", fake_run)

    sampler = MIPPresolvedAdaptiveBulkSASampler(seed=0, shots=4, steps=4, return_stats=True)
    result, stats = sampler.run(([[0.0] * 4 for _ in range(4)], {"x": 0, "y": 1, "z": 2, "w": 3}), return_stats=True)

    assert result[0][0] == {"x": 1, "y": 0, "z": 0, "w": 1}
    assert result[0][1] == -3.0
    assert result[0][2] == 6
    assert stats["presolve_stats"]["block_count"] == 2
    assert stats["reduced_solve_stats"]["block_count"] == 2
    assert len(stats["reduced_solve_stats"]["mip_block_results"]) == 2


@pytest.mark.parametrize(
    "case_id,backend_mode,pool_mode,block_mode,include_diverse",
    [
        ("MP-01", "rust", "none", "single", True),
        ("MP-02", "rust", "frequency", "multi", False),
        ("MP-03", "rust", "pair", "single", False),
        ("MP-04", "python", "none", "multi", True),
        ("MP-05", "python", "frequency", "single", False),
        ("MP-06", "python", "pair", "multi", True),
    ],
)
def test_mip_presolved_sampler_pairwise_cases(
    case_id, backend_mode, pool_mode, block_mode, include_diverse, monkeypatch
):
    del case_id

    include_diverse_calls = []

    def fake_run(self, qubomix, shots=None, return_stats=None, include_diverse=None):
        del shots, return_stats
        include_diverse_calls.append(include_diverse)
        _, local_index_map = qubomix
        names = set(local_index_map.keys())
        if names == {"x", "y", "z", "w"}:
            return (
                [[{"x": 1, "y": 0, "z": 1, "w": 0}, -4.0, 1]],
                {
                    "best_energy": -4.0,
                    "strategy_weights": {},
                    "log_entries": [],
                    "clamp_mode": "none",
                },
            )
        if names == {"x", "y"}:
            return (
                [[{"x": 1, "y": 0}, -1.0, 2]],
                {
                    "best_energy": -1.0,
                    "strategy_weights": {},
                    "log_entries": [],
                    "clamp_mode": "none",
                },
            )
        if names == {"z", "w"}:
            return (
                [[{"z": 0, "w": 1}, -2.0, 3]],
                {
                    "best_energy": -2.0,
                    "strategy_weights": {},
                    "log_entries": [],
                    "clamp_mode": "none",
                },
            )
        raise AssertionError(f"unexpected local index map: {local_index_map}")

    plan = {
        "reduced_matrix": [
            [-0.2, 0.0, 0.0, 0.0],
            [0.0, -0.2, 0.0, 0.0],
            [0.0, 0.0, -0.2, 0.0],
            [0.0, 0.0, 0.0, -0.2],
        ],
        "active_indices": [0, 1, 2, 3],
        "hard_fixed_indices": [],
        "hard_fixed_values": [],
        "soft_fixed_indices": [],
        "fix_confidence": [0.5, 0.5, 0.5, 0.5],
        "aggregation_src": [],
        "aggregation_dst": [],
        "aggregation_relation": [],
        "aggregation_strength": [],
        "component_ids": [0, 0, 0, 0] if block_mode == "single" else [0, 0, 1, 1],
        "block_membership": [0, 0, 0, 0] if block_mode == "single" else [0, 0, 1, 1],
        "boundary_indices": [],
        "frontier_indices": [],
        "branch_candidate_indices": [],
        "branch_candidate_scores": [],
        "stats": {"original_size": 4.0, "reduced_size": 4.0},
    }

    monkeypatch.setattr(mip_mod.AdaptiveBulkSASampler, "run", fake_run)

    if backend_mode == "rust":
        monkeypatch.setattr(rb, "mip_presolve_available", lambda: True)
        monkeypatch.setattr(rb, "try_mip_presolve_plan", lambda *args, **kwargs: plan)
    else:
        monkeypatch.setattr(rb, "mip_presolve_available", lambda: False)

    pool_summary = None
    if pool_mode == "frequency":
        pool_summary = {"variable_frequency": np.array([0.5, 0.5, 0.5, 0.5], dtype=float)}
    elif pool_mode == "pair":
        pool_summary = {
            "variable_frequency": np.array([0.0, 1.0, 0.0, 1.0], dtype=float),
            "pair_correlation": np.array(
                [
                    [0.0, 0.95, 0.0, 0.0],
                    [0.95, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.95],
                    [0.0, 0.0, 0.95, 0.0],
                ],
                dtype=float,
            ),
        }

    sampler = MIPPresolvedAdaptiveBulkSASampler(seed=0, shots=4, steps=4, return_stats=True)
    result, stats = sampler.run(
        ([[0.0] * 4 for _ in range(4)], {"x": 0, "y": 1, "z": 2, "w": 3}),
        return_stats=True,
        include_diverse=include_diverse,
        pool_summary=pool_summary,
    )

    assert result
    assert all(flag == include_diverse for flag in include_diverse_calls)

    if backend_mode == "rust":
        assert stats["presolve_backend"] == "rust"
        assert stats["presolve_stats"]["pool_summary_used"] == (pool_mode != "none")
        assert stats["presolve_stats"]["block_count"] == (1 if block_mode == "single" else 2)
        if block_mode == "multi":
            assert stats["reduced_solve_stats"]["block_count"] == 2
            assert len(stats["reduced_solve_stats"]["mip_block_results"]) == 2
        else:
            assert stats["presolve_stats"]["block_count"] == 1
    else:
        assert stats["presolve_backend"] == "python"
        assert stats["presolve_stats"]["reduced_size"] == 4
