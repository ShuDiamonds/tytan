from pytest import approx

from tytan import Compile, symbols
from tytan.adaptive_sa import PresolvedAdaptiveBulkSASampler


def test_presolved_sampler_passes_through_without_presolve():
    x, y = symbols("x y")
    qubo_data, _ = Compile((x + y - 1) ** 2).get_qubo()
    qubo_mix, index_map = qubo_data
    sampler = PresolvedAdaptiveBulkSASampler(
        seed=0,
        shots=4,
        steps=8,
        presolve=False,
        return_stats=True,
    )
    result, stats = sampler.run((qubo_mix, index_map), return_stats=True)
    assert isinstance(result, list)
    assert result
    assert stats["reduced_problem_size"] == len(index_map)
    assert stats["normalization_info"]["after"]


def test_presolved_sampler_reports_stats_and_restores_with_hard_fix():
    qmatrix = [[-3.0, 0.0], [0.0, -0.4]]
    index_map = {"x": 0, "y": 1}
    sampler = PresolvedAdaptiveBulkSASampler(
        seed=0,
        shots=4,
        steps=8,
        presolve=True,
        enable_hard_fix=True,
        enable_probing=True,
        return_stats=True,
    )
    result, stats = sampler.run((qmatrix, index_map), return_stats=True)
    assert result
    assert stats["presolve_stats"]["hard_fix_count"] >= 1
    assert stats["probing_summary"]["strength"] == "medium"
    assert stats["reduced_problem_size"] <= len(index_map)
    for check in stats["restore_energy_check"]:
        assert isinstance(check["restored_energy"], float)
        assert isinstance(check["reported_energy"], float)


def test_presolved_sampler_accepts_diversity_and_restart_controls():
    qmatrix = [[0.0, 0.0], [0.0, 0.0]]
    index_map = {"x": 0, "y": 1}
    sampler = PresolvedAdaptiveBulkSASampler(
        seed=0,
        shots=4,
        steps=4,
        include_diverse=False,
        stall_steps=1,
        restart_ratio=0.5,
        restart_min_flips=1,
        restart_burnin_steps=0,
        restart_diversity_threshold=100.0,
        return_stats=True,
    )
    result, stats = sampler.run((qmatrix, index_map), return_stats=True, include_diverse=False)
    assert result
    assert stats["reduced_solve_stats"]["restart_count"] >= 1
