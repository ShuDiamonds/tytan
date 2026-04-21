from tytan import Compile, symbols
from tytan.adaptive_sa import AdaptiveBulkSASampler
from tytan import _rust_backend as rb


def test_adaptive_bulk_sa_produces_results_and_stats():
    x, y, z = symbols("x y z")
    qubo, _ = Compile((x + y + z - 1) ** 2).get_qubo()
    sampler = AdaptiveBulkSASampler(seed=0, shots=6, steps=32, enable_clamp=True, return_stats=True)
    result, stats = sampler.run(qubo, return_stats=True)
    assert isinstance(result, list)
    assert result
    assert "best_energy" in stats
    assert "strategy_weights" in stats
    assert stats["log_entries"]


def test_adaptive_bulk_sa_logger_contains_expected_fields():
    x, y, z = symbols("x y z")
    qubo, _ = Compile((x + y + z - 1) ** 2).get_qubo()
    sampler = AdaptiveBulkSASampler(seed=0, shots=3, steps=8, enable_clamp=True, return_stats=True)
    _, stats = sampler.run(qubo, return_stats=True)
    entry = stats["log_entries"][0]
    assert "strategy" in entry
    assert "temperature" in entry
    assert "improvements" in entry


def test_adaptive_bulk_sa_clamp_mode_reflected():
    x, y, z = symbols("x y z")
    qubo, _ = Compile((x + y + z - 1) ** 2).get_qubo()
    sampler = AdaptiveBulkSASampler(seed=0, shots=3, steps=4, enable_clamp=True, clamp_mode="hard", return_stats=True)
    _, stats = sampler.run(qubo, return_stats=True)
    assert stats["clamp_mode"] == "hard"


def test_adaptive_bulk_sa_can_skip_diverse_results():
    x, y, z = symbols("x y z")
    qubo, _ = Compile((x + y + z - 1) ** 2).get_qubo()
    sampler = AdaptiveBulkSASampler(
        seed=0,
        shots=4,
        steps=8,
        include_diverse=False,
        return_stats=True,
    )
    result, stats = sampler.run(qubo, return_stats=True, include_diverse=False)
    assert isinstance(result, list)
    assert result
    assert stats["best_energy"] is not None


def test_adaptive_bulk_sa_triggers_restart_on_stall():
    qmatrix = [[0.0, 0.0], [0.0, 0.0]]
    index_map = {0: 0, 1: 1}
    sampler = AdaptiveBulkSASampler(
        seed=0,
        shots=4,
        steps=4,
        stall_steps=1,
        restart_ratio=0.5,
        restart_min_flips=1,
        restart_burnin_steps=0,
        restart_diversity_threshold=100.0,
        return_stats=True,
    )
    _, stats = sampler.run((qmatrix, index_map), return_stats=True)
    assert stats["restart_count"] >= 1
    assert "pool_mean_pairwise_distance" in stats
    assert "state_diversity" in stats


def test_adaptive_bulk_sa_uses_rust_fast_path(monkeypatch):
    x, y = symbols("x y")
    qubo, _ = Compile((x + y - 1) ** 2).get_qubo()

    called = {"rust": 0}

    def fake_rust_core(*args, **kwargs):
        called["rust"] += 1
        return [[{"x": 1, "y": 0}, -1.0, 1]], {"best_energy": -1.0, "strategy_weights": {}, "log_entries": [], "restart_count": 0}

    monkeypatch.setattr(rb, "adaptive_bulk_sa_available", lambda: True)
    monkeypatch.setattr(rb, "try_adaptive_bulk_sa", fake_rust_core)

    sampler = AdaptiveBulkSASampler(seed=0, shots=2, steps=2, enable_clamp=False, return_stats=True)
    result, stats = sampler.run(qubo, return_stats=True)

    assert called["rust"] == 1
    assert result[0][0]["x"] == 1
    assert stats["best_energy"] == -1.0
