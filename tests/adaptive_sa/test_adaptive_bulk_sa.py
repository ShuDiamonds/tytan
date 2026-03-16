from tytan import Compile, symbols
from tytan.adaptive_sa import AdaptiveBulkSASampler


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
