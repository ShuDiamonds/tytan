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
