import numpy as np
import pytest

from tytan import _rust_backend as rb


@pytest.mark.skipif(not rb.adaptive_bulk_sa_available(), reason="Adaptive Rust core not installed")
def test_adaptive_bulk_rust_core_returns_rows_and_stats():
    qmatrix = np.array([[0.0, -1.0], [0.0, 0.0]], dtype=float)
    rows, stats = rb.try_adaptive_bulk_sa(
        qmatrix,
        ["x", "y"],
        shots=4,
        steps=4,
        batch_size=4,
        init_temp=5.0,
        end_temp=0.1,
        schedule="linear",
        adaptive=True,
        strategy_configs=None,
        epsilon=0.2,
        include_diverse=True,
        pool_max_entries=16,
        near_dup_hamming=1,
        replace_margin=1e-6,
        stall_steps=2,
        restart_ratio=0.5,
        restart_min_flips=1,
        restart_burnin_steps=0,
        restart_diversity_threshold=10.0,
        novelty_weight=0.05,
        seed=0,
    )
    assert rows
    assert stats["rust_core"] is True
    assert "best_energy" in stats
    assert "strategy_weights" in stats
