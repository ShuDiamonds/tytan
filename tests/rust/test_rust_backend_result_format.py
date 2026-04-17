import numpy as np
import pytest

from tytan import _rust_backend as rb


@pytest.mark.skipif(not rb.rust_available(), reason="Rust extension not installed")
def test_aggregate_result_format_matches_contract():
    states = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    energies = np.array([-1.0, -1.0, 0.5], dtype=float)

    rows = rb.try_aggregate_results(states, energies, ["x", "y"])
    assert rows is not None
    assert isinstance(rows, list)
    assert len(rows) == 2

    first = rows[0]
    assert isinstance(first, list)
    assert len(first) == 3
    assert isinstance(first[0], dict)
    assert "x" in first[0]
    assert "y" in first[0]
    assert isinstance(first[1], float)
    assert isinstance(first[2], int)
