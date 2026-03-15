import numpy as np

from tytan.adaptive_sa import ReducedQuboMapper


def test_reduced_qubo_mapper_restores_states_with_fixed_variables():
    index_map = {"a": 0, "b": 1}
    mapper = ReducedQuboMapper(index_map)
    mapper.register_fixed("a", 1)
    mapper.update_active_indices([1])

    restored = mapper.restore_results([[{"b": 1}, -2.0, 3]])
    assert mapper.reduced_index_map == {"b": 0}
    state, energy, count = restored[0]
    assert state["a"] == 1
    assert state["b"] == 1
    assert energy == -2.0
    assert count == 3

    np.testing.assert_array_equal(mapper.full_state_array(state), np.array([1.0, 1.0]))
