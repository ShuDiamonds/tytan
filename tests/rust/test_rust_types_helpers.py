import numpy as np

from tytan import _rust_types as rt


def test_state_vector_and_matrix_conversion():
    vec = rt.as_state_vector([0, 1, 1])
    mat = rt.as_state_matrix([[0, 1], [1, 0]])

    assert isinstance(vec, np.ndarray)
    assert isinstance(mat, np.ndarray)
    assert vec.dtype == float
    assert mat.dtype == float
    np.testing.assert_array_equal(vec, np.array([0.0, 1.0, 1.0]))
    np.testing.assert_array_equal(mat, np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_energy_and_index_vector_conversion():
    energies = rt.as_energy_vector([1, 2.5, -3])
    indices = rt.as_index_vector([0, 2, 3])

    assert energies.dtype == float
    assert indices.dtype == np.int64
    np.testing.assert_array_equal(energies, np.array([1.0, 2.5, -3.0]))
    np.testing.assert_array_equal(indices, np.array([0, 2, 3], dtype=np.int64))


def test_normalize_hobomix_converts_matrix_to_float():
    qmatrix = np.array([[1, 2], [0, 3]], dtype=int)
    index_map = {"x": 0, "y": 1}

    normalized_matrix, normalized_index_map = rt.normalize_hobomix((qmatrix, index_map))

    assert normalized_matrix.dtype == float
    np.testing.assert_array_equal(normalized_matrix, np.array([[1.0, 2.0], [0.0, 3.0]]))
    assert normalized_index_map == index_map
