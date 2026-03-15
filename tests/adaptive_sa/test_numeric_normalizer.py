import numpy as np

from tytan.adaptive_sa import NumericNormalizer


def test_numeric_normalizer_rescales_and_prunes():
    matrix = np.array([[10.0, 0.0], [0.0, 0.005]])
    normalizer = NumericNormalizer(small_coeff_threshold=0.01)
    normalized, info = normalizer.normalize(matrix)
    assert np.isclose(info["scale_factor"], 0.1)
    assert normalized[0, 0] == 1.0
    assert normalized[1, 1] == 0.0
    assert info["pruned_coefficients"] == 1
    assert info["after"]["absmax"] <= 1.0
