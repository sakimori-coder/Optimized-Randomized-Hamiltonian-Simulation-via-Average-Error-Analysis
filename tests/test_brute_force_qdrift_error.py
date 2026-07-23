import numpy as np

from brute_force_qdrift_error import brute_force_qdrift_error
from lch import LCH
from lcp import LCP


def test_direct_centered_second_moment_for_x_plus_z() -> None:
    result = brute_force_qdrift_error(
        LCP({"X": 1.0, "Z": 1.0}),
        time=1.0,
        epsilon=0.1,
    )

    np.testing.assert_allclose(result.sampling_probabilities, [0.5, 0.5])
    assert np.isclose(result.normalized_variance_constant, 0.5)
    assert np.isclose(result.variance_constant, 2.0)
    assert result.error_bound <= 0.1


def test_single_pauli_has_zero_centered_second_moment() -> None:
    result = brute_force_qdrift_error(
        LCP({"Y": -2.5}),
        time=1.0,
        epsilon=0.1,
    )

    assert result.normalized_variance_constant == 0.0
    assert result.variance_constant == 0.0
    assert result.required_steps == 1
    assert result.error_bound == 0.0


def test_accepts_lch_and_ignores_evolution_costs() -> None:
    result = brute_force_qdrift_error(
        LCH(
            [
                (1.0, "X", 10.0),
                (1.0, "Z", 20.0),
            ]
        ),
        time=1.0,
        epsilon=0.1,
    )

    np.testing.assert_allclose(result.sampling_probabilities, [0.5, 0.5])
    assert np.isclose(result.normalized_variance_constant, 0.5)
    assert np.isclose(result.variance_constant, 2.0)
