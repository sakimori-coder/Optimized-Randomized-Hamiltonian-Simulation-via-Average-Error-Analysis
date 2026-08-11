import numpy as np
import pytest
from qulacs import QuantumState

import average_trace_distance.comparison as comparison_module
from average_trace_distance.comparison import (
    estimate_qdrift_haar_average_trace_distances,
)
from average_trace_distance.monte_carlo import (
    haar_random_state,
)
from operators import LCH, LCP


def test_haar_random_state_is_generated_by_qulacs_seed() -> None:
    expected_rng = np.random.default_rng(123)
    qulacs_seed = int(expected_rng.integers(0, 1 << 31))
    expected_state = QuantumState(4)
    expected_state.set_Haar_random_state(qulacs_seed)

    actual = haar_random_state(4, np.random.default_rng(123))

    np.testing.assert_array_equal(actual, expected_state.get_vector())
    assert np.linalg.norm(actual) == pytest.approx(1.0)


def test_multiple_decompositions_share_states_and_use_step_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zero = LCP({}, num_qubits=1)
    x_decomposition = LCH(
        [(1.0, LCP({"X": 1.0})), (-1.0, LCP({"X": 1.0}))]
    )
    z_decomposition = LCH(
        [(1.0, LCP({"Z": 1.0})), (-1.0, LCP({"Z": 1.0}))]
    )
    states = iter(
        [
            np.array([1.0, 0.0], dtype=complex),
            np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0),
        ]
    )
    calls = 0

    def fixed_haar_state(_num_qubits, _rng):
        nonlocal calls
        calls += 1
        return next(states).copy()

    monkeypatch.setattr(
        comparison_module,
        "haar_random_state",
        fixed_haar_state,
    )

    x_result, z_result = estimate_qdrift_haar_average_trace_distances(
        zero,
        total_time=0.2,
        number_of_steps=2,
        qdrift_decompositions=[x_decomposition, z_decomposition],
        num_initial_states=2,
    )

    expected = np.sin(0.2) ** 2
    assert calls == 2
    assert x_result.step_time == pytest.approx(0.1)
    np.testing.assert_allclose(x_result.values, [expected, 0.0], atol=1e-12)
    np.testing.assert_allclose(z_result.values, [0.0, expected], atol=1e-12)
    assert x_result.mean == pytest.approx(expected / 2.0)
    assert z_result.mean == pytest.approx(expected / 2.0)


def test_single_pauli_qdrift_is_exact_for_every_initial_state() -> None:
    hamiltonian = LCH([(0.7, LCP({"X": 1.0}))])

    estimate = estimate_qdrift_haar_average_trace_distances(
        hamiltonian.lcp,
        total_time=0.23,
        number_of_steps=1,
        qdrift_decompositions=[hamiltonian],
        num_initial_states=4,
        seed=11,
    )[0]

    np.testing.assert_allclose(estimate.values, 0.0, atol=1e-12)
    assert estimate.mean == pytest.approx(0.0, abs=1e-12)


def test_fixed_seed_reproduces_every_trace_distance() -> None:
    hamiltonian = LCH(
        [
            (0.8, LCP({"X": 1.0})),
            (-0.3, LCP({"Z": 1.0})),
        ]
    )
    arguments = dict(
        target_hamiltonian=hamiltonian.lcp,
        total_time=0.2,
        number_of_steps=1,
        qdrift_decompositions=[hamiltonian],
        num_initial_states=4,
        seed=123,
    )

    first = estimate_qdrift_haar_average_trace_distances(**arguments)[0]
    second = estimate_qdrift_haar_average_trace_distances(**arguments)[0]

    np.testing.assert_array_equal(first.values, second.values)
    assert first.mean == second.mean
    assert first.sample_standard_deviation == second.sample_standard_deviation
    assert first.standard_error == second.standard_error


def test_single_sample_has_zero_sample_deviation_and_standard_error() -> None:
    hamiltonian = LCH(
        [
            (0.8, LCP({"X": 1.0})),
            (-0.3, LCP({"Z": 1.0})),
        ]
    )

    estimate = estimate_qdrift_haar_average_trace_distances(
        hamiltonian.lcp,
        total_time=0.2,
        number_of_steps=1,
        qdrift_decompositions=[hamiltonian],
        num_initial_states=1,
        seed=5,
    )[0]

    assert estimate.values.shape == (1,)
    assert estimate.sample_standard_deviation == 0.0
    assert estimate.standard_error == 0.0


def test_low_rank_and_dense_methods_agree_for_the_same_haar_states() -> None:
    hamiltonian = LCH(
        [
            (0.5, LCP({"XI": 1.0})),
            (0.2, LCP({"IZ": 1.0})),
            (-0.4, LCP({"ZZ": 1.0})),
        ]
    )
    arguments = dict(
        target_hamiltonian=hamiltonian.lcp,
        total_time=0.15,
        number_of_steps=1,
        qdrift_decompositions=[hamiltonian],
        num_initial_states=3,
        seed=17,
    )

    low_rank = estimate_qdrift_haar_average_trace_distances(
        trace_distance_method="low_rank",
        **arguments,
    )[0]
    dense = estimate_qdrift_haar_average_trace_distances(
        trace_distance_method="dense",
        **arguments,
    )[0]

    np.testing.assert_allclose(low_rank.values, dense.values, atol=1e-12)
    assert low_rank.mean == pytest.approx(dense.mean, abs=1e-12)
