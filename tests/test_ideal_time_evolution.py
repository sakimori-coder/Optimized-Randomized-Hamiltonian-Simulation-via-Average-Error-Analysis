import numpy as np
import pytest
from scipy.linalg import expm

from average_trace_distance.ideal_time_evolution import (
    ideal_time_evolution,
    lcp_linear_operator,
)
from operators import LCP


def _dense(operator) -> np.ndarray:
    factors = operator.factors
    return factors @ factors.conj().T


def test_single_x_has_known_evolution() -> None:
    initial_state = np.array([1.0, 0.0])

    density = ideal_time_evolution(
        LCP({"X": 1.0}),
        np.pi / 2,
        initial_state,
    )

    np.testing.assert_allclose(
        _dense(density),
        np.array([[0.0, 0.0], [0.0, 1.0]]),
        atol=1e-12,
    )
    assert density.rank_upper_bound == 1


@pytest.mark.parametrize(
    ("pauli", "target_index"),
    [("XI", 2), ("IX", 1)],
)
def test_leftmost_pauli_is_most_significant_qubit(
    pauli: str,
    target_index: int,
) -> None:
    initial_state = np.array([1.0, 0.0, 0.0, 0.0])
    expected = np.zeros((4, 4), dtype=complex)
    expected[target_index, target_index] = 1.0

    density = ideal_time_evolution(
        LCP({pauli: 1.0}),
        np.pi / 2,
        initial_state,
    )

    np.testing.assert_allclose(_dense(density), expected, atol=1e-12)


def test_matches_dense_exponential_for_noncommuting_hamiltonian() -> None:
    hamiltonian = LCP(
        {
            "II": 0.31,
            "XI": 0.7,
            "IZ": -0.4,
            "YY": 0.23,
        }
    )
    initial_state = np.array(
        [1.0 + 2.0j, -3.0 + 0.5j, 0.2 - 1.0j, 2.0 + 0.3j],
        dtype=complex,
    )
    initial_state /= np.linalg.norm(initial_state)
    time = 0.37

    expected = expm(-1j * time * hamiltonian.to_matrix()) @ initial_state
    density = ideal_time_evolution(hamiltonian, time, initial_state)

    np.testing.assert_allclose(
        _dense(density),
        np.outer(expected, expected.conj()),
        rtol=1e-12,
        atol=1e-12,
    )


def test_identity_is_applied_as_a_global_phase() -> None:
    initial_state = np.array([1.0, 2.0j, -0.5, 0.25j], dtype=complex)
    initial_state /= np.linalg.norm(initial_state)
    time = 0.4

    density = ideal_time_evolution(
        LCP({"II": 2.0}),
        time,
        initial_state,
    )

    np.testing.assert_allclose(
        _dense(density),
        np.outer(initial_state, initial_state.conj()),
        atol=1e-12,
    )


def test_linear_operator_matches_lcp_matrix_and_adjoint() -> None:
    hamiltonian = LCP(
        {
            "XI": 0.7,
            "YZ": -0.4,
            "II": 0.1,
        }
    )
    vector = np.array([1.0, 2.0j, 3.0, -0.5j])
    vectors = np.column_stack((vector, vector[::-1]))
    operator = lcp_linear_operator(hamiltonian)
    matrix = hamiltonian.to_matrix()

    np.testing.assert_allclose(operator @ vector, matrix @ vector, atol=1e-12)
    np.testing.assert_allclose(
        operator @ vectors,
        matrix @ vectors,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        operator.adjoint() @ vector,
        matrix.conj().T @ vector,
        atol=1e-12,
    )


def test_evolution_does_not_materialize_hamiltonian_or_modify_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hamiltonian = LCP({"X": 0.7, "Z": -0.2})
    initial_state = np.array([1.0, 1.0j]) / np.sqrt(2)
    original_state = initial_state.copy()

    def fail_if_materialized(*_args: object, **_kwargs: object) -> None:
        pytest.fail("ideal evolution must not materialize the Hamiltonian")

    monkeypatch.setattr(LCP, "to_matrix", fail_if_materialized)
    monkeypatch.setattr(LCP, "to_csr", fail_if_materialized)

    density = ideal_time_evolution(hamiltonian, 0.3, initial_state)

    np.testing.assert_array_equal(initial_state, original_state)
    np.testing.assert_allclose(np.trace(_dense(density)), 1.0, atol=1e-12)


def test_zero_time_returns_initial_density_and_copies_input() -> None:
    initial_state = np.array([1.0, 0.0])

    density = ideal_time_evolution(LCP({"X": 1.0}), 0.0, initial_state)
    initial_state[:] = [0.0, 1.0]

    np.testing.assert_array_equal(
        _dense(density),
        np.array([[1.0, 0.0], [0.0, 0.0]]),
    )
