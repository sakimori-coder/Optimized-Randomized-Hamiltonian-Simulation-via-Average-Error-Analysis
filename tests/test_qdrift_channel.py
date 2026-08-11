import numpy as np
import pytest
from scipy.linalg import expm

import average_trace_distance.qdrift_channel as qdrift_channel_module
from average_trace_distance.ideal_time_evolution import ideal_time_evolution
from operators import LCH, LCP
from average_trace_distance.qdrift_channel import (
    prepare_qdrift_channel,
    qdrift_channel_output,
)


def _dense(operator) -> np.ndarray:
    factors = operator.factors
    return factors @ factors.conj().T


def test_equal_x_z_channel_maps_zero_to_maximally_mixed_state() -> None:
    hamiltonian = LCH(
        [
            (1.0, LCP({"X": 1.0}), 1.0),
            (1.0, LCP({"Z": 1.0}), 1.0),
        ]
    )

    density = qdrift_channel_output(
        hamiltonian,
        np.pi / 4,
        np.array([1.0, 0.0]),
    )

    np.testing.assert_allclose(_dense(density), 0.5 * np.eye(2), atol=1e-12)
    assert density.rank_upper_bound == 2


def test_channel_matches_explicit_weighted_branch_evolution() -> None:
    hamiltonian = LCH(
        [
            (2.0, LCP({"XI": 1.0}), 100.0),
            (-1.0, LCP({"IZ": 1.0}), 200.0),
        ]
    )
    initial_state = np.array([1.0, 1.0j, -0.5, 0.25j], dtype=complex)
    initial_state /= np.linalg.norm(initial_state)
    time = 0.23
    lambda_sum = 3.0
    expected = np.zeros((4, 4), dtype=complex)
    for probability, sign, pauli in (
        (2.0 / 3.0, 1.0, "XI"),
        (1.0 / 3.0, -1.0, "IZ"),
    ):
        unitary = expm(
            -1j
            * lambda_sum
            * time
            * sign
            * LCP({pauli: 1.0}).to_matrix()
        )
        state = unitary @ initial_state
        expected += probability * np.outer(state, state.conj())

    density = qdrift_channel_output(hamiltonian, time, initial_state)

    np.testing.assert_allclose(_dense(density), expected, atol=1e-12)
    np.testing.assert_allclose(np.trace(_dense(density)), 1.0, atol=1e-12)


def test_channel_supports_multi_pauli_lcp_branches() -> None:
    branches = [
        LCP({"XI": 0.5, "IZ": -0.25}),
        LCP({"YY": 1.0}),
    ]
    hamiltonian = LCH(
        [
            (2.0, branches[0], 3.0),
            (-1.0, branches[1], 4.0),
        ]
    )
    initial_state = np.array([1.0, 1.0j, -0.5, 0.25j], dtype=complex)
    initial_state /= np.linalg.norm(initial_state)
    time = 0.17
    lambda_sum = 3.0
    expected = np.zeros((4, 4), dtype=complex)
    for probability, sign, branch in (
        (2.0 / 3.0, 1.0, branches[0]),
        (1.0 / 3.0, -1.0, branches[1]),
    ):
        state = (
            expm(-1j * lambda_sum * time * sign * branch.to_matrix())
            @ initial_state
        )
        expected += probability * np.outer(state, state.conj())

    density = qdrift_channel_output(hamiltonian, time, initial_state)

    np.testing.assert_allclose(_dense(density), expected, atol=1e-11)
    assert density.rank_upper_bound == 2


def test_prepared_channel_reuses_branch_circuits_and_one_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branches = [
        LCP({"II": 0.13, "XI": 0.5, "IX": -0.25, "XX": 0.2}),
        LCP({"YY": 0.4, "ZZ": -0.3}),
    ]
    hamiltonian = LCH(
        [
            (2.0, branches[0], 3.0),
            (-1.0, branches[1], 4.0),
        ]
    )
    time = 0.19
    preparation_count = 0
    original_prepare = qdrift_channel_module._commuting_lcp_circuit

    def counting_prepare(operator: LCP, branch_time: float):
        nonlocal preparation_count
        preparation_count += 1
        return original_prepare(operator, branch_time)

    monkeypatch.setattr(
        qdrift_channel_module,
        "_commuting_lcp_circuit",
        counting_prepare,
    )
    prepared = prepare_qdrift_channel(hamiltonian, time)
    assert preparation_count == len(branches)

    workspace_count = 0
    original_quantum_state = qdrift_channel_module.QuantumState

    def counting_quantum_state(num_qubits: int):
        nonlocal workspace_count
        workspace_count += 1
        return original_quantum_state(num_qubits)

    monkeypatch.setattr(
        qdrift_channel_module,
        "QuantumState",
        counting_quantum_state,
    )
    initial_states = [
        np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
        np.array([1.0, 2.0j, -0.5, 0.25j], dtype=complex),
    ]
    initial_states[1] /= np.linalg.norm(initial_states[1])

    for call_index, initial_state in enumerate(initial_states, start=1):
        expected = np.zeros((4, 4), dtype=complex)
        for probability, sign, branch in (
            (2.0 / 3.0, 1.0, branches[0]),
            (1.0 / 3.0, -1.0, branches[1]),
        ):
            evolved = (
                expm(-1j * 3.0 * time * sign * branch.to_matrix())
                @ initial_state
            )
            expected += probability * np.outer(evolved, evolved.conj())

        density = prepared.output(initial_state)

        np.testing.assert_allclose(_dense(density), expected, atol=1e-12)
        assert workspace_count == call_index
        assert preparation_count == len(branches)


def test_single_term_qdrift_equals_ideal_density() -> None:
    hamiltonian = LCH([(-0.7, LCP({"Y": 1.0}), 3.0)])
    initial_state = np.array([1.0, 1.0j]) / np.sqrt(2)

    qdrift_density = qdrift_channel_output(
        hamiltonian,
        0.4,
        initial_state,
    )
    ideal_density = ideal_time_evolution(
        hamiltonian.lcp,
        0.4,
        initial_state,
    )

    np.testing.assert_allclose(
        _dense(qdrift_density),
        _dense(ideal_density),
        atol=1e-12,
    )


def test_zero_coefficient_and_zero_time_need_no_special_case() -> None:
    initial_state = np.array([1.0, 1.0j]) / np.sqrt(2)
    expected = np.outer(initial_state, initial_state.conj())

    zero_coefficient = qdrift_channel_output(
        LCH(
            [
                (1.0, LCP({"I": 1.0}), 1.0),
                (0.0, LCP({"X": 1.0}), 1.0),
            ]
        ),
        0.3,
        initial_state,
    )
    zero_time = qdrift_channel_output(
        LCH(
            [
                (1.0, LCP({"X": 1.0}), 1.0),
                (1.0, LCP({"Z": 1.0}), 1.0),
            ]
        ),
        0.0,
        initial_state,
    )

    np.testing.assert_allclose(
        _dense(zero_coefficient),
        expected,
        atol=1e-12,
    )
    np.testing.assert_allclose(_dense(zero_time), expected, atol=1e-12)


def test_rejects_unnormalized_initial_state() -> None:
    with pytest.raises(ValueError, match="normalized"):
        qdrift_channel_output(
            LCH([(1.0, LCP({"X": 1.0}), 1.0)]),
            1.0,
            np.array([2.0, 0.0]),
        )
