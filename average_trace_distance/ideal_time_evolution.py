r"""Matrix-free ideal time evolution for average trace-distance experiments."""

from __future__ import annotations

import numpy as np
from qulacs import GeneralQuantumOperator, QuantumState
from scipy.sparse.linalg import LinearOperator, expm_multiply

from operators import LCP

from .density_operator import DensityOperator


def lcp_linear_operator(hamiltonian: LCP) -> LinearOperator:
    r"""Represent an LCP as a Qulacs-backed SciPy ``LinearOperator``.

    One operator application costs ``O(L 2**n)`` for ``L`` active Pauli
    terms and ``n`` qubits.  The full ``2**n`` by ``2**n`` matrix is never
    constructed.  The returned operator reuses Qulacs workspaces and must
    not be applied concurrently from multiple threads.
    """
    if not isinstance(hamiltonian, LCP):
        raise TypeError("hamiltonian must be an LCP")

    dimension = 1 << hamiltonian.num_qubits
    operator = GeneralQuantumOperator(hamiltonian.num_qubits)
    for pauli, coefficient in hamiltonian.terms.items():
        if coefficient == 0.0:
            continue
        qulacs_pauli = " ".join(
            f"{symbol} {hamiltonian.num_qubits - 1 - position}"
            for position, symbol in enumerate(pauli)
            if symbol != "I"
        )
        operator.add_operator(coefficient, qulacs_pauli)

    source_state = QuantumState(hamiltonian.num_qubits)
    work_state = QuantumState(hamiltonian.num_qubits)
    destination_state = QuantumState(hamiltonian.num_qubits)

    def apply(vectors: np.ndarray) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.complex128)
        is_vector = array.ndim == 1
        matrix = array.reshape(dimension, -1)
        result = np.empty(matrix.shape, dtype=np.complex128)

        for column in range(matrix.shape[1]):
            source_state.load(matrix[:, column])
            operator.apply_to_state(
                work_state,
                source_state,
                destination_state,
            )
            result[:, column] = destination_state.get_vector()

        return result[:, 0] if is_vector else result

    return LinearOperator(
        shape=(dimension, dimension),
        matvec=apply,
        rmatvec=apply,
        matmat=apply,
        rmatmat=apply,
        dtype=np.complex128,
    )


def ideal_time_evolution(
    hamiltonian: LCP,
    time: float,
    initial_state: np.ndarray,
) -> DensityOperator:
    r"""Return the ideal evolved density operator as a rank-one operator."""
    state = np.asarray(initial_state, dtype=np.complex128)
    terms = hamiltonian.terms
    identity = "I" * hamiltonian.num_qubits
    identity_coefficient = terms.pop(identity, 0.0)
    active_terms = {
        pauli: coefficient
        for pauli, coefficient in terms.items()
        if coefficient != 0
    }
    global_phase = np.exp(-1j * time * identity_coefficient)

    if time == 0.0 or not active_terms:
        return DensityOperator(
            [global_phase * state],
            [1.0],
        )

    traceless_hamiltonian = LCP(
        active_terms,
        num_qubits=hamiltonian.num_qubits,
    )
    generator = (-1j * time) * lcp_linear_operator(traceless_hamiltonian)
    evolved = expm_multiply(
        generator,
        state,
        traceA=0.0,
    )
    evolved_state = global_phase * np.asarray(evolved, dtype=np.complex128)
    return DensityOperator(
        [evolved_state],
        [1.0],
    )
