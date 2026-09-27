"""Ideal time evolution exp(-i t H)|psi> for a Pauli Hamiltonian."""

import numpy as np
from qulacs import GeneralQuantumOperator, QuantumState
from scipy.sparse.linalg import LinearOperator, expm_multiply

from operators import LCP


def ideal_time_evolved_state(
    hamiltonian: LCP,
    time: float,
    initial_state: QuantumState,
) -> QuantumState:
    """Return a new state ``exp(-1j*time*H)|psi>``; leave initial_state unchanged."""
    num_qubits = hamiltonian.num_qubits
    dimension = 1 << num_qubits
    operator = GeneralQuantumOperator(num_qubits)
    for pauli, coefficient in hamiltonian.terms.items():
        qulacs_pauli = " ".join(
            f"{symbol} {num_qubits - 1 - position}"
            for position, symbol in enumerate(pauli)
            if symbol != "I"
        )
        operator.add_operator(coefficient, qulacs_pauli)

    source = QuantumState(num_qubits)
    work = QuantumState(num_qubits)
    destination = QuantumState(num_qubits)

    def apply(vector):
        source.load(vector.reshape(-1))
        operator.apply_to_state(work, source, destination)
        return destination.get_vector()

    # H is Hermitian, so its adjoint uses the same operation.
    matrix = LinearOperator(
        (dimension, dimension), matvec=apply, rmatvec=apply, dtype=np.complex128,
    )
    # Only the identity Pauli string has a nonzero trace.
    trace = dimension * hamiltonian.terms.get("I" * num_qubits, 0.0)
    evolved = expm_multiply(
        -1j * time * matrix, initial_state.get_vector(), traceA=-1j * time * trace,
    )
    state = QuantumState(num_qubits)
    state.load(evolved)
    return state
