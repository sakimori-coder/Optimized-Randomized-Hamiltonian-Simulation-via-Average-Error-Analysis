r"""Qulacs state-vector primitives used by the three qDRIFT metrics."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray
from qulacs import GeneralQuantumOperator, QuantumCircuit, QuantumState
from scipy.sparse.linalg import LinearOperator, expm_multiply

from operators import LCH, LCP


_QULACS_PAULI_IDS = {"X": 1, "Y": 2, "Z": 3}


@dataclass(frozen=True)
class PreparedQDriftBranches:
    """Sampling probabilities and Qulacs circuits for one qDRIFT step."""

    num_qubits: int
    probabilities: NDArray[np.float64]
    circuits: tuple[QuantumCircuit, ...]


def haar_random_state(
    num_qubits: int,
    rng: np.random.Generator,
) -> NDArray[np.complex128]:
    """Draw one Haar-random normalized state with Qulacs."""
    state = QuantumState(num_qubits)
    state.set_Haar_random_state(int(rng.integers(0, 1 << 31)))
    return np.array(state.get_vector(), dtype=np.complex128, copy=True)


def ideal_time_evolved_state(
    hamiltonian: LCP,
    time: float,
    initial_state: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Return ``exp(-1j*time*H)|psi>`` without constructing a dense matrix."""
    state = _validated_initial_state(hamiltonian.num_qubits, initial_state)
    terms = hamiltonian.terms
    identity = "I" * hamiltonian.num_qubits
    identity_coefficient = terms.pop(identity, 0.0)
    active_terms = {
        pauli: coefficient
        for pauli, coefficient in terms.items()
        if coefficient != 0.0
    }
    global_phase = np.exp(-1j * time * identity_coefficient)
    if time == 0.0 or not active_terms:
        return global_phase * state

    generator = (-1j * time) * _lcp_linear_operator(
        LCP(active_terms, num_qubits=hamiltonian.num_qubits)
    )
    evolved = expm_multiply(generator, state, traceA=0.0)
    return global_phase * np.asarray(evolved, dtype=np.complex128)


def prepare_qdrift_branches(
    decomposition: LCH,
    step_time: float,
) -> PreparedQDriftBranches:
    r"""Prepare ``exp(-i step_time B_j)`` for every qDRIFT sample.

    Pauli strings inside each grouped sample commute, so its exponential is
    exactly a product of Pauli rotations.  Identity strings are omitted here;
    their QPE-relevant global phases are restored by the trajectory sampler.
    """
    if not isinstance(step_time, Real) or isinstance(step_time, bool):
        raise TypeError("step_time must be a real number")
    step_time = float(step_time)
    if not np.isfinite(step_time):
        raise ValueError("step_time must be finite")

    probabilities = decomposition.sampling_probabilities()
    probabilities.setflags(write=False)
    branch_time = decomposition.coefficient_one_norm() * step_time
    circuits = tuple(
        _commuting_lcp_circuit(
            operator,
            branch_time * float(np.sign(coefficient)),
        )
        for coefficient, operator in decomposition.terms
    )
    return PreparedQDriftBranches(
        num_qubits=decomposition.num_qubits,
        probabilities=probabilities,
        circuits=circuits,
    )


def _lcp_linear_operator(hamiltonian: LCP) -> LinearOperator:
    dimension = 1 << hamiltonian.num_qubits
    operator = GeneralQuantumOperator(hamiltonian.num_qubits)
    for pauli, coefficient in hamiltonian.term_items():
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
            operator.apply_to_state(work_state, source_state, destination_state)
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


def _commuting_lcp_circuit(hamiltonian: LCP, time: float) -> QuantumCircuit:
    circuit = QuantumCircuit(hamiltonian.num_qubits)
    for pauli, coefficient in hamiltonian.term_items():
        indices = [
            hamiltonian.num_qubits - 1 - position
            for position, symbol in enumerate(pauli)
            if symbol != "I"
        ]
        if not indices:
            continue
        pauli_ids = [
            _QULACS_PAULI_IDS[symbol]
            for symbol in pauli
            if symbol != "I"
        ]
        circuit.add_multi_Pauli_rotation_gate(
            indices,
            pauli_ids,
            -2.0 * time * coefficient,
        )
    return circuit


def _validated_initial_state(
    num_qubits: int,
    initial_state: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    dimension = 1 << num_qubits
    state = np.asarray(initial_state, dtype=np.complex128)
    if state.ndim != 1 or state.shape != (dimension,):
        raise ValueError(f"initial_state must have shape ({dimension},)")
    if not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain only finite values")
    if not np.isclose(np.linalg.norm(state), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("initial_state must be normalized")
    return np.array(state, dtype=np.complex128, copy=True)


__all__ = [
    "PreparedQDriftBranches",
    "haar_random_state",
    "ideal_time_evolved_state",
    "prepare_qdrift_branches",
]
