r"""One-step qDRIFT channel outputs for average trace-distance calculations."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray
from qulacs import QuantumCircuit, QuantumState

from operators import LCH, LCP

from .density_operator import DensityOperator


_QULACS_PAULI_IDS = {"X": 1, "Y": 2, "Z": 3}


@dataclass(frozen=True)
class PreparedQDriftChannel:
    """Reusable branch circuits for one LCH qDRIFT channel step."""

    num_qubits: int
    probabilities: NDArray[np.float64]
    circuits: tuple[QuantumCircuit, ...]

    def output(self, initial_state: np.ndarray) -> DensityOperator:
        """Apply every prepared branch to one initial state."""
        state_vector = _validated_initial_state(
            self.num_qubits,
            initial_state,
        )
        dimension = 1 << self.num_qubits
        evolved_states = np.empty(
            (len(self.circuits), dimension),
            dtype=np.complex128,
        )
        workspace = QuantumState(self.num_qubits)
        for index, circuit in enumerate(self.circuits):
            workspace.load(state_vector)
            circuit.update_quantum_state(workspace)
            evolved_states[index] = workspace.get_vector()

        return DensityOperator(
            evolved_states,
            self.probabilities,
        )


def qdrift_channel_output(
    hamiltonian: LCH,
    time: float,
    initial_state: np.ndarray,
) -> DensityOperator:
    r"""Return the output density operator of one LCH qDRIFT step.

    For ``H = sum_j c_j A_j`` with LCP samples ``A_j``, this function uses

    ``lambda = sum_j |c_j|``, ``p_j = |c_j| / lambda``, and
    ``H_j = sign(c_j) A_j``.  The input ``time`` is the duration of this one
    step, so branch ``j`` applies ``exp(-1j * lambda * time * H_j)``.
    Pauli strings within every LCP sample ``A_j`` are assumed to commute, so
    each branch is evaluated as an exact sequence of Pauli rotations.  This
    commutativity precondition is not checked.

    The returned density operator stores one state vector per LCP term.  It
    does not represent a composition of multiple independently sampled
    qDRIFT steps.
    """
    return prepare_qdrift_channel(hamiltonian, time).output(initial_state)


def prepare_qdrift_channel(
    hamiltonian: LCH,
    time: float,
) -> PreparedQDriftChannel:
    r"""Prepare all branch circuits of one LCH qDRIFT channel step.

    For branch ``j``, the prepared circuit implements
    ``exp(-1j * Lambda * time * sign(c_j) * A_j)``.  Pauli strings inside
    every LCP ``A_j`` are assumed to commute pairwise; this is not checked.
    """
    if not isinstance(hamiltonian, LCH):
        raise TypeError("hamiltonian must be an LCH")
    if not isinstance(time, Real) or isinstance(time, bool):
        raise TypeError("time must be a real number")
    time = float(time)
    if not np.isfinite(time):
        raise ValueError("time must be finite")

    terms = hamiltonian.terms
    if any(not np.isfinite(coefficient) for coefficient, _ in terms):
        raise ValueError("hamiltonian coefficients must be finite")
    probabilities = hamiltonian.sampling_probabilities()
    probabilities.setflags(write=False)
    lambda_sum = hamiltonian.coefficient_one_norm()
    branch_time = lambda_sum * time
    if not np.isfinite(branch_time):
        raise ValueError("lambda * time must be finite")

    circuits = tuple(
        _commuting_lcp_circuit(
            operator,
            branch_time * float(np.sign(coefficient)),
        )
        for coefficient, operator in terms
    )
    return PreparedQDriftChannel(
        num_qubits=hamiltonian.num_qubits,
        probabilities=probabilities,
        circuits=circuits,
    )


def _commuting_lcp_circuit(
    hamiltonian: LCP,
    time: float,
) -> QuantumCircuit:
    r"""Build ``exp(-1j*time*H)`` from commuting Pauli rotations."""
    num_qubits = hamiltonian.num_qubits
    circuit = QuantumCircuit(num_qubits)

    for pauli, coefficient in hamiltonian.terms.items():
        indices = [
            num_qubits - 1 - position
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
    initial_state: np.ndarray,
) -> NDArray[np.complex128]:
    """Validate and copy one normalized state vector."""
    if not isinstance(initial_state, np.ndarray):
        raise TypeError("initial_state must be a NumPy array")
    dimension = 1 << num_qubits
    if initial_state.ndim != 1 or initial_state.shape != (dimension,):
        raise ValueError(f"initial_state must have shape ({dimension},)")
    state = np.array(initial_state, dtype=np.complex128, copy=True)
    if not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain only finite values")
    if not np.isclose(np.linalg.norm(state), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("initial_state must be normalized")
    return state
