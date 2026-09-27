"""Sample and execute one qDRIFT trajectory from commuting Pauli groups."""

import numpy as np
from qulacs import QuantumCircuit, QuantumState

from operators import LCH


def sample_qdrift_state(
    decomposition: LCH, total_time, number_of_steps, initial_state: QuantumState, *, rng=None,
) -> QuantumState:
    """Return one qDRIFT-evolved copy of initial_state for H = sum_j H_j.

    Each H_j must contain mutually commuting Pauli terms. Sample j with
    p_j proportional to its HS norm and apply exp(-i total_time H_j / (N p_j)).
    Pass a NumPy Generator as rng to draw reproducible successive trajectories.
    """
    rng = np.random.default_rng(rng)
    probabilities = decomposition.sampling_probabilities()
    samples = rng.choice(len(decomposition.terms), size=number_of_steps, p=probabilities)
    step_time = total_time / number_of_steps
    circuit = QuantumCircuit(decomposition.num_qubits)
    # The leftmost Pauli character acts on the most significant qubit.
    qubits = list(reversed(range(decomposition.num_qubits)))
    for j in samples:
        for pauli, coefficient in decomposition.terms[j].terms.items():
            # Include I (ID 0) so identity terms retain their global phase.
            pauli_ids = ["IXYZ".index(symbol) for symbol in pauli]
            # Qulacs uses exp(+i angle P / 2).
            angle = -2 * step_time * coefficient / probabilities[j]
            circuit.add_multi_Pauli_rotation_gate(qubits, pauli_ids, angle)
    state = initial_state.copy()
    circuit.update_quantum_state(state)
    return state
