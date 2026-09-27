"""Compare Pauli qDRIFT (LCP) with a supplied commuting grouping (LCH).

Both inputs must represent the same Hamiltonian. Grouped qDRIFT samples
H_j with probability proportional to its normalized HS norm.
"""

import math
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import partial
from multiprocessing import get_context

from operators import LCH, LCP


def calculate_i_m(hamiltonian: LCP, grouped: LCH) -> float:
    """Return I_M = tau(M_single) / tau(M_grouped) from coefficients only.

    tau(M) = Lambda**2 - tau(H**2), with tau(H**2) = sum_P a_P**2.
    No state vectors, trajectory sampling or worker processes are needed.
    """
    h_squared = math.fsum(a * a for a in hamiltonian.terms.values())
    lambda_single = math.fsum(abs(a) for a in hamiltonian.terms.values())
    lambda_grouped = grouped.lambda_sum()
    return (lambda_single**2 - h_squared) / (lambda_grouped**2 - h_squared)


def estimate_i_r_and_i_sig(
    hamiltonian: LCP,
    grouped: LCH,
    *,
    total_time: float,
    number_of_steps: int,
    num_initial_states: int = 20,
    num_trajectories: int = 200,
    seed: int = 42,
    num_workers: int = 1,
) -> tuple[float, float]:
    """Return (I_r, I_sig), each the single/grouped mean-error ratio.

    Both methods share Haar inputs and ideal outputs, with independent
    qDRIFT trajectories. Average complex signals before taking their error.
    Trajectories are divided into num_workers batches on this node.
    Fix seed and num_workers to reproduce samples. Initial states are processed
    in order.
    """
    import numpy as np
    from qulacs import QuantumState
    from qulacs.state import inner_product

    from ideal_time_evolution import ideal_time_evolved_state

    rng = np.random.default_rng(seed)

    single = LCH(
        [LCP({word: a}) for word, a in hamiltonian.terms.items()],
        num_qubits=hamiltonian.num_qubits,
    )
    size, remainder = divmod(num_trajectories, num_workers)
    chunk_sizes = [size + (index < remainder) for index in range(num_workers)]
    single_infidelities = np.empty(num_initial_states)
    grouped_infidelities = np.empty(num_initial_states)
    single_signal_errors = np.empty(num_initial_states)
    grouped_signal_errors = np.empty(num_initial_states)

    pool = (
        ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context("fork"))
        if num_workers > 1 else nullcontext()
    )
    with pool as executor:
        map_chunks = executor.map if executor is not None else map
        for state_index in range(num_initial_states):
            initial = QuantumState(hamiltonian.num_qubits)
            initial.set_Haar_random_state(seed + state_index)
            ideal = ideal_time_evolved_state(hamiltonian, total_time, initial)
            ideal_signal = inner_product(initial, ideal)

            # Single Pauli qDRIFT.
            sample_single = partial(
                _trajectory_sums, single, total_time, number_of_steps, initial, ideal,
            )
            single_sums = list(map_chunks(
                sample_single, chunk_sizes, rng.integers(2**63, size=num_workers),
            ))
            mean_fidelity = sum(fidelity for fidelity, _ in single_sums) / num_trajectories
            mean_signal = sum(signal for _, signal in single_sums) / num_trajectories
            single_infidelities[state_index] = 1 - mean_fidelity
            single_signal_errors[state_index] = abs(ideal_signal - mean_signal)

            # Grouped qDRIFT.
            sample_grouped = partial(
                _trajectory_sums, grouped, total_time, number_of_steps, initial, ideal,
            )
            grouped_sums = list(map_chunks(
                sample_grouped, chunk_sizes, rng.integers(2**63, size=num_workers),
            ))
            mean_fidelity = sum(fidelity for fidelity, _ in grouped_sums) / num_trajectories
            mean_signal = sum(signal for _, signal in grouped_sums) / num_trajectories
            grouped_infidelities[state_index] = 1 - mean_fidelity
            grouped_signal_errors[state_index] = abs(ideal_signal - mean_signal)

    return (
        float(np.mean(single_infidelities) / np.mean(grouped_infidelities)),
        float(np.mean(single_signal_errors) / np.mean(grouped_signal_errors)),
    )


def _trajectory_sums(
    decomposition, total_time, number_of_steps, initial, ideal, count, seed,
):
    """Sum fidelities and complex signals for one trajectory chunk."""
    import numpy as np
    from qulacs.state import inner_product

    from qdrift_trajectory import sample_qdrift_state

    rng = np.random.default_rng(seed)
    fidelity_sum, signal_sum = 0.0, 0j
    for _ in range(count):
        final = sample_qdrift_state(
            decomposition, total_time, number_of_steps, initial, rng=rng,
        )
        fidelity_sum += abs(inner_product(ideal, final)) ** 2
        signal_sum += inner_product(initial, final)
    return fidelity_sum, signal_sum
