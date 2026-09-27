import numpy as np
import pytest
from scipy.linalg import expm
from qulacs import QuantumState

from improvement_factors import calculate_i_m, estimate_i_r_and_i_sig
from operators import LCH, LCP


@pytest.fixture
def example():
    hamiltonian = LCP({"XI": 3, "IZ": -4, "ZZ": 2})
    grouped = LCH([LCP({"XI": 3, "IZ": -4}), LCP({"ZZ": 2})])
    identity = np.eye(2)
    x = np.array([[0, 1], [1, 0]])
    z = np.diag([1, -1])
    single_matrices = (3 * np.kron(x, identity), -4 * np.kron(identity, z),
                      2 * np.kron(z, z))
    grouped_matrices = (single_matrices[0] + single_matrices[1], single_matrices[2])
    return hamiltonian, grouped, single_matrices, grouped_matrices


def test_i_m_matches_dense_variance_trace(example):
    hamiltonian, grouped, single_matrices, grouped_matrices = example
    matrix = sum(single_matrices)
    traces = []
    for branches in (single_matrices, grouped_matrices):
        norms = np.array([np.linalg.norm(branch) / 2 for branch in branches])
        probabilities = norms / sum(norms)
        variance = sum(branch @ branch / p for branch, p in zip(branches, probabilities))
        variance -= matrix @ matrix
        traces.append(np.trace(variance) / 4)
    assert traces == pytest.approx([52, 20])
    assert calculate_i_m(hamiltonian, grouped) == pytest.approx(traces[0] / traces[1])


@pytest.mark.parametrize("num_workers,num_trajectories", [(1, 11), (2, 11), (4, 2)])
def test_sampled_factors_match_dense_trajectories(example, num_workers, num_trajectories):
    hamiltonian, grouped, single_matrices, grouped_matrices = example
    time, steps, num_states = 0.4, 3, 3
    ideal_unitary = expm(-1j * time * sum(single_matrices))

    trajectory_seeds = np.random.default_rng(18).integers(
        2**63, size=(num_states, 2, num_workers),
    )

    chunk_sizes = [len(chunk) for chunk in np.array_split(np.arange(num_trajectories), num_workers)]
    mean_errors = []
    for decomposition_index, branches in enumerate((single_matrices, grouped_matrices)):
        norms = np.array([np.linalg.norm(branch) / 2 for branch in branches])
        probabilities = norms / sum(norms)
        unitaries = [expm(-1j * time / steps * branch / p)
                     for branch, p in zip(branches, probabilities)]
        errors = []
        for state_index in range(num_states):
            initial_state = QuantumState(2)
            initial_state.set_Haar_random_state(18 + state_index)
            initial = initial_state.get_vector()
            ideal = ideal_unitary @ initial
            fidelities, signals = [], []
            for chunk_index, chunk_size in enumerate(chunk_sizes):
                rng = np.random.default_rng(
                    trajectory_seeds[state_index, decomposition_index, chunk_index],
                )
                sequences = rng.choice(
                    len(branches), size=(chunk_size, steps), p=probabilities,
                )
                for sequence in sequences:
                    state = initial.copy()
                    for branch_index in sequence:
                        state = unitaries[branch_index] @ state
                    fidelities.append(abs(np.vdot(ideal, state)) ** 2)
                    signals.append(np.vdot(initial, state))
            errors.append((1 - np.mean(fidelities),
                           abs(np.vdot(initial, ideal) - np.mean(signals))))
        mean_errors.append(np.mean(errors, axis=0))
    expected = mean_errors[0] / mean_errors[1]

    actual = estimate_i_r_and_i_sig(
        hamiltonian, grouped, total_time=time, number_of_steps=steps,
        num_initial_states=num_states, num_trajectories=num_trajectories,
        seed=18, num_workers=num_workers,
    )
    assert actual == pytest.approx(expected, rel=1e-10, abs=1e-12)
