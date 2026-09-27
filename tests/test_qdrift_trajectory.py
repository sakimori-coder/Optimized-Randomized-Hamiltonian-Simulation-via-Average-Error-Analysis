import numpy as np
import pytest
from qulacs import QuantumState
from scipy.linalg import expm

from operators import LCH, LCP
from qdrift_trajectory import sample_qdrift_state



def test_sampled_states_match_dense_products_and_advance_rng():
    identity = np.eye(2)
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.diag([1, -1])
    decomposition = LCH([
        LCP({"II": -0.136, "XI": -0.48, "IZ": 0.32}),
        LCP({"XY": 0.91, "YX": -0.26}),
    ])
    matrices = [
        -0.136 * np.eye(4) - 0.48 * np.kron(x, identity) + 0.32 * np.kron(identity, z),
        0.91 * np.kron(x, y) - 0.26 * np.kron(y, x),
    ]
    norms = np.array([np.linalg.norm(matrix) / 2 for matrix in matrices])
    probabilities = norms / sum(norms)
    total_time, steps = 0.73, 7
    branches = [expm(-1j * total_time / steps * matrix / p)
                for matrix, p in zip(matrices, probabilities)]
    rng, reference_rng = np.random.default_rng(42), np.random.default_rng(42)
    initial_vector = np.array([1 + 2j, -0.3j, -0.4 + 0.1j, 0.7])
    initial_vector /= np.linalg.norm(initial_vector)
    initial = QuantumState(2)
    initial.load(initial_vector)
    outputs = []
    for _ in range(2):
        samples = reference_rng.choice(2, size=steps, p=probabilities)
        assert set(samples) == {0, 1}
        expected = initial_vector.copy()
        for j in samples:
            expected = branches[j] @ expected
        final = sample_qdrift_state(decomposition, total_time, steps, initial, rng=rng)
        assert isinstance(final, QuantumState)
        assert final is not initial
        np.testing.assert_allclose(final.get_vector(), expected, rtol=0, atol=3e-15)
        np.testing.assert_array_equal(initial.get_vector(), initial_vector)
        outputs.append(final.get_vector())
    assert not np.allclose(outputs[0], outputs[1])


@pytest.mark.parametrize("steps", [1, 5])
def test_single_commuting_group_is_exact_including_identity(steps):
    hamiltonian = LCP({"II": 0.3, "ZI": -0.8, "IZ": 0.2})
    decomposition = LCH([hamiltonian])
    z = np.diag([1, -1])
    matrix = 0.3 * np.eye(4) - 0.8 * np.kron(z, np.eye(2)) + 0.2 * np.kron(np.eye(2), z)
    initial_vector = np.array([1, 1j, -1, -1j]) / 2
    initial = QuantumState(2)
    initial.load(initial_vector)
    final = sample_qdrift_state(decomposition, 0.9, steps, initial)
    np.testing.assert_allclose(final.get_vector(), expm(-0.9j * matrix) @ initial_vector,
                               rtol=0, atol=3e-15)
    np.testing.assert_array_equal(initial.get_vector(), initial_vector)
