import numpy as np
import pytest
from qulacs import QuantumState
from scipy.linalg import expm

from ideal_time_evolution import ideal_time_evolved_state
from operators import LCP


@pytest.mark.parametrize("time", [0.73, 0.0, -0.4])
def test_ideal_state_matches_dense_evolution_and_preserves_input(time):
    # Noncommuting terms, Y, negative coefficients and identity expose
    # errors in tensor ordering, rotation signs and the overall phase.
    identity = np.eye(2)
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.diag([1, -1])
    hamiltonian = LCP({"II": -0.136, "XI": -0.48, "IZ": 0.32,
                       "XY": 0.91, "YX": -0.26})
    matrix = (-0.136 * np.eye(4) - 0.48 * np.kron(x, identity)
              + 0.32 * np.kron(identity, z) + 0.91 * np.kron(x, y)
              - 0.26 * np.kron(y, x))
    vector = np.array([1 + 2j, -0.3j, -0.4 + 0.1j, 0.7])
    vector /= np.linalg.norm(vector)
    initial = QuantumState(2)
    initial.load(vector)

    evolved = ideal_time_evolved_state(hamiltonian, time, initial)

    assert isinstance(evolved, QuantumState)
    assert evolved is not initial
    np.testing.assert_allclose(evolved.get_vector(), expm(-1j * time * matrix) @ vector,
                               rtol=0, atol=3e-15)
    np.testing.assert_array_equal(initial.get_vector(), vector)
