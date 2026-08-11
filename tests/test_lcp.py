import numpy as np
import pytest
import networkx as nx
from scipy import sparse

from operators import LCP


def test_keeps_zero_and_small_coefficients():
    hamiltonian = LCP({"I": 0, "X": 1e-20, "Y": 2 + 0j})

    assert hamiltonian.terms == {"I": 0.0, "X": 1e-20, "Y": 2.0}
    assert all(isinstance(value, float) for value in hamiltonian.terms.values())


def test_subtraction_combines_coefficients_without_removing_zero():
    left = LCP({"X": 2, "Z": 3})
    right = LCP({"X": 2, "Y": 1})

    difference = left - right

    assert difference.terms == {"X": 0.0, "Z": 3.0, "Y": -1.0}


def test_subtraction_rejects_different_qubit_counts():
    with pytest.raises(ValueError, match="same number of qubits"):
        LCP({"X": 1}) - LCP({"XX": 1})


def test_scalar_multiplication_from_both_sides():
    hamiltonian = LCP({"X": 2, "Z": -1})

    assert (0.5 * hamiltonian).terms == {"X": 1.0, "Z": -0.5}
    assert (hamiltonian * 2).terms == {"X": 4.0, "Z": -2.0}
    assert (hamiltonian * 0).terms == {"X": 0.0, "Z": 0.0}

    with pytest.raises(ValueError, match="scalar must be real"):
        hamiltonian * 2j


def test_commutation_graph_contains_all_pauli_strings_and_coefficients():
    hamiltonian = LCP({"II": 0.5, "XX": 1, "YY": -2, "XI": 3})

    graph = hamiltonian.commutation_graph()

    assert isinstance(graph, nx.Graph)
    assert set(graph.nodes) == {"II", "XX", "YY", "XI"}
    assert graph.nodes["YY"]["coefficient"] == -2
    assert nx.number_of_selfloops(graph) == 0


def test_commutation_graph_edges_represent_general_pauli_commutation():
    hamiltonian = LCP({"II": 1, "XX": 1, "YY": 1, "XI": 1})

    graph = hamiltonian.commutation_graph()

    # Identity commutes with every string. XX and YY commute because they
    # anti-commute on two qubits, whereas XI anti-commutes with YY.
    assert set(graph.edges) == {
        ("II", "XX"),
        ("II", "YY"),
        ("II", "XI"),
        ("XX", "YY"),
        ("XX", "XI"),
    }


def test_commutation_graph_of_empty_lcp_is_empty():
    graph = LCP({}, num_qubits=2).commutation_graph()

    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


def test_dense_matrix():
    hamiltonian = LCP({"ZI": 0.5, "IZ": 0.5, "XX": 1.0})
    expected = np.array(
        [
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, -1],
        ],
        dtype=complex,
    )

    np.testing.assert_allclose(hamiltonian.to_matrix(), expected)


def test_csr_matrix_matches_dense_matrix():
    hamiltonian = LCP({"ZI": 0.5, "IZ": 0.5, "XX": 1.0})
    csr = hamiltonian.to_csr()

    assert sparse.isspmatrix_csr(csr)
    np.testing.assert_allclose(csr.toarray(), hamiltonian.to_matrix())


def test_coefficient_one_norm():
    hamiltonian = LCP({"II": 0, "XX": -2, "YZ": 3})

    assert hamiltonian.coefficient_one_norm() == pytest.approx(5.0)


def test_operator_norm_uses_largest_singular_value():
    hamiltonian = LCP({"X": 1, "Y": 1})

    assert hamiltonian.operator_norm() == pytest.approx(np.sqrt(2))


def test_operator_norm_for_multi_qubit_sparse_matrix():
    hamiltonian = LCP({"XI": 1, "YI": 1})

    assert hamiltonian.operator_norm() == pytest.approx(np.sqrt(2))


def test_operator_norm_of_zero_operator():
    hamiltonian = LCP({"X": 0}, num_qubits=1)

    assert hamiltonian.operator_norm() == 0.0


def test_empty_hamiltonian_requires_num_qubits():
    with pytest.raises(ValueError, match="num_qubits"):
        LCP({})

    hamiltonian = LCP({}, num_qubits=2)
    np.testing.assert_array_equal(hamiltonian.to_matrix(), np.zeros((4, 4)))
    assert hamiltonian.to_csr().shape == (4, 4)


def test_rejects_invalid_terms():
    with pytest.raises(ValueError, match="invalid Pauli"):
        LCP({"XA": 1})
    with pytest.raises(ValueError, match="same length"):
        LCP({"X": 1, "ZZ": 2})
    with pytest.raises(TypeError, match="coefficient"):
        LCP({"X": "1"})
    with pytest.raises(ValueError, match="coefficients must be real"):
        LCP({"X": 1.0j})
