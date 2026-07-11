from itertools import combinations

import numpy as np
import pytest

from commuting_groups import decompose_into_commuting_groups
from lcp import LCP


def test_decomposition_partitions_all_terms_without_duplicates():
    hamiltonian = LCP(
        {
            "XI": 1.0,
            "IX": 2.0,
            "ZI": 3.0,
            "IZ": 4.0,
            "YY": 5.0,
        }
    )

    groups = decompose_into_commuting_groups(hamiltonian)
    grouped_paulis = [pauli for group in groups for pauli in group.terms]

    assert set(grouped_paulis) == set(hamiltonian.terms)
    assert len(grouped_paulis) == len(set(grouped_paulis))

    reconstructed = sum(
        (group.to_matrix() for group in groups),
        np.zeros_like(hamiltonian.to_matrix()),
    )
    np.testing.assert_allclose(reconstructed, hamiltonian.to_matrix())


def test_every_group_is_pairwise_commuting():
    hamiltonian = LCP(
        {
            "II": 0.5,
            "XX": 1.0,
            "YY": -2.0,
            "XI": 3.0,
            "IZ": 4.0,
        }
    )

    groups = decompose_into_commuting_groups(hamiltonian)
    graph = hamiltonian.commutation_graph()

    for group in groups:
        for left, right in combinations(group.terms, 2):
            assert graph.has_edge(left, right)


def test_fully_commuting_hamiltonian_produces_one_group():
    hamiltonian = LCP({"II": 1, "ZI": 2, "IZ": 3, "ZZ": 4})

    groups = decompose_into_commuting_groups(hamiltonian)

    assert len(groups) == 1
    assert groups[0].terms == hamiltonian.terms


def test_empty_hamiltonian_produces_no_groups():
    assert decompose_into_commuting_groups(LCP({}, num_qubits=3)) == []


def test_rejects_non_lcp_input():
    with pytest.raises(TypeError, match="must be an LCP"):
        decompose_into_commuting_groups({"X": 1})
