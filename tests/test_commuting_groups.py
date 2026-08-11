from itertools import combinations

import numpy as np
import pytest

from grouping.commuting import (
    chemistry_commuting_groups,
    decompose_into_commuting_groups,
    greedy_commuting_groups,
    group_commuting_terms,
)
from operators import LCP


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


def test_greedy_groups_are_commuting_and_obey_size_limit() -> None:
    hamiltonian = LCP(
        {
            "II": 0.5,
            "XX": 1.0,
            "YY": -2.0,
            "XI": 3.0,
            "IZ": 4.0,
        }
    )

    groups = greedy_commuting_groups(hamiltonian, max_group_size=2)

    assert {
        pauli
        for group in groups
        for pauli in group.terms
    } == set(hamiltonian.terms)
    assert all(len(group.terms) <= 2 for group in groups)
    assert all(
        LCP._pauli_strings_commute(left, right)
        for group in groups
        for left, right in combinations(group.terms, 2)
    )


def test_chemistry_grouping_keeps_z_sector_and_orbital_pair_seeds() -> None:
    hamiltonian = LCP(
        {
            "ZZ": 3.0,
            "XX": 2.0,
            "YY": -1.0,
        }
    )

    groups = chemistry_commuting_groups(hamiltonian)

    assert [group.terms for group in groups] == [
        {"ZZ": 3.0},
        {"XX": 2.0, "YY": -1.0},
    ]


def test_chemistry_grouping_does_not_split_z_sector_at_size_limit() -> None:
    hamiltonian = LCP(
        {
            "ZII": 4.0,
            "IZI": 3.0,
            "IIZ": 2.0,
            "XXI": 1.0,
            "YYI": -0.5,
        }
    )

    groups = chemistry_commuting_groups(hamiltonian, max_group_size=2)

    assert groups[0].terms == {
        "ZII": 4.0,
        "IZI": 3.0,
        "IIZ": 2.0,
    }
    assert len(groups[0].terms) > 2
    assert all(len(group.terms) <= 2 for group in groups[1:])


def test_chemistry_grouping_separates_y_parity_sectors() -> None:
    hamiltonian = LCP(
        {
            "XX": 4.0,
            "YY": 3.0,
            "XY": 2.0,
            "YX": 1.0,
        }
    )

    groups = chemistry_commuting_groups(hamiltonian)

    assert [group.terms for group in groups] == [
        {"XX": 4.0, "YY": 3.0},
        {"XY": 2.0, "YX": 1.0},
    ]


def test_chemistry_grouping_packs_disjoint_four_orbital_supports() -> None:
    hamiltonian = LCP(
        {
            "XXXXIIII": 4.0,
            "YYYYIIII": 3.0,
            "IIIIXXXX": 2.0,
            "IIIIYYYY": 1.0,
        }
    )

    groups = chemistry_commuting_groups(hamiltonian)

    assert len(groups) == 1
    assert groups[0].terms == hamiltonian.terms


def test_chemistry_grouping_partitions_terms_and_obeys_size_limit() -> None:
    hamiltonian = LCP(
        {
            "ZZII": 5.0,
            "IZZI": 4.0,
            "XXII": 3.0,
            "YYII": 2.0,
            "IIXX": 1.0,
            "IIYY": 0.5,
        }
    )

    groups = chemistry_commuting_groups(
        hamiltonian,
        max_group_size=2,
    )
    grouped_paulis = [pauli for group in groups for pauli in group.terms]

    assert set(grouped_paulis) == set(hamiltonian.terms)
    assert len(grouped_paulis) == len(set(grouped_paulis))
    assert all(len(group.terms) <= 2 for group in groups)
    assert all(
        LCP._pauli_strings_commute(left, right)
        for group in groups
        for left, right in combinations(group.terms, 2)
    )


def test_grouping_dispatch_selects_chemistry_method() -> None:
    hamiltonian = LCP({"ZZ": 3.0, "XX": 2.0, "YY": -1.0})

    direct = chemistry_commuting_groups(hamiltonian)
    dispatched = group_commuting_terms(
        hamiltonian,
        method="chemistry",
    )

    assert [group.terms for group in dispatched] == [
        group.terms for group in direct
    ]
