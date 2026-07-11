"""Decompose an LCP Hamiltonian into commuting groups."""

from __future__ import annotations

import networkx as nx

from lcp import LCP


def decompose_into_commuting_groups(hamiltonian: LCP) -> list[LCP]:
    """Partition a Hamiltonian into pairwise-commuting LCP Hamiltonians.

    Maximal cliques of the commutation graph are enumerated with NetworkX's
    Bron--Kerbosch-based ``find_cliques`` implementation.  A greedy clique
    cover is then constructed: at each step, the clique containing the most
    as-yet-unassigned Pauli strings is selected.

    Every Pauli term is assigned to exactly one returned group, so the sum of
    the groups equals ``hamiltonian``.  The number of groups is not guaranteed
    to be minimal.

    Args:
        hamiltonian: LCP Hamiltonian to decompose.

    Returns:
        A list of LCP Hamiltonians.  All Pauli strings within each element
        commute pairwise.  An empty Hamiltonian produces an empty list.
    """
    if not isinstance(hamiltonian, LCP):
        raise TypeError("hamiltonian must be an LCP")

    graph = hamiltonian.commutation_graph()
    if graph.number_of_nodes() == 0:
        return []

    maximal_cliques = [frozenset(clique) for clique in nx.find_cliques(graph)]
    unassigned = set(graph.nodes)
    groups: list[LCP] = []
    terms = hamiltonian.terms

    while unassigned:
        # The secondary key makes ties deterministic with respect to the
        # order returned by find_cliques.
        _, selected_clique = max(
            enumerate(maximal_cliques),
            key=lambda item: (
                len(item[1] & unassigned),
                -item[0],
            ),
        )

        selected_paulis = [
            pauli_string
            for pauli_string in terms
            if pauli_string in selected_clique and pauli_string in unassigned
        ]
        group_terms = {
            pauli_string: terms[pauli_string]
            for pauli_string in selected_paulis
        }
        groups.append(LCP(group_terms, num_qubits=hamiltonian.num_qubits))
        unassigned.difference_update(selected_paulis)

    return groups
