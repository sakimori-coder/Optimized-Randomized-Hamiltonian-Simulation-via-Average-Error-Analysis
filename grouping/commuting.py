"""Algorithms for partitioning an LCP into commuting Pauli groups."""

from __future__ import annotations

from typing import Literal

import networkx as nx

from operators import LCP


GroupingMethod = Literal["greedy", "chemistry"]


def group_commuting_terms(
    hamiltonian: LCP,
    *,
    method: GroupingMethod,
    max_group_size: int | None = None,
) -> list[LCP]:
    """Partition an LCP with the selected deterministic grouping method."""
    if method == "greedy":
        return greedy_commuting_groups(
            hamiltonian,
            max_group_size=max_group_size,
        )
    if method == "chemistry":
        return chemistry_commuting_groups(
            hamiltonian,
            max_group_size=max_group_size,
        )
    raise ValueError("method must be 'greedy' or 'chemistry'")


def greedy_commuting_groups(
    hamiltonian: LCP,
    *,
    max_group_size: int | None = None,
) -> list[LCP]:
    """Partition an LCP into pairwise-commuting groups.

    Pauli strings are visited in lexicographic order.  Each term is placed in
    the largest existing compatible group, or starts a new group when no such
    group exists.  ``max_group_size=None`` leaves group size uncapped.  The
    result is deterministic but is not guaranteed to use the minimum possible
    number of groups.
    """
    if not isinstance(hamiltonian, LCP):
        raise TypeError("hamiltonian must be an LCP")
    group_size_limit = _group_size_limit(max_group_size, len(hamiltonian.terms))

    grouped_terms: list[dict[str, float]] = []
    terms = hamiltonian.terms
    for pauli in sorted(terms):
        coefficient = terms[pauli]
        compatible = [
            index
            for index, group in enumerate(grouped_terms)
            if len(group) < group_size_limit
            and all(
                LCP._pauli_strings_commute(pauli, other)
                for other in group
            )
        ]
        if compatible:
            selected = max(
                compatible,
                key=lambda index: (len(grouped_terms[index]), -index),
            )
            grouped_terms[selected][pauli] = coefficient
        else:
            grouped_terms.append({pauli: coefficient})

    return [
        LCP(group, num_qubits=hamiltonian.num_qubits)
        for group in grouped_terms
    ]


def chemistry_commuting_groups(
    hamiltonian: LCP,
    *,
    max_group_size: int | None = None,
) -> list[LCP]:
    r"""Group a Jordan--Wigner molecular Hamiltonian by orbital support.

    For a Pauli string ``P``, define its non-diagonal support as
    ``S(P) = {q: P[q] is X or Y}``.  Strings with equal
    ``(S(P), number_of_Y(P) mod 2)`` commute, so they first form indivisible
    seed fragments.  This naturally collects the Z-only sector, hopping and
    controlled-hopping terms with the same orbital pair, and double
    excitations with the same four-orbital support.

    The Z-only sector is kept separate.  Four-orbital seed fragments with
    disjoint supports are packed first as a Baranyai-type heuristic.  The
    remaining fragments are then processed in descending coefficient
    one-norm and inserted into the first fully commuting compatible group.
    This last step is a deterministic sorted-insertion heuristic, not an
    optimal graph coloring.

    The Z-only sector always remains one group.  ``max_group_size`` is a
    project-specific practical cap for the non-diagonal seed fragments; it
    may therefore be exceeded by the Z-only group.  ``None`` leaves all
    chemistry-derived groups uncapped.
    """
    if not isinstance(hamiltonian, LCP):
        raise TypeError("hamiltonian must be an LCP")

    terms = hamiltonian.terms
    group_size_limit = _group_size_limit(max_group_size, len(terms))
    if not terms:
        return []

    ordered_paulis = sorted(
        terms,
        key=lambda pauli: (-abs(terms[pauli]), pauli),
    )
    seed_terms: dict[tuple[tuple[int, ...], int], list[tuple[str, float]]] = {}
    for pauli in ordered_paulis:
        support = tuple(
            qubit
            for qubit, symbol in enumerate(pauli)
            if symbol in "XY"
        )
        key = (support, pauli.count("Y") % 2)
        seed_terms.setdefault(key, []).append((pauli, terms[pauli]))

    z_only_items = seed_terms.pop(((), 0), [])
    z_only_groups = [dict(z_only_items)] if z_only_items else []

    four_orbital_fragments: list[tuple[dict[str, float], frozenset[int]]] = []
    other_fragments: list[dict[str, float]] = []
    for (support, _), items in sorted(seed_terms.items()):
        for fragment in _chunk_items(items, group_size_limit):
            if len(support) == 4:
                four_orbital_fragments.append(
                    (fragment, frozenset(support))
                )
            else:
                other_fragments.append(fragment)

    baranyai_fragments = _pack_disjoint_four_orbital_fragments(
        four_orbital_fragments,
        group_size_limit,
    )
    merged_groups = _merge_fully_commuting_fragments(
        [*baranyai_fragments, *other_fragments],
        group_size_limit,
    )

    return [
        LCP(group, num_qubits=hamiltonian.num_qubits)
        for group in [*z_only_groups, *merged_groups]
    ]


def _group_size_limit(
    max_group_size: int | None,
    num_terms: int,
) -> int:
    if max_group_size is None:
        return max(1, num_terms)
    if not isinstance(max_group_size, int) or isinstance(max_group_size, bool):
        raise TypeError("max_group_size must be an integer or None")
    if max_group_size <= 0:
        raise ValueError("max_group_size must be positive")
    return max_group_size


def _chunk_items(
    items: list[tuple[str, float]],
    group_size_limit: int,
) -> list[dict[str, float]]:
    return [
        dict(items[start : start + group_size_limit])
        for start in range(0, len(items), group_size_limit)
    ]


def _fragment_sort_key(
    fragment: dict[str, float],
) -> tuple[float, int, tuple[str, ...]]:
    return (
        -sum(abs(coefficient) for coefficient in fragment.values()),
        -len(fragment),
        tuple(sorted(fragment)),
    )


def _fragments_commute(
    left: dict[str, float],
    right: dict[str, float],
) -> bool:
    return all(
        LCP._pauli_strings_commute(left_pauli, right_pauli)
        for left_pauli in left
        for right_pauli in right
    )


def _pack_disjoint_four_orbital_fragments(
    fragments: list[tuple[dict[str, float], frozenset[int]]],
    group_size_limit: int,
) -> list[dict[str, float]]:
    packed: list[tuple[dict[str, float], set[int]]] = []
    for fragment, support in sorted(
        fragments,
        key=lambda item: (_fragment_sort_key(item[0]), tuple(item[1])),
    ):
        for group, used_orbitals in packed:
            if (
                len(group) + len(fragment) <= group_size_limit
                and used_orbitals.isdisjoint(support)
                and _fragments_commute(group, fragment)
            ):
                group.update(fragment)
                used_orbitals.update(support)
                break
        else:
            packed.append((dict(fragment), set(support)))
    return [group for group, _ in packed]


def _merge_fully_commuting_fragments(
    fragments: list[dict[str, float]],
    group_size_limit: int,
) -> list[dict[str, float]]:
    groups: list[dict[str, float]] = []
    for fragment in sorted(fragments, key=_fragment_sort_key):
        for group in groups:
            if (
                len(group) + len(fragment) <= group_size_limit
                and _fragments_commute(group, fragment)
            ):
                group.update(fragment)
                break
        else:
            groups.append(dict(fragment))
    return groups


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
