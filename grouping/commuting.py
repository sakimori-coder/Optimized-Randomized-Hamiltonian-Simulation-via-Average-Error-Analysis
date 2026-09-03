r"""The fixed chemistry-depth-one grouping used by this project."""

from __future__ import annotations

import math

from operators import LCH, LCP


_PACKING_CANDIDATES_PER_SIZE = 16
_INDEPENDENCE_CANDIDATES = 8


def build_chemistry_depth1_frobenius_lch(
    hamiltonian: LCP,
    *,
    max_group_size: int | None = None,
) -> LCH:
    r"""Build the only grouped-qDRIFT decomposition supported here.

    Each depth-one group ``G_g = sum_P a_P P`` gets Frobenius weight
    ``h_g = sqrt(sum_P a_P**2)`` and sampled operator ``G_g / h_g``.
    """
    groups = chemistry_depth_one_groups(
        hamiltonian,
        max_group_size=max_group_size,
    )
    terms: list[tuple[float, LCP]] = []
    for group in groups:
        weight = math.sqrt(
            math.fsum(value * value for value in group.terms.values())
        )
        if weight != 0.0:
            terms.append((weight, group * (1.0 / weight)))
    return LCH(terms, num_qubits=hamiltonian.num_qubits)


def chemistry_depth_one_groups(
    hamiltonian: LCP,
    *,
    max_group_size: int | None = None,
) -> list[LCP]:
    r"""Partition a JW Hamiltonian into commuting depth-one groups in O(nL).

    Canonical four-orbital strings are split into independent A/B packets and
    packed only across disjoint orbital supports.  Remaining commuting seeds
    are partitioned by bounded binary-basis insertions.  Every active group is
    F_2-linearly independent and therefore has Pauli-rotation depth one.
    """
    if not isinstance(hamiltonian, LCP):
        raise TypeError("hamiltonian must be an LCP")
    terms = hamiltonian.terms
    if not terms:
        return []
    group_size_limit = _group_size_limit(max_group_size, len(terms))

    four_orbital_terms: dict[
        tuple[tuple[int, ...], int, int],
        list[tuple[str, float]],
    ] = {}
    remaining_seed_terms: dict[
        tuple[tuple[int, ...], int],
        list[tuple[str, float]],
    ] = {}
    for pauli, coefficient in terms.items():
        support, y_parity, packet = _pauli_key(pauli)
        if packet is None:
            remaining_seed_terms.setdefault(
                (support, y_parity), []
            ).append((pauli, coefficient))
        else:
            four_orbital_terms.setdefault(
                (support, y_parity, packet), []
            ).append((pauli, coefficient))

    fragments: list[
        tuple[dict[str, float], tuple[int, ...], int, int, int]
    ] = []
    for (support, y_parity, packet), items in four_orbital_terms.items():
        ordered = sorted(items, key=lambda item: (-abs(item[1]), item[0]))
        for chunk_index, fragment in enumerate(
            _chunk_items(ordered, group_size_limit)
        ):
            fragments.append(
                (fragment, support, y_parity, packet, chunk_index)
            )
    fragments.sort(
        key=lambda item: (
            -round(sum(abs(value) for value in item[0].values()), 12),
            -len(item[0]),
            item[1],
            item[2],
            item[3],
            item[4],
        )
    )
    grouped_terms = _pack_disjoint_fragments(
        [(fragment, support) for fragment, support, _, _, _ in fragments],
        num_qubits=hamiltonian.num_qubits,
        group_size_limit=group_size_limit,
    )
    for items in remaining_seed_terms.values():
        grouped_terms.extend(
            _independent_seed_groups(
                items,
                group_size_limit=group_size_limit,
            )
        )
    return [
        LCP(group, num_qubits=hamiltonian.num_qubits)
        for group in grouped_terms
    ]


def _pauli_key(pauli: str) -> tuple[tuple[int, ...], int, int | None]:
    support: list[int] = []
    active_symbols: list[str] = []
    z_mask = 0
    number_of_y = 0
    for qubit, symbol in enumerate(pauli):
        if symbol in "XY":
            support.append(qubit)
            active_symbols.append(symbol)
            number_of_y += symbol == "Y"
        elif symbol == "Z":
            z_mask |= 1 << qubit
    support_tuple = tuple(support)
    y_parity = number_of_y % 2
    if len(support) != 4:
        return support_tuple, y_parity, None

    first, second, third, fourth = support
    expected_z_mask = (
        _integer_range_mask(first + 1, second)
        | _integer_range_mask(third + 1, fourth)
    )
    if z_mask != expected_z_mask:
        return support_tuple, y_parity, None
    if y_parity == 1:
        packet = 0 if number_of_y == 1 else 1
    elif number_of_y == 0:
        packet = 0
    elif number_of_y == 4:
        packet = 1
    else:
        packet = 0 if active_symbols[0] == "Y" else 1
    return support_tuple, y_parity, packet


def _integer_range_mask(start: int, stop: int) -> int:
    return ((1 << stop) - 1) ^ ((1 << start) - 1)


def _pack_disjoint_fragments(
    fragments: list[tuple[dict[str, float], tuple[int, ...]]],
    *,
    num_qubits: int,
    group_size_limit: int,
) -> list[dict[str, float]]:
    if not fragments:
        return []
    maximum_open_size = min(group_size_limit - 1, num_qubits)
    groups: list[dict[str, float]] = []
    used_orbitals: list[set[int]] = []
    bucket_positions: list[int] = []
    size_buckets: list[list[int]] = [
        [] for _ in range(maximum_open_size + 1)
    ]
    nonempty_size_mask = 0

    def add_to_bucket(group_index: int) -> None:
        nonlocal nonempty_size_mask
        size = len(groups[group_index])
        bucket_positions[group_index] = len(size_buckets[size])
        size_buckets[size].append(group_index)
        nonempty_size_mask |= 1 << size

    def remove_from_bucket(group_index: int, size: int) -> None:
        nonlocal nonempty_size_mask
        bucket = size_buckets[size]
        position = bucket_positions[group_index]
        last_index = bucket.pop()
        if position < len(bucket):
            bucket[position] = last_index
            bucket_positions[last_index] = position
        bucket_positions[group_index] = -1
        if not bucket:
            nonempty_size_mask &= ~(1 << size)

    for fragment, support in fragments:
        fragment_size = len(fragment)
        largest_size = min(
            maximum_open_size,
            group_size_limit - fragment_size,
        )
        selected: int | None = None
        eligible_sizes = nonempty_size_mask & (
            (1 << (largest_size + 1)) - 2
        )
        while eligible_sizes:
            size = eligible_sizes.bit_length() - 1
            eligible_sizes ^= 1 << size
            bucket = size_buckets[size]
            first_position = max(
                -1,
                len(bucket) - _PACKING_CANDIDATES_PER_SIZE - 1,
            )
            for position in range(len(bucket) - 1, first_position, -1):
                group_index = bucket[position]
                if used_orbitals[group_index].isdisjoint(support):
                    selected = group_index
                    break
            if selected is not None:
                break

        if selected is None:
            group_index = len(groups)
            groups.append(dict(fragment))
            used_orbitals.append(set(support))
            bucket_positions.append(-1)
            if (
                fragment_size < group_size_limit
                and len(support) + 4 <= num_qubits
            ):
                add_to_bucket(group_index)
            continue

        old_size = len(groups[selected])
        remove_from_bucket(selected, old_size)
        groups[selected].update(fragment)
        used_orbitals[selected].update(support)
        if (
            len(groups[selected]) < group_size_limit
            and len(used_orbitals[selected]) + 4 <= num_qubits
        ):
            add_to_bucket(selected)
    return groups


def _independent_seed_groups(
    terms: list[tuple[str, float]],
    *,
    group_size_limit: int,
) -> list[dict[str, float]]:
    groups: list[dict[str, float]] = []
    bases: list[tuple[int, ...]] = []
    open_group_indices: list[int] = []
    open_positions: list[int] = []

    def remove_open_group(group_index: int) -> None:
        position = open_positions[group_index]
        last_index = open_group_indices.pop()
        if position < len(open_group_indices):
            open_group_indices[position] = last_index
            open_positions[last_index] = position
        open_positions[group_index] = -1

    for pauli, coefficient in _radix_sort_pauli_items(terms):
        vector = (
            0
            if coefficient == 0.0 or all(symbol == "I" for symbol in pauli)
            else _pauli_to_binary_vector(pauli)
        )
        selected: int | None = None
        extended_basis: tuple[int, ...] | None = None
        first_position = max(
            -1,
            len(open_group_indices) - _INDEPENDENCE_CANDIDATES - 1,
        )
        for position in range(len(open_group_indices) - 1, first_position, -1):
            group_index = open_group_indices[position]
            candidate_basis = (
                bases[group_index]
                if vector == 0
                else _extend_binary_basis(bases[group_index], vector)
            )
            if vector == 0 or len(candidate_basis) > len(bases[group_index]):
                selected = group_index
                extended_basis = candidate_basis
                break

        if selected is None:
            group_index = len(groups)
            groups.append({pauli: coefficient})
            bases.append(() if vector == 0 else (vector,))
            open_positions.append(-1)
            if group_size_limit > 1:
                open_positions[group_index] = len(open_group_indices)
                open_group_indices.append(group_index)
            continue

        groups[selected][pauli] = coefficient
        assert extended_basis is not None
        bases[selected] = extended_basis
        if len(groups[selected]) >= group_size_limit:
            remove_open_group(selected)
    return groups


def _pauli_to_binary_vector(pauli: str) -> int:
    num_qubits = len(pauli)
    vector = 0
    for qubit, symbol in enumerate(pauli):
        if symbol in "XY":
            vector |= 1 << qubit
        if symbol in "YZ":
            vector |= 1 << (num_qubits + qubit)
    return vector


def _extend_binary_basis(
    basis: tuple[int, ...],
    vector: int,
) -> tuple[int, ...]:
    for row in basis:
        vector = min(vector, vector ^ row)
    if vector == 0:
        return basis
    rows = [min(row, row ^ vector) for row in basis]
    rows.append(vector)
    rows.sort(reverse=True)
    return tuple(rows)


def _group_size_limit(max_group_size: int | None, num_terms: int) -> int:
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


def _radix_sort_pauli_items(
    items: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    if len(items) < 2:
        return list(items)
    ordered = list(items)
    alphabet = "IXYZ"
    for qubit in range(len(ordered[0][0]) - 1, -1, -1):
        buckets: dict[str, list[tuple[str, float]]] = {
            symbol: [] for symbol in alphabet
        }
        for item in ordered:
            buckets[item[0][qubit]].append(item)
        ordered = [item for symbol in alphabet for item in buckets[symbol]]
    return ordered


__all__ = [
    "build_chemistry_depth1_frobenius_lch",
    "chemistry_depth_one_groups",
]
