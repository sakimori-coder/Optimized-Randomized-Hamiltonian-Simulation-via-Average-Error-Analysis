"""Grouping for the JW Hamiltonians from hamiltonians.chemistry and .syk.

Four-site X/Y terms are assumed to have the canonical Jordan--Wigner Z
strings between the first/second and third/fourth sites, as these models do.
"""

from operators import LCH, LCP


# The paper's independent four-site packets: A_even, B_even, A_odd, B_odd.
_PACKET = {
    word: packet
    for packet, words in enumerate((
        "XXXX YYXX YXYX YXXY",
        "YYYY XXYY XYXY XYYX",
        "YXXX XYXX XXYX XXXY",
        "XYYY YXYY YYXY YYYX",
    ))
    for word in words.split()
}


def build_fermionic_lch(hamiltonian):
    """Return H = sum_j H_j, keeping the original coefficients in each H_j."""
    return LCH(group_fermionic_hamiltonian(hamiltonian), num_qubits=hamiltonian.num_qubits)


def group_fermionic_hamiltonian(hamiltonian):
    """Return commuting, binary-independent groups as a list of LCPs."""
    packets = {}
    seeds = {}
    for pauli, coefficient in hamiltonian.terms.items():
        support = tuple(q for q, symbol in enumerate(pauli) if symbol in "XY")
        if len(support) == 4:
            word = "".join(pauli[q] for q in support)
            packets.setdefault((support, _PACKET[word]), []).append((pauli, coefficient))
        else:
            key = (support, pauli.count("Y") % 2)
            seeds.setdefault(key, []).append((pauli, coefficient))

    ordered_packets = [
        (dict(sorted(items, key=lambda item: (-abs(item[1]), item[0]))), support, packet)
        for (support, packet), items in packets.items()
    ]
    ordered_packets.sort(key=lambda item: (
        -round(sum(abs(value) for value in item[0].values()), 12),
        -len(item[0]), item[1], item[2],
    ))
    n = hamiltonian.num_qubits
    groups = _pack_disjoint_packets(ordered_packets, n)
    for items in seeds.values():
        groups.extend(_independent_groups(items, n))
    return [LCP(group, num_qubits=n) for group in groups]


def _pack_disjoint_packets(packets, n):
    """Try up to 16 groups of each size, largest first (paper's greedy rule)."""
    groups = []
    buckets = {}  # Number of terms -> open (group, used sites) pairs.
    for terms, support, _ in packets:
        for size in sorted(buckets, reverse=True):
            bucket = buckets[size]
            for i in reversed(range(max(0, len(bucket) - 16), len(bucket))):
                group, used = bucket[i]
                if used.isdisjoint(support):
                    # Swap with the last entry to retain the benchmark's candidate order.
                    bucket[i] = bucket[-1]
                    bucket.pop()
                    if not bucket:
                        del buckets[size]
                    break
            else:
                continue
            break
        else:
            group, used = {}, set()
            groups.append(group)
        group.update(terms)
        used.update(support)
        # Disjoint packets have at most one term per site, hence at most n terms.
        if len(used) + 4 <= n:
            buckets.setdefault(len(group), []).append((group, used))
    return groups


def _independent_groups(terms, n):
    """The seed already commutes; keep only rank-increasing additions."""
    groups = []
    open_groups = []
    for pauli, coefficient in sorted(terms):
        x = sum(1 << q for q, symbol in enumerate(pauli) if symbol in "XY")
        z = sum(1 << q for q, symbol in enumerate(pauli) if symbol in "YZ")
        vector = (x | (z << n)) if coefficient else 0
        # Inspect up to eight nonfull groups, as in the paper.
        for i in reversed(range(max(0, len(open_groups) - 8), len(open_groups))):
            group, basis = open_groups[i]
            if vector == 0 or _add_to_basis(basis, vector):
                if len(group) + 1 == n:
                    open_groups[i] = open_groups[-1]
                    open_groups.pop()
                break
        else:
            group, basis = {}, {}
            _add_to_basis(basis, vector)
            groups.append(group)
            if n > 1:
                open_groups.append((group, basis))
        group[pauli] = coefficient
    return groups


def _add_to_basis(basis, vector):
    """Binary Gaussian elimination; update the pivot rows only if rank grows."""
    while vector:
        pivot = vector.bit_length()
        if pivot not in basis:
            basis[pivot] = vector
            return True
        vector ^= basis[pivot]
    return False
