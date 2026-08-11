r"""Exact Pauli-rotation depth for a commuting Hamiltonian.

Cost model
----------
Clifford basis changes are free.  In one rotation layer, a linearly
independent commuting set of Pauli operators can be Clifford-conjugated to
different single-qubit Z operators and rotated in parallel.  Consequently,
the minimum depth is the minimum number of F_2-linearly-independent sets
into which the non-trivial Pauli terms can be partitioned.

By Edmonds' matroid partition theorem, this depth is

    max over non-empty A of ceil(|A| / rank_F2(A)).

The implementation is exact.  It enumerates the flats of the represented
binary linear matroid; this is generally much smaller than enumerating all
subsets, but still has exponential worst-case complexity.
"""

from __future__ import annotations

from collections.abc import Iterable

from operators import LCP


def _pauli_to_binary_vector(pauli_string: str) -> int:
    """Encode a Pauli string as the integer bit vector (x | z)."""
    num_qubits = len(pauli_string)
    vector = 0
    for qubit, symbol in enumerate(pauli_string):
        if symbol in "XY":
            vector |= 1 << qubit
        if symbol in "YZ":
            vector |= 1 << (num_qubits + qubit)
    return vector


def _reduce_with_basis(vector: int, basis: tuple[int, ...]) -> int:
    """Reduce a binary vector by a canonical XOR basis."""
    for row in basis:
        vector = min(vector, vector ^ row)
    return vector


def _extend_basis(basis: tuple[int, ...], vector: int) -> tuple[int, ...]:
    """Return the canonical XOR basis after inserting one vector."""
    vector = _reduce_with_basis(vector, basis)
    if vector == 0:
        return basis

    rows = [min(row, row ^ vector) for row in basis]
    rows.append(vector)
    rows.sort(reverse=True)
    return tuple(rows)


def _closure_mask(vectors: list[int], basis: tuple[int, ...]) -> int:
    """Return the set of input vectors contained in the span of ``basis``."""
    mask = 0
    for index, vector in enumerate(vectors):
        if _reduce_with_basis(vector, basis) == 0:
            mask |= 1 << index
    return mask


def _maximum_rank_density_ceiling(vectors: Iterable[int]) -> int:
    """Compute max ceil(|A| / rank(A)) exactly over all non-empty subsets."""
    vector_list = list(vectors)
    if not vector_list:
        return 0

    # The objective cannot decrease when A is replaced by its closure, since
    # closure preserves rank and can only add elements.  It is therefore
    # sufficient to enumerate distinct flats instead of all 2**m subsets.
    empty_basis: tuple[int, ...] = ()
    stack = [(empty_basis, 0)]
    visited_closures = {0}
    minimum_depth = 0

    while stack:
        basis, closure = stack.pop()
        rank = len(basis)
        if rank:
            cardinality = closure.bit_count()
            depth = (cardinality + rank - 1) // rank
            minimum_depth = max(minimum_depth, depth)

        outside = ((1 << len(vector_list)) - 1) ^ closure
        while outside:
            element_bit = outside & -outside
            element_index = element_bit.bit_length() - 1
            outside ^= element_bit

            extended_basis = _extend_basis(basis, vector_list[element_index])
            extended_closure = _closure_mask(vector_list, extended_basis)
            if extended_closure not in visited_closures:
                visited_closures.add(extended_closure)
                stack.append((extended_basis, extended_closure))

    return minimum_depth


def minimum_pauli_rotation_depth(hamiltonian: LCP) -> int:
    """Return the exact minimum Pauli-rotation depth of a commuting LCP.

    Coefficients must be real because ``hamiltonian`` represents a Hermitian
    operator.  Identity terms contribute only a global phase and exactly-zero
    coefficient terms require no rotation, so both are omitted from the cost.

    Args:
        hamiltonian: Pairwise-commuting Hermitian LCP Hamiltonian.

    Returns:
        Exact minimum depth under the cost model described in this module.

    Raises:
        TypeError: If ``hamiltonian`` is not an LCP.
        ValueError: If a coefficient is non-real or active Pauli terms do not
            commute pairwise.
    """
    if not isinstance(hamiltonian, LCP):
        raise TypeError("hamiltonian must be an LCP")

    terms = hamiltonian.terms
    if any(coefficient.imag != 0 for coefficient in terms.values()):
        raise ValueError("Hamiltonian coefficients must be real")

    active_paulis = [
        pauli_string
        for pauli_string, coefficient in terms.items()
        if coefficient != 0 and any(symbol != "I" for symbol in pauli_string)
    ]

    graph = hamiltonian.commutation_graph()
    for index, left in enumerate(active_paulis):
        for right in active_paulis[index + 1 :]:
            if not graph.has_edge(left, right):
                raise ValueError("Hamiltonian terms must commute pairwise")

    binary_vectors = [_pauli_to_binary_vector(pauli) for pauli in active_paulis]
    return _maximum_rank_density_ceiling(binary_vectors)
