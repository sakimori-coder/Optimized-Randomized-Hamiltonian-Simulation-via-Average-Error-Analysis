import pytest

from grouping import (
    build_fermionic_lch,
    group_fermionic_hamiltonian,
)
from operators import LCP


def _pauli_strings_commute(left, right):
    """Equal-length Pauli strings commute iff local anticommutations are even."""
    anti_commuting_positions = sum(
        a != "I" and b != "I" and a != b
        for a, b in zip(left, right)
    )
    return anti_commuting_positions % 2 == 0


def _binary_rank(paulis: list[str]) -> int:
    basis: list[int] = []
    n = len(paulis[0]) if paulis else 0
    for pauli in paulis:
        vector = 0
        for q, symbol in enumerate(pauli):
            if symbol in "XY":
                vector |= 1 << q
            if symbol in "YZ":
                vector |= 1 << (n + q)
        for row in basis:
            vector = min(vector, vector ^ row)
        if vector:
            basis = [min(row, row ^ vector) for row in basis]
            basis.append(vector)
            basis.sort(reverse=True)
    return len(basis)


def test_fermionic_groups_are_commuting_independent_and_bounded() -> None:
    hamiltonian = LCP(
        {
            "XXXX": 1.0,
            "YYXX": 0.9,
            "YXYX": 0.8,
            "YXXY": 0.7,
            "YYYY": 0.6,
            "XXYY": 0.5,
            "ZZII": 0.4,
            "IZZI": 0.3,
        }
    )
    groups = group_fermionic_hamiltonian(hamiltonian)

    flattened = {}
    for group in groups:
        paulis = list(group.terms)
        assert len(paulis) <= hamiltonian.num_qubits
        assert all(
            _pauli_strings_commute(left, right)
            for index, left in enumerate(paulis)
            for right in paulis[index + 1 :]
        )
        active = [pauli for pauli in paulis if set(pauli) != {"I"}]
        assert _binary_rank(active) == len(active)
        flattened.update(group.terms)
    assert flattened == hamiltonian.terms


def test_all_four_site_packets_with_jordan_wigner_z_strings() -> None:
    # A_even, B_even, A_odd, B_odd on sites 0, 2, 4, 6.
    # Sites 1 and 5 carry JW Z strings; the middle gap at site 3 is I.
    packets = [
        {"XZXIXZX", "YZYIXZX", "YZXIYZX", "YZXIXZY"},
        {"YZYIYZY", "XZXIYZY", "XZYIXZY", "XZYIYZX"},
        {"YZXIXZX", "XZYIXZX", "XZXIYZX", "XZXIXZY"},
        {"XZYIYZY", "YZXIYZY", "YZYIXZY", "YZYIYZX"},
    ]
    target = LCP({
        pauli: (-1) ** i * (i + 1)
        for i, pauli in enumerate(sorted(set.union(*packets)))
    })

    groups = group_fermionic_hamiltonian(target)

    assert {frozenset(group.terms) for group in groups} == {
        frozenset(packet) for packet in packets
    }
    assert sum(len(group.terms) for group in groups) == 16
    assert {pauli: value for group in groups for pauli, value in group.terms.items()} == target.terms
    for group in groups:
        paulis = list(group.terms)
        assert _binary_rank(paulis) == 4
        assert all(
            _pauli_strings_commute(left, right)
            for index, left in enumerate(paulis)
            for right in paulis[index + 1 :]
        )


def test_packets_with_interleaved_disjoint_sites_are_packed_together() -> None:
    # These A_even packets have disjoint X/Y sites (0, 2, 4, 6) and
    # (1, 3, 5, 7), even though their JW Z strings cross the other packet.
    target = LCP(dict(zip([
        "XZXIXZXI", "YZYIXZXI", "YZXIYZXI", "YZXIXZYI",
        "IXZXIXZX", "IYZYIXZX", "IYZXIYZX", "IYZXIXZY",
    ], [8.0, -7.0, 6.0, -5.0, 4.0, -3.0, 2.0, -1.0])))

    groups = group_fermionic_hamiltonian(target)

    assert len(groups) == 1
    assert groups[0].terms == target.terms
    paulis = list(groups[0].terms)
    assert _binary_rank(paulis) == 8
    assert all(
        _pauli_strings_commute(left, right)
        for index, left in enumerate(paulis)
        for right in paulis[index + 1 :]
    )


def test_dependent_diagonal_seed_terms_are_separated_without_loss() -> None:
    # All three commute, but Z_0 Z_1 is dependent on Z_0 and Z_1.
    # Four qubits leave enough capacity: independence must force the split.
    target = LCP({"ZIII": 3.0, "IZII": -2.0, "ZZII": 1.0})

    groups = group_fermionic_hamiltonian(target)

    assert len(groups) == 2
    assert sorted(len(group.terms) for group in groups) == [1, 2]
    assert _binary_rank(list(target.terms)) == 2
    assert {pauli: value for group in groups for pauli, value in group.terms.items()} == target.terms
    for group in groups:
        paulis = list(group.terms)
        assert _binary_rank(paulis) == len(paulis)
        assert all(
            _pauli_strings_commute(left, right)
            for index, left in enumerate(paulis)
            for right in paulis[index + 1 :]
        )


def test_grouped_lch_keeps_hj_coefficients_and_uses_hs_probabilities() -> None:
    target = LCP({"ZI": 3.0, "IZ": -4.0, "XY": -12.0})
    grouped = build_fermionic_lch(
        target,
    )

    assert [operator.terms for operator in grouped.terms] == [
        {"ZI": 3.0, "IZ": -4.0},
        {"XY": -12.0},
    ]
    assert grouped.lcp.terms == target.terms
    assert grouped.hs_norms() == pytest.approx([5.0, 12.0])
    assert grouped.lambda_sum() == pytest.approx(17.0)
    assert grouped.sampling_probabilities() == pytest.approx([5.0 / 17, 12.0 / 17])
