import math

import pytest

from grouping import (
    build_chemistry_depth1_frobenius_lch,
    chemistry_depth_one_groups,
)
from operators import LCP


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


def test_depth1_groups_are_commuting_independent_and_bounded() -> None:
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
    groups = chemistry_depth_one_groups(hamiltonian, max_group_size=4)

    flattened = {}
    for group in groups:
        paulis = list(group.terms)
        assert len(paulis) <= 4
        assert all(
            LCP.pauli_strings_commute(left, right)
            for index, left in enumerate(paulis)
            for right in paulis[index + 1 :]
        )
        active = [pauli for pauli in paulis if set(pauli) != {"I"}]
        assert _binary_rank(active) == len(active)
        flattened.update(group.terms)
    assert flattened == hamiltonian.terms


def test_grouped_lch_uses_only_frobenius_weights() -> None:
    target = LCP({"XI": 3.0, "IZ": 4.0})
    grouped = build_chemistry_depth1_frobenius_lch(
        target,
        max_group_size=2,
    )

    assert grouped.lcp.terms == pytest.approx(target.terms)
    for weight, operator in grouped.terms:
        assert weight == pytest.approx(
            math.sqrt(sum((weight * value) ** 2 for value in operator.terms.values()))
        )

