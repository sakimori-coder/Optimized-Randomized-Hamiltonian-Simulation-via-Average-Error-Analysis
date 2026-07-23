import pytest

from brute_force_qdrift_comparison import compare_brute_force_qdrift_costs
from lch import LCH


def test_single_commuting_group_is_exact_after_grouping() -> None:
    hamiltonian = LCH(
        [
            (1.0, "ZII", 1.0),
            (1.0, "IZI", 1.0),
            (1.0, "IIZ", 1.0),
            (-1.0, "ZZZ", 1.0),
        ]
    )

    comparison = compare_brute_force_qdrift_costs(
        hamiltonian,
        time=1.0,
        epsilon=0.1,
    )

    assert len(comparison.groups) == 1
    assert comparison.grouped.error.lambda_sum == pytest.approx(2.0)
    assert comparison.grouped.error.normalized_variance_constant == pytest.approx(0.0)
    assert comparison.grouped.error.required_steps == 1
    assert comparison.grouped.expected_depth_per_step == pytest.approx(2.0)
    assert comparison.grouped.expected_total_depth == pytest.approx(2.0)


def test_noncommuting_single_paulis_give_same_two_decompositions() -> None:
    comparison = compare_brute_force_qdrift_costs(
        LCH([(1.0, "X", 1.0), (1.0, "Z", 1.0)]),
        time=1.0,
        epsilon=0.1,
    )

    assert len(comparison.groups) == 2
    assert comparison.grouped.error.variance_constant == pytest.approx(
        comparison.pauli.error.variance_constant
    )
    assert comparison.grouped.expected_total_depth == pytest.approx(
        comparison.pauli.expected_total_depth
    )
