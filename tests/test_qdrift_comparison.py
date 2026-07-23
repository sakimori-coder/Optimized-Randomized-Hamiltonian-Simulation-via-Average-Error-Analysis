import pytest

from lcp import LCP
from qdrift_comparison import compare_qdrift_implementations


def test_grouping_can_reduce_total_expected_rotation_depth() -> None:
    # The four commuting Paulis have one even dependency.  With the chosen
    # signs, no joint eigenstate can align all four contributions, giving
    # ||H|| = 2 although the coefficient one-norm is 4.
    hamiltonian = LCP(
        {
            "ZII": 1.0,
            "IZI": 1.0,
            "IIZ": 1.0,
            "ZZZ": -1.0,
        }
    )

    comparison = compare_qdrift_implementations(
        hamiltonian,
        time=1.0,
        epsilon=1.0,
    )

    assert comparison.pauli.lambda_sum == pytest.approx(4.0)
    assert comparison.pauli_strings == ("ZII", "IZI", "IIZ", "ZZZ")
    assert comparison.pauli.steps == 32
    assert comparison.pauli.expected_total_depth == pytest.approx(32.0)

    assert comparison.grouped.num_sampling_terms == 1
    assert comparison.grouped.lambda_sum == pytest.approx(2.0)
    # Iterative sparse singular-value calculations can overestimate h_g by a
    # few ulps and therefore push an exact integer ceiling up by one.
    assert comparison.grouped.steps in {8, 9}
    assert comparison.grouped.expected_depth_per_step == pytest.approx(2.0)
    assert comparison.grouped.expected_total_depth == pytest.approx(
        2.0 * comparison.grouped.steps
    )


def test_empty_hamiltonian_has_zero_cost() -> None:
    comparison = compare_qdrift_implementations(
        LCP({}, num_qubits=2),
        time=1.0,
        epsilon=0.1,
    )

    assert comparison.pauli.steps == 0
    assert comparison.grouped.steps == 0
    assert comparison.pauli.expected_total_depth == 0.0
    assert comparison.grouped.expected_total_depth == 0.0
