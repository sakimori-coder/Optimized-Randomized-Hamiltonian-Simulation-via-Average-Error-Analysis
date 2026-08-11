import pytest

from diamond_distance import (
    qdrift_diamond_distance_bound,
    qdrift_diamond_distance_bounds,
)
from operators import LCH, LCP


def _pauli(pauli: str) -> LCP:
    return LCP({pauli: 1.0})


@pytest.mark.parametrize(
    ("number_of_steps", "expected_second_order", "expected_remainder"),
    [
        (1, 0.02, (4.0 / 3.0) * 0.2**3),
        (4, 0.00125, (4.0 / 3.0) * 0.2**3 / 64.0),
    ],
)
def test_diamond_distance_bound_for_x_z_decomposition(
    number_of_steps: int,
    expected_second_order: float,
    expected_remainder: float,
) -> None:
    target = LCP({"X": 1.0, "Z": 1.0})
    decomposition = LCH([(1.0, _pauli("X")), (1.0, _pauli("Z"))])

    bound = qdrift_diamond_distance_bound(
        target,
        0.1,
        number_of_steps,
        decomposition,
        variance_method="exact",
    )

    assert bound.target_hamiltonian is target
    assert bound.qdrift_decomposition is decomposition
    assert bound.total_time == 0.1
    assert bound.number_of_steps == number_of_steps
    assert bound.step_time == pytest.approx(0.1 / number_of_steps)
    assert bound.lambda_sum == pytest.approx(2.0)
    assert bound.normalized_variance_bound == pytest.approx(0.5)
    assert bound.variance_constant == pytest.approx(2.0)
    assert bound.sampling_probabilities == pytest.approx([0.5, 0.5])
    assert bound.step_second_order_bound == pytest.approx(expected_second_order)
    assert bound.step_taylor_remainder_bound == pytest.approx(expected_remainder)
    assert bound.step_diamond_distance_upper_bound == pytest.approx(
        expected_second_order + expected_remainder
    )


def test_multiple_decompositions_preserve_input_order() -> None:
    target = LCP({"ZI": 1.0, "IZ": 1.0, "XX": 1.0})
    pauli = LCH(
        [
            (1.0, _pauli("ZI")),
            (1.0, _pauli("IZ")),
            (1.0, _pauli("XX")),
        ]
    )
    grouped = LCH(
        [
            (2.0, LCP({"ZI": 0.5, "IZ": 0.5})),
            (1.0, _pauli("XX")),
        ]
    )

    grouped_bound, pauli_bound = qdrift_diamond_distance_bounds(
        target,
        0.1,
        2,
        [grouped, pauli],
        variance_method="exact",
    )

    assert grouped_bound.qdrift_decomposition is grouped
    assert pauli_bound.qdrift_decomposition is pauli
    assert grouped_bound.step_time == pytest.approx(0.05)
    assert pauli_bound.step_time == pytest.approx(0.05)
    assert grouped_bound.normalized_variance_bound == pytest.approx(4.0 / 9.0)
    assert pauli_bound.normalized_variance_bound == pytest.approx(8.0 / 9.0)
    assert grouped_bound.variance_constant == pytest.approx(4.0)
    assert pauli_bound.variance_constant == pytest.approx(8.0)
    assert grouped_bound.sampling_probabilities == pytest.approx([2 / 3, 1 / 3])
    assert pauli_bound.sampling_probabilities == pytest.approx([1 / 3] * 3)
    assert grouped_bound.step_diamond_distance_upper_bound == pytest.approx(
        0.0145
    )
    assert pauli_bound.step_diamond_distance_upper_bound == pytest.approx(
        0.0245
    )


def test_diamond_distance_bound_is_zero_for_zero_variance() -> None:
    target = LCP({"Y": 2.5})
    bound = qdrift_diamond_distance_bound(
        target,
        100.0,
        1,
        LCH([(2.5, _pauli("Y"))]),
        variance_method="exact",
    )

    assert bound.normalized_variance_bound == 0.0
    assert bound.variance_constant == 0.0
    assert bound.step_second_order_bound == 0.0
    assert bound.step_taylor_remainder_bound == 0.0
    assert bound.step_diamond_distance_upper_bound == 0.0


def test_diamond_remainder_survives_second_order_underflow() -> None:
    def tiny_positive_variance(_decomposition) -> float:
        return 2e-300

    target = LCP({"X": 1.0, "Z": 1e-300})
    bound = qdrift_diamond_distance_bound(
        target,
        1e-20,
        1,
        LCH([(1.0, _pauli("X")), (1e-300, _pauli("Z"))]),
        variance_method=tiny_positive_variance,
    )

    assert bound.step_second_order_bound == 0.0
    assert bound.step_taylor_remainder_bound == pytest.approx(
        (4.0 / 3.0) * 1e-60
    )
    assert bound.step_diamond_distance_upper_bound == pytest.approx(
        bound.step_taylor_remainder_bound
    )


def test_diamond_distance_bound_is_even_in_time() -> None:
    target = LCP({"X": 1.0, "Z": 1.0})
    decomposition = LCH([(1.0, _pauli("X")), (1.0, _pauli("Z"))])
    positive = qdrift_diamond_distance_bound(
        target,
        0.1,
        3,
        decomposition,
        variance_method="exact",
    )
    negative = qdrift_diamond_distance_bound(
        target,
        -0.1,
        3,
        decomposition,
        variance_method="exact",
    )

    assert negative.step_second_order_bound == positive.step_second_order_bound
    assert (
        negative.step_taylor_remainder_bound
        == positive.step_taylor_remainder_bound
    )
    assert (
        negative.step_diamond_distance_upper_bound
        == positive.step_diamond_distance_upper_bound
    )


@pytest.mark.parametrize("number_of_steps", [0, -1])
def test_diamond_distance_bound_rejects_nonpositive_steps(
    number_of_steps: int,
) -> None:
    with pytest.raises(ValueError, match="number_of_steps must be positive"):
        qdrift_diamond_distance_bound(
            _pauli("X"),
            0.1,
            number_of_steps,
            LCH([(1.0, _pauli("X"))]),
        )
