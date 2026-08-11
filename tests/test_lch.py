import numpy as np
import pytest

from operators import LCH, LCP


def test_lch_terms_are_lcps_and_lcp_property_flattens_them() -> None:
    first = LCP({"XI": 0.5, "IZ": -0.25})
    second = LCP({"XI": -0.5, "YY": 1.0})

    hamiltonian = LCH(
        [
            (2.0, first, 3.0),
            (-1.0, second, 5.0),
        ]
    )

    assert len(hamiltonian) == 2
    assert all(isinstance(operator, LCP) for _, operator in hamiltonian.terms)
    assert hamiltonian.lcp.terms == {
        "XI": 1.5,
        "IZ": -0.5,
        "YY": -1.0,
    }
    assert hamiltonian.evolution_costs == [3.0, 5.0]
    np.testing.assert_allclose(
        hamiltonian.to_matrix(),
        hamiltonian.lcp.to_matrix(),
    )


def test_lch_preserves_duplicate_samples_and_outer_distribution() -> None:
    sample = LCP({"X": 1.0})
    hamiltonian = LCH(
        [
            (1.0, sample, 2.0),
            (-0.5, sample, 7.0),
        ]
    )

    assert len(hamiltonian) == 2
    assert hamiltonian.lcp.terms == {"X": 0.5}
    assert hamiltonian.coefficient_one_norm() == pytest.approx(1.5)
    assert hamiltonian.sampling_probabilities() == pytest.approx(
        [2.0 / 3.0, 1.0 / 3.0]
    )
    assert hamiltonian.get_operator(0).terms == {"X": 1.0}
    assert hamiltonian.get_term_with_cost(1)[2] == 7.0


def test_lch_arithmetic_preserves_decomposition_order() -> None:
    left = LCH([(1.0, LCP({"X": 1.0}), 2.0)])
    right = LCH([(0.5, LCP({"Z": 1.0}), 3.0)])

    added = left + right
    subtracted = left - right
    scaled = -2.0 * left

    assert [coefficient for coefficient, _, _ in added.lcp_terms] == [1.0, 0.5]
    assert added.evolution_costs == [2.0, 3.0]
    assert added.lcp.terms == {"X": 1.0, "Z": 0.5}
    assert [coefficient for coefficient, _, _ in subtracted.lcp_terms] == [
        1.0,
        -0.5,
    ]
    assert subtracted.lcp.terms == {"X": 1.0, "Z": -0.5}
    assert scaled.lcp.terms == {"X": -2.0}
    assert scaled.evolution_costs == [2.0]


def test_lch_requires_lcp_terms_with_equal_qubit_counts() -> None:
    with pytest.raises(TypeError, match="must be an LCP"):
        LCH([(1.0, "X", 1.0)])
    with pytest.raises(ValueError, match="same number of qubits"):
        LCH(
            [
                (1.0, LCP({"X": 1.0}), 1.0),
                (1.0, LCP({"ZZ": 1.0}), 1.0),
            ]
        )
    with pytest.raises(ValueError, match="must be real"):
        LCH([(1.0j, LCP({"X": 1.0}), 1.0)])


def test_empty_lch_requires_num_qubits() -> None:
    with pytest.raises(ValueError, match="num_qubits"):
        LCH([])

    empty = LCH([], num_qubits=3)
    assert len(empty) == 0
    assert empty.lcp.terms == {}
    assert empty.dimension == 8


def test_format_decomposition_shows_outer_and_inner_coefficients() -> None:
    hamiltonian = LCH(
        [
            (
                4.0,
                LCP({"ZI": 0.25, "IZ": -0.5}),
                7.0,
            )
        ]
    )

    output = hamiltonian.format_decomposition(precision=6)

    assert "LCH(num_qubits=2, num_terms=1, coefficient_one_norm=4)" in output
    assert (
        "term[0]: outer_coefficient=4, sampling_probability=1, "
        "evolution_cost=7"
    ) in output
    assert (
        "ZI: inner_coefficient=+0.25, weighted_coefficient=+1" in output
    )
    assert (
        "IZ: inner_coefficient=-0.5, weighted_coefficient=-2" in output
    )


def test_format_decomposition_preserves_duplicate_outer_terms() -> None:
    hamiltonian = LCH(
        [
            (2.0, LCP({"X": 0.5}), 1.0),
            (-3.0, LCP({"X": 0.25}), 2.0),
        ]
    )

    output = hamiltonian.format_decomposition(precision=6)

    assert output.count("  X:") == 2
    assert "term[0]: outer_coefficient=2" in output
    assert "weighted_coefficient=+1" in output
    assert "term[1]: outer_coefficient=-3" in output
    assert "weighted_coefficient=-0.75" in output


def test_format_decomposition_limits_large_output() -> None:
    hamiltonian = LCH(
        [
            (1.0, LCP({"XI": 1.0, "IZ": 2.0}), 3.0),
            (2.0, LCP({"YY": 1.0}), 4.0),
        ]
    )

    output = hamiltonian.format_decomposition(
        max_terms=1,
        max_pauli_terms=1,
    )

    assert "XI:" in output
    assert "IZ:" not in output
    assert "1 Pauli terms omitted" in output
    assert "1 LCH terms omitted" in output


def test_string_representation_handles_empty_lch() -> None:
    output = str(LCH([], num_qubits=3))

    assert "LCH(num_qubits=3, num_terms=0" in output
    assert "(empty decomposition)" in output
    assert "object at 0x" not in output
