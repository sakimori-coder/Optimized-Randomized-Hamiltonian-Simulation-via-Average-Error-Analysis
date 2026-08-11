import pytest

from grouping import build_grouped_lch
from operators import LCP


def test_grouped_lch_contains_group_weights_and_normalized_groups() -> None:
    hamiltonian = LCP(
        {
            "ZII": 1.0,
            "IZI": 1.0,
            "IIZ": 1.0,
            "ZZZ": -1.0,
        }
    )

    grouped = build_grouped_lch(
        hamiltonian,
        grouping_method="greedy",
        group_norm_method="lp",
    )

    assert len(grouped.terms) == 1
    weight, normalized_group = grouped.terms[0]
    assert weight == pytest.approx(2.0)
    assert normalized_group.terms == {
        "IIZ": 0.5,
        "IZI": 0.5,
        "ZII": 0.5,
        "ZZZ": -0.5,
    }
    assert normalized_group.operator_norm() == pytest.approx(1.0)
    assert grouped.lcp.terms == hamiltonian.terms


def test_noncommuting_terms_become_different_lch_samples() -> None:
    hamiltonian = LCP({"X": 2.0, "Z": -0.5})

    grouped = build_grouped_lch(
        hamiltonian,
        grouping_method="greedy",
        group_norm_method="coefficient_l1",
    )

    assert [
        (weight, group.terms)
        for weight, group in grouped.terms
    ] == [
        (2.0, {"X": 1.0}),
        (0.5, {"Z": -1.0}),
    ]
    assert grouped.lcp.terms == hamiltonian.terms


def test_group_norm_method_controls_the_outer_lch_weight() -> None:
    hamiltonian = LCP(
        {
            "ZII": 1.0,
            "IZI": 1.0,
            "IIZ": 1.0,
            "ZZZ": -1.0,
        }
    )

    coefficient_l1 = build_grouped_lch(
        hamiltonian,
        group_norm_method="coefficient_l1",
    )
    lp = build_grouped_lch(
        hamiltonian,
        group_norm_method="lp",
    )

    assert coefficient_l1.terms[0][0] == pytest.approx(4.0)
    assert lp.terms[0][0] == pytest.approx(2.0)


def test_frobenius_weights_define_the_sampling_probabilities() -> None:
    hamiltonian = LCP(
        {
            "IX": 4.0,
            "XI": 3.0,
            "IZ": 0.8,
            "ZI": 0.6,
        }
    )

    grouped = build_grouped_lch(
        hamiltonian,
        grouping_method="greedy",
        group_norm_method="frobenius",
    )

    weights = [weight for weight, _ in grouped.terms]
    assert weights == pytest.approx([10.0, 2.0])
    assert grouped.sampling_probabilities() == pytest.approx(
        [5.0 / 6.0, 1.0 / 6.0]
    )
    assert all(
        normalized_group.operator_norm() <= 1.0 + 1e-12
        for _, normalized_group in grouped.terms
    )
    assert grouped.lcp.terms == pytest.approx(hamiltonian.terms)


def test_empty_lcp_produces_empty_lch() -> None:
    grouped = build_grouped_lch(
        LCP({}, num_qubits=3),
    )

    assert grouped.terms == []
    assert grouped.num_qubits == 3
