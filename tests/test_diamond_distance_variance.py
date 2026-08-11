import numpy as np
import pytest

from diamond_distance import variance
from operators import LCH, LCP
from diamond_distance.variance import (
    anticommuting_bound_from_samples,
    estimate_lch_centered_second_moment_norm,
    lch_centered_second_moment_pauli_coefficients,
    pauli_l1_bound_from_samples,
    pauli_moment_sdp_bound_from_samples,
)


def _pauli(pauli: str) -> LCP:
    return LCP({pauli: 1.0})


def test_lch_exact_estimator_is_separate_from_error_and_cost_parameters() -> None:
    hamiltonian = LCH(
        [(1.0, _pauli("X"), 10.0), (1.0, _pauli("Z"), 20.0)]
    )

    exact = estimate_lch_centered_second_moment_norm(
        hamiltonian,
        method="exact",
    )

    assert np.isclose(exact, 0.5)


def test_accepts_a_custom_estimator_callable() -> None:
    hamiltonian = LCH([(1.0, _pauli("X"), 1.0)])

    value = estimate_lch_centered_second_moment_norm(
        hamiltonian,
        method=lambda decomposition: 0.25
        if decomposition.lambda_sum == 1.0
        else 1.0,
    )

    assert value == 0.25


@pytest.mark.parametrize("method", ["direct", "triangle", "uniform_bound"])
def test_removed_method_names_are_rejected(method: str) -> None:
    with pytest.raises(ValueError, match="unknown method"):
        estimate_lch_centered_second_moment_norm(
            LCH([(1.0, _pauli("X"), 1.0)]),
            method=method,  # type: ignore[arg-type]
        )


def test_contraction_bound_is_one_without_materializing_matrices(
    monkeypatch,
) -> None:
    hamiltonian = LCH([(1.0, _pauli("X" * 100), 1.0)])

    def fail_if_materialized(_hamiltonian):
        pytest.fail("contraction_bound must not materialize Pauli matrices")

    monkeypatch.setattr(
        variance,
        "lch_qdrift_decomposition",
        fail_if_materialized,
    )

    value = estimate_lch_centered_second_moment_norm(
        hamiltonian,
        method="contraction_bound",
    )

    assert value == 1.0


def test_contraction_bound_is_zero_for_zero_hamiltonian() -> None:
    hamiltonian = LCH([(0.0, _pauli("X"), 1.0)])

    value = estimate_lch_centered_second_moment_norm(
        hamiltonian,
        method="contraction_bound",
    )

    assert value == 0.0


def test_lch_rejects_non_real_coefficient() -> None:
    with pytest.raises(ValueError, match="must be real"):
        LCH([(1.0j, _pauli("X"), 1.0)])


def test_polynomial_pauli_bounds_dominate_exact_value() -> None:
    hamiltonian = LCH(
        [
            (0.7, _pauli("XI"), 1.0),
            (-0.4, _pauli("IZ"), 1.0),
            (0.2, _pauli("YY"), 1.0),
            (0.9, _pauli("ZX"), 1.0),
        ]
    )

    exact = estimate_lch_centered_second_moment_norm(
        hamiltonian,
        method="exact",
    )
    estimates = {
        method: estimate_lch_centered_second_moment_norm(
            hamiltonian,
            method=method,
        )
        for method in (
            "pauli_l1_bound",
            "anticommuting_bound",
            "sdp_bound",
        )
    }

    for estimate in estimates.values():
        assert exact <= estimate + 1e-9
        assert estimate <= 1.0
    assert estimates["anticommuting_bound"] <= estimates["pauli_l1_bound"]


def test_pauli_l1_bound_combines_products_before_absolute_values() -> None:
    hamiltonian = LCH(
        [(1.0, _pauli("X"), 1.0), (1.0, _pauli("Z"), 1.0)]
    )

    pauli_l1 = estimate_lch_centered_second_moment_norm(
        hamiltonian,
        method="pauli_l1_bound",
    )
    anticommuting = estimate_lch_centered_second_moment_norm(
        hamiltonian,
        method="anticommuting_bound",
    )

    assert np.isclose(pauli_l1, 0.5)
    assert np.isclose(anticommuting, 0.5)


def test_pauli_l1_bound_supports_grouped_pauli_samples() -> None:
    samples = [
        LCP({"ZI": 0.5, "IZ": 0.5}),
        LCP({"XX": 1.0}),
    ]

    bound = pauli_l1_bound_from_samples([1.0, 1.0], samples)

    # V = 3/8 I + 1/8 ZZ, whose coefficient 1-norm and operator norm are 1/2.
    assert bound == pytest.approx(0.5)


def test_general_multi_pauli_lch_matches_from_samples_estimators() -> None:
    operators = [
        LCP({"ZI": 0.5, "IZ": 0.5}),
        LCP({"XX": 0.5, "YY": 0.5}),
    ]
    coefficients = [1.5, -0.75]
    hamiltonian = LCH(
        [
            (coefficient, operator, 1.0)
            for coefficient, operator in zip(coefficients, operators)
        ]
    )
    weights = [abs(coefficient) for coefficient in coefficients]
    signed_samples = [
        np.sign(coefficient) * operator
        for coefficient, operator in zip(coefficients, operators)
    ]

    decomposition = variance.qdrift_decomposition_from_samples(
        weights,
        [sample.to_csr() for sample in signed_samples],
    )
    direct_exact = variance.estimate_centered_second_moment_norm(
        decomposition,
        method="exact",
    )
    assert estimate_lch_centered_second_moment_norm(
        hamiltonian,
        method="exact",
    ) == pytest.approx(direct_exact, abs=1e-10)

    estimators = {
        "pauli_l1_bound": pauli_l1_bound_from_samples,
        "anticommuting_bound": anticommuting_bound_from_samples,
        "sdp_bound": pauli_moment_sdp_bound_from_samples,
    }
    for method, from_samples in estimators.items():
        assert estimate_lch_centered_second_moment_norm(
            hamiltonian,
            method=method,
        ) == pytest.approx(
            from_samples(weights, signed_samples),
            abs=1e-7,
        )


def test_pauli_l1_bound_is_zero_for_one_group_sample() -> None:
    bound = pauli_l1_bound_from_samples(
        [2.0],
        [LCP({"ZI": 0.5, "IZ": 0.5})],
    )

    assert bound == 0.0
    assert anticommuting_bound_from_samples(
        [2.0],
        [LCP({"ZI": 0.5, "IZ": 0.5})],
    ) == 0.0


def test_anticommuting_bound_supports_grouped_pauli_samples() -> None:
    samples = [
        LCP({"IZ": 2.0 / 3.0, "XZ": 1.0 / 3.0}),
        LCP({"YY": 1.0}),
        LCP({"ZI": 1.0}),
    ]
    weights = [1.5, 1.0, 1.0]

    pauli_l1 = pauli_l1_bound_from_samples(weights, samples)
    anticommuting = anticommuting_bound_from_samples(weights, samples)

    assert pauli_l1 == pytest.approx(44.0 / 49.0)
    assert anticommuting == pytest.approx((80.0 + 4.0 * np.sqrt(61.0)) / 147.0)
    assert anticommuting < pauli_l1


def test_symbolic_pauli_products_combine_with_the_correct_phases() -> None:
    hamiltonian = LCH(
        [
            (1.0, _pauli("XX"), 1.0),
            (1.0, _pauli("YY"), 1.0),
            (1.0, _pauli("ZI"), 1.0),
            (1.0, _pauli("IZ"), 1.0),
        ]
    )

    coefficients = lch_centered_second_moment_pauli_coefficients(hamiltonian)
    sdp_bound = estimate_lch_centered_second_moment_norm(
        hamiltonian,
        method="sdp_bound",
    )

    # XX*YY=-ZZ and ZI*IZ=+ZZ, so their variance contributions cancel.
    assert coefficients == {"II": 0.75 + 0.0j}
    assert sdp_bound == pytest.approx(0.75, abs=1e-7)


def test_anticommuting_bound_improves_the_pauli_l1_bound() -> None:
    hamiltonian = LCH(
        [
            (1.0, _pauli("I"), 1.0),
            (1.0, _pauli("X"), 1.0),
            (1.0, _pauli("Z"), 1.0),
        ]
    )

    pauli_l1 = estimate_lch_centered_second_moment_norm(
        hamiltonian,
        method="pauli_l1_bound",
    )
    anticommuting = estimate_lch_centered_second_moment_norm(
        hamiltonian,
        method="anticommuting_bound",
    )

    assert pauli_l1 == 1.0
    assert np.isclose(anticommuting, (6.0 + 2.0 * np.sqrt(2.0)) / 9.0)


def test_pauli_moment_sdp_uses_anticommutation_relations() -> None:
    hamiltonian = LCH(
        [(3.0, _pauli("X"), 1.0), (1.0, _pauli("Z"), 1.0)]
    )

    value = estimate_lch_centered_second_moment_norm(
        hamiltonian,
        method="sdp_bound",
    )

    # For anticommuting X and Z, V=2*p*(1-p)*I.  The previous
    # probability-only SDP returned 4*p*(1-p), which is twice as large.
    assert value == pytest.approx(2.0 * 0.75 * 0.25, abs=1e-7)


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        ([LCP({"X": 1.0}), LCP({"Z": 1.0})], 0.5),
        ([LCP({"ZI": 1.0}), LCP({"IZ": 1.0})], 1.0),
        ([LCP({"X": 1.0}), LCP({"X": 1.0})], 0.0),
        ([LCP({"X": 1.0}), LCP({"X": -1.0})], 1.0),
    ],
)
def test_pauli_moment_sdp_distinguishes_pauli_relations(
    samples: list[LCP],
    expected: float,
) -> None:
    assert pauli_moment_sdp_bound_from_samples(
        [1.0, 1.0],
        samples,
    ) == pytest.approx(expected, abs=1e-7)


def test_pauli_moment_sdp_supports_grouped_pauli_samples() -> None:
    samples = [
        LCP({"ZI": 0.5, "IZ": 0.5}),
        LCP({"XX": 1.0}),
    ]

    bound = pauli_moment_sdp_bound_from_samples([1.0, 1.0], samples)

    # V = 3/8 I + 1/8 ZZ, so its operator norm is 1/2.
    assert bound == pytest.approx(0.5, abs=1e-7)


def test_pauli_moment_sdp_bound_dominates_exact_grouped_variance() -> None:
    samples = [
        LCP({"IZ": 2.0 / 3.0, "XZ": 1.0 / 3.0}),
        LCP({"YY": 1.0}),
        LCP({"ZI": 1.0}),
    ]
    weights = [1.5, 1.0, 1.0]
    decomposition = variance.qdrift_decomposition_from_samples(
        weights,
        [sample.to_csr() for sample in samples],
    )
    exact = variance.estimate_centered_second_moment_norm(
        decomposition,
        method="exact",
    )

    bound = pauli_moment_sdp_bound_from_samples(weights, samples)

    assert exact <= bound + 1e-7
    assert bound <= 1.0


@pytest.mark.parametrize(
    "method",
    ["pauli_l1_bound", "anticommuting_bound", "sdp_bound"],
)
def test_polynomial_bounds_do_not_materialize_pauli_matrices(
    monkeypatch,
    method,
) -> None:
    hamiltonian = LCH(
        [
            (1.0, _pauli("X" * 1000), 1.0),
            (-0.5, _pauli("Z" * 1000), 1.0),
        ]
    )

    def fail_if_materialized(*_args, **_kwargs):
        pytest.fail(f"{method} must not materialize Pauli matrices")

    for owner, attribute in (
        (variance, "lch_qdrift_decomposition"),
        (LCH, "to_csr"),
        (LCH, "to_matrix"),
        (LCP, "to_csr"),
        (LCP, "to_matrix"),
    ):
        monkeypatch.setattr(owner, attribute, fail_if_materialized)

    value = estimate_lch_centered_second_moment_norm(
        hamiltonian,
        method=method,
    )

    assert 0.0 <= value <= 1.0
