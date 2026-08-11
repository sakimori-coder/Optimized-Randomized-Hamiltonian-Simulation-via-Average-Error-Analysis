import numpy as np
import pytest

from operators import LCH, LCP
from main import compare_pauli_and_grouped_qdrift


def _pauli_lch(
    terms: list[tuple[float, str, float]],
    *,
    num_qubits: int | None = None,
) -> LCH:
    if num_qubits is None:
        num_qubits = len(terms[0][1])
    return LCH(
        [
            (
                coefficient,
                LCP({pauli: 1.0}, num_qubits=num_qubits),
                cost,
            )
            for coefficient, pauli, cost in terms
        ],
        num_qubits=num_qubits,
    )


@pytest.mark.parametrize(
    ("variance_method", "expected_variance", "expected_constant", "expected_steps"),
    [
        ("contraction_bound", 1.0, 16.0, 30),
        ("sdp_bound", 0.375, 6.0, 11),
        ("exact", 0.375, 6.0, 11),
        ("pauli_l1_bound", 0.375, 6.0, 11),
        ("anticommuting_bound", 0.375, 6.0, 11),
    ],
)
def test_variance_method_is_applied_to_both_decompositions(
    variance_method: str,
    expected_variance: float,
    expected_constant: float,
    expected_steps: int,
) -> None:
    # X and Z cannot share a commuting group, so both decompositions have
    # weights (3, 1).  This isolates variance-method dispatch from grouping.
    hamiltonian = _pauli_lch([(3.0, "X", 1.0), (1.0, "Z", 1.0)])

    comparison = compare_pauli_and_grouped_qdrift(
        hamiltonian,
        time=1.0,
        epsilon=1.1,
        variance_method=variance_method,  # type: ignore[arg-type]
        group_norm_method="lp",
    )

    assert comparison.variance_method == variance_method
    for cost in (comparison.pauli, comparison.grouped):
        assert cost.variance_method == variance_method
        assert cost.lambda_sum == pytest.approx(4.0)
        assert cost.normalized_variance_bound == pytest.approx(expected_variance)
        assert cost.variance_constant == pytest.approx(expected_constant)
        assert cost.one_step_second_order_bound == pytest.approx(
            expected_constant
        )
        assert cost.one_step_taylor_remainder_bound > 0.0
        assert cost.one_step_certified_upper_bound == pytest.approx(1.0)
        assert cost.steps == expected_steps
        assert cost.sampling_probabilities == pytest.approx([0.75, 0.25])
        assert cost.expected_depth_per_step == pytest.approx(1.0)
        assert cost.expected_total_depth == pytest.approx(float(expected_steps))


def test_multi_pauli_outer_term_is_rebuilt_as_standard_pauli_qdrift() -> None:
    hamiltonian = LCH(
        [
            (
                2.0,
                LCP({"X": 0.5, "Z": 0.5}),
                99.0,
            )
        ]
    )

    comparison = compare_pauli_and_grouped_qdrift(
        hamiltonian,
        time=1.0,
        epsilon=1.0,
        variance_method="exact",
        group_norm_method="lp",
        max_group_size=1,
    )

    assert comparison.pauli_strings == ("X", "Z")
    for cost in (comparison.pauli, comparison.grouped):
        assert cost.num_sampling_terms == 2
        assert cost.lambda_sum == pytest.approx(2.0)
        assert cost.normalized_variance_bound == pytest.approx(0.5)
        assert cost.sampling_probabilities == pytest.approx([0.5, 0.5])
        assert cost.rotation_depths == pytest.approx([1.0, 1.0])
        assert cost.expected_depth_per_step == pytest.approx(1.0)


def test_group_norm_method_changes_group_weight_and_total_depth() -> None:
    # ZII * IZI * IIZ = ZZZ.  Consequently the true/LP norm of this block is
    # 2, while its coefficient one-norm is 4.  Its rotation depth is 2.
    hamiltonian = _pauli_lch(
        [
            (1.0, "ZII", 1.0),
            (1.0, "IZI", 1.0),
            (1.0, "IIZ", 1.0),
            (-1.0, "ZZZ", 1.0),
        ]
    )

    coefficient_l1 = compare_pauli_and_grouped_qdrift(
        hamiltonian,
        time=1.0,
        epsilon=1.0,
        variance_method="contraction_bound",
        group_norm_method="coefficient_l1",
    )
    lp = compare_pauli_and_grouped_qdrift(
        hamiltonian,
        time=1.0,
        epsilon=1.0,
        variance_method="contraction_bound",
        group_norm_method="lp",
    )

    assert coefficient_l1.group_norm_method == "coefficient_l1"
    assert lp.group_norm_method == "lp"
    assert len(coefficient_l1.groups) == len(lp.groups) == 1
    assert coefficient_l1.groups[0].normalization_weight == pytest.approx(4.0)
    assert lp.groups[0].normalization_weight == pytest.approx(2.0)
    assert coefficient_l1.groups[0].rotation_depth == 2
    assert lp.groups[0].rotation_depth == 2

    # The Pauli decomposition is independent of group normalization.
    assert coefficient_l1.pauli.lambda_sum == pytest.approx(4.0)
    assert lp.pauli.lambda_sum == pytest.approx(4.0)
    assert coefficient_l1.pauli.steps == lp.pauli.steps == 32
    assert coefficient_l1.pauli.expected_total_depth == pytest.approx(32.0)
    assert lp.pauli.expected_total_depth == pytest.approx(32.0)

    assert coefficient_l1.grouped.lambda_sum == pytest.approx(4.0)
    assert coefficient_l1.grouped.steps == 32
    assert coefficient_l1.grouped.expected_depth_per_step == pytest.approx(2.0)
    assert coefficient_l1.grouped.expected_total_depth == pytest.approx(64.0)

    assert lp.grouped.lambda_sum == pytest.approx(2.0)
    assert lp.grouped.steps == 8
    assert lp.grouped.expected_depth_per_step == pytest.approx(2.0)
    assert lp.grouped.expected_total_depth == pytest.approx(16.0)


def test_grouped_pauli_l1_bound_improves_contraction_bound() -> None:
    hamiltonian = _pauli_lch(
        [
            (0.5, "ZI", 1.0),
            (0.5, "IZ", 1.0),
            (1.0, "XX", 1.0),
        ]
    )

    comparison = compare_pauli_and_grouped_qdrift(
        hamiltonian,
        time=1.0,
        epsilon=1.0,
        variance_method="pauli_l1_bound",
        group_norm_method="lp",
    )

    assert comparison.pauli.normalized_variance_bound == pytest.approx(0.75)
    assert comparison.pauli.steps == 6
    assert comparison.grouped.normalized_variance_bound == pytest.approx(0.5)
    assert comparison.grouped.steps == 4
    assert comparison.grouped.expected_total_depth == pytest.approx(4.0)


def test_grouped_anticommuting_bound_improves_pauli_l1_bound() -> None:
    hamiltonian = _pauli_lch(
        [
            (1.0, "YY", 1.0),
            (1.0, "IZ", 1.0),
            (0.5, "XZ", 1.0),
            (1.0, "ZI", 1.0),
        ]
    )

    pauli_l1 = compare_pauli_and_grouped_qdrift(
        hamiltonian,
        time=1.0,
        epsilon=1.2,
        variance_method="pauli_l1_bound",
        group_norm_method="lp",
    )
    anticommuting = compare_pauli_and_grouped_qdrift(
        hamiltonian,
        time=1.0,
        epsilon=1.2,
        variance_method="anticommuting_bound",
        group_norm_method="lp",
    )

    assert pauli_l1.grouped.normalized_variance_bound == pytest.approx(44.0 / 49.0)
    assert anticommuting.grouped.normalized_variance_bound == pytest.approx(
        (80.0 + 4.0 * np.sqrt(61.0)) / 147.0
    )
    assert pauli_l1.grouped.steps == 19
    assert anticommuting.grouped.steps == 16


@pytest.mark.parametrize(
    "variance_method",
    ["pauli_l1_bound", "anticommuting_bound", "sdp_bound"],
)
def test_grouped_symbolic_bounds_are_matrix_free_at_large_n(
    monkeypatch: pytest.MonkeyPatch,
    variance_method: str,
) -> None:
    num_qubits = 1000
    suffix = "I" * (num_qubits - 2)
    hamiltonian = _pauli_lch(
        [
            (0.5, "ZI" + suffix, 1.0),
            (0.5, "IZ" + suffix, 1.0),
            (1.0, "XX" + suffix, 1.0),
        ]
    )

    def fail_if_materialized(*_args: object, **_kwargs: object) -> None:
        pytest.fail(f"grouped {variance_method} must not materialize a 2**n matrix")

    for owner, attribute in (
        (LCP, "to_matrix"),
        (LCP, "to_csr"),
        (LCP, "operator_norm"),
        (LCH, "to_matrix"),
        (LCH, "to_csr"),
        (LCH, "operator_norm"),
    ):
        monkeypatch.setattr(owner, attribute, fail_if_materialized)

    comparison = compare_pauli_and_grouped_qdrift(
        hamiltonian,
        time=1.0,
        epsilon=1.1,
        variance_method=variance_method,  # type: ignore[arg-type]
        group_norm_method="lp",
    )

    assert comparison.pauli.normalized_variance_bound == pytest.approx(0.75)
    assert comparison.pauli.steps == 6
    assert comparison.grouped.normalized_variance_bound == pytest.approx(0.5)
    assert comparison.grouped.steps == 4


def test_large_contraction_lp_comparison_does_not_materialize_matrices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_qubits = 1000
    suffix = "I" * (num_qubits - 3)
    hamiltonian = _pauli_lch(
        [
            (1.0, "ZII" + suffix, 1.0),
            (1.0, "IZI" + suffix, 1.0),
            (1.0, "IIZ" + suffix, 1.0),
            (-1.0, "ZZZ" + suffix, 1.0),
        ]
    )

    def fail_if_materialized(*_args: object, **_kwargs: object) -> None:
        pytest.fail("contraction_bound + lp must not materialize a 2**n matrix")

    monkeypatch.setattr(LCP, "to_matrix", fail_if_materialized)
    monkeypatch.setattr(LCP, "to_csr", fail_if_materialized)
    monkeypatch.setattr(LCP, "operator_norm", fail_if_materialized)
    monkeypatch.setattr(LCH, "to_matrix", fail_if_materialized)
    monkeypatch.setattr(LCH, "to_csr", fail_if_materialized)
    monkeypatch.setattr(LCH, "operator_norm", fail_if_materialized)

    comparison = compare_pauli_and_grouped_qdrift(
        hamiltonian,
        time=1.0,
        epsilon=0.1,
        variance_method="contraction_bound",
        group_norm_method="lp",
    )

    assert comparison.pauli.lambda_sum == pytest.approx(4.0)
    assert comparison.pauli.steps == 320
    assert comparison.pauli.expected_total_depth == pytest.approx(320.0)
    assert comparison.grouped.lambda_sum == pytest.approx(2.0)
    assert comparison.grouped.steps == 80
    assert comparison.grouped.expected_total_depth == pytest.approx(160.0)
    assert np.isfinite(comparison.grouped.expected_total_depth)


def test_empty_hamiltonian_has_zero_cost_for_both_decompositions() -> None:
    comparison = compare_pauli_and_grouped_qdrift(
        _pauli_lch([], num_qubits=1000),
        time=1.0,
        epsilon=0.1,
        variance_method="contraction_bound",
        group_norm_method="lp",
    )

    assert comparison.pauli_strings == ()
    assert comparison.groups == ()
    assert comparison.grouped_to_pauli_depth_ratio == 0.0
    for cost in (comparison.pauli, comparison.grouped):
        assert cost.num_sampling_terms == 0
        assert cost.lambda_sum == 0.0
        assert cost.normalized_variance_bound == 0.0
        assert cost.variance_constant == 0.0
        assert cost.one_step_second_order_bound == 0.0
        assert cost.one_step_taylor_remainder_bound == 0.0
        assert cost.one_step_certified_upper_bound == 0.0
        assert cost.steps == 0
        assert cost.sampling_probabilities.size == 0
        assert cost.rotation_depths.size == 0
        assert cost.expected_depth_per_step == 0.0
        assert cost.expected_total_depth == 0.0


def test_comparison_can_select_chemistry_aware_grouping() -> None:
    hamiltonian = _pauli_lch(
        [
            (3.0, "ZZ", 1.0),
            (2.0, "XX", 1.0),
            (-1.0, "YY", 1.0),
        ]
    )

    greedy = compare_pauli_and_grouped_qdrift(
        hamiltonian,
        time=1.0,
        epsilon=1.0,
        variance_method="contraction_bound",
        group_norm_method="coefficient_l1",
        grouping_method="greedy",
    )
    chemistry = compare_pauli_and_grouped_qdrift(
        hamiltonian,
        time=1.0,
        epsilon=1.0,
        variance_method="contraction_bound",
        group_norm_method="coefficient_l1",
        grouping_method="chemistry",
    )

    assert greedy.grouping_method == "greedy"
    assert len(greedy.groups) == 1
    assert chemistry.grouping_method == "chemistry"
    assert [group.pauli_strings for group in chemistry.groups] == [
        ("ZZ",),
        ("XX", "YY"),
    ]
