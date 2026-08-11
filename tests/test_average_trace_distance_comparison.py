import numpy as np
import pytest

from average_trace_distance_comparison import (
    compare_average_trace_distances,
)
from operators import LCH, LCP


def test_single_commuting_group_is_exact_for_one_channel_step() -> None:
    hamiltonian = LCH(
        [
            (1.0, LCP({"II": 1.0}), 0.0),
            (1.0, LCP({"ZI": 1.0}), 1.0),
            (1.0, LCP({"IZ": 1.0}), 1.0),
            (-1.0, LCP({"ZZ": 1.0}), 1.0),
        ]
    )

    comparison = compare_average_trace_distances(
        hamiltonian,
        total_time=0.35,
        number_of_steps=1,
        num_initial_states=5,
        group_norm_method="coefficient_l1",
        max_group_size=4,
        seed=11,
    )

    assert len(comparison.groups) == 1
    # Here the coefficient-l1 group weight is 4, while ||H||_op is 2.
    # A single grouped branch is still exp(-i*time*H) exactly.
    assert comparison.group_weights == (4.0,)
    assert comparison.grouped.mean == pytest.approx(0.0, abs=1e-10)
    assert comparison.grouped_bound.tau_v == 0.0
    assert comparison.grouped_bound.step_average_trace_distance_upper_bound == 0.0
    assert comparison.operator_variance_method == "exact"
    assert comparison.grouped_diamond_bound.normalized_variance_bound == 0.0
    assert comparison.grouped_diamond_bound.step_second_order_bound == 0.0
    assert (
        comparison.grouped_diamond_bound.step_diamond_distance_upper_bound
        == 0.0
    )
    assert comparison.pauli_diamond_bound.step_second_order_bound > 0.0
    assert comparison.pauli.mean > 1e-4
    assert comparison.pauli.values.shape == (5,)
    assert comparison.grouped.values.shape == (5,)
    assert isinstance(comparison.pauli_hamiltonian, LCH)
    assert isinstance(comparison.grouped_hamiltonian, LCH)
    assert "ZI:" in comparison.grouped_hamiltonian.format_decomposition()


def test_single_term_groups_reproduce_pauli_qdrift() -> None:
    hamiltonian = LCH(
        [
            (0.8, LCP({"X": 1.0}), 1.0),
            (-0.3, LCP({"Z": 1.0}), 1.0),
        ]
    )

    comparison = compare_average_trace_distances(
        hamiltonian,
        total_time=0.2,
        number_of_steps=1,
        num_initial_states=4,
        group_norm_method="lp",
        max_group_size=1,
        trace_distance_method="low_rank",
        seed=5,
    )

    np.testing.assert_allclose(
        comparison.grouped.values,
        comparison.pauli.values,
        atol=1e-10,
    )
    assert comparison.grouped_bound.tau_v == pytest.approx(
        comparison.pauli_bound.tau_v
    )
    assert comparison.grouped_bound.tau_v_squared == pytest.approx(
        comparison.pauli_bound.tau_v_squared
    )
    assert comparison.grouped_diamond_bound.normalized_variance_bound == (
        pytest.approx(
            comparison.pauli_diamond_bound.normalized_variance_bound
        )
    )
    assert comparison.grouped_diamond_bound.step_second_order_bound == (
        pytest.approx(comparison.pauli_diamond_bound.step_second_order_bound)
    )
    assert (
        comparison.grouped_diamond_bound.step_diamond_distance_upper_bound
        == pytest.approx(
            comparison.pauli_diamond_bound.step_diamond_distance_upper_bound
        )
    )


def test_multi_pauli_outer_term_uses_flattened_standard_pauli_channel() -> None:
    hamiltonian = LCH(
        [
            (
                2.0,
                LCP({"X": 0.5, "Z": 0.5}),
                99.0,
            )
        ]
    )

    comparison = compare_average_trace_distances(
        hamiltonian,
        total_time=0.2,
        number_of_steps=1,
        num_initial_states=4,
        group_norm_method="lp",
        max_group_size=1,
        seed=17,
    )

    assert comparison.num_pauli_terms == 2
    assert comparison.groups == (("X",), ("Z",))
    assert comparison.group_weights == pytest.approx((1.0, 1.0))
    np.testing.assert_allclose(
        comparison.pauli.values,
        comparison.grouped.values,
        atol=1e-10,
    )
    assert comparison.pauli.mean > 1e-6


def test_fixed_seed_reproduces_monte_carlo_samples() -> None:
    hamiltonian = LCH(
        [
            (0.5, LCP({"XI": 1.0}), 1.0),
            (0.2, LCP({"IX": 1.0}), 1.0),
            (-0.4, LCP({"ZI": 1.0}), 1.0),
        ]
    )
    arguments = dict(
        total_time=0.15,
        number_of_steps=1,
        num_initial_states=3,
        group_norm_method="coefficient_l1",
        max_group_size=2,
        seed=123,
    )

    first = compare_average_trace_distances(hamiltonian, **arguments)
    second = compare_average_trace_distances(hamiltonian, **arguments)

    np.testing.assert_array_equal(first.pauli.values, second.pauli.values)
    np.testing.assert_array_equal(first.grouped.values, second.grouped.values)
