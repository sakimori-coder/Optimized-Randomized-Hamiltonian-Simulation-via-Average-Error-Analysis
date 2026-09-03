import numpy as np
import pytest

from grouping import build_chemistry_depth1_frobenius_lch
from hamiltonians.pauli import build_lch_from_lcp_unit_cost
from operators import LCH, LCP
from qdrift_trajectory_metrics import (
    calculate_tau_m,
    estimate_qdrift_haar_trajectory_metrics,
)


def test_pauli_tau_m_matches_closed_form() -> None:
    target = LCP({"X": 1.0, "Z": 1.0})
    result = calculate_tau_m(build_lch_from_lcp_unit_cost(target))

    assert result.lambda_sum == pytest.approx(2.0)
    assert result.tau_h_squared == pytest.approx(2.0)
    assert result.tau_m == pytest.approx(2.0)


def test_one_group_trajectory_is_exact_including_identity_phase() -> None:
    target = LCP({"I": 0.7, "X": -0.2})
    estimate = estimate_qdrift_haar_trajectory_metrics(
        target,
        1.3,
        7,
        [LCH([(1.0, target)])],
        num_initial_states=3,
        num_trajectories=4,
        seed=5,
    )[0]

    np.testing.assert_allclose(
        estimate.approximate_signals,
        estimate.ideal_signals,
        atol=2e-12,
    )
    assert estimate.mean_infidelity < 2e-14
    assert estimate.mean_absolute_qpe_signal_error < 2e-12


def test_parallel_metrics_are_reproducible_across_worker_counts() -> None:
    target = LCP({"XI": 1.0, "IZ": 0.7})
    decompositions = (
        build_lch_from_lcp_unit_cost(target),
        build_chemistry_depth1_frobenius_lch(target, max_group_size=2),
    )
    arguments = dict(
        target_hamiltonian=target,
        total_time=0.4,
        number_of_steps=3,
        qdrift_decompositions=decompositions,
        num_initial_states=4,
        num_trajectories=9,
        trajectory_chunks_per_state=3,
        seed=18,
    )
    two = estimate_qdrift_haar_trajectory_metrics(
        **arguments,
        num_workers=2,
    )
    three = estimate_qdrift_haar_trajectory_metrics(
        **arguments,
        num_workers=3,
    )

    for left, right in zip(two, three):
        np.testing.assert_array_equal(left.ideal_signals, right.ideal_signals)
        np.testing.assert_allclose(
            left.approximate_signals,
            right.approximate_signals,
            rtol=0.0,
            atol=2e-16,
        )


def test_global_state_indices_make_node_shards_reproducible() -> None:
    target = LCP({"XI": 1.0, "IZ": 0.7})
    decompositions = (
        build_lch_from_lcp_unit_cost(target),
        build_chemistry_depth1_frobenius_lch(target, max_group_size=2),
    )
    arguments = dict(
        target_hamiltonian=target,
        total_time=0.4,
        number_of_steps=3,
        qdrift_decompositions=decompositions,
        num_initial_states=4,
        num_trajectories=9,
        trajectory_chunks_per_state=3,
        seed=18,
        num_workers=2,
    )
    full = estimate_qdrift_haar_trajectory_metrics(**arguments)
    even = estimate_qdrift_haar_trajectory_metrics(
        **arguments,
        initial_state_indices=(0, 2),
    )
    odd = estimate_qdrift_haar_trajectory_metrics(
        **arguments,
        initial_state_indices=(1, 3),
    )

    for complete, even_part, odd_part in zip(full, even, odd):
        reconstructed_infidelity = np.empty(4)
        reconstructed_infidelity[(0, 2),] = even_part.infidelity.values
        reconstructed_infidelity[(1, 3),] = odd_part.infidelity.values
        reconstructed_qpe = np.empty(4)
        reconstructed_qpe[(0, 2),] = (
            even_part.absolute_qpe_signal_error.values
        )
        reconstructed_qpe[(1, 3),] = odd_part.absolute_qpe_signal_error.values
        np.testing.assert_allclose(
            reconstructed_infidelity,
            complete.infidelity.values,
            rtol=0.0,
            atol=2e-16,
        )
        np.testing.assert_allclose(
            reconstructed_qpe,
            complete.absolute_qpe_signal_error.values,
            rtol=0.0,
            atol=2e-16,
        )
