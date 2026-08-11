import numpy as np
import pytest

from average_trace_distance.variance_bound import (
    qdrift_haar_average_trace_distance_bound,
    qdrift_haar_average_trace_distance_bounds,
)
from operators import LCH, LCP


def test_tau_moments_match_known_grouped_example() -> None:
    hamiltonian = LCH(
        [
            (1.0, LCP({"ZI": 0.5, "IZ": 0.5})),
            (1.0, LCP({"XX": 1.0})),
        ]
    )

    result = qdrift_haar_average_trace_distance_bound(
        hamiltonian.lcp,
        0.2,
        1,
        hamiltonian,
    )

    # V = 3/8 II + 1/8 ZZ.
    assert result.lambda_sum == pytest.approx(2.0)
    assert result.tau_v == pytest.approx(3.0 / 8.0)
    assert result.tau_v_squared == pytest.approx(5.0 / 32.0)
    expected_normalized = 0.5 * (3.0 / 8.0 + np.sqrt(5.0 / 32.0))
    assert result.normalized_average_variance_bound == pytest.approx(
        expected_normalized
    )
    assert result.step_second_order_bound == pytest.approx(
        0.4**2 * expected_normalized
    )


def test_certified_bound_includes_pauli_l1_taylor_remainder() -> None:
    hamiltonian = LCH(
        [
            (1.0, LCP({"X": 1.0})),
            (1.0, LCP({"Z": 1.0})),
        ]
    )

    result = qdrift_haar_average_trace_distance_bound(
        hamiltonian.lcp,
        0.1,
        1,
        hamiltonian,
    )

    # V = I/2, Lambda*time = 0.2, and both sample norm bounds are one.
    assert result.step_second_order_bound == pytest.approx(0.02)
    assert result.step_taylor_remainder_bound == pytest.approx(
        (4.0 / 3.0) * 0.2**3
    )
    assert result.step_average_trace_distance_upper_bound == pytest.approx(
        result.step_second_order_bound + result.step_taylor_remainder_bound
    )


def test_certified_bound_is_clipped_at_one() -> None:
    hamiltonian = LCH(
        [
            (1.0, LCP({"X": 1.0})),
            (1.0, LCP({"Z": 1.0})),
        ]
    )

    result = qdrift_haar_average_trace_distance_bound(
        hamiltonian.lcp,
        1.0,
        1,
        hamiltonian,
    )

    assert (
        result.step_second_order_bound + result.step_taylor_remainder_bound
        > 1.0
    )
    assert result.step_average_trace_distance_upper_bound == 1.0


def test_zero_variance_has_zero_finite_time_bound() -> None:
    hamiltonian = LCH([(3.0, LCP({"X": 1.0}))])

    result = qdrift_haar_average_trace_distance_bound(
        hamiltonian.lcp,
        2.0,
        1,
        hamiltonian,
    )

    assert result.tau_v == 0.0
    assert result.tau_v_squared == 0.0
    assert result.step_second_order_bound == 0.0
    assert result.step_taylor_remainder_bound == 0.0
    assert result.step_average_trace_distance_upper_bound == 0.0


def test_bound_does_not_materialize_hilbert_space_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hamiltonian = LCH(
        [
            (1.0, LCP({"X" * 100: 1.0})),
            (1.0, LCP({"Z" * 100: 1.0})),
        ]
    )

    def fail_if_materialized(*_args, **_kwargs):
        pytest.fail("the tau bound must remain matrix-free")

    monkeypatch.setattr(LCP, "to_matrix", fail_if_materialized)
    monkeypatch.setattr(LCP, "to_csr", fail_if_materialized)

    result = qdrift_haar_average_trace_distance_bound(
        hamiltonian.lcp,
        0.1,
        1,
        hamiltonian,
    )

    assert result.tau_v == pytest.approx(0.5)
    # X**100 and Z**100 commute, so V also contains a Pauli-product term.
    assert result.tau_v_squared == pytest.approx(0.5)


def test_multiple_bounds_preserve_order_and_use_step_time() -> None:
    target = LCP({"ZI": 1.0, "IZ": 1.0, "XX": 1.0})
    grouped = LCH(
        [
            (2.0, LCP({"ZI": 0.5, "IZ": 0.5})),
            (1.0, LCP({"XX": 1.0})),
        ]
    )
    pauli = LCH(
        [
            (1.0, LCP({"ZI": 1.0})),
            (1.0, LCP({"IZ": 1.0})),
            (1.0, LCP({"XX": 1.0})),
        ]
    )

    grouped_bound, pauli_bound = qdrift_haar_average_trace_distance_bounds(
        target,
        total_time=0.2,
        number_of_steps=2,
        qdrift_decompositions=[grouped, pauli],
    )

    assert grouped_bound.qdrift_decomposition is grouped
    assert pauli_bound.qdrift_decomposition is pauli
    assert grouped_bound.step_time == pytest.approx(0.1)
    assert grouped_bound.tau_v == pytest.approx(1.0 / 3.0)
    assert grouped_bound.tau_v_squared == pytest.approx(10.0 / 81.0)
    assert pauli_bound.tau_v == pytest.approx(2.0 / 3.0)
    assert pauli_bound.tau_v_squared == pytest.approx(40.0 / 81.0)
    assert grouped_bound.step_second_order_bound == pytest.approx(
        0.0308113883008419
    )
    assert pauli_bound.step_second_order_bound == pytest.approx(
        0.0616227766016838
    )


def test_bound_scales_with_inverse_step_count() -> None:
    target = LCP({"X": 1.0, "Z": 1.0})
    decomposition = LCH(
        [(1.0, LCP({"X": 1.0})), (1.0, LCP({"Z": 1.0}))]
    )
    one_step = qdrift_haar_average_trace_distance_bound(
        target,
        0.1,
        1,
        decomposition,
    )
    two_steps = qdrift_haar_average_trace_distance_bound(
        target,
        0.1,
        2,
        decomposition,
    )

    assert two_steps.tau_v == one_step.tau_v
    assert two_steps.tau_v_squared == one_step.tau_v_squared
    assert two_steps.step_second_order_bound == pytest.approx(
        one_step.step_second_order_bound / 4.0
    )
    assert two_steps.step_taylor_remainder_bound == pytest.approx(
        one_step.step_taylor_remainder_bound / 8.0
    )
