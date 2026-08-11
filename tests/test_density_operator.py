import numpy as np
import pytest

import average_trace_distance.density_operator as density_operator_module
from average_trace_distance.density_operator import (
    DensityOperator,
    trace_distance,
)


def _dense(operator: DensityOperator) -> np.ndarray:
    factors = operator.factors
    return factors @ factors.conj().T


def _ensemble_density(
    states: np.ndarray,
    probabilities: np.ndarray,
) -> DensityOperator:
    return DensityOperator(states.T, probabilities)


def _pure_density(state: np.ndarray) -> DensityOperator:
    return DensityOperator([state], [1.0])


def test_factors_match_weighted_outer_products() -> None:
    states = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0j],
        ],
        dtype=complex,
    )
    probabilities = np.array([0.25, 0.75])

    density = _ensemble_density(states, probabilities)
    probabilities[:] = [1.0, 0.0]

    np.testing.assert_allclose(
        _dense(density),
        np.diag([0.25, 0.75]),
        atol=1e-12,
    )
    np.testing.assert_array_equal(density.states, states.T)
    np.testing.assert_array_equal(
        density.probabilities,
        np.array([0.25, 0.75]),
    )
    assert density.rank_upper_bound == 2
    assert density.shape == (2, 2)


def test_pure_density_operator_copies_the_state() -> None:
    state = np.array([1.0, 0.0])

    density = _pure_density(state)
    state[:] = [0.0, 1.0]

    np.testing.assert_array_equal(
        _dense(density),
        np.array([[1.0, 0.0], [0.0, 0.0]]),
    )
    assert density.rank_upper_bound == 1


def test_trace_distance_matches_dense_eigenvalue_calculation() -> None:
    rng = np.random.default_rng(7)
    left_states = (
        rng.normal(size=(8, 2)) + 1j * rng.normal(size=(8, 2))
    )
    left_states /= np.linalg.norm(left_states, axis=0)
    right_state = rng.normal(size=8) + 1j * rng.normal(size=8)
    right_state /= np.linalg.norm(right_state)
    left = _ensemble_density(
        left_states,
        np.array([0.3, 0.7]),
    )
    right = _pure_density(right_state)
    dense_difference = _dense(left) - _dense(right)
    expected = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(dense_difference)))

    assert trace_distance(left, right, method="low_rank") == pytest.approx(
        expected,
        abs=1e-12,
    )
    assert trace_distance(left, right, method="dense") == pytest.approx(
        expected,
        abs=1e-12,
    )


def test_dense_uses_one_extremal_eigenvalue_for_pure_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(23)
    states = rng.normal(size=(8, 3)) + 1j * rng.normal(size=(8, 3))
    states /= np.linalg.norm(states, axis=0)
    pure_state = rng.normal(size=8) + 1j * rng.normal(size=8)
    pure_state /= np.linalg.norm(pure_state)
    mixed = _ensemble_density(
        states,
        np.array([0.2, 0.3, 0.5]),
    )
    pure = _pure_density(pure_state)
    expected = trace_distance(mixed, pure, method="low_rank")
    subsets: list[list[int]] = []
    original_eigh = density_operator_module.eigh

    def recording_eigh(*args: object, **kwargs: object):
        subsets.append(kwargs["subset_by_index"])
        return original_eigh(*args, **kwargs)

    monkeypatch.setattr(density_operator_module, "eigh", recording_eigh)

    mixed_minus_pure = trace_distance(mixed, pure, method="dense")
    pure_minus_mixed = trace_distance(pure, mixed, method="dense")

    assert subsets == [[7, 7], [7, 7]]
    assert mixed_minus_pure == pytest.approx(expected, abs=1e-12)
    assert pure_minus_mixed == pytest.approx(expected, abs=1e-12)


def test_dense_pure_branch_handles_unequal_traces() -> None:
    mixed = _ensemble_density(
        np.array([[2.0, 0.0], [0.0, 1.0]], dtype=complex),
        np.array([0.25, 0.75]),
    )
    pure = _pure_density(np.array([0.5, 0.0]))
    difference = _dense(mixed) - _dense(pure)
    expected = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(difference)))

    assert trace_distance(mixed, pure, method="dense") == pytest.approx(
        expected,
        abs=1e-12,
    )
    assert trace_distance(pure, mixed, method="dense") == pytest.approx(
        expected,
        abs=1e-12,
    )
    assert trace_distance(mixed, pure, method="low_rank") == pytest.approx(
        expected,
        abs=1e-12,
    )
    assert trace_distance(pure, mixed, method="low_rank") == pytest.approx(
        expected,
        abs=1e-12,
    )


def test_orthogonal_pure_states_have_unit_trace_distance() -> None:
    zero = _pure_density(np.array([1.0, 0.0]))
    one = _pure_density(np.array([0.0, 1.0]))

    assert trace_distance(zero, one) == pytest.approx(1.0)
    assert trace_distance(zero, zero) == pytest.approx(0.0, abs=1e-15)


def test_low_rank_uses_one_extremal_eigenvalue_for_pure_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(19)
    states = rng.normal(size=(8, 3)) + 1j * rng.normal(size=(8, 3))
    states /= np.linalg.norm(states, axis=0)
    pure_state = rng.normal(size=8) + 1j * rng.normal(size=8)
    pure_state /= np.linalg.norm(pure_state)
    mixed = _ensemble_density(
        states,
        np.array([0.2, 0.3, 0.5]),
    )
    pure = _pure_density(pure_state)
    expected = trace_distance(mixed, pure, method="dense")
    subsets: list[list[int]] = []
    original_eigh = density_operator_module.eigh

    def recording_eigh(*args: object, **kwargs: object):
        subsets.append(kwargs["subset_by_index"])
        return original_eigh(*args, **kwargs)

    monkeypatch.setattr(density_operator_module, "eigh", recording_eigh)

    mixed_minus_pure = trace_distance(mixed, pure, method="low_rank")
    pure_minus_mixed = trace_distance(pure, mixed, method="low_rank")

    assert subsets == [[3, 3], [3, 3]]
    assert mixed_minus_pure == pytest.approx(expected, abs=1e-12)
    assert pure_minus_mixed == pytest.approx(expected, abs=1e-12)
