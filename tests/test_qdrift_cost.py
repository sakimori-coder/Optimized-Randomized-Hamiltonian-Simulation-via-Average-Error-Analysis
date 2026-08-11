import numpy as np
import pytest

from operators import LCH, LCP
from qdrift_cost import qdrift_cost


def _pauli(pauli: str) -> LCP:
    return LCP({pauli: 1.0})


def test_variance_method_controls_qdrift_cost() -> None:
    hamiltonian = LCH(
        [
            (1.0, _pauli("X"), 10.0),
            (1.0, _pauli("Z"), 20.0),
        ]
    )

    exact = qdrift_cost(
        hamiltonian,
        time=1.0,
        epsilon=0.1,
        variance_method="exact",
    )
    contraction = qdrift_cost(
        hamiltonian,
        time=1.0,
        epsilon=0.1,
        variance_method="contraction_bound",
    )

    assert exact.normalized_variance_bound == pytest.approx(0.5)
    assert exact.variance_constant == pytest.approx(2.0)
    assert exact.steps == 40
    assert contraction.normalized_variance_bound == pytest.approx(1.0)
    assert contraction.variance_constant == pytest.approx(4.0)
    assert contraction.steps == 80
    assert exact.sampling_probabilities == pytest.approx([0.5, 0.5])
    assert exact.per_step_cost == pytest.approx(15.0)
    assert exact.total_cost == pytest.approx(600.0)
    assert exact.one_step_second_order_bound == pytest.approx(2.0)
    assert exact.one_step_certified_upper_bound == pytest.approx(1.0)


def test_accepts_a_custom_variance_estimator() -> None:
    hamiltonian = LCH(
        [(2.0, _pauli("X"), 1.0), (1.0, _pauli("Z"), 4.0)]
    )

    def custom_variance(_decomposition) -> float:
        return 0.25

    result = qdrift_cost(
        hamiltonian,
        time=2.0,
        epsilon=0.5,
        variance_method=custom_variance,
    )

    assert result.variance_method == "custom_variance"
    assert result.lambda_sum == pytest.approx(3.0)
    assert result.normalized_variance_bound == pytest.approx(0.25)
    assert result.variance_constant == pytest.approx(2.25)
    assert result.steps == 36
    assert result.per_step_cost == pytest.approx(2.0)


def test_nonzero_single_term_uses_one_step_when_variance_is_zero() -> None:
    hamiltonian = LCH([(2.5, _pauli("Y"), 3.0)])

    result = qdrift_cost(
        hamiltonian,
        time=1.0,
        epsilon=0.1,
        variance_method="exact",
    )

    assert result.normalized_variance_bound == 0.0
    assert result.variance_constant == 0.0
    assert result.steps == 1
    assert result.total_cost == pytest.approx(3.0)


def test_zero_hamiltonian_has_zero_cost() -> None:
    hamiltonian = LCH([(0.0, _pauli("X"), 10.0)], num_qubits=1)

    result = qdrift_cost(hamiltonian, time=1.0, epsilon=0.1)

    assert result.lambda_sum == 0.0
    assert result.normalized_variance_bound == 0.0
    assert result.variance_constant == 0.0
    assert result.steps == 0
    assert result.sampling_probabilities == pytest.approx([0.0])
    assert result.total_cost == 0.0


def test_matrix_free_variance_method_does_not_materialize_paulis(
    monkeypatch,
) -> None:
    hamiltonian = LCH(
        [
            (1.0, _pauli("X" * 1000), 1.0),
            (-0.5, _pauli("Z" * 1000), 2.0),
        ]
    )

    def fail_if_materialized(*_args, **_kwargs) -> None:
        pytest.fail("matrix-free qDRIFT cost must not materialize Pauli matrices")

    monkeypatch.setattr(LCP, "to_csr", fail_if_materialized)
    monkeypatch.setattr(LCP, "to_matrix", fail_if_materialized)

    result = qdrift_cost(
        hamiltonian,
        time=1.0,
        epsilon=0.1,
        variance_method="sdp_bound",
    )

    assert np.isfinite(result.normalized_variance_bound)
    assert result.steps > 0


def test_time_zero_has_zero_steps() -> None:
    result = qdrift_cost(
        LCH([(1.0, _pauli("X"), 2.0), (1.0, _pauli("Z"), 3.0)]),
        time=0.0,
        epsilon=0.1,
        variance_method="contraction_bound",
    )

    assert result.steps == 0
    assert result.normalized_variance_bound == 1.0
    assert result.variance_constant == pytest.approx(4.0)
    assert result.total_cost == 0.0


def test_zero_coefficient_term_is_excluded_from_sampling_cost() -> None:
    hamiltonian = LCH(
        [(1.0, _pauli("X"), 2.0), (0.0, _pauli("Z"), 1e9)]
    )

    result = qdrift_cost(
        hamiltonian,
        time=1.0,
        epsilon=0.1,
        variance_method="exact",
    )

    assert result.sampling_probabilities == pytest.approx([1.0, 0.0])
    assert result.per_step_cost == pytest.approx(2.0)


def test_zero_variance_avoids_overflow_for_large_coefficient() -> None:
    result = qdrift_cost(
        LCH([(1e200, _pauli("X"), 1.0)]),
        time=1.0,
        epsilon=0.1,
        variance_method="exact",
    )

    assert result.variance_constant == 0.0
    assert result.steps == 1


@pytest.mark.parametrize(
    ("coefficient", "time"),
    [(1e-200, 1e200), (1e200, 1e-200)],
)
def test_steps_combine_lambda_and_time_before_squaring(
    coefficient: float,
    time: float,
) -> None:
    result = qdrift_cost(
        LCH([(coefficient, _pauli("X"), 1.0)]),
        time=time,
        epsilon=0.1,
        variance_method="contraction_bound",
    )

    assert result.steps == 20


@pytest.mark.parametrize("epsilon", [0.0, -1.0, np.inf, np.nan])
def test_rejects_invalid_epsilon(epsilon: float) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        qdrift_cost(
            LCH([(1.0, _pauli("X"), 1.0)]),
            time=1.0,
            epsilon=epsilon,
        )


def test_rejects_unknown_variance_method() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        qdrift_cost(
            LCH([(1.0, _pauli("X"), 1.0)]),
            time=1.0,
            epsilon=0.1,
            variance_method="unknown",  # type: ignore[arg-type]
        )
