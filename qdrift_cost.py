r"""qDRIFT cost calculation for an LCH."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from lch import LCH
from qdrift_variance_estimator import (
    BuiltInMethod as VarianceMethodName,
    VarianceEstimator,
    estimate_lch_centered_second_moment_norm,
)


VarianceMethod = VarianceMethodName | VarianceEstimator


@dataclass(frozen=True)
class QDriftCost:
    """qDRIFT step count and expected implementation cost.

    ``normalized_variance_bound`` is the selected estimate ``B_hat`` of

    ``||sum_j p_j (H / Lambda - H_j)^2||_op``.

    ``variance_constant`` is the dimensionful value ``Lambda**2 * B_hat``.
    """

    steps: int
    time: float
    epsilon: float
    variance_method: str
    lambda_sum: float
    normalized_variance_bound: float
    variance_constant: float
    sampling_probabilities: NDArray[np.float64]
    per_step_cost: float

    @property
    def total_cost(self) -> float:
        """Return the expected implementation cost over all steps."""
        return float(self.steps * self.per_step_cost)


def qdrift_cost(
    hamiltonian: LCH,
    time: float,
    epsilon: float,
    *,
    variance_method: VarianceMethod = "contraction_bound",
) -> QDriftCost:
    r"""Calculate qDRIFT cost for an LCH.

    Terms are sampled with ``p_j = |c_j| / Lambda``.  The selected variance
    estimator supplies the normalized estimate ``B_hat`` and the required
    number of steps is

    ``ceil(2 * Lambda**2 * B_hat * |time|**2 / epsilon)``.

    A nonzero evolution uses at least one step.  Per-step cost is the sampling
    average of the evolution costs stored in the LCH.  Returned probabilities
    follow the LCH term order after zero-coefficient terms are removed.
    """
    if not isinstance(hamiltonian, LCH):
        raise TypeError("hamiltonian must be an LCH")
    if not isinstance(time, Real) or isinstance(time, bool):
        raise TypeError("time must be a real number")
    if not isinstance(epsilon, Real) or isinstance(epsilon, bool):
        raise TypeError("epsilon must be a real number")

    time = float(time)
    epsilon = float(epsilon)
    if not np.isfinite(time):
        raise ValueError("time must be finite")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")

    active_terms = [
        (abs(coefficient), float(cost))
        for coefficient, _, cost in hamiltonian.lcp_terms
        if coefficient != 0
    ]
    if active_terms:
        maximum_weight = max(weight for weight, _ in active_terms)
        scaled_weights = np.asarray(
            [weight / maximum_weight for weight, _ in active_terms],
            dtype=float,
        )
        scaled_sum = math.fsum(float(weight) for weight in scaled_weights)
        probabilities = scaled_weights / scaled_sum
        lambda_sum = float(maximum_weight * scaled_sum)
        per_step_cost = float(
            np.dot(
                probabilities,
                np.asarray([cost for _, cost in active_terms], dtype=float),
            )
        )
    else:
        probabilities = np.array([], dtype=float)
        lambda_sum = 0.0
        per_step_cost = 0.0

    normalized_variance_bound = estimate_lch_centered_second_moment_norm(
        hamiltonian,
        method=variance_method,
    )
    variance_constant = (
        0.0
        if normalized_variance_bound == 0.0
        else float(lambda_sum * lambda_sum * normalized_variance_bound)
    )

    if lambda_sum == 0.0 or time == 0.0:
        steps = 0
    elif normalized_variance_bound == 0.0:
        steps = 1
    else:
        lambda_time = lambda_sum * abs(time)
        steps = max(
            1,
            math.ceil(
                2.0
                * normalized_variance_bound
                * lambda_time
                * lambda_time
                / epsilon
            ),
        )

    method_name = (
        variance_method
        if isinstance(variance_method, str)
        else getattr(variance_method, "__name__", "custom")
    )
    return QDriftCost(
        steps=steps,
        time=time,
        epsilon=epsilon,
        variance_method=method_name,
        lambda_sum=lambda_sum,
        normalized_variance_bound=normalized_variance_bound,
        variance_constant=variance_constant,
        sampling_probabilities=probabilities,
        per_step_cost=per_step_cost,
    )
