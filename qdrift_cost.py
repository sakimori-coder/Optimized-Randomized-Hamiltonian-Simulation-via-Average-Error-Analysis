r"""qDRIFT cost calculation for an LCH."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from diamond_distance import (
    VarianceMethod,
    qdrift_diamond_distance_bound,
)
from operators import LCH


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
    one_step_second_order_bound: float
    one_step_taylor_remainder_bound: float
    one_step_certified_upper_bound: float

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
    follow the LCH term order; zero-coefficient terms have probability zero.
    """
    if not isinstance(epsilon, Real) or isinstance(epsilon, bool):
        raise TypeError("epsilon must be a real number")

    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")

    one_step_bound = qdrift_diamond_distance_bound(
        hamiltonian.lcp,
        time,
        1,
        hamiltonian,
        variance_method=variance_method,
    )
    time = one_step_bound.total_time
    evolution_costs = np.asarray(hamiltonian.evolution_costs, dtype=float)
    if len(evolution_costs) > 0:
        per_step_cost = float(
            np.dot(
                one_step_bound.sampling_probabilities,
                evolution_costs,
            )
        )
    else:
        per_step_cost = 0.0

    if one_step_bound.lambda_sum == 0.0 or time == 0.0:
        steps = 0
    elif one_step_bound.normalized_variance_bound == 0.0:
        steps = 1
    else:
        steps = max(
            1,
            math.ceil(
                2.0
                * one_step_bound.step_second_order_bound
                / epsilon
            ),
        )

    return QDriftCost(
        steps=steps,
        time=time,
        epsilon=epsilon,
        variance_method=one_step_bound.variance_method,
        lambda_sum=one_step_bound.lambda_sum,
        normalized_variance_bound=one_step_bound.normalized_variance_bound,
        variance_constant=one_step_bound.variance_constant,
        sampling_probabilities=one_step_bound.sampling_probabilities,
        per_step_cost=per_step_cost,
        one_step_second_order_bound=one_step_bound.step_second_order_bound,
        one_step_taylor_remainder_bound=(
            one_step_bound.step_taylor_remainder_bound
        ),
        one_step_certified_upper_bound=(
            one_step_bound.step_diamond_distance_upper_bound
        ),
    )
