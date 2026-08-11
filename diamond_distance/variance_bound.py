r"""Variance-based diamond-distance bound for one qDRIFT decomposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from operators import LCH, LCP
from .variance import (
    BuiltInMethod as VarianceMethodName,
    VarianceEstimator,
    estimate_lch_centered_second_moment_norm,
)


VarianceMethod = VarianceMethodName | VarianceEstimator


@dataclass(frozen=True)
class QDriftDiamondDistanceBound:
    r"""Worst-case error bound for one qDRIFT decomposition.

    The distance is ``0.5 * ||E_qdrift - U_ideal||_diamond`` and therefore
    lies in ``[0, 1]``.  The qDRIFT LCH is assumed to be a decomposition of
    ``target_hamiltonian``; this experiment-level assumption is not checked.

    ``normalized_variance_bound`` is ``B`` and ``variance_constant`` is
    ``Lambda**2 * B``.  All three error fields are local errors for the
    time step ``step_time = total_time / number_of_steps``; they are not
    multiplied by ``number_of_steps``.
    """

    target_hamiltonian: LCP
    qdrift_decomposition: LCH
    total_time: float
    number_of_steps: int
    step_time: float
    variance_method: str
    lambda_sum: float
    normalized_variance_bound: float
    variance_constant: float
    sampling_probabilities: NDArray[np.float64]
    step_second_order_bound: float
    step_taylor_remainder_bound: float
    step_diamond_distance_upper_bound: float


def qdrift_diamond_distance_bound(
    target_hamiltonian: LCP,
    total_time: float,
    number_of_steps: int,
    qdrift_decomposition: LCH,
    *,
    variance_method: VarianceMethod = "exact",
) -> QDriftDiamondDistanceBound:
    r"""Bound one qDRIFT step of duration ``total_time / number_of_steps``.

    For

    ``B >= ||sum_j p_j (target_hamiltonian / Lambda - H_j)^2||_op``,

    with ``delta = total_time / number_of_steps``, the step error's
    second-order term is

    ``(Lambda * delta)**2 * B``.

    Here ``qdrift_decomposition`` must satisfy

    ``qdrift_decomposition.lcp == target_hamiltonian``.

    This mathematical assumption is not checked at runtime.  Without it,
    there is a first-order representation error that the centered variance
    does not capture.  The finite-time bound adds a
    Pauli-coefficient-one-norm Taylor remainder proportional to
    ``abs(delta)**3`` and is clipped at one.  No factor of
    ``number_of_steps`` is applied: the returned distance is between the
    ideal and qDRIFT channels for this single time step.
    """
    total_time = float(total_time)
    if number_of_steps <= 0:
        raise ValueError("number_of_steps must be positive")
    step_time = total_time / number_of_steps

    lambda_sum = qdrift_decomposition.coefficient_one_norm()
    probabilities = qdrift_decomposition.sampling_probabilities()

    normalized_variance_bound = estimate_lch_centered_second_moment_norm(
        qdrift_decomposition,
        method=variance_method,
    )
    variance_constant = (
        0.0
        if normalized_variance_bound == 0.0
        else float(lambda_sum * lambda_sum * normalized_variance_bound)
    )
    if normalized_variance_bound == 0.0:
        step_second_order_bound = 0.0
        step_remainder_bound = 0.0
    else:
        scaled_time_step = lambda_sum * abs(step_time)
        step_second_order_bound = float(
            scaled_time_step
            * scaled_time_step
            * normalized_variance_bound
        )
        sample_norm_bounds = np.asarray(
            [
                operator.coefficient_one_norm()
                for _, operator in qdrift_decomposition.terms
            ],
            dtype=float,
        )
        mean_cubed_norm_bound = float(
            np.dot(probabilities, sample_norm_bounds**3)
        )
        target_norm_bound = (
            target_hamiltonian.coefficient_one_norm() / lambda_sum
        )
        step_remainder_bound = float(
            (2.0 / 3.0)
            * scaled_time_step
            * scaled_time_step
            * scaled_time_step
            * (mean_cubed_norm_bound + target_norm_bound**3)
        )

    method_name = (
        variance_method
        if isinstance(variance_method, str)
        else getattr(variance_method, "__name__", "custom")
    )
    return QDriftDiamondDistanceBound(
        target_hamiltonian=target_hamiltonian,
        qdrift_decomposition=qdrift_decomposition,
        total_time=total_time,
        number_of_steps=number_of_steps,
        step_time=step_time,
        variance_method=method_name,
        lambda_sum=lambda_sum,
        normalized_variance_bound=normalized_variance_bound,
        variance_constant=variance_constant,
        sampling_probabilities=probabilities,
        step_second_order_bound=step_second_order_bound,
        step_taylor_remainder_bound=step_remainder_bound,
        step_diamond_distance_upper_bound=min(
            1.0,
            step_second_order_bound + step_remainder_bound,
        ),
    )
