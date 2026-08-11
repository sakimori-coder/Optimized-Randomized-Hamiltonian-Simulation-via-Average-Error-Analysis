r"""Variance-based Haar-average trace-distance bounds for qDRIFT."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from diamond_distance.variance import (
    lch_centered_second_moment_pauli_coefficients,
)
from operators import LCH, LCP


@dataclass(frozen=True)
class QDriftHaarAverageTraceDistanceBound:
    r"""Haar-average error bound for one qDRIFT time step.

    The qDRIFT LCH is assumed to be a decomposition of
    ``target_hamiltonian``.  This experiment-level assumption is documented
    rather than checked at runtime.

    For the normalized centered second moment

    ``V = sum_j p_j (target_hamiltonian / Lambda - H_j)**2``,

    ``normalized_average_variance_bound`` is

    ``C = (tau(V) + sqrt(tau(V**2))) / 2``.

    All three error fields are local errors for
    ``step_time = total_time / number_of_steps``.  They are not multiplied by
    ``number_of_steps``.
    """

    target_hamiltonian: LCP
    qdrift_decomposition: LCH
    total_time: float
    number_of_steps: int
    step_time: float
    lambda_sum: float
    sampling_probabilities: NDArray[np.float64]
    tau_v: float
    tau_v_squared: float
    normalized_average_variance_bound: float
    step_second_order_bound: float
    step_taylor_remainder_bound: float
    step_average_trace_distance_upper_bound: float


def qdrift_haar_average_trace_distance_bound(
    target_hamiltonian: LCP,
    total_time: float,
    number_of_steps: int,
    qdrift_decomposition: LCH,
) -> QDriftHaarAverageTraceDistanceBound:
    r"""Bound one qDRIFT step of duration ``total_time / number_of_steps``.

    With ``delta = total_time / number_of_steps``, the second-order bound is

    ``(Lambda * delta)**2 * (tau(V) + sqrt(tau(V**2))) / 2``.

    Here ``qdrift_decomposition`` must satisfy

    ``qdrift_decomposition.lcp == target_hamiltonian``.

    Without this assumption, a first-order representation error is present
    and the centered variance does not describe the full error.  The returned
    finite-time bound adds a Pauli-coefficient-one-norm Taylor remainder of
    order ``abs(delta)**3`` and is clipped at one.  No factor of
    ``number_of_steps`` is applied.
    """
    total_time = float(total_time)
    if number_of_steps <= 0:
        raise ValueError("number_of_steps must be positive")
    step_time = total_time / number_of_steps

    lambda_sum = qdrift_decomposition.coefficient_one_norm()
    probabilities = qdrift_decomposition.sampling_probabilities()
    variance_coefficients = (
        lch_centered_second_moment_pauli_coefficients(
            qdrift_decomposition
        )
    )

    identity = "I" * qdrift_decomposition.num_qubits
    tau_v = max(
        0.0,
        float(variance_coefficients.get(identity, 0.0j).real),
    )
    tau_v_squared = math.fsum(
        float(abs(coefficient) ** 2)
        for coefficient in variance_coefficients.values()
    )
    normalized_bound = 0.5 * (
        tau_v + math.sqrt(tau_v_squared)
    )

    if not variance_coefficients:
        step_second_order_bound = 0.0
        step_remainder_bound = 0.0
    else:
        scaled_time_step = lambda_sum * abs(step_time)
        step_second_order_bound = float(
            scaled_time_step * scaled_time_step * normalized_bound
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

    return QDriftHaarAverageTraceDistanceBound(
        target_hamiltonian=target_hamiltonian,
        qdrift_decomposition=qdrift_decomposition,
        total_time=total_time,
        number_of_steps=number_of_steps,
        step_time=step_time,
        lambda_sum=lambda_sum,
        sampling_probabilities=probabilities,
        tau_v=tau_v,
        tau_v_squared=tau_v_squared,
        normalized_average_variance_bound=normalized_bound,
        step_second_order_bound=step_second_order_bound,
        step_taylor_remainder_bound=step_remainder_bound,
        step_average_trace_distance_upper_bound=min(
            1.0,
            step_second_order_bound + step_remainder_bound,
        ),
    )


def qdrift_haar_average_trace_distance_bounds(
    target_hamiltonian: LCP,
    total_time: float,
    number_of_steps: int,
    qdrift_decompositions: Sequence[LCH],
) -> list[QDriftHaarAverageTraceDistanceBound]:
    r"""Return one bound per qDRIFT decomposition in input order.

    Every LCH must be an exact decomposition of ``target_hamiltonian``.  Each
    result bounds one step of duration ``total_time / number_of_steps``, not
    the accumulated error after all steps.
    """
    return [
        qdrift_haar_average_trace_distance_bound(
            target_hamiltonian,
            total_time,
            number_of_steps,
            qdrift_decomposition,
        )
        for qdrift_decomposition in qdrift_decompositions
    ]
