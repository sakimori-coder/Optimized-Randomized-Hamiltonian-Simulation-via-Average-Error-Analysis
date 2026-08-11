r"""Compare several qDRIFT decompositions on shared Haar-random states."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from operators import LCH, LCP

from .density_operator import TraceDistanceMethod, trace_distance
from .ideal_time_evolution import ideal_time_evolution
from .monte_carlo import (
    QDriftHaarAverageTraceDistanceEstimate,
    haar_random_state,
)
from .qdrift_channel import prepare_qdrift_channel
from .statistics import TraceDistanceStatistics


def estimate_qdrift_haar_average_trace_distances(
    target_hamiltonian: LCP,
    total_time: float,
    number_of_steps: int,
    qdrift_decompositions: Sequence[LCH],
    *,
    num_initial_states: int = 100,
    trace_distance_method: TraceDistanceMethod = "low_rank",
    seed: int | None = None,
) -> list[QDriftHaarAverageTraceDistanceEstimate]:
    r"""Estimate every decomposition's one-step Haar-average error.

    The step duration is ``step_time = total_time / number_of_steps``.  For
    each sampled initial state, the ideal state is evolved once for this step
    and compared with every qDRIFT channel.  Consequently all decompositions
    use exactly the same Haar states and ideal evolutions, enabling paired
    comparisons.

    Every LCH is assumed to be an exact decomposition of
    ``target_hamiltonian``::

        qdrift_decomposition.lcp == target_hamiltonian

    This experiment-level assumption is documented rather than checked at
    runtime.  Each returned result describes one step; the channel error is
    not accumulated over ``number_of_steps``.
    """
    if number_of_steps <= 0:
        raise ValueError("number_of_steps must be positive")

    total_time = float(total_time)
    step_time = total_time / number_of_steps
    decompositions = tuple(qdrift_decompositions)
    channels = tuple(
        prepare_qdrift_channel(decomposition, step_time)
        for decomposition in decompositions
    )
    distances = np.empty(
        (len(decompositions), num_initial_states),
        dtype=float,
    )
    rng = np.random.default_rng(seed)

    for sample_index in range(num_initial_states):
        initial_state = haar_random_state(target_hamiltonian.num_qubits, rng)
        ideal_density = ideal_time_evolution(
            target_hamiltonian,
            step_time,
            initial_state,
        )

        for decomposition_index, channel in enumerate(channels):
            qdrift_density = channel.output(initial_state)
            distances[decomposition_index, sample_index] = trace_distance(
                qdrift_density,
                ideal_density,
                method=trace_distance_method,
            )

    return [
        QDriftHaarAverageTraceDistanceEstimate(
            target_hamiltonian=target_hamiltonian,
            qdrift_decomposition=decomposition,
            total_time=total_time,
            number_of_steps=number_of_steps,
            step_time=step_time,
            num_initial_states=num_initial_states,
            trace_distance_method=trace_distance_method,
            statistics=TraceDistanceStatistics.from_values(
                distances[decomposition_index]
            ),
        )
        for decomposition_index, decomposition in enumerate(decompositions)
    ]
