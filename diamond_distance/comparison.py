r"""Evaluate several qDRIFT decompositions against one target evolution."""

from __future__ import annotations

from collections.abc import Sequence

from operators import LCH, LCP

from .variance_bound import (
    QDriftDiamondDistanceBound,
    VarianceMethod,
    qdrift_diamond_distance_bound,
)


def qdrift_diamond_distance_bounds(
    target_hamiltonian: LCP,
    total_time: float,
    number_of_steps: int,
    qdrift_decompositions: Sequence[LCH],
    *,
    variance_method: VarianceMethod = "exact",
) -> list[QDriftDiamondDistanceBound]:
    r"""Return one bound per qDRIFT decomposition, preserving input order.

    Every LCH must be an exact decomposition of ``target_hamiltonian``:

    ``qdrift_decomposition.lcp == target_hamiltonian``.

    This assumption is deliberately documented rather than checked at
    runtime so that the numerical experiment remains explicit and readable.
    Each result bounds one step of duration
    ``total_time / number_of_steps``; it is not the accumulated error of all
    steps.
    """
    return [
        qdrift_diamond_distance_bound(
            target_hamiltonian,
            total_time,
            number_of_steps,
            qdrift_decomposition,
            variance_method=variance_method,
        )
        for qdrift_decomposition in qdrift_decompositions
    ]
