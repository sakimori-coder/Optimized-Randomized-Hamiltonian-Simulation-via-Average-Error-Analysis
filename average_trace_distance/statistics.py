"""Summary statistics for sampled trace distances."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TraceDistanceStatistics:
    """Finite-sample estimate of a mean trace distance."""

    mean: float
    sample_standard_deviation: float
    standard_error: float
    values: NDArray[np.float64]

    @classmethod
    def from_values(
        cls,
        values: NDArray[np.float64],
    ) -> TraceDistanceStatistics:
        """Calculate sample statistics from individual trace distances."""
        values = np.array(values, dtype=float, copy=True)
        values.setflags(write=False)
        sample_standard_deviation = (
            0.0
            if values.size == 1
            else float(np.std(values, ddof=1))
        )
        return cls(
            mean=float(np.mean(values)),
            sample_standard_deviation=sample_standard_deviation,
            standard_error=(
                sample_standard_deviation / math.sqrt(values.size)
            ),
            values=values,
        )
