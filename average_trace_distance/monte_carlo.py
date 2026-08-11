"""Data structures and Haar sampling for trace-distance experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qulacs import QuantumState

from operators import LCH, LCP

from .density_operator import TraceDistanceMethod
from .statistics import TraceDistanceStatistics


@dataclass(frozen=True)
class QDriftHaarAverageTraceDistanceEstimate:
    """Monte Carlo estimate for one qDRIFT decomposition and one time step."""

    target_hamiltonian: LCP
    qdrift_decomposition: LCH
    total_time: float
    number_of_steps: int
    step_time: float
    num_initial_states: int
    trace_distance_method: TraceDistanceMethod
    statistics: TraceDistanceStatistics

    @property
    def mean(self) -> float:
        """Return the finite-sample Haar mean."""
        return self.statistics.mean

    @property
    def sample_standard_deviation(self) -> float:
        """Return the sample standard deviation."""
        return self.statistics.sample_standard_deviation

    @property
    def standard_error(self) -> float:
        """Return the standard error of the finite-sample mean."""
        return self.statistics.standard_error

    @property
    def values(self) -> NDArray[np.float64]:
        """Return the trace distance for every sampled initial state."""
        return self.statistics.values


def haar_random_state(
    num_qubits: int,
    rng: np.random.Generator,
) -> NDArray[np.complex128]:
    """Draw one Haar-random state vector with Qulacs."""
    state = QuantumState(num_qubits)
    state.set_Haar_random_state(int(rng.integers(0, 1 << 31)))
    return np.array(state.get_vector(), dtype=np.complex128, copy=True)
