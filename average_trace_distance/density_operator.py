r"""Density operators used by average trace-distance calculations."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh


TraceDistanceMethod = Literal["low_rank", "dense"]


class DensityOperator:
    r"""Represent ``rho = sum_i p_i |psi_i><psi_i|`` as an ensemble.

    ``states[i]`` is ``|psi_i>`` and ``probabilities[i]`` is ``p_i``.
    """

    def __init__(
        self,
        states: ArrayLike,
        probabilities: ArrayLike,
    ) -> None:
        state_vectors = np.array(states, dtype=np.complex128, copy=True)
        probability_distribution = np.array(
            probabilities,
            dtype=np.float64,
            copy=True,
        )
        state_vectors.setflags(write=False)
        probability_distribution.setflags(write=False)
        self._states = state_vectors
        self._probabilities = probability_distribution

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the represented density operator."""
        dimension = self._states.shape[1]
        return dimension, dimension

    @property
    def rank_upper_bound(self) -> int:
        """Return the ensemble size, an upper bound on the rank."""
        return self._states.shape[0]

    @property
    def states(self) -> NDArray[np.complex128]:
        """Return the ensemble state vectors in input order."""
        return self._states.copy()

    @property
    def probabilities(self) -> NDArray[np.float64]:
        """Return the ensemble probability distribution."""
        return self._probabilities.copy()

    @property
    def factors(self) -> NDArray[np.complex128]:
        r"""Return ``F`` satisfying ``rho = F F^\dagger``."""
        weighted_states = (
            self._states
            * np.sqrt(self._probabilities)[:, np.newaxis]
        )
        return weighted_states.T


def trace_distance(
    left: DensityOperator,
    right: DensityOperator,
    *,
    method: TraceDistanceMethod = "low_rank",
) -> float:
    r"""Return ``0.5 * ||left - right||_1``.

    ``low_rank`` uses a reduced QR factorization.  If ``left=F F^\dagger``
    and ``right=G G^\dagger``, the nonzero eigenvalues of their difference
    are obtained from ``R diag(I, -I) R^\dagger``.  When either input is
    pure, only the relevant extremal eigenvalue is computed.  For two mixed
    inputs, every nonzero eigenvalue is computed.

    ``dense`` constructs the full difference matrix.  When either input is
    pure, it computes only the relevant extremal eigenvalue.  For two mixed
    inputs, it computes every eigenvalue.  This is useful when the total
    number of factors is at least the Hilbert-space dimension.

    """
    if not isinstance(left, DensityOperator):
        raise TypeError("left must be a DensityOperator")
    if not isinstance(right, DensityOperator):
        raise TypeError("right must be a DensityOperator")
    if left.shape != right.shape:
        raise ValueError("density operators must have equal shapes")
    if method not in ("low_rank", "dense"):
        raise ValueError("method must be 'low_rank' or 'dense'")
    if left.rank_upper_bound > right.rank_upper_bound:
        left, right = right, left

    if method == "low_rank":
        return _low_rank_trace_distance(left, right)
    return _dense_trace_distance(left, right)


def _low_rank_trace_distance(
    left: DensityOperator,
    right: DensityOperator,
) -> float:
    """Compute trace distance from the stored low-rank factors."""
    left_factors = left.factors
    right_factors = right.factors
    combined = np.column_stack((left_factors, right_factors))
    if combined.shape[1] == 0:
        return 0.0

    _, triangular = np.linalg.qr(combined, mode="reduced")
    signs = np.concatenate(
        (
            np.ones(left_factors.shape[1]),
            -np.ones(right_factors.shape[1]),
        )
    )
    reduced_difference = (
        triangular * signs[np.newaxis, :]
    ) @ triangular.conj().T
    reduced_difference = 0.5 * (
        reduced_difference + reduced_difference.conj().T
    )

    dimension = reduced_difference.shape[0]
    trace_difference = float(np.trace(reduced_difference).real)
    if left.rank_upper_bound == 1:
        maximum_eigenvalue = eigh(
            reduced_difference,
            eigvals_only=True,
            subset_by_index=[dimension - 1, dimension - 1],
            overwrite_a=True,
            check_finite=False,
        )[0]
        distance = (
            max(float(maximum_eigenvalue), 0.0)
            - 0.5 * trace_difference
        )
        return max(0.0, distance)

    eigenvalues = np.linalg.eigvalsh(reduced_difference)
    return float(0.5 * np.sum(np.abs(eigenvalues)))


def _dense_trace_distance(
    left: DensityOperator,
    right: DensityOperator,
) -> float:
    """Compute trace distance from the full dense difference matrix."""
    left_factors = left.factors
    right_factors = right.factors
    difference = (
        left_factors @ left_factors.conj().T
        - right_factors @ right_factors.conj().T
    )
    difference = 0.5 * (difference + difference.conj().T)

    dimension = difference.shape[0]
    trace_difference = float(np.trace(difference).real)
    if left.rank_upper_bound == 1:
        maximum_eigenvalue = eigh(
            difference,
            eigvals_only=True,
            subset_by_index=[dimension - 1, dimension - 1],
            overwrite_a=True,
            check_finite=False,
        )[0]
        distance = (
            max(float(maximum_eigenvalue), 0.0)
            - 0.5 * trace_difference
        )
        return max(0.0, distance)

    eigenvalues = np.linalg.eigvalsh(difference)
    return float(0.5 * np.sum(np.abs(eigenvalues)))
