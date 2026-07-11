"""qDRIFT cost helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from lch import LCH


StepMode = Literal["theory", "variance", "hybrid"]


@dataclass(frozen=True)
class QDriftCost:
    """Result of a qDRIFT step/cost estimate."""

    steps: int
    strategy: StepMode
    time: float
    epsilon: float
    lambda_sum: float
    variance_constant: float
    theory_steps: int
    variance_steps: int
    sampling_probabilities: NDArray[np.float64]
    per_step_cost: float

    @property
    def total_cost(self) -> float:
        """Expected total implementation cost for all steps."""
        return float(self.steps * self.per_step_cost)

    @property
    def sampled_evolution_cost(self) -> float:
        """Alias for ``total_cost / steps`` when steps > 0."""
        return self.per_step_cost


def qdrift_cost(
    hamiltonian: LCH,
    time: float,
    epsilon: float,
    *,
    strategy: StepMode = "hybrid",
    epsilon_tolerance: float = 1e-12,
) -> QDriftCost:
    """Estimate a qDRIFT step count using coefficient-only sampling weights.

    The sampling weight of term ``j`` is
    ``abs(coeff_j)``.

    Two estimates are used:
    1) standard bound with ``Lambda = sum_j abs(coeff_j)``
       -> ``N >= 2*(Lambda |t|)^2 / eps``
    2) a variance-like bound from the second-order BCH term:
       ``E[A^2]-E[A]^2``, where
       ``A_j = -i * Lambda * phase_j * H_j``.

    The returned ``steps`` is:
    - ``theory``: only estimate 1
    - ``variance``: only estimate 2
    - ``hybrid``: max(estimate 1, estimate 2)
    """
    if not isinstance(hamiltonian, LCH):
        raise TypeError("hamiltonian must be an LCH")
    if not isinstance(time, (int, float, np.floating, np.integer)):
        raise TypeError("time must be a real number")
    if not isinstance(epsilon, (int, float, np.floating, np.integer)):
        raise TypeError("epsilon must be a real number")
    if epsilon <= epsilon_tolerance:
        raise ValueError("epsilon must be positive")

    num_terms = len(hamiltonian)
    if num_terms == 0:
        return QDriftCost(
            steps=0,
            strategy=strategy,
            time=float(time),
            epsilon=float(epsilon),
            lambda_sum=0.0,
            variance_constant=0.0,
            theory_steps=0,
            variance_steps=0,
            sampling_probabilities=np.array([], dtype=float),
            per_step_cost=0.0,
        )

    active_terms: list[tuple[complex, float]] | list[tuple[complex, sparse.csr_matrix, float]]
    if strategy == "theory":
        active_terms = []
        for coefficient, _, cost in hamiltonian.lcp_terms:
            weight = abs(coefficient)
            if weight <= 0.0:
                continue
            active_terms.append((coefficient, cost))
    else:
        active_terms = []
        for index in range(num_terms):
            coefficient, operator, cost = hamiltonian.get_term_with_cost(index)
            weight = abs(coefficient)
            if weight <= 0.0:
                continue
            active_terms.append((coefficient, operator, cost))

    if not active_terms:
        return QDriftCost(
            steps=0,
            strategy=strategy,
            time=float(time),
            epsilon=float(epsilon),
            lambda_sum=0.0,
            variance_constant=0.0,
            theory_steps=0,
            variance_steps=0,
            sampling_probabilities=np.array([], dtype=float),
            per_step_cost=0.0,
        )

    if strategy == "theory":
        lambda_sum = float(sum(abs(coefficient) for coefficient, _ in active_terms))
    else:
        lambda_sum = float(sum(abs(coefficient) for coefficient, _, __ in active_terms))
    probabilities = np.array(
        [
            abs(coefficient) / lambda_sum
            for coefficient, *_ in active_terms
        ],
        dtype=float,
    )

    if strategy == "theory":
        mean_generator = None
        variance_constant = 0.0
    else:
        mean_generator = sparse.csr_matrix((hamiltonian.dimension, hamiltonian.dimension), dtype=complex)
        second_moment = sparse.csr_matrix((hamiltonian.dimension, hamiltonian.dimension), dtype=complex)

        for probability, (coefficient, operator, _) in zip(probabilities, active_terms):
            phase = coefficient / abs(coefficient) if coefficient != 0 else 1.0
            generator = (-1j * lambda_sum * phase) * operator
            mean_generator += float(probability) * generator
            second_moment += float(probability) * (generator @ generator)

        variance_matrix = second_moment - (mean_generator @ mean_generator)
        variance_constant = _sparse_operator_norm(variance_matrix)
    abs_time = abs(time)
    if abs_time < epsilon_tolerance:
        steps = 0
        variance_steps = 0
        theory_steps = 0
        variance_constant = 0.0
    else:
        theory_steps = int(np.ceil(2.0 * (lambda_sum * abs_time) ** 2 / float(epsilon)))
        if strategy == "theory":
            variance_steps = 0
        else:
            variance_steps = int(np.ceil(2.0 * variance_constant * (abs_time**2) / float(epsilon)))
        if strategy == "theory":
            steps = theory_steps
        elif strategy == "variance":
            steps = max(1, variance_steps)
        elif strategy == "hybrid":
            steps = max(theory_steps, variance_steps)
        else:
            raise ValueError("strategy must be one of {'theory', 'variance', 'hybrid'}")

    valid_costs = [cost for _, cost in active_terms] if strategy == "theory" else [cost for _, _, cost in active_terms]
    per_step_cost = float(np.dot(probabilities, np.array(valid_costs, dtype=float)))

    return QDriftCost(
        steps=steps,
        strategy=strategy,
        time=float(time),
        epsilon=float(epsilon),
        lambda_sum=lambda_sum,
        variance_constant=variance_constant,
        theory_steps=theory_steps if abs_time >= epsilon_tolerance else 0,
        variance_steps=variance_steps if abs_time >= epsilon_tolerance else 0,
        sampling_probabilities=probabilities,
        per_step_cost=per_step_cost,
    )


def _sparse_operator_norm(matrix: sparse.spmatrix) -> float:
    sparse_matrix = sparse.csr_matrix(matrix)
    if sparse_matrix.shape[0] == 0:
        return 0.0
    if sparse_matrix.shape[0] <= 2:
        return float(np.linalg.norm(sparse_matrix.toarray(), ord=2))
    largest = sparse.linalg.svds(
        sparse_matrix,
        k=1,
        which="LM",
        return_singular_vectors=False,
    )[0]
    return float(abs(largest))
