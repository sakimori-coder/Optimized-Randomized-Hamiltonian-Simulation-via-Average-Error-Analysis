r"""Direct evaluation of a centered second-order qDRIFT error constant.

This module explicitly constructs

    sum_j p_j (H / Lambda - H_j)^2,

where ``p_j = |c_j| / Lambda`` and
``H_j = (c_j / |c_j|) P_j``.  It intentionally performs the sparse matrix
squares instead of bounding their norms term by term.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from lch import LCH
from lcp import LCP
from qdrift_variance_estimator import (
    QDriftDecomposition,
    estimate_centered_second_moment_norm,
    lch_qdrift_decomposition,
    qdrift_decomposition_from_samples,
)


@dataclass(frozen=True)
class BruteForceQDriftError:
    """Result of directly evaluating the centered qDRIFT matrix."""

    lambda_sum: float
    sampling_probabilities: NDArray[np.float64]
    normalized_variance_constant: float
    variance_constant: float
    evaluated_steps: int
    error_bound: float
    required_steps: int
    theory_steps: int


def brute_force_qdrift_error(
    hamiltonian: LCP | LCH,
    *,
    time: float,
    epsilon: float,
    steps: int | None = None,
) -> BruteForceQDriftError:
    r"""Directly compute ``||sum_j p_j (H/Lambda - H_j)^2||_op``.

    The dimensional second-order constant is

        ``Lambda**2 * normalized_variance_constant``.

    Consistently with :func:`qdrift_cost.qdrift_cost`, the reported error
    bound for ``N`` steps is ``2 * variance_constant * time**2 / N`` and the
    required step count is its ceiling for the requested ``epsilon``.

    Args:
        hamiltonian: Pauli linear-combination Hamiltonian.  For an ``LCH``,
            stored evolution costs do not enter this error calculation.
        time: Evolution time.
        epsilon: Target error used to calculate ``required_steps``.
        steps: Optional step count at which ``error_bound`` is evaluated.
            When omitted, ``required_steps`` is used.
    """
    if isinstance(hamiltonian, LCH):
        lch_hamiltonian = hamiltonian
    elif isinstance(hamiltonian, LCP):
        lch_hamiltonian = LCH(
            [
                (coefficient, pauli, 0.0)
                for pauli, coefficient in hamiltonian.terms.items()
            ],
            num_qubits=hamiltonian.num_qubits,
        )
    else:
        raise TypeError("hamiltonian must be an LCP or LCH")
    return _qdrift_error_from_decomposition(
        lch_qdrift_decomposition(lch_hamiltonian),
        time=time,
        epsilon=epsilon,
        steps=steps,
    )


def brute_force_qdrift_error_from_decomposition(
    weights: Sequence[float],
    sampled_hamiltonians: Sequence[sparse.spmatrix],
    *,
    time: float,
    epsilon: float,
    steps: int | None = None,
) -> BruteForceQDriftError:
    r"""Evaluate a qDRIFT decomposition ``H = sum_j weights[j] H_j``.

    All weights must be non-negative.  The sampled operators may be Pauli
    strings, normalized commuting groups, or any equally shaped matrices.
    The centered-moment estimation itself is delegated to
    :mod:`qdrift_variance_estimator`.
    """
    decomposition = qdrift_decomposition_from_samples(
        weights,
        sampled_hamiltonians,
    )
    return _qdrift_error_from_decomposition(
        decomposition,
        time=time,
        epsilon=epsilon,
        steps=steps,
    )


def _qdrift_error_from_decomposition(
    decomposition: QDriftDecomposition,
    *,
    time: float,
    epsilon: float,
    steps: int | None,
) -> BruteForceQDriftError:
    """Convert a separately estimated variance constant into an error cost."""
    if not isinstance(time, Real):
        raise TypeError("time must be a real number")
    if not isinstance(epsilon, Real):
        raise TypeError("epsilon must be a real number")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if steps is not None and (
        not isinstance(steps, Integral) or isinstance(steps, bool) or steps <= 0
    ):
        raise ValueError("steps must be a positive integer")
    lambda_sum = decomposition.lambda_sum
    abs_time = abs(float(time))

    if lambda_sum == 0.0:
        return BruteForceQDriftError(
            lambda_sum=0.0,
            sampling_probabilities=np.array([], dtype=float),
            normalized_variance_constant=0.0,
            variance_constant=0.0,
            evaluated_steps=0 if steps is None else int(steps),
            error_bound=0.0,
            required_steps=0,
            theory_steps=0,
        )

    normalized_constant = estimate_centered_second_moment_norm(
        decomposition,
        method="exact",
    )
    variance_constant = float(lambda_sum**2 * normalized_constant)
    error_prefactor = 2.0 * variance_constant * abs_time**2

    if abs_time == 0.0:
        required_steps = 0
    else:
        required_steps = max(1, int(np.ceil(error_prefactor / float(epsilon))))
    evaluated_steps = required_steps if steps is None else int(steps)
    error_bound = (
        0.0 if evaluated_steps == 0 else float(error_prefactor / evaluated_steps)
    )
    theory_steps = (
        0
        if abs_time == 0.0
        else int(np.ceil(2.0 * (lambda_sum * abs_time) ** 2 / float(epsilon)))
    )

    return BruteForceQDriftError(
        lambda_sum=lambda_sum,
        sampling_probabilities=decomposition.probabilities,
        normalized_variance_constant=normalized_constant,
        variance_constant=variance_constant,
        evaluated_steps=evaluated_steps,
        error_bound=error_bound,
        required_steps=required_steps,
        theory_steps=theory_steps,
    )


def main() -> None:
    """Run the exact variance calculation for a random Pauli Hamiltonian."""
    parser = ArgumentParser(
        description="Directly evaluate the centered second-order qDRIFT error"
    )
    parser.add_argument("--num-terms", type=int, default=6)
    parser.add_argument("--num-qubits", type=int, default=3)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Reuse the random-Hamiltonian generator used by the other examples.
    from main import build_lch_from_lcp_unit_cost, build_random_lcp

    hamiltonian = build_lch_from_lcp_unit_cost(
        build_random_lcp(
            num_terms=args.num_terms,
            num_qubits=args.num_qubits,
            seed=args.seed,
        )
    )
    result = brute_force_qdrift_error(
        hamiltonian,
        time=args.time,
        epsilon=args.epsilon,
        steps=args.steps,
    )

    print(f"lambda={result.lambda_sum:.10f}")
    print(
        "normalized_variance_constant="
        f"{result.normalized_variance_constant:.10e}"
    )
    print(f"variance_constant={result.variance_constant:.10e}")
    print(f"evaluated_steps={result.evaluated_steps}")
    print(f"error_bound={result.error_bound:.10e}")
    print(f"required_steps={result.required_steps}")
    print(f"theory_steps={result.theory_steps}")


if __name__ == "__main__":
    main()
