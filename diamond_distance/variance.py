r"""Estimate the normalized centered second moment used by qDRIFT bounds.

The target quantity is

    || sum_j p_j (H / Lambda - H_j)^2 ||_op,

independently of evolution time, target error, step count, and gate cost.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from operators import LCH, LCP


@dataclass(frozen=True)
class QDriftDecomposition:
    """Normalized qDRIFT sampling data for ``H = Lambda E[H_j]``."""

    lambda_sum: float
    probabilities: NDArray[np.float64]
    sampled_hamiltonians: tuple[sparse.csr_matrix, ...]
    normalized_hamiltonian: sparse.csr_matrix | None


BuiltInMethod = Literal[
    "exact",
    "contraction_bound",
    "pauli_l1_bound",
    "anticommuting_bound",
    "sdp_bound",
]
VarianceEstimator = Callable[[QDriftDecomposition], float]


def lch_qdrift_decomposition(hamiltonian: LCH) -> QDriftDecomposition:
    r"""Build ``p_j`` and signed LCP samples ``H_j`` from an LCH.

    For ``H = sum_j c_j A_j``, this uses
    ``p_j=|c_j|/Lambda`` and ``H_j=(c_j/|c_j|)A_j``.
    LCH evolution costs do not enter the decomposition.
    """
    if not isinstance(hamiltonian, LCH):
        raise TypeError("hamiltonian must be an LCH")

    active_terms = [
        (coefficient, operator)
        for coefficient, operator, _ in hamiltonian.lcp_terms
        if coefficient != 0
    ]
    weights = [abs(coefficient) for coefficient, _ in active_terms]
    sampled_hamiltonians = [
        (
            math.copysign(1.0, coefficient) * operator
        ).to_csr()
        for coefficient, operator in active_terms
    ]
    return qdrift_decomposition_from_samples(weights, sampled_hamiltonians)


def qdrift_decomposition_from_samples(
    weights: Sequence[float],
    sampled_hamiltonians: Sequence[sparse.spmatrix],
) -> QDriftDecomposition:
    """Build normalized sampling data from ``H=sum_j weights[j] H_j``."""
    if len(weights) != len(sampled_hamiltonians):
        raise ValueError("weights and sampled_hamiltonians must have equal length")

    active_samples: list[tuple[float, sparse.csr_matrix]] = []
    matrix_shape: tuple[int, int] | None = None
    for weight, sampled_hamiltonian in zip(weights, sampled_hamiltonians):
        if not isinstance(weight, Real):
            raise TypeError("weights must be real numbers")
        if weight < 0.0:
            raise ValueError("weights must be non-negative")
        matrix = sparse.csr_matrix(sampled_hamiltonian, dtype=complex)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("sampled Hamiltonians must be square matrices")
        if matrix_shape is None:
            matrix_shape = matrix.shape
        elif matrix.shape != matrix_shape:
            raise ValueError("sampled Hamiltonians must have equal shapes")
        if weight != 0.0:
            active_samples.append((float(weight), matrix))

    lambda_sum = float(sum(weight for weight, _ in active_samples))
    if lambda_sum == 0.0:
        return QDriftDecomposition(
            lambda_sum=0.0,
            probabilities=np.array([], dtype=float),
            sampled_hamiltonians=(),
            normalized_hamiltonian=None,
        )

    assert matrix_shape is not None
    probabilities = np.asarray(
        [weight / lambda_sum for weight, _ in active_samples],
        dtype=float,
    )
    samples = tuple(matrix for _, matrix in active_samples)
    normalized_hamiltonian = sparse.csr_matrix(matrix_shape, dtype=complex)
    for probability, sample in zip(probabilities, samples):
        normalized_hamiltonian += float(probability) * sample

    return QDriftDecomposition(
        lambda_sum=lambda_sum,
        probabilities=probabilities,
        sampled_hamiltonians=samples,
        normalized_hamiltonian=normalized_hamiltonian,
    )


def estimate_lch_centered_second_moment_norm(
    hamiltonian: LCH,
    *,
    method: BuiltInMethod | VarianceEstimator = "exact",
) -> float:
    r"""Estimate ``||sum_j p_j(H/Lambda-H_j)^2||_op`` for an LCH.

    Built-in methods:

    - ``exact``: explicitly sum each centered sparse matrix square and
      numerically compute the resulting operator norm.
    - ``contraction_bound``: return the matrix-free upper bound ``1`` for
      Hermitian contractions ``H_j``.
    - ``pauli_l1_bound``: symbolically expand the variance in the Pauli basis
      and use the coefficient one-norm after combining equal products.
    - ``anticommuting_bound``: greedily partition the symbolic non-identity
      terms into pairwise-anticommuting sets and use their exact Euclidean
      norms followed by a triangle inequality.
    - ``sdp_bound``: solve a level-1 Pauli moment SDP using Pauli-product,
      commutation, and sample-contraction constraints.

    All four matrix-free bound methods assume that every signed LCP sample is
    a Hermitian contraction.  ``LCH`` does not normalize or verify this
    precondition; use ``exact`` for a general unnormalized decomposition.

    A custom callable taking :class:`QDriftDecomposition` may be supplied to
    experiment with another estimator without changing cost code.
    """
    if not isinstance(hamiltonian, LCH):
        raise TypeError("hamiltonian must be an LCH")

    # This branch must precede matrix materialization: its purpose is to be
    # polynomial in the Pauli input size rather than exponential in qubits.
    matrix_free_methods = {
        "contraction_bound",
        "pauli_l1_bound",
        "anticommuting_bound",
        "sdp_bound",
    }
    if isinstance(method, str) and method in matrix_free_methods:
        weights, samples = _active_lch_samples(hamiltonian, method=method)
        if not weights:
            return 0.0
        if method == "contraction_bound":
            return 1.0
        if method == "pauli_l1_bound":
            return pauli_l1_bound_from_samples(weights, samples)
        if method == "anticommuting_bound":
            return anticommuting_bound_from_samples(weights, samples)
        return pauli_moment_sdp_bound_from_samples(weights, samples)

    if not callable(method) and method != "exact":
        raise ValueError(
            "unknown method; expected 'exact', 'contraction_bound', "
            "'pauli_l1_bound', 'anticommuting_bound', or 'sdp_bound'"
        )

    decomposition = lch_qdrift_decomposition(hamiltonian)
    return estimate_centered_second_moment_norm(decomposition, method=method)


def lch_centered_second_moment_pauli_coefficients(
    hamiltonian: LCH,
) -> dict[str, complex]:
    r"""Return the matrix-free Pauli expansion of the centered second moment.

    This helper exposes the shared symbolic representation so additional
    polynomial-time estimators can be prototyped without constructing
    ``2**n`` dimensional matrices.
    """
    if not isinstance(hamiltonian, LCH):
        raise TypeError("hamiltonian must be an LCH")
    weights, samples = _active_lch_samples(
        hamiltonian,
        method="symbolic Pauli expansion",
    )
    return _centered_second_moment_pauli_coefficients_from_samples(
        weights,
        samples,
    )


def pauli_l1_bound_from_samples(
    weights: Sequence[float],
    sampled_hamiltonians: Sequence[LCP],
) -> float:
    r"""Bound a Pauli-sample centered second moment by coefficient 1-norm.

    The samples define ``p_j = weights[j] / sum(weights)`` and

    ``V = sum_j p_j H_j**2 - (sum_j p_j H_j)**2``.

    ``V`` is expanded symbolically in the Pauli basis before absolute values
    are taken.  The samples are assumed to be Hermitian contractions, so the
    returned value is also capped by the contraction bound ``1``.
    """
    coefficients = _centered_second_moment_pauli_coefficients_from_samples(
        weights,
        sampled_hamiltonians,
    )
    return min(
        1.0,
        float(sum(abs(coefficient) for coefficient in coefficients.values())),
    )


def anticommuting_bound_from_samples(
    weights: Sequence[float],
    sampled_hamiltonians: Sequence[LCP],
) -> float:
    r"""Bound a Pauli-sample centered second moment by anticommuting groups.

    The centered second moment is first expanded in the Pauli basis.  Its
    non-identity terms are then greedily partitioned into pairwise-
    anticommuting sets, whose operator norms are their coefficient Euclidean
    norms.  Samples are assumed to be Hermitian contractions.
    """
    coefficients = _centered_second_moment_pauli_coefficients_from_samples(
        weights,
        sampled_hamiltonians,
    )
    if not coefficients:
        return 0.0
    return min(
        1.0,
        _anticommuting_partition_bound(
            coefficients,
            num_qubits=sampled_hamiltonians[0].num_qubits,
        ),
    )


def pauli_moment_sdp_bound_from_samples(
    weights: Sequence[float],
    sampled_hamiltonians: Sequence[LCP],
    *,
    solver: str | None = None,
) -> float:
    r"""Bound a Pauli-sample centered second moment by a level-1 SDP.

    For Pauli coefficient columns ``Z`` and sampling probabilities ``p``, the
    coefficient covariance is

    ``C = Z @ (diag(p) - outer(p, p)) @ Z.T``.

    The SDP maximizes its contraction against a symmetrized Pauli moment
    matrix.  Equal Pauli products share one moment, anticommuting entries
    vanish, and every sample ``A_j`` obeys ``<A_j**2> <= 1``.  Samples are
    therefore assumed to be Hermitian contractions.

    The numerical solver is applied to the dual Pauli sum-of-squares problem.
    Its Gram matrix and non-negative multipliers are repaired after solving;
    any remaining Pauli coefficient mismatch is added through a coefficient
    one-norm bound.  No ``2**n`` dimensional Hamiltonian matrix is built.
    """
    probabilities, paulis, sample_coefficients = _pauli_sample_matrix(
        weights,
        sampled_hamiltonians,
    )
    if probabilities.size == 0:
        return 0.0

    product_paulis, product_map = _symmetrized_pauli_product_map(paulis)
    mean = sample_coefficients @ probabilities
    centered = sample_coefficients - mean[:, np.newaxis]
    covariance = (centered * probabilities[np.newaxis, :]) @ centered.T
    covariance = 0.5 * (covariance + covariance.T)
    variance_coefficients = np.asarray(
        product_map @ covariance.reshape(-1),
        dtype=float,
    ).reshape(-1)
    if not np.any(variance_coefficients):
        return 0.0

    identity = "I" * len(paulis[0])
    identity_coefficients = np.zeros(len(product_paulis), dtype=float)
    identity_coefficients[product_paulis.index(identity)] = 1.0
    sample_square_coefficients = np.column_stack(
        [
            np.asarray(
                product_map @ np.outer(column, column).reshape(-1),
                dtype=float,
            ).reshape(-1)
            for column in sample_coefficients.T
        ]
    )
    contraction_coefficients = (
        identity_coefficients[:, np.newaxis] - sample_square_coefficients
    )

    coefficient_l1_bound = float(
        np.nextafter(
            math.fsum(abs(float(value)) for value in variance_coefficients),
            np.inf,
        )
    )
    probability_only_bound = _probability_only_sdp_bound(probabilities)
    fallback_bound = min(1.0, probability_only_bound, coefficient_l1_bound)
    dual_bound = _solve_pauli_moment_sdp_dual(
        product_map=product_map,
        variance_coefficients=variance_coefficients,
        contraction_coefficients=contraction_coefficients,
        identity_coefficients=identity_coefficients,
        solver=solver,
    )
    if dual_bound is None:
        return float(fallback_bound)
    return float(min(fallback_bound, dual_bound))


def estimate_centered_second_moment_norm(
    decomposition: QDriftDecomposition,
    *,
    method: BuiltInMethod | VarianceEstimator = "exact",
) -> float:
    """Estimate the centered second-moment norm of a prepared decomposition."""
    if not isinstance(decomposition, QDriftDecomposition):
        raise TypeError("decomposition must be a QDriftDecomposition")
    if callable(method):
        value = float(method(decomposition))
    elif method == "exact":
        value = _exact_estimator(decomposition)
    elif method == "contraction_bound":
        # The prepared decomposition is assumed to consist of Hermitian
        # contractions.  This precondition cannot in general be verified
        # without computing operator norms.
        value = 1.0 if decomposition.lambda_sum != 0.0 else 0.0
    elif method in {"pauli_l1_bound", "anticommuting_bound", "sdp_bound"}:
        sample_functions = {
            "pauli_l1_bound": "pauli_l1_bound_from_samples",
            "anticommuting_bound": "anticommuting_bound_from_samples",
            "sdp_bound": "pauli_moment_sdp_bound_from_samples",
        }
        raise ValueError(
            f"{method} requires Pauli metadata; use the LCH entry point or "
            f"{sample_functions[method]}"
        )
    else:
        raise ValueError(
            "unknown method; expected 'exact', 'contraction_bound', "
            "'pauli_l1_bound', 'anticommuting_bound', or 'sdp_bound'"
        )
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("variance estimator must return a finite non-negative value")
    return value


def _exact_estimator(decomposition: QDriftDecomposition) -> float:
    if decomposition.normalized_hamiltonian is None:
        return 0.0
    centered_second_moment = sparse.csr_matrix(
        decomposition.normalized_hamiltonian.shape,
        dtype=complex,
    )
    for probability, sample in zip(
        decomposition.probabilities,
        decomposition.sampled_hamiltonians,
    ):
        centered = decomposition.normalized_hamiltonian - sample
        centered_second_moment += float(probability) * (centered @ centered)
    return sparse_operator_norm(centered_second_moment)


def _active_lch_samples(
    hamiltonian: LCH,
    *,
    method: object,
) -> tuple[list[float], list[LCP]]:
    """Return qDRIFT weights and signed LCP samples without matrices."""
    weights: list[float] = []
    samples: list[LCP] = []
    for coefficient, operator, _ in hamiltonian.lcp_terms:
        if coefficient == 0:
            continue
        if not np.isfinite(coefficient):
            raise ValueError(f"{method} requires finite LCH coefficients")
        weights.append(abs(coefficient))
        samples.append(math.copysign(1.0, coefficient) * operator)
    return weights, samples


def _centered_second_moment_pauli_coefficients_from_samples(
    weights: Sequence[float],
    sampled_hamiltonians: Sequence[LCP],
) -> dict[str, complex]:
    """Return the Pauli coefficients of a general centered second moment."""
    if len(weights) != len(sampled_hamiltonians):
        raise ValueError("weights and sampled_hamiltonians must have equal length")

    num_qubits: int | None = None
    active_samples: list[tuple[float, list[tuple[str, float]]]] = []
    for weight, sampled_hamiltonian in zip(weights, sampled_hamiltonians):
        if not isinstance(weight, Real):
            raise TypeError("weights must be real numbers")
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("weights must be finite and non-negative")
        if not isinstance(sampled_hamiltonian, LCP):
            raise TypeError("sampled_hamiltonians must contain LCP instances")
        if num_qubits is None:
            num_qubits = sampled_hamiltonian.num_qubits
        elif sampled_hamiltonian.num_qubits != num_qubits:
            raise ValueError("sampled Hamiltonians must act on equal qubit counts")

        real_terms: list[tuple[str, float]] = []
        for pauli, coefficient in sampled_hamiltonian.terms.items():
            if not np.isfinite(coefficient.real) or not np.isfinite(
                coefficient.imag
            ):
                raise ValueError("sampled Hamiltonian coefficients must be finite")
            if coefficient.imag != 0.0:
                raise ValueError("sampled Hamiltonian coefficients must be real")
            if coefficient != 0.0:
                real_terms.append((pauli, float(coefficient.real)))
        if weight != 0.0:
            active_samples.append((float(weight), real_terms))

    if not active_samples:
        return {}

    maximum_weight = max(weight for weight, _ in active_samples)
    scaled_weights = [weight / maximum_weight for weight, _ in active_samples]
    scaled_sum = math.fsum(scaled_weights)
    probabilities = [weight / scaled_sum for weight in scaled_weights]
    assert num_qubits is not None
    identity = "I" * num_qubits

    mean_coefficients: dict[str, complex] = {}
    variance_coefficients: dict[str, complex] = {}
    for probability, (_, terms) in zip(probabilities, active_samples):
        for pauli, coefficient in terms:
            _accumulate_pauli_coefficient(
                mean_coefficients,
                pauli,
                probability * coefficient,
            )
        _accumulate_pauli_square(
            variance_coefficients,
            terms,
            scale=probability,
            identity=identity,
        )

    _accumulate_pauli_square(
        variance_coefficients,
        list(mean_coefficients.items()),
        scale=-1.0,
        identity=identity,
    )
    return variance_coefficients


def _accumulate_pauli_square(
    output: dict[str, complex],
    terms: Sequence[tuple[str, complex]],
    *,
    scale: float,
    identity: str,
) -> None:
    """Accumulate ``scale * (sum_j terms[j])**2`` into ``output``."""
    for index, (left_pauli, left_coefficient) in enumerate(terms):
        _accumulate_pauli_coefficient(
            output,
            identity,
            scale * left_coefficient**2,
        )
        for right_pauli, right_coefficient in terms[index + 1 :]:
            if not LCP._pauli_strings_commute(left_pauli, right_pauli):
                continue
            phase, product = _multiply_pauli_strings(left_pauli, right_pauli)
            _accumulate_pauli_coefficient(
                output,
                product,
                scale * 2.0 * left_coefficient * right_coefficient * phase,
            )


def _multiply_pauli_strings(left: str, right: str) -> tuple[complex, str]:
    """Return ``phase, product`` satisfying ``left * right = phase * product``."""
    if len(left) != len(right):
        raise ValueError("Pauli strings must have equal lengths")
    multiplication_table: dict[tuple[str, str], tuple[complex, str]] = {
        ("X", "Y"): (1.0j, "Z"),
        ("Y", "X"): (-1.0j, "Z"),
        ("Y", "Z"): (1.0j, "X"),
        ("Z", "Y"): (-1.0j, "X"),
        ("Z", "X"): (1.0j, "Y"),
        ("X", "Z"): (-1.0j, "Y"),
    }
    phase = 1.0 + 0.0j
    product_symbols: list[str] = []
    for left_symbol, right_symbol in zip(left, right):
        if left_symbol == "I":
            product_symbols.append(right_symbol)
        elif right_symbol == "I":
            product_symbols.append(left_symbol)
        elif left_symbol == right_symbol:
            product_symbols.append("I")
        else:
            local_phase, product_symbol = multiplication_table[
                (left_symbol, right_symbol)
            ]
            phase *= local_phase
            product_symbols.append(product_symbol)
    return phase, "".join(product_symbols)


def _accumulate_pauli_coefficient(
    coefficients: dict[str, complex],
    pauli: str,
    value: complex,
) -> None:
    updated = coefficients.get(pauli, 0.0j) + value
    if updated == 0.0:
        coefficients.pop(pauli, None)
    else:
        coefficients[pauli] = updated


def _anticommuting_partition_bound(
    coefficients: dict[str, complex],
    *,
    num_qubits: int,
) -> float:
    """Bound a Hermitian Pauli sum using greedy anticommuting groups."""
    identity = "I" * num_qubits
    identity_bound = abs(coefficients.get(identity, 0.0j))
    non_identity_terms: list[tuple[str, float]] = []
    for pauli, coefficient in coefficients.items():
        if pauli == identity:
            continue
        if abs(coefficient.imag) > 1e-12 * max(1.0, abs(coefficient.real)):
            raise ValueError("variance Pauli coefficients must be real")
        non_identity_terms.append((pauli, float(coefficient.real)))

    # First-fit with a largest-compatible-norm tie break is deterministic and
    # polynomial.  Each group is guaranteed pairwise anticommuting.
    non_identity_terms.sort(key=lambda item: (-abs(item[1]), item[0]))
    groups: list[list[tuple[str, float]]] = []
    group_squared_norms: list[float] = []
    for pauli, coefficient in non_identity_terms:
        compatible = [
            index
            for index, group in enumerate(groups)
            if all(
                not LCP._pauli_strings_commute(pauli, other_pauli)
                for other_pauli, _ in group
            )
        ]
        if compatible:
            selected = max(compatible, key=lambda index: group_squared_norms[index])
            groups[selected].append((pauli, coefficient))
            group_squared_norms[selected] += coefficient**2
        else:
            groups.append([(pauli, coefficient)])
            group_squared_norms.append(coefficient**2)

    return float(identity_bound + sum(np.sqrt(value) for value in group_squared_norms))


def _pauli_sample_matrix(
    weights: Sequence[float],
    sampled_hamiltonians: Sequence[LCP],
) -> tuple[NDArray[np.float64], tuple[str, ...], NDArray[np.float64]]:
    """Return probabilities, Pauli support, and sample coefficient columns."""
    if len(weights) != len(sampled_hamiltonians):
        raise ValueError("weights and sampled_hamiltonians must have equal length")

    num_qubits: int | None = None
    active_samples: list[tuple[float, dict[str, float]]] = []
    for weight, sampled_hamiltonian in zip(weights, sampled_hamiltonians):
        if not isinstance(weight, Real):
            raise TypeError("weights must be real numbers")
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("weights must be finite and non-negative")
        if not isinstance(sampled_hamiltonian, LCP):
            raise TypeError("sampled_hamiltonians must contain LCP instances")
        if num_qubits is None:
            num_qubits = sampled_hamiltonian.num_qubits
        elif sampled_hamiltonian.num_qubits != num_qubits:
            raise ValueError("sampled Hamiltonians must act on equal qubit counts")

        real_terms: dict[str, float] = {}
        for pauli, coefficient in sampled_hamiltonian.terms.items():
            if not np.isfinite(coefficient.real) or not np.isfinite(
                coefficient.imag
            ):
                raise ValueError("sampled Hamiltonian coefficients must be finite")
            if coefficient.imag != 0.0:
                raise ValueError("sampled Hamiltonian coefficients must be real")
            if coefficient != 0.0:
                real_terms[pauli] = float(coefficient.real)
        if weight != 0.0:
            active_samples.append((float(weight), real_terms))

    if not active_samples:
        return (
            np.array([], dtype=float),
            (),
            np.zeros((0, 0), dtype=float),
        )

    assert num_qubits is not None
    identity = "I" * num_qubits
    support = {
        pauli
        for _, terms in active_samples
        for pauli in terms
    }
    paulis = (identity, *sorted(support - {identity}))
    pauli_indices = {pauli: index for index, pauli in enumerate(paulis)}
    coefficients = np.zeros((len(paulis), len(active_samples)), dtype=float)
    for sample_index, (_, terms) in enumerate(active_samples):
        for pauli, coefficient in terms.items():
            coefficients[pauli_indices[pauli], sample_index] = coefficient

    maximum_weight = max(weight for weight, _ in active_samples)
    scaled_weights = [weight / maximum_weight for weight, _ in active_samples]
    scaled_sum = math.fsum(scaled_weights)
    probabilities = np.asarray(
        [weight / scaled_sum for weight in scaled_weights],
        dtype=float,
    )
    return probabilities, paulis, coefficients


def _symmetrized_pauli_product_map(
    paulis: Sequence[str],
) -> tuple[tuple[str, ...], sparse.csr_matrix]:
    r"""Map a real symmetric coefficient matrix to a Pauli anticommutator sum."""
    matrix_size = len(paulis)
    commuting_products: list[tuple[int, int, float, str]] = []
    product_support: set[str] = set()
    for left_index, left in enumerate(paulis):
        for right_index in range(left_index, matrix_size):
            phase, product = _multiply_pauli_strings(left, paulis[right_index])
            if phase.imag != 0.0:
                # The two ordered products cancel for a real symmetric matrix.
                continue
            real_phase = float(phase.real)
            commuting_products.append(
                (left_index, right_index, real_phase, product)
            )
            product_support.add(product)

    identity = "I" * len(paulis[0])
    product_paulis = (identity, *sorted(product_support - {identity}))
    product_indices = {
        pauli: index for index, pauli in enumerate(product_paulis)
    }
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for left_index, right_index, phase, product in commuting_products:
        rows.append(product_indices[product])
        columns.append(left_index * matrix_size + right_index)
        values.append(phase)
        if left_index != right_index:
            rows.append(product_indices[product])
            columns.append(right_index * matrix_size + left_index)
            values.append(phase)

    product_map = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(len(product_paulis), matrix_size * matrix_size),
        dtype=float,
    ).tocsr()
    return product_paulis, product_map


def _solve_pauli_moment_sdp_dual(
    *,
    product_map: sparse.csr_matrix,
    variance_coefficients: NDArray[np.float64],
    contraction_coefficients: NDArray[np.float64],
    identity_coefficients: NDArray[np.float64],
    solver: str | None,
) -> float | None:
    r"""Solve and repair the dual Pauli SOS certificate."""
    matrix_size = math.isqrt(product_map.shape[1])
    gram = cp.Variable((matrix_size, matrix_size), symmetric=True)
    contraction_multipliers = cp.Variable(
        contraction_coefficients.shape[1],
        nonneg=True,
    )
    upper_bound = cp.Variable()
    sos_coefficients = product_map @ cp.reshape(
        gram,
        (matrix_size * matrix_size,),
        order="C",
    )
    constraints = [
        gram >> 0,
        sos_coefficients
        + contraction_coefficients @ contraction_multipliers
        + variance_coefficients
        == upper_bound * identity_coefficients,
    ]
    problem = cp.Problem(cp.Minimize(upper_bound), constraints)

    installed_solvers = set(cp.installed_solvers())
    if solver is not None:
        if solver not in installed_solvers:
            raise ValueError(f"CVXPY solver is not installed: {solver!r}")
        solver_candidates = [solver]
    else:
        solver_candidates = [
            candidate
            for candidate in ("CLARABEL", "SCS")
            if candidate in installed_solvers
        ]

    for candidate in solver_candidates:
        try:
            if candidate == "CLARABEL":
                problem.solve(
                    solver=candidate,
                    tol_gap_abs=1e-9,
                    tol_gap_rel=1e-9,
                    tol_feas=1e-9,
                    max_iter=500,
                )
            elif candidate == "SCS":
                problem.solve(
                    solver=candidate,
                    eps=1e-7,
                    max_iters=100_000,
                )
            else:
                problem.solve(solver=candidate)
        except cp.error.SolverError:
            if solver is not None:
                raise
            continue

        if (
            problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
            and gram.value is not None
            and contraction_multipliers.value is not None
            and upper_bound.value is not None
        ):
            return _repair_pauli_sos_certificate(
                product_map=product_map,
                variance_coefficients=variance_coefficients,
                contraction_coefficients=contraction_coefficients,
                identity_coefficients=identity_coefficients,
                gram=np.asarray(gram.value, dtype=float),
                contraction_multipliers=np.asarray(
                    contraction_multipliers.value,
                    dtype=float,
                ),
                upper_bound=float(upper_bound.value),
            )
        if solver is not None:
            raise RuntimeError(f"Pauli moment SDP failed with status {problem.status!r}")
    return None


def _repair_pauli_sos_certificate(
    *,
    product_map: sparse.csr_matrix,
    variance_coefficients: NDArray[np.float64],
    contraction_coefficients: NDArray[np.float64],
    identity_coefficients: NDArray[np.float64],
    gram: NDArray[np.float64],
    contraction_multipliers: NDArray[np.float64],
    upper_bound: float,
) -> float | None:
    """Project a numerical dual solution and pay for its Pauli residual."""
    if (
        not np.all(np.isfinite(gram))
        or not np.all(np.isfinite(contraction_multipliers))
        or not np.isfinite(upper_bound)
    ):
        return None

    symmetric_gram = 0.5 * (gram + gram.T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric_gram)
    except np.linalg.LinAlgError:
        return None
    positive_eigenvalues = np.maximum(eigenvalues, 0.0)
    repaired_gram = (eigenvectors * positive_eigenvalues) @ eigenvectors.T
    repaired_multipliers = np.maximum(contraction_multipliers, 0.0)
    sos_coefficients = np.asarray(
        product_map @ repaired_gram.reshape(-1),
        dtype=float,
    ).reshape(-1)
    residual = (
        upper_bound * identity_coefficients
        - variance_coefficients
        - sos_coefficients
        - contraction_coefficients @ repaired_multipliers
    )
    residual_bound = math.fsum(abs(float(value)) for value in residual)
    scale = max(
        1.0,
        abs(upper_bound),
        math.fsum(abs(float(value)) for value in variance_coefficients),
        math.fsum(abs(float(value)) for value in sos_coefficients),
        math.fsum(
            abs(float(value))
            for value in contraction_coefficients @ repaired_multipliers
        ),
    )
    rounding_allowance = (
        256.0
        * np.finfo(float).eps
        * max(1, len(residual), len(gram))
        * scale
    )
    certified_bound = upper_bound + residual_bound + rounding_allowance
    if certified_bound < 0.0:
        return None
    return float(np.nextafter(certified_bound, np.inf))


def _probability_only_sdp_bound(
    probabilities: Sequence[float],
) -> float:
    """Return the legacy diagonal-majorant bound used as a safe fallback."""
    if len(probabilities) == 0:
        return 0.0
    maximum_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    maximum_probability = float(probabilities[maximum_index])
    remaining_probability = math.fsum(
        float(probability)
        for index, probability in enumerate(probabilities)
        if index != maximum_index
    )
    if remaining_probability == 0.0:
        return 0.0
    if remaining_probability >= maximum_probability:
        return 1.0
    imbalance = maximum_probability - remaining_probability
    value = 1.0 - imbalance * imbalance
    return min(1.0, float(value + 32.0 * np.finfo(float).eps))


def sparse_operator_norm(matrix: sparse.spmatrix) -> float:
    """Return the largest singular value of a sparse matrix."""
    csr_matrix = sparse.csr_matrix(matrix)
    csr_matrix.eliminate_zeros()
    if csr_matrix.nnz == 0:
        return 0.0
    if csr_matrix.shape[0] <= 2:
        return float(np.linalg.norm(csr_matrix.toarray(), ord=2))
    largest = sparse.linalg.svds(
        csr_matrix,
        k=1,
        which="LM",
        return_singular_vectors=False,
        tol=1e-7,
    )[0]
    return float(abs(largest))
