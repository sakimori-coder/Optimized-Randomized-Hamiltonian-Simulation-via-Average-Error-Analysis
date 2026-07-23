r"""Operator-norm estimates for a commuting Pauli Hamiltonian."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from lcp import LCP


BuiltInMethod = Literal["coefficient_l1", "lp", "exact"]

_PAULI_PRODUCT: dict[tuple[str, str], tuple[int, str]] = {
    ("I", "I"): (0, "I"),
    ("I", "X"): (0, "X"),
    ("I", "Y"): (0, "Y"),
    ("I", "Z"): (0, "Z"),
    ("X", "I"): (0, "X"),
    ("Y", "I"): (0, "Y"),
    ("Z", "I"): (0, "Z"),
    ("X", "X"): (0, "I"),
    ("Y", "Y"): (0, "I"),
    ("Z", "Z"): (0, "I"),
    ("X", "Y"): (1, "Z"),
    ("Y", "X"): (3, "Z"),
    ("Y", "Z"): (1, "X"),
    ("Z", "Y"): (3, "X"),
    ("Z", "X"): (1, "Y"),
    ("X", "Z"): (3, "Y"),
}


def _binary_pauli(pauli: str) -> int:
    """Encode a Pauli string as its binary symplectic vector."""
    num_qubits = len(pauli)
    vector = 0
    for qubit, symbol in enumerate(pauli):
        if symbol in "XY":
            vector |= 1 << qubit
        if symbol in "YZ":
            vector |= 1 << (num_qubits + qubit)
    return vector


def _dependency_basis(paulis: list[str]) -> list[int]:
    """Return a GF(2) basis of multiplicative Pauli dependencies."""
    pivots: dict[int, tuple[int, int]] = {}
    dependencies: list[int] = []

    for term_index, pauli in enumerate(paulis):
        vector = _binary_pauli(pauli)
        combination = 1 << term_index

        while vector:
            pivot = vector.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = (vector, combination)
                break
            pivot_vector, pivot_combination = pivots[pivot]
            vector ^= pivot_vector
            combination ^= pivot_combination
        else:
            dependencies.append(combination)

    return dependencies


def _dependency_parity(paulis: list[str], dependency: int) -> int:
    r"""Return ``b`` such that the selected Pauli product is ``(-1)**b I``."""
    product = ["I"] * len(paulis[0])
    phase = 0

    for term_index, pauli in enumerate(paulis):
        if not dependency & (1 << term_index):
            continue
        for qubit, right_symbol in enumerate(pauli):
            phase_increment, product[qubit] = _PAULI_PRODUCT[
                product[qubit], right_symbol
            ]
            phase = (phase + phase_increment) % 4

    return phase // 2


def _parity_trellis(
    num_terms: int,
    paulis: list[str],
    dependencies: list[int],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Build an extended LP for the intersection of parity polytopes."""
    relations: list[tuple[tuple[int, ...], int, int]] = []
    num_variables = num_terms
    for dependency in dependencies:
        support = tuple(
            index
            for index in range(num_terms)
            if dependency & (1 << index)
        )
        relations.append(
            (support, _dependency_parity(paulis, dependency), num_variables)
        )
        num_variables += 4 * len(support)

    rows: list[int] = []
    columns: list[int] = []
    values: list[int] = []
    right_hand_side: list[int] = []

    def add_entry(row: int, column: int, value: int) -> None:
        rows.append(row)
        columns.append(column)
        values.append(value)

    for support, required_parity, flow_start in relations:
        length = len(support)

        def flow_index(layer: int, parity: int, bit: int) -> int:
            return flow_start + 4 * layer + 2 * parity + bit

        # Unit flow from state (0, 0) to state (length, required_parity).
        # The sink balance is redundant and is omitted.
        for layer in range(length + 1):
            for parity in (0, 1):
                if layer == length and parity == required_parity:
                    continue

                row = len(right_hand_side)
                if layer < length:
                    add_entry(row, flow_index(layer, parity, 0), 1)
                    add_entry(row, flow_index(layer, parity, 1), 1)
                if layer > 0:
                    add_entry(row, flow_index(layer - 1, parity, 0), -1)
                    add_entry(row, flow_index(layer - 1, 1 - parity, 1), -1)

                right_hand_side.append(1 if layer == 0 and parity == 0 else 0)

        # q_j is the flow that chooses bit 1 at the corresponding layer.
        for layer, term_index in enumerate(support):
            row = len(right_hand_side)
            add_entry(row, term_index, 1)
            add_entry(row, flow_index(layer, 0, 1), -1)
            add_entry(row, flow_index(layer, 1, 1), -1)
            right_hand_side.append(0)

    equalities = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(len(right_hand_side), num_variables),
        dtype=float,
    ).tocsr()
    return equalities, np.asarray(right_hand_side, dtype=float)


def _lagrangian_upper_bound(
    objective: list[Fraction],
    constant: Fraction,
    equalities: sparse.csr_matrix,
    right_hand_side: np.ndarray,
    multipliers: np.ndarray,
    multiplier_scale: Fraction = Fraction(1),
) -> Fraction:
    r"""Certify an upper bound for ``constant + max objective @ x``.

    Every LP variable obeys ``0 <= x_i <= 1``.  For arbitrary multipliers
    ``y``, Lagrangian relaxation gives

    ``max objective @ x <= y @ rhs + sum_i max(0, objective-E.T@y)``.
    """
    multiplier_fractions = [
        Fraction.from_float(float(value)) * multiplier_scale
        for value in multipliers
    ]
    bound = constant + sum(
        (
            Fraction.from_float(float(rhs)) * multiplier
            for rhs, multiplier in zip(right_hand_side, multiplier_fractions)
        ),
        start=Fraction(0),
    )

    transposed = equalities.tocsc()
    for column, coefficient in enumerate(objective):
        reduced_coefficient = coefficient
        for position in range(
            transposed.indptr[column],
            transposed.indptr[column + 1],
        ):
            row = transposed.indices[position]
            reduced_coefficient -= (
                int(transposed.data[position]) * multiplier_fractions[row]
            )
        if reduced_coefficient > 0:
            bound += reduced_coefficient

    return bound


def _solve_lp_direction(
    solver_objective: np.ndarray,
    certificate_objective: list[Fraction],
    constant: Fraction,
    equalities: sparse.csr_matrix,
    right_hand_side: np.ndarray,
    objective_scale: Fraction,
) -> Fraction:
    """Solve one direction and return a dual-certified upper bound."""
    result = linprog(
        -solver_objective,
        A_eq=equalities,
        b_eq=right_hand_side,
        bounds=(0.0, 1.0),
        method="highs",
    )

    multipliers = [np.zeros(equalities.shape[0], dtype=float)]
    if result.success:
        marginals = np.asarray(result.eqlin.marginals, dtype=float)
        multipliers.extend((marginals, -marginals))

    return min(
        _lagrangian_upper_bound(
            certificate_objective,
            constant,
            equalities,
            right_hand_side,
            candidate,
            objective_scale,
        )
        for candidate in multipliers
    )


def _float_rounding_up(value: Fraction) -> float:
    """Convert an exact rational to a float without rounding downward."""
    try:
        result = float(value)
    except OverflowError:
        return float("inf") if value > 0 else float("-inf")
    if not np.isfinite(result):
        return result
    if Fraction.from_float(result) < value:
        result = float(np.nextafter(result, np.inf))
    return result


def _coefficient_l1_bound(hamiltonian: LCP) -> float:
    """Return the coefficient 1-norm with outward-rounded summation."""
    exact_sum = sum(
        (
            Fraction.from_float(float(abs(coefficient)))
            for coefficient in hamiltonian.terms.values()
        ),
        start=Fraction(0),
    )
    return _float_rounding_up(exact_sum)


def _lp_relaxation_bound(hamiltonian: LCP) -> float:
    identity = "I" * hamiltonian.num_qubits
    constant_coefficient = 0.0
    active_terms: list[tuple[str, float]] = []

    for pauli, coefficient in sorted(hamiltonian.terms.items()):
        if coefficient == 0:
            continue
        if pauli == identity:
            constant_coefficient = float(coefficient.real)
        else:
            active_terms.append((pauli, float(coefficient.real)))

    if not active_terms:
        return abs(constant_coefficient)

    paulis = [pauli for pauli, _ in active_terms]
    coefficients = np.asarray(
        [coefficient for _, coefficient in active_terms],
        dtype=float,
    )
    coefficient_fractions = [
        Fraction.from_float(float(coefficient)) for coefficient in coefficients
    ]
    dependencies = _dependency_basis(paulis)

    positive_constant = Fraction.from_float(constant_coefficient) + sum(
        coefficient_fractions,
        start=Fraction(0),
    )
    num_variables = len(coefficients) + 4 * sum(
        dependency.bit_count() for dependency in dependencies
    )
    positive_certificate_objective = [
        -2 * coefficient for coefficient in coefficient_fractions
    ] + [Fraction(0)] * (num_variables - len(coefficients))

    if not dependencies:
        positive_cube_bound = positive_constant + sum(
            (
                max(Fraction(0), coefficient)
                for coefficient in positive_certificate_objective
            ),
            start=Fraction(0),
        )
        negative_cube_bound = -positive_constant + sum(
            (
                max(Fraction(0), -coefficient)
                for coefficient in positive_certificate_objective
            ),
            start=Fraction(0),
        )
        return _float_rounding_up(max(positive_cube_bound, negative_cube_bound))

    equalities, right_hand_side = _parity_trellis(
        len(coefficients),
        paulis,
        dependencies,
    )
    coefficient_scale = float(np.max(np.abs(coefficients)))
    solver_objective = np.zeros(num_variables, dtype=float)
    solver_objective[: len(coefficients)] = -2.0 * (
        coefficients / coefficient_scale
    )
    objective_scale = Fraction.from_float(coefficient_scale)
    positive_bound = _solve_lp_direction(
        solver_objective,
        positive_certificate_objective,
        positive_constant,
        equalities,
        right_hand_side,
        objective_scale,
    )
    negative_bound = _solve_lp_direction(
        -solver_objective,
        [-coefficient for coefficient in positive_certificate_objective],
        -positive_constant,
        equalities,
        right_hand_side,
        objective_scale,
    )
    return _float_rounding_up(max(positive_bound, negative_bound))


def estimate_commuting_operator_norm_upper_bound(
    hamiltonian: LCP,
    *,
    method: BuiltInMethod = "coefficient_l1",
) -> float:
    r"""Estimate ``||hamiltonian||_op`` with the selected method.

    ``coefficient_l1`` returns the matrix-free bound ``sum_j |c_j|``.
    ``lp`` tightens that bound using Pauli-product parity relations and a
    matrix-free linear-programming relaxation.  ``exact`` constructs the
    ``2**n`` dimensional CSR matrix and numerically computes its largest
    singular value.

    The input is assumed to be a bounded, pairwise-commuting Hermitian LCP.
    """
    if method == "coefficient_l1":
        return _coefficient_l1_bound(hamiltonian)
    if method == "lp":
        return _lp_relaxation_bound(hamiltonian)
    if method == "exact":
        return hamiltonian.operator_norm()
    raise ValueError(
        "unknown method; expected 'coefficient_l1', 'lp', or 'exact'"
    )
