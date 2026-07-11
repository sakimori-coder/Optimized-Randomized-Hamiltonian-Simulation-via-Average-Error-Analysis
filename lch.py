"""Linear combination of Pauli Hamiltonian terms with per-term evolution cost."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Number

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from lcp import LCP


class LCH:
    """Linear combination of Pauli terms, represented as an internal :class:`LCP`.

    Each item is ``(coefficient, pauli_string, evolution_cost)``.
    Costs are stored separately from :class:`LCP` coefficients.

    Args:
        terms: Iterable of ``(coefficient, operator)`` or
            ``(coefficient, operator, evolution_cost)`` tuples where the operator is
            an n-qubit Pauli string such as ``"ZIXX"``.
        num_qubits: Number of qubits, optional. If omitted, it is inferred from Pauli-string length.
        default_evolution_cost: Evolution cost for terms specified without explicit cost.
    """

    def __init__(
        self,
        terms: Iterable[tuple] | None = None,
        *,
        num_qubits: int | None = None,
        default_evolution_cost: float = 0.0,
    ) -> None:
        if num_qubits is not None and (
            not isinstance(num_qubits, int)
            or isinstance(num_qubits, bool)
            or num_qubits < 0
        ):
            raise ValueError("num_qubits must be a non-negative integer")
        if not isinstance(default_evolution_cost, Number):
            raise TypeError("default_evolution_cost must be a number")

        raw_terms = [] if terms is None else list(terms)
        if not isinstance(raw_terms, list):
            # ``list(terms)`` above guarantees list, this check keeps type errors
            # explicit when users pass non-iterables as terms.
            raise TypeError(
                "terms must be an iterable of (coefficient, operator[, evolution_cost]) tuples"
            )

        inferred_num_qubits = num_qubits
        validated_terms: dict[str, complex] = {}
        validated_costs: dict[str, float] = {}

        for index, raw_term in enumerate(raw_terms):
            if not isinstance(raw_term, tuple):
                raise TypeError(
                    "each term must be a tuple of (coefficient, operator[, evolution_cost])"
                )
            if len(raw_term) == 2:
                coefficient, operator = raw_term
                evolution_cost = default_evolution_cost
            elif len(raw_term) == 3:
                coefficient, operator, evolution_cost = raw_term
            else:
                raise TypeError(
                    "each term must be a tuple of (coefficient, operator[, evolution_cost])"
                )
            if not isinstance(coefficient, Number):
                raise TypeError(
                    f"coefficient of term at index {index} must be a number"
                )
            if not isinstance(evolution_cost, Number):
                raise TypeError(
                    f"evolution_cost of term at index {index} must be a number"
                )

            if not isinstance(operator, str):
                raise TypeError(
                    f"operator of term at index {index} must be a Pauli string"
                )

            # Delegate Pauli validation to LCP.
            LCP._validate_pauli_string(operator)
            if inferred_num_qubits is None:
                inferred_num_qubits = len(operator)

            if len(operator) != inferred_num_qubits:
                raise ValueError("all Pauli strings must have the same length as num_qubits")

            if operator in validated_terms:
                raise ValueError(
                    f"duplicate Pauli string at index {index}: {operator!r}"
                )

            validated_terms[operator] = complex(coefficient)
            validated_costs[operator] = float(evolution_cost)

        if not validated_terms and inferred_num_qubits is None:
            raise ValueError("terms is empty; num_qubits must be provided")

        self._lcp = LCP(validated_terms, num_qubits=inferred_num_qubits)
        self._costs = validated_costs
        self._default_evolution_cost = float(default_evolution_cost)

    @staticmethod
    def _validate_operator_index(index: int, num_terms: int) -> None:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= num_terms:
            raise IndexError("operator index out of range")

    @property
    def lcp(self) -> LCP:
        """Return the internal :class:`LCP` representation."""
        return self._lcp

    def _pauli_terms_in_order(self) -> list[tuple[str, complex]]:
        return list(self._lcp.terms.items())

    def _materialize_operator(self, operator: str) -> sparse.csr_matrix:
        return _pauli_to_csr(operator)

    @property
    def terms(self) -> list[tuple[complex, sparse.csr_matrix]]:
        """Return a list of ``(coefficient, operator_matrix)``."""
        return [
            (coefficient, self._materialize_operator(pauli))
            for pauli, coefficient in self._pauli_terms_in_order()
        ]

    @property
    def lcp_terms(self) -> list[tuple[complex, str, float]]:
        """Return terms as ``(coefficient, pauli_string, evolution_cost)``."""
        return [
            (coefficient, pauli, self._costs[pauli])
            for pauli, coefficient in self._pauli_terms_in_order()
        ]

    @property
    def evolution_costs(self) -> list[float]:
        """Return a copied list of evolution costs in Pauli-term order."""
        return [float(self._costs[pauli]) for pauli, _ in self._pauli_terms_in_order()]

    @property
    def default_evolution_cost(self) -> float:
        """Return the default evolution cost used when not explicitly specified."""
        return self._default_evolution_cost

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits represented by this Hamiltonian."""
        return self._lcp.num_qubits

    @property
    def dimension(self) -> int:
        """Return the matrix dimension of each term."""
        return 1 << self._lcp.num_qubits

    def __len__(self) -> int:
        return len(self._lcp.terms)

    def __iter__(self):
        return iter(self.terms)

    def _resolve_cost(self, pauli: str, other: "LCH") -> float:
        self_cost = self._costs[pauli] if pauli in self._costs else None
        other_cost = other._costs[pauli] if pauli in other._costs else None
        if self_cost is not None and other_cost is not None and not np.isclose(self_cost, other_cost):
            raise ValueError(
                "cannot combine LCHs with different evolution costs for the same Pauli string: "
                f"{pauli!r}"
            )
        if self_cost is not None:
            return float(self_cost)
        if other_cost is not None:
            return float(other_cost)
        raise ValueError(f"missing Pauli cost for {pauli!r}")

    def __add__(self, other: "LCH") -> "LCH":
        """Add two linear-combination Hamiltonians."""
        if not isinstance(other, LCH):
            return NotImplemented
        if self.dimension != other.dimension:
            raise ValueError("LCH operands must have the same dimension")

        result_terms_map: dict[str, complex] = self._lcp.terms
        for pauli, coefficient in other._lcp.terms.items():
            result_terms_map[pauli] = result_terms_map.get(pauli, 0j) + coefficient

        combined_terms = [
            (coefficient, pauli, self._resolve_cost(pauli, other))
            for pauli, coefficient in result_terms_map.items()
        ]
        return LCH(
            combined_terms,
            num_qubits=self.num_qubits,
        )

    def __sub__(self, other: "LCH") -> "LCH":
        """Subtract two linear-combination Hamiltonians."""
        if not isinstance(other, LCH):
            return NotImplemented
        if self.dimension != other.dimension:
            raise ValueError("LCH operands must have the same dimension")

        result_terms_map: dict[str, complex] = self._lcp.terms
        for pauli, coefficient in other._lcp.terms.items():
            result_terms_map[pauli] = result_terms_map.get(pauli, 0j) - coefficient

        combined_terms = [
            (coefficient, pauli, self._resolve_cost(pauli, other))
            for pauli, coefficient in result_terms_map.items()
        ]
        return LCH(
            combined_terms,
            num_qubits=self.num_qubits,
        )

    def __mul__(self, scalar: complex) -> "LCH":
        """Multiply every coefficient by a scalar."""
        if not isinstance(scalar, Number):
            return NotImplemented
        return LCH(
            [
                (coefficient * scalar, pauli, self._costs[pauli])
                for pauli, coefficient in self._pauli_terms_in_order()
            ],
            num_qubits=self.num_qubits,
        )

    def __rmul__(self, scalar: complex) -> "LCH":
        """Multiply every coefficient by a scalar."""
        return self * scalar

    def coefficient_one_norm(self) -> float:
        """Return the coefficient 1-norm ``sum(abs(coefficient))``."""
        return float(sum(abs(coefficient) for _, coefficient in self._pauli_terms_in_order()))

    def sampling_probabilities(self) -> NDArray[np.float64]:
        """Return sampling probabilities proportional to absolute coefficients."""
        if not self._lcp.terms:
            return np.array([], dtype=float)

        abs_coefficients = np.array(
            [abs(coefficient) for _, coefficient in self._pauli_terms_in_order()],
            dtype=float,
        )
        total = float(abs_coefficients.sum())
        if total == 0.0:
            return np.zeros_like(abs_coefficients)
        return abs_coefficients / total

    def sample_term_index(self, rng: np.random.Generator | None = None) -> int:
        """Sample one operator index using the absolute-coefficient distribution."""
        if not self._pauli_terms_in_order():
            raise ValueError("cannot sample from an empty LCH")

        probabilities = self.sampling_probabilities()
        if probabilities.size > 0 and not np.isclose(probabilities.sum(), 1.0):
            if np.isclose(probabilities.sum(), 0.0):
                raise ValueError("cannot sample when total coefficient magnitude is zero")
            probabilities = probabilities / probabilities.sum()

        generator = np.random.default_rng() if rng is None else rng
        return int(generator.choice(len(self), p=probabilities))

    def sample_term(self, rng: np.random.Generator | None = None) -> tuple[complex, sparse.csr_matrix]:
        """Return one sampled term as ``(coefficient, operator)``."""
        index = self.sample_term_index(rng=rng)
        coefficient, operator, _ = self.get_term_with_cost(index)
        return coefficient, operator

    def sample_term_with_cost(
        self, rng: np.random.Generator | None = None
    ) -> tuple[complex, sparse.csr_matrix, float]:
        """Return one sampled term as ``(coefficient, operator, evolution_cost)``."""
        return self.get_term_with_cost(self.sample_term_index(rng=rng))

    def get_operator(self, index: int) -> sparse.csr_matrix:
        """Return a copy of the operator at ``index``."""
        self._validate_operator_index(index, len(self))
        pauli, _ = self._pauli_terms_in_order()[index]
        return self._materialize_operator(pauli)

    def get_coefficient(self, index: int) -> complex:
        """Return the coefficient at ``index``."""
        self._validate_operator_index(index, len(self))
        _, coefficient = self._pauli_terms_in_order()[index]
        return coefficient

    def get_term(self, index: int) -> tuple[complex, sparse.csr_matrix]:
        """Return a single term by index as ``(coefficient, operator)``."""
        self._validate_operator_index(index, len(self))
        pauli, coefficient = self._pauli_terms_in_order()[index]
        return coefficient, self._materialize_operator(pauli)

    def get_term_cost(self, index: int) -> float:
        """Return the evolution cost at ``index``."""
        self._validate_operator_index(index, len(self))
        pauli, _ = self._pauli_terms_in_order()[index]
        return self._costs[pauli]

    def get_term_with_cost(self, index: int) -> tuple[complex, sparse.csr_matrix, float]:
        """Return a single term by index as ``(coefficient, operator, evolution_cost)``."""
        self._validate_operator_index(index, len(self))
        pauli, coefficient = self._pauli_terms_in_order()[index]
        return coefficient, self._materialize_operator(pauli), self._costs[pauli]

    def get_pauli_term(self, index: int) -> tuple[complex, str, float]:
        """Return a single Pauli-term by index."""
        self._validate_operator_index(index, len(self))
        pauli, coefficient = self._pauli_terms_in_order()[index]
        return coefficient, pauli, self._costs[pauli]

    def set_term_cost(self, index: int, evolution_cost: float) -> None:
        """Set the evolution cost at ``index``."""
        self._validate_operator_index(index, len(self))
        if not isinstance(evolution_cost, Number):
            raise TypeError("evolution_cost must be a number")
        pauli, _ = self._pauli_terms_in_order()[index]
        self._costs[pauli] = float(evolution_cost)

    def to_csr(self) -> sparse.csr_matrix:
        """Combine all terms into one CSR matrix."""
        result = sparse.csr_matrix((self.dimension, self.dimension), dtype=complex)
        for pauli, coefficient in self._lcp.terms.items():
            result = result + coefficient * self._materialize_operator(pauli)
        return result.tocsr()

    def to_matrix(self) -> NDArray[np.complex128]:
        """Return a dense complex matrix."""
        return self.to_csr().toarray()

    def operator_norm(self) -> float:
        """Return the spectral norm (largest singular value)."""
        matrix = self.to_csr()
        if matrix.nnz == 0:
            return 0.0
        if matrix.shape[0] <= 2:
            return float(np.linalg.norm(matrix.toarray(), ord=2))

        largest = sparse.linalg.svds(
            matrix,
            k=1,
            which="LM",
            return_singular_vectors=False,
        )[0]
        return float(abs(largest))


def _pauli_to_csr(pauli_string: str) -> sparse.csr_matrix:
    if len(pauli_string) == 0:
        return sparse.csr_matrix([[1]], dtype=complex)

    result = sparse.csr_matrix([[1]], dtype=complex)
    for symbol in pauli_string:
        result = sparse.kron(result, _PAULI_OPERATORS[symbol], format="csr")
    return result.tocsr()


_PAULI_OPERATORS: dict[str, sparse.csr_matrix] = {
    "I": sparse.csr_matrix(np.array([[1, 0], [0, 1]], dtype=complex)),
    "X": sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex)),
    "Y": sparse.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex)),
    "Z": sparse.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex)),
}
