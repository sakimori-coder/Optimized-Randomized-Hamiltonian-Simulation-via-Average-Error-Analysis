r"""Linear combinations of LCP Hamiltonians with per-term evolution costs."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Number

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from .lcp import LCP


class LCH:
    r"""Represent ``H = sum_j coefficients[j] * operators[j]``.

    Every sampled operator is an :class:`LCP`.  The decomposition is preserved
    in term order because different LCP decompositions of the same flattened
    Hamiltonian generally define different qDRIFT channels.

    Each input item is ``(coefficient, lcp)`` or
    ``(coefficient, lcp, evolution_cost)``.

    ``LCH`` does not normalize the inner LCPs.  When a qDRIFT estimator assumes
    Hermitian contractions, callers must supply ``||operator||_op <= 1`` terms
    (for example, ``G_g / h_g`` with ``h_g`` an operator-norm upper bound).
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
        default_cost = _real_value(
            default_evolution_cost,
            name="default_evolution_cost",
        )

        try:
            raw_terms = [] if terms is None else list(terms)
        except TypeError as error:
            raise TypeError(
                "terms must be an iterable of "
                "(coefficient, LCP[, evolution_cost]) tuples"
            ) from error

        inferred_num_qubits = num_qubits
        validated_terms: list[tuple[float, LCP]] = []
        validated_costs: list[float] = []
        for index, raw_term in enumerate(raw_terms):
            if not isinstance(raw_term, tuple) or len(raw_term) not in (2, 3):
                raise TypeError(
                    "each term must be a tuple of "
                    "(coefficient, LCP[, evolution_cost])"
                )

            coefficient, operator = raw_term[:2]
            evolution_cost = raw_term[2] if len(raw_term) == 3 else default_cost
            coefficient_value = _real_value(
                coefficient,
                name=f"coefficient of term at index {index}",
            )
            cost_value = _real_value(
                evolution_cost,
                name=f"evolution_cost of term at index {index}",
            )
            if not isinstance(operator, LCP):
                raise TypeError(
                    f"operator of term at index {index} must be an LCP"
                )

            if inferred_num_qubits is None:
                inferred_num_qubits = operator.num_qubits
            elif operator.num_qubits != inferred_num_qubits:
                raise ValueError(
                    "all LCP terms must act on the same number of qubits"
                )

            validated_terms.append(
                (
                    coefficient_value,
                    LCP(operator.terms, num_qubits=operator.num_qubits),
                )
            )
            validated_costs.append(cost_value)

        if inferred_num_qubits is None:
            raise ValueError("terms is empty; num_qubits must be provided")

        self._num_qubits = inferred_num_qubits
        self._terms = validated_terms
        self._costs = validated_costs
        self._default_evolution_cost = default_cost
        self._lcp = _flatten_lcp_terms(
            validated_terms,
            num_qubits=inferred_num_qubits,
        )

    @staticmethod
    def _validate_operator_index(index: int, num_terms: int) -> None:
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= num_terms
        ):
            raise IndexError("operator index out of range")

    @property
    def lcp(self) -> LCP:
        r"""Return the flattened Hamiltonian ``sum_j c_j H_j``."""
        return self._lcp

    @property
    def terms(self) -> list[tuple[float, LCP]]:
        """Return the ordered decomposition as ``(coefficient, LCP)`` pairs."""
        return list(self._terms)

    @property
    def lcp_terms(self) -> list[tuple[float, LCP, float]]:
        """Return ``(coefficient, LCP, evolution_cost)`` tuples."""
        return [
            (coefficient, operator, float(cost))
            for (coefficient, operator), cost in zip(self._terms, self._costs)
        ]

    @property
    def evolution_costs(self) -> list[float]:
        """Return evolution costs in LCP-term order."""
        return [float(cost) for cost in self._costs]

    @property
    def default_evolution_cost(self) -> float:
        """Return the default cost for terms given without an explicit cost."""
        return self._default_evolution_cost

    @property
    def num_qubits(self) -> int:
        """Return the common number of qubits of all LCP terms."""
        return self._num_qubits

    @property
    def dimension(self) -> int:
        """Return the Hilbert-space dimension."""
        return 1 << self._num_qubits

    def __len__(self) -> int:
        return len(self._terms)

    def __iter__(self):
        return iter(self.terms)

    def __str__(self) -> str:
        """Return a readable, size-limited decomposition."""
        return self.format_decomposition(
            precision=6,
            max_terms=20,
            max_pauli_terms=20,
        )

    def format_decomposition(
        self,
        *,
        precision: int = 10,
        max_terms: int | None = None,
        max_pauli_terms: int | None = None,
    ) -> str:
        """Format the ordered outer-LCH and inner-LCP decomposition.

        ``weighted_coefficient`` is the contribution from that particular
        LCH term, namely ``outer_coefficient * inner_coefficient``.  It need
        not equal the coefficient in the flattened Hamiltonian when the same
        Pauli string occurs in multiple outer terms.
        """
        terms = self.lcp_terms
        probabilities = self.sampling_probabilities()
        lines = [
            "LCH("
            f"num_qubits={self.num_qubits}, "
            f"num_terms={len(terms)}, "
            "coefficient_one_norm="
            f"{self.coefficient_one_norm():.{precision}g}"
            ")"
        ]
        if not terms:
            lines.append("  (empty decomposition)")
            return "\n".join(lines)

        term_limit = (
            len(terms)
            if max_terms is None
            else min(len(terms), max(0, max_terms))
        )
        for index, (outer, operator, cost) in enumerate(terms[:term_limit]):
            lines.extend(
                [
                    "",
                    (
                        f"term[{index}]: "
                        f"outer_coefficient={outer:.{precision}g}, "
                        "sampling_probability="
                        f"{probabilities[index]:.{precision}g}, "
                        f"evolution_cost={cost:.{precision}g}"
                    ),
                ]
            )
            pauli_terms = list(operator.terms.items())
            pauli_limit = (
                len(pauli_terms)
                if max_pauli_terms is None
                else min(len(pauli_terms), max(0, max_pauli_terms))
            )
            if not pauli_terms:
                lines.append("  (empty LCP)")
            for pauli, inner in pauli_terms[:pauli_limit]:
                lines.append(
                    f"  {pauli}: "
                    f"inner_coefficient={inner:+.{precision}g}, "
                    "weighted_coefficient="
                    f"{outer * inner:+.{precision}g}"
                )
            if pauli_limit < len(pauli_terms):
                lines.append(
                    f"  ... {len(pauli_terms) - pauli_limit} "
                    "Pauli terms omitted"
                )

        if term_limit < len(terms):
            lines.extend(
                [
                    "",
                    f"... {len(terms) - term_limit} LCH terms omitted",
                ]
            )
        return "\n".join(lines)

    def __add__(self, other: LCH) -> LCH:
        """Concatenate two decompositions of Hamiltonians on equal qubits."""
        if not isinstance(other, LCH):
            return NotImplemented
        if self.num_qubits != other.num_qubits:
            raise ValueError("LCH operands must act on equal qubit counts")
        return LCH(
            [*self.lcp_terms, *other.lcp_terms],
            num_qubits=self.num_qubits,
            default_evolution_cost=self.default_evolution_cost,
        )

    def __sub__(self, other: LCH) -> LCH:
        """Subtract another LCH while preserving both term decompositions."""
        if not isinstance(other, LCH):
            return NotImplemented
        if self.num_qubits != other.num_qubits:
            raise ValueError("LCH operands must act on equal qubit counts")
        return LCH(
            [
                *self.lcp_terms,
                *[
                    (-coefficient, operator, cost)
                    for coefficient, operator, cost in other.lcp_terms
                ],
            ],
            num_qubits=self.num_qubits,
            default_evolution_cost=self.default_evolution_cost,
        )

    def __mul__(self, scalar: complex) -> LCH:
        """Multiply every outer LCH coefficient by a real scalar."""
        if not isinstance(scalar, Number):
            return NotImplemented
        scalar_value = _real_value(scalar, name="LCH scalar")
        return LCH(
            [
                (scalar_value * coefficient, operator, cost)
                for coefficient, operator, cost in self.lcp_terms
            ],
            num_qubits=self.num_qubits,
            default_evolution_cost=self.default_evolution_cost,
        )

    def __rmul__(self, scalar: complex) -> LCH:
        """Multiply every outer LCH coefficient by a real scalar."""
        return self * scalar

    def coefficient_one_norm(self) -> float:
        """Return the outer coefficient one-norm ``sum_j |c_j|``."""
        return float(sum(abs(coefficient) for coefficient, _ in self._terms))

    def sampling_probabilities(self) -> NDArray[np.float64]:
        """Return probabilities proportional to outer coefficient magnitudes."""
        if not self._terms:
            return np.array([], dtype=float)
        maximum = max(abs(coefficient) for coefficient, _ in self._terms)
        if maximum == 0.0:
            return np.zeros(len(self._terms), dtype=float)
        scaled = np.asarray(
            [
                abs(coefficient) / maximum
                for coefficient, _ in self._terms
            ],
            dtype=float,
        )
        return scaled / np.sum(scaled)

    def sample_term_index(self, rng: np.random.Generator | None = None) -> int:
        """Sample one LCP-term index from the coefficient distribution."""
        if not self._terms:
            raise ValueError("cannot sample from an empty LCH")
        probabilities = self.sampling_probabilities()
        if np.isclose(np.sum(probabilities), 0.0):
            raise ValueError(
                "cannot sample when total coefficient magnitude is zero"
            )
        probabilities = probabilities / np.sum(probabilities)
        generator = np.random.default_rng() if rng is None else rng
        return int(generator.choice(len(self), p=probabilities))

    def sample_term(
        self,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, LCP]:
        """Return a sampled ``(coefficient, LCP)`` pair."""
        return self.get_term(self.sample_term_index(rng=rng))

    def sample_term_with_cost(
        self,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, LCP, float]:
        """Return a sampled ``(coefficient, LCP, evolution_cost)`` tuple."""
        return self.get_term_with_cost(self.sample_term_index(rng=rng))

    def get_operator(self, index: int) -> LCP:
        """Return the LCP operator at ``index``."""
        self._validate_operator_index(index, len(self))
        return self._terms[index][1]

    def get_coefficient(self, index: int) -> float:
        """Return the outer coefficient at ``index``."""
        self._validate_operator_index(index, len(self))
        return self._terms[index][0]

    def get_term(self, index: int) -> tuple[float, LCP]:
        """Return ``(coefficient, LCP)`` at ``index``."""
        self._validate_operator_index(index, len(self))
        return self._terms[index]

    def get_term_cost(self, index: int) -> float:
        """Return the evolution cost at ``index``."""
        self._validate_operator_index(index, len(self))
        return self._costs[index]

    def get_term_with_cost(self, index: int) -> tuple[float, LCP, float]:
        """Return ``(coefficient, LCP, evolution_cost)`` at ``index``."""
        self._validate_operator_index(index, len(self))
        coefficient, operator = self._terms[index]
        return coefficient, operator, self._costs[index]

    def set_term_cost(self, index: int, evolution_cost: float) -> None:
        """Set the evolution cost at ``index``."""
        self._validate_operator_index(index, len(self))
        self._costs[index] = _real_value(
            evolution_cost,
            name="evolution_cost",
        )

    def to_csr(self) -> sparse.csr_matrix:
        """Return the flattened Hamiltonian as a CSR matrix."""
        return self._lcp.to_csr()

    def to_matrix(self) -> NDArray[np.complex128]:
        """Return the flattened Hamiltonian as a dense matrix."""
        return self._lcp.to_matrix()

    def operator_norm(self) -> float:
        """Return the flattened Hamiltonian spectral norm."""
        return self._lcp.operator_norm()


def _real_value(value: object, *, name: str) -> float:
    """Validate and convert a real numeric value."""
    if not isinstance(value, Number):
        raise TypeError(f"{name} must be a number")
    numeric = complex(value)
    if numeric.imag != 0.0:
        raise ValueError(f"{name} must be real")
    return float(numeric.real)


def _flatten_lcp_terms(
    terms: Iterable[tuple[float, LCP]],
    *,
    num_qubits: int,
) -> LCP:
    """Return the weighted Pauli sum without changing the LCH decomposition."""
    flattened: dict[str, float] = {}
    for outer_coefficient, operator in terms:
        for pauli, inner_coefficient in operator.terms.items():
            flattened[pauli] = (
                flattened.get(pauli, 0.0)
                + outer_coefficient * inner_coefficient
            )
    return LCP(flattened, num_qubits=num_qubits)
