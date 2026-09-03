"""Minimal real Pauli-sum representation."""

from __future__ import annotations

from collections.abc import ItemsView, Mapping
from numbers import Number


_VALID_PAULIS = frozenset("IXYZ")


class LCP:
    """Represent ``sum_P coefficient[P] P`` on a fixed number of qubits."""

    def __init__(
        self,
        terms: Mapping[str, complex] | None = None,
        *,
        num_qubits: int | None = None,
    ) -> None:
        if num_qubits is not None and (
            not isinstance(num_qubits, int)
            or isinstance(num_qubits, bool)
            or num_qubits < 0
        ):
            raise ValueError("num_qubits must be a non-negative integer")
        terms = {} if terms is None else terms
        if not isinstance(terms, Mapping):
            raise TypeError("terms must map Pauli strings to coefficients")

        inferred = num_qubits
        validated: dict[str, float] = {}
        for pauli, coefficient in terms.items():
            if not isinstance(pauli, str) or set(pauli) - _VALID_PAULIS:
                raise ValueError(f"invalid Pauli string: {pauli!r}")
            if not isinstance(coefficient, Number):
                raise TypeError("Pauli coefficients must be numbers")
            numeric = complex(coefficient)
            if numeric.imag != 0.0:
                raise ValueError("Pauli coefficients must be real")
            if inferred is None:
                inferred = len(pauli)
            if len(pauli) != inferred:
                raise ValueError("all Pauli strings must have equal length")
            validated[pauli] = float(numeric.real)
        if inferred is None:
            raise ValueError("num_qubits is required for an empty Pauli sum")
        self._terms = validated
        self._num_qubits = inferred

    @property
    def terms(self) -> dict[str, float]:
        return dict(self._terms)

    def term_items(self) -> ItemsView[str, float]:
        return self._terms.items()

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    def __len__(self) -> int:
        return len(self._terms)

    def __mul__(self, scalar: complex) -> LCP:
        if not isinstance(scalar, Number):
            return NotImplemented
        numeric = complex(scalar)
        if numeric.imag != 0.0:
            raise ValueError("LCP scalar must be real")
        return LCP(
            {
                pauli: coefficient * numeric.real
                for pauli, coefficient in self._terms.items()
            },
            num_qubits=self.num_qubits,
        )

    def __rmul__(self, scalar: complex) -> LCP:
        return self * scalar

    @staticmethod
    def pauli_strings_commute(left: str, right: str) -> bool:
        """Return whether two equal-length Pauli strings commute."""
        if len(left) != len(right):
            raise ValueError("Pauli strings must have equal length")
        anti_commuting_positions = sum(
            a != "I" and b != "I" and a != b
            for a, b in zip(left, right)
        )
        return anti_commuting_positions % 2 == 0
