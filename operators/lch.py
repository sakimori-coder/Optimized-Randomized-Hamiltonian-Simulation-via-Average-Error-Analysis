"""Minimal qDRIFT decomposition into Pauli-sum samples."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Number

import numpy as np
from numpy.typing import NDArray

from .lcp import LCP


class LCH:
    """Represent ``H = sum_j c_j A_j`` for qDRIFT sampling."""

    def __init__(
        self,
        terms: Iterable[tuple[float, LCP]] | None = None,
        *,
        num_qubits: int | None = None,
    ) -> None:
        raw_terms = [] if terms is None else list(terms)
        inferred = num_qubits
        validated: list[tuple[float, LCP]] = []
        for raw in raw_terms:
            if not isinstance(raw, tuple) or len(raw) != 2:
                raise TypeError("each LCH term must be (coefficient, LCP)")
            coefficient, operator = raw
            if not isinstance(coefficient, Number):
                raise TypeError("outer coefficients must be numbers")
            numeric = complex(coefficient)
            if numeric.imag != 0.0:
                raise ValueError("outer coefficients must be real")
            if not isinstance(operator, LCP):
                raise TypeError("each sampled operator must be an LCP")
            if inferred is None:
                inferred = operator.num_qubits
            if operator.num_qubits != inferred:
                raise ValueError("all sampled operators must use equal qubits")
            validated.append(
                (
                    float(numeric.real),
                    LCP(operator.terms, num_qubits=operator.num_qubits),
                )
            )
        if inferred is None:
            raise ValueError("num_qubits is required for an empty LCH")
        self._num_qubits = inferred
        self._terms = validated
        flattened: dict[str, float] = {}
        for outer, operator in validated:
            for pauli, inner in operator.term_items():
                flattened[pauli] = flattened.get(pauli, 0.0) + outer * inner
        self._lcp = LCP(flattened, num_qubits=inferred)

    @property
    def lcp(self) -> LCP:
        return self._lcp

    @property
    def terms(self) -> list[tuple[float, LCP]]:
        return list(self._terms)

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    def __len__(self) -> int:
        return len(self._terms)

    def coefficient_one_norm(self) -> float:
        return float(sum(abs(coefficient) for coefficient, _ in self._terms))

    def sampling_probabilities(self) -> NDArray[np.float64]:
        if not self._terms:
            return np.array([], dtype=np.float64)
        maximum = max(abs(coefficient) for coefficient, _ in self._terms)
        if maximum == 0.0:
            return np.zeros(len(self._terms), dtype=np.float64)
        scaled = np.asarray(
            [abs(coefficient) / maximum for coefficient, _ in self._terms],
            dtype=np.float64,
        )
        return scaled / np.sum(scaled)
