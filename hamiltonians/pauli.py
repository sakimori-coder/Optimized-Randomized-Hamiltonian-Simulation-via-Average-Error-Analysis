"""Memory-efficient single-Pauli qDRIFT decomposition."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from operators import LCH, LCP


class SinglePauliLCH(LCH):
    """Store one shared LCP and materialize single-Pauli samples on demand."""

    def __init__(self, hamiltonian: LCP) -> None:
        if not isinstance(hamiltonian, LCP):
            raise TypeError("hamiltonian must be an LCP")
        self._lcp = hamiltonian
        self._num_qubits = hamiltonian.num_qubits

    def iter_pauli_terms(self):
        return iter(self._lcp.term_items())

    @property
    def lcp(self) -> LCP:
        return self._lcp

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def terms(self) -> list[tuple[float, LCP]]:
        return [
            (coefficient, LCP({pauli: 1.0}, num_qubits=self.num_qubits))
            for pauli, coefficient in self.iter_pauli_terms()
        ]

    def __len__(self) -> int:
        return len(self._lcp)

    def coefficient_one_norm(self) -> float:
        return math.fsum(
            abs(coefficient) for _, coefficient in self.iter_pauli_terms()
        )

    def sampling_probabilities(self) -> NDArray[np.float64]:
        magnitudes = np.fromiter(
            (abs(value) for _, value in self.iter_pauli_terms()),
            dtype=np.float64,
            count=len(self),
        )
        total = float(np.sum(magnitudes))
        return magnitudes / total if total != 0.0 else np.zeros_like(magnitudes)


def build_lch_from_lcp_unit_cost(hamiltonian: LCP) -> SinglePauliLCH:
    return SinglePauliLCH(hamiltonian)


__all__ = ["SinglePauliLCH", "build_lch_from_lcp_unit_cost"]
