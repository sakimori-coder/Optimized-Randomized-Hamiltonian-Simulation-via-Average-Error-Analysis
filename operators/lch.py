"""qDRIFT decomposition H = sum_j H_j, with Pauli sums H_j."""

import math

import numpy as np

from .lcp import LCP


class LCH:
    """Store Hamiltonian terms on the same qubits, including their coefficients.

    For an empty decomposition, supply num_qubits explicitly.
    """

    def __init__(self, terms=(), *, num_qubits=None):
        self.terms = tuple(terms)
        self.num_qubits = (
            self.terms[0].num_qubits if num_qubits is None else num_qubits
        )

        # Reconstruct H = sum_j H_j by collecting equal Pauli strings.
        pauli_terms = {}
        for operator in self.terms:
            for pauli, coefficient in operator.terms.items():
                pauli_terms[pauli] = pauli_terms.get(pauli, 0.0) + coefficient
        self.lcp = LCP(pauli_terms, num_qubits=self.num_qubits)

    def hs_norms(self):
        """h_j = sqrt(tau(H_j**2)) = sqrt(sum_P a_jP**2), tau = Tr / 2**n."""
        return np.array([operator.hs_norm() for operator in self.terms], dtype=float)

    def lambda_sum(self):
        """Lambda = sum_j h_j."""
        return math.fsum(self.hs_norms())

    def sampling_probabilities(self):
        """p_j = h_j / Lambda; a zero decomposition has no samples."""
        weights = self.hs_norms()
        total = math.fsum(weights)
        return weights / total if total != 0.0 else weights
