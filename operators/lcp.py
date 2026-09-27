"""Real Pauli sum H = sum_P a_P P."""

import math


class LCP:
    """Store real coefficients for equal-length strings of I, X, Y, Z.

    For an empty sum, supply num_qubits explicitly.
    """

    def __init__(self, terms=None, *, num_qubits=None):
        terms = {} if terms is None else terms
        self.terms = {pauli: float(coefficient) for pauli, coefficient in terms.items()}
        self.num_qubits = (
            len(next(iter(self.terms))) if num_qubits is None else num_qubits
        )

    def hs_norm(self):
        """Normalized HS norm: sqrt(Tr(H**2) / 2**n) = sqrt(sum_P a_P**2)."""
        return math.sqrt(math.fsum(value * value for value in self.terms.values()))
