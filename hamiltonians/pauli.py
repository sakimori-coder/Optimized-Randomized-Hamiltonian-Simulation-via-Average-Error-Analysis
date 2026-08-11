r"""Conversions between Pauli Hamiltonian representations."""

from __future__ import annotations

from operators import LCH, LCP


def build_lch_from_lcp_unit_cost(hamiltonian: LCP) -> LCH:
    """Convert an LCP into single-Pauli qDRIFT sampling terms."""
    return LCH(
        [
            (
                coefficient,
                LCP(
                    {pauli: 1.0},
                    num_qubits=hamiltonian.num_qubits,
                ),
                0.0 if all(symbol == "I" for symbol in pauli) else 1.0,
            )
            for pauli, coefficient in hamiltonian.terms.items()
        ],
        num_qubits=hamiltonian.num_qubits,
    )
