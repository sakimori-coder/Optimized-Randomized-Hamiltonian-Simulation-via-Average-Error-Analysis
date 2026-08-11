r"""Random Pauli Hamiltonian generators."""

from __future__ import annotations

import numpy as np

from operators import LCP


def _random_pauli_string(
    num_qubits: int,
    rng: np.random.Generator,
) -> str:
    paulis = np.array(["I", "X", "Y", "Z"])
    return "".join(rng.choice(paulis, size=num_qubits).tolist())


def build_random_lcp(
    num_terms: int,
    num_qubits: int,
    *,
    seed: int | None = None,
) -> LCP:
    """Build a random LCP with unique Pauli strings and real coefficients."""
    if not isinstance(num_terms, int) or num_terms < 0:
        raise ValueError("num_terms must be a non-negative integer")
    if not isinstance(num_qubits, int) or num_qubits < 0:
        raise ValueError("num_qubits must be a non-negative integer")
    if num_terms > 4**num_qubits:
        raise ValueError("num_terms cannot exceed 4^num_qubits")

    rng = np.random.default_rng(seed)
    terms: dict[str, float] = {}
    while len(terms) < num_terms:
        pauli = _random_pauli_string(num_qubits, rng)
        if pauli not in terms:
            terms[pauli] = float(rng.normal())
    return LCP(terms, num_qubits=num_qubits)
