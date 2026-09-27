r"""Quartic SYK Hamiltonian H = sum_{a<b<c<d} J_abcd gamma_a gamma_b gamma_c gamma_d.

The Majoranas satisfy {gamma_a, gamma_b} = 2 delta_ab I, and the independent
Gaussian couplings have variance 3! J**2 / (2n)**3 for n qubits.
Jordan--Wigner uses gamma_2j = Z_0 ... Z_(j-1) X_j and gamma_(2j+1) with Y_j.
Site 0 is the leftmost Pauli character, preserving the experiment convention.
"""

import math
from itertools import combinations

import numpy as np

from operators import LCP

from . import jordan_wigner


def generate(num_qubits, *, coupling_scale=1.0, seed=42):
    """Return one seeded SYK realization directly as an LCP."""
    num_majoranas = 2 * num_qubits
    coupling_std = math.sqrt(6) * coupling_scale / num_majoranas ** 1.5
    rng = np.random.default_rng(seed)
    couplings = rng.normal(0.0, coupling_std, math.comb(num_majoranas, 4))

    terms = {}
    for quartet, coupling in zip(combinations(range(num_majoranas), 4), couplings):
        if coupling == 0.0:
            continue
        phase, x, z = jordan_wigner.majorana_product(quartet)
        pauli = jordan_wigner.pauli_string(x, z, num_qubits)[::-1]
        terms[pauli] = coupling * phase.real

    return LCP(terms, num_qubits=num_qubits)
