r"""Jordan--Wigner algebra shared by SYK and real molecular Hamiltonians.

A Pauli word is represented by integer masks (x, z): I=(0,0), X=(1,0),
Z=(0,1), Y=(1,1). Bit q denotes spin orbital q; strings put q0 on the right.
Thus P(x,z) = i**popcount(x & z) X**x Z**z.
"""

from itertools import combinations_with_replacement

import numpy as np

from operators import LCP


# Render eight qubits at a time; avoid a Python loop over every character
# for each of the millions of Pauli terms in the larger Hamiltonians.
_PAULI_BLOCKS = tuple(
    "".join("IXZY"[((x >> q) & 1) + 2 * ((z >> q) & 1)] for q in range(7, -1, -1))
    for z in range(256) for x in range(256)
)


def pauli_string(x, z, num_qubits):
    return "".join(
        _PAULI_BLOCKS[((x >> q) & 255) + 256 * ((z >> q) & 255)]
        for q in range(8 * ((num_qubits - 1) // 8), -1, -8)
    )[-num_qubits:]


def pauli_product(x1, z1, x2, z2):
    """Return (phase, x, z) for P(x1,z1) P(x2,z2)."""
    x, z = x1 ^ x2, z1 ^ z2
    exponent = (
        (x1 & z1).bit_count() + (x2 & z2).bit_count()
        - (x & z).bit_count() + 2 * (z1 & x2).bit_count()
    )
    return (1, 1j, -1, -1j)[exponent % 4], x, z


def majorana_product(indices):
    """Return (phase, x, z) for the ordered product of Majorana operators."""
    x = z = 0
    phase = 1
    for index in indices:
        bit = 1 << (index // 2)
        # gamma_2q = X_q Z_<q; gamma_(2q+1) = i X_q Z_<=q.
        if z & bit:  # Move the next X through the accumulated Z factors.
            phase = -phase
        if index % 2:
            phase *= 1j
        x ^= bit
        z ^= bit - 1 + (index % 2) * bit
    return phase * (-1j) ** (x & z).bit_count(), x, z


def _density(p, q):
    """P_pq = (E_pq + E_qp)/2 - delta_pq I, E_pq = sum_spin a†_p a_q."""
    terms = {}
    for spin in range(2):
        a, b = 2 * p + spin, 2 * q + spin
        # P_pq = sum_spin i/4 (gamma_2a gamma_(2b+1) - gamma_(2a+1) gamma_2b).
        for indices, factor in (((2 * a, 2 * b + 1), 0.25j),
                                ((2 * a + 1, 2 * b), -0.25j)):
            phase, x, z = majorana_product(indices)
            terms[x, z] = terms.get((x, z), 0.0) + (factor * phase).real
    return [(x, z, value) for (x, z), value in terms.items() if value != 0.0]


def from_spatial_integrals(one_body, two_body, *, coefficient_tolerance=1e-12):
    r"""Convert real spatial integrals h_pq, (pq|rs) to an identity-free LCP.

    With the traceless densities P_pq defined above, up to a scalar identity,
      H = sum_pq h_eff[p,q] P_pq + 1/2 sum_pqrs (pq|rs) P_pq P_rs,
      h_eff[p,q] = h[p,q] + sum_r (pq|rr) - 1/2 sum_r (pr|rq).
    The two-electron integrals have the usual eightfold permutation symmetry.
    """
    num_orbitals = one_body.shape[0]
    effective = (
        0.5 * (one_body + one_body.T)
        + np.einsum("pqrr->pq", two_body)
        - 0.5 * np.einsum("prrq->pq", two_body)
    )
    coefficients = {}
    pairs = []
    for p in range(num_orbitals):
        for q in range(p + 1):
            weight = 1 if p == q else 2
            terms = _density(p, q)
            pairs.append((p, q, weight, terms))
            value = float(effective[p, q]) * weight
            for x, z, factor in terms:
                coefficients[x, z] = coefficients.get((x, z), 0.0) + value * factor

    for (p, q, w1, first), (r, s, w2, second) in combinations_with_replacement(pairs, 2):
        value = float(two_body[p, q, r, s]) * w1 * w2
        if value == 0.0:
            continue
        if (p, q) == (r, s):
            value *= 0.5
        for x1, z1, c1 in first:
            for x2, z2, c2 in second:
                phase, x, z = pauli_product(x1, z1, x2, z2)
                # Symmetric pair exchange cancels anticommuting products.
                if phase.real == 0 or (x == 0 and z == 0):
                    continue
                coefficients[x, z] = (
                    coefficients.get((x, z), 0.0) + value * c1 * c2 * phase.real
                )

    # Sum every contribution before applying the experiment's Pauli cutoff.
    return LCP(
        {pauli_string(x, z, 2 * num_orbitals): value
         for (x, z), value in coefficients.items()
         if abs(value) > coefficient_tolerance},
        num_qubits=2 * num_orbitals,
    )
