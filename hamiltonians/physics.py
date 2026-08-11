r"""Lattice-model Hamiltonian generators."""

from __future__ import annotations

from operators import LCP


def _pauli_string(
    num_qubits: int,
    operators: dict[int, str],
) -> str:
    symbols = ["I"] * num_qubits
    for site, operator in operators.items():
        symbols[site] = operator
    return "".join(symbols)


def _neighbor_pairs(
    num_qubits: int,
    periodic: bool,
) -> list[tuple[int, int]]:
    pairs = [(site, site + 1) for site in range(num_qubits - 1)]
    if periodic and num_qubits > 2:
        pairs.append((num_qubits - 1, 0))
    return pairs


def transverse_field_ising_chain(
    num_qubits: int,
    *,
    coupling: float = 1.0,
    transverse_field: float = 1.0,
    periodic: bool = False,
) -> LCP:
    r"""Return ``-J sum_i Z_i Z_(i+1) - h sum_i X_i``."""
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    terms: dict[str, float] = {}
    if coupling != 0.0:
        terms.update(
            {
                _pauli_string(
                    num_qubits,
                    {left: "Z", right: "Z"},
                ): -coupling
                for left, right in _neighbor_pairs(num_qubits, periodic)
            }
        )
    if transverse_field != 0.0:
        terms.update(
            {
                _pauli_string(num_qubits, {site: "X"}): -transverse_field
                for site in range(num_qubits)
            }
        )
    return LCP(terms, num_qubits=num_qubits)


def heisenberg_chain(
    num_qubits: int,
    *,
    coupling: float = 1.0,
    longitudinal_field: float = 0.0,
    periodic: bool = False,
) -> LCP:
    r"""Return ``J sum_i (XX+YY+ZZ) + h sum_i Z_i``."""
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    terms: dict[str, float] = {}
    if coupling != 0.0:
        for left, right in _neighbor_pairs(num_qubits, periodic):
            for operator in "XYZ":
                terms[
                    _pauli_string(
                        num_qubits,
                        {left: operator, right: operator},
                    )
                ] = coupling
    if longitudinal_field != 0.0:
        terms.update(
            {
                _pauli_string(num_qubits, {site: "Z"}): longitudinal_field
                for site in range(num_qubits)
            }
        )
    return LCP(terms, num_qubits=num_qubits)


def schwinger_model(
    num_qubits: int,
    *,
    x: float = 1.0,
    mu: float = 0.0,
    background_field: float = 0.0,
    include_identity: bool = False,
) -> LCP:
    r"""Return the dimensionless open-boundary Schwinger spin Hamiltonian.

    Gauge links are eliminated with Gauss's law after mapping staggered
    fermions to ``num_qubits`` spins.  With zero-based site indices, the
    generated Hamiltonian is

    ``W = x/2 sum_n (X_n X_(n+1) + Y_n Y_(n+1))``

    ``  + mu/2 sum_n [I + (-1)^n Z_n]``

    ``  + sum_n [l0 + 1/2 sum_(k<=n) (Z_k + (-1)^k)]^2``,

    where the last sum runs over the ``num_qubits - 1`` gauge links and
    ``l0 = background_field``.  By default, all identity terms are removed
    because they only produce a global phase.
    """
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")

    terms: dict[str, float] = {}

    for left, right in _neighbor_pairs(num_qubits, periodic=False):
        _add_term(
            terms,
            _pauli_string(num_qubits, {left: "X", right: "X"}),
            x / 2.0,
        )
        _add_term(
            terms,
            _pauli_string(num_qubits, {left: "Y", right: "Y"}),
            x / 2.0,
        )

    for site in range(num_qubits):
        _add_term(
            terms,
            _pauli_string(num_qubits, {site: "Z"}),
            mu * ((-1.0) ** site) / 2.0,
        )

    identity_coefficient = mu * num_qubits / 2.0
    for link in range(num_qubits - 1):
        constant = background_field + (0.5 if link % 2 == 0 else 0.0)
        identity_coefficient += constant**2 + (link + 1) / 4.0
        for site in range(link + 1):
            _add_term(
                terms,
                _pauli_string(num_qubits, {site: "Z"}),
                constant,
            )

    for left in range(num_qubits - 2):
        for right in range(left + 1, num_qubits - 1):
            _add_term(
                terms,
                _pauli_string(
                    num_qubits,
                    {left: "Z", right: "Z"},
                ),
                (num_qubits - 1 - right) / 2.0,
            )

    if include_identity:
        _add_term(terms, "I" * num_qubits, identity_coefficient)

    return LCP(terms, num_qubits=num_qubits)


def _add_term(
    terms: dict[str, float],
    pauli: str,
    coefficient: float,
) -> None:
    """Accumulate one coefficient and remove exact cancellations."""
    if coefficient == 0.0:
        return
    total = terms.get(pauli, 0.0) + coefficient
    if total == 0.0:
        terms.pop(pauli, None)
    else:
        terms[pauli] = total
