"""Hamiltonian generators grouped by problem family.

Molecular generators live in :mod:`hamiltonians.chemistry` because they use
optional PySCF and OpenFermion dependencies.
"""

from hamiltonians.pauli import build_lch_from_lcp_unit_cost
from hamiltonians.physics import (
    heisenberg_chain,
    schwinger_model,
    transverse_field_ising_chain,
)
from hamiltonians.random import build_random_lcp

__all__ = [
    "build_lch_from_lcp_unit_cost",
    "build_random_lcp",
    "heisenberg_chain",
    "schwinger_model",
    "transverse_field_ising_chain",
]
