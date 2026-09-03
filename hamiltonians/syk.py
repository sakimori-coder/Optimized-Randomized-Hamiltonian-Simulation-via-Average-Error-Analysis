r"""Quartic Majorana Sachdev--Ye--Kitaev Hamiltonians.

The implementation uses Jordan--Wigner Majorana operators satisfying
``{gamma_a, gamma_b} = 2 delta_ab I``.  ``num_qubits`` qubits therefore
represent ``2 * num_qubits`` Majorana modes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from numbers import Real

import numpy as np

from operators import LCP


_PAULI_PRODUCT: dict[tuple[str, str], tuple[complex, str]] = {
    ("I", "I"): (1.0, "I"),
    ("I", "X"): (1.0, "X"),
    ("I", "Y"): (1.0, "Y"),
    ("I", "Z"): (1.0, "Z"),
    ("X", "I"): (1.0, "X"),
    ("Y", "I"): (1.0, "Y"),
    ("Z", "I"): (1.0, "Z"),
    ("X", "X"): (1.0, "I"),
    ("Y", "Y"): (1.0, "I"),
    ("Z", "Z"): (1.0, "I"),
    ("X", "Y"): (1.0j, "Z"),
    ("Y", "X"): (-1.0j, "Z"),
    ("Y", "Z"): (1.0j, "X"),
    ("Z", "Y"): (-1.0j, "X"),
    ("Z", "X"): (1.0j, "Y"),
    ("X", "Z"): (-1.0j, "Y"),
}


@dataclass(frozen=True)
class GeneratedSykHamiltonian:
    """One reproducible disorder realization of the quartic SYK model."""

    num_qubits: int
    num_majoranas: int
    terms: tuple[tuple[float, str], ...]
    coupling_scale: float
    coupling_standard_deviation: float
    seed: int | None

    @property
    def identity_pauli(self) -> str:
        return "I" * self.num_qubits

    @property
    def identity_coefficient(self) -> float:
        return 0.0

    def to_lcp(self, *, include_identity: bool = False) -> LCP:
        """Return the Jordan--Wigner Pauli Hamiltonian as an ``LCP``."""
        del include_identity  # Quartic products of distinct modes are traceless.
        return LCP(
            {
                pauli: coefficient
                for coefficient, pauli in self.terms
            },
            num_qubits=self.num_qubits,
        )


@dataclass(frozen=True)
class SykHamiltonianPreset:
    """Parameters defining one seeded quartic Majorana SYK realization."""

    num_qubits: int
    coupling_scale: float = 1.0
    seed: int | None = 42

    def __post_init__(self) -> None:
        _validate_parameters(self.num_qubits, self.coupling_scale)

    @property
    def name(self) -> str:
        return f"syk_q4_{self.num_majoranas}_majorana_jw"

    @property
    def description(self) -> str:
        return (
            "quartic Majorana SYK, "
            f"{self.num_majoranas} Majoranas, Gaussian coupling scale "
            f"J={self.coupling_scale}, seed={self.seed}, Jordan-Wigner"
        )

    @property
    def num_majoranas(self) -> int:
        return 2 * self.num_qubits

    def generate(self) -> GeneratedSykHamiltonian:
        """Generate this disorder realization."""
        return _generate_syk_hamiltonian(
            self.num_qubits,
            float(self.coupling_scale),
            self.seed,
        )

def _validate_parameters(num_qubits: int, coupling_scale: float) -> None:
    if (
        not isinstance(num_qubits, int)
        or isinstance(num_qubits, bool)
        or num_qubits < 2
    ):
        raise ValueError("num_qubits must be an integer of at least two")
    if (
        not isinstance(coupling_scale, Real)
        or isinstance(coupling_scale, bool)
        or not math.isfinite(float(coupling_scale))
        or float(coupling_scale) < 0.0
    ):
        raise ValueError("coupling_scale must be finite and non-negative")


def _generate_syk_hamiltonian(
    num_qubits: int,
    coupling_scale: float,
    seed: int | None,
) -> GeneratedSykHamiltonian:
    _validate_parameters(num_qubits, coupling_scale)
    num_majoranas = 2 * num_qubits
    coupling_std = (
        math.sqrt(math.factorial(3))
        * coupling_scale
        / (num_majoranas ** 1.5)
    )
    majoranas = tuple(
        _jordan_wigner_majorana(index, num_qubits)
        for index in range(num_majoranas)
    )
    quartets = combinations(range(num_majoranas), 4)
    rng = np.random.default_rng(seed)
    couplings = rng.normal(
        loc=0.0,
        scale=coupling_std,
        size=math.comb(num_majoranas, 4),
    )

    # Products of distinct Majorana subsets are distinct Pauli-basis elements.
    # Preserve the deterministic combinations order instead of building a
    # multi-million-entry dictionary and sorting it for large sweeps.
    terms: list[tuple[float, str]] = []
    for quartet, coupling in zip(quartets, couplings):
        phase, pauli = _multiply_pauli_strings(
            tuple(majoranas[index] for index in quartet)
        )
        if not math.isclose(phase.imag, 0.0, abs_tol=1e-12):
            raise ArithmeticError("quartic Majorana product is not Hermitian")
        coefficient = float(coupling * phase.real)
        if coefficient != 0.0:
            terms.append((coefficient, pauli))

    return GeneratedSykHamiltonian(
        num_qubits=num_qubits,
        num_majoranas=num_majoranas,
        terms=tuple(terms),
        coupling_scale=coupling_scale,
        coupling_standard_deviation=coupling_std,
        seed=seed,
    )


def _jordan_wigner_majorana(index: int, num_qubits: int) -> str:
    site, parity = divmod(index, 2)
    return "Z" * site + ("X" if parity == 0 else "Y") + "I" * (
        num_qubits - site - 1
    )


def _multiply_pauli_strings(paulis: tuple[str, ...]) -> tuple[complex, str]:
    if not paulis:
        return 1.0, ""
    product = "I" * len(paulis[0])
    phase = 1.0 + 0.0j
    for pauli in paulis:
        symbols: list[str] = []
        for left, right in zip(product, pauli):
            local_phase, symbol = _PAULI_PRODUCT[(left, right)]
            phase *= local_phase
            symbols.append(symbol)
        product = "".join(symbols)
    return phase, product
