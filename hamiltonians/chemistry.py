r"""Generate molecular Hamiltonians with PySCF and OpenFermion.

PySCF computes molecular integrals from a geometry, OpenFermion applies the
Jordan--Wigner transformation, and the resulting Pauli terms are converted to
the project's :class:`operators.LCP` representation.

The identity term is excluded from generated LCP objects by default because it
contributes only a global phase.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeAlias

import numpy as np

from operators import LCP


Atom: TypeAlias = tuple[str, tuple[float, float, float]]
Geometry: TypeAlias = tuple[Atom, ...]


def _normalized_geometry(
    geometry: Sequence[tuple[str, Sequence[float]]],
) -> Geometry:
    normalized: Geometry = tuple(
        (atom, tuple(float(value) for value in coordinates))
        for atom, coordinates in geometry
    )
    if any(len(coordinates) != 3 for _, coordinates in normalized):
        raise ValueError("each atom must have three Cartesian coordinates")
    return normalized


def _validated_nonnegative_real(value: float, *, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


@dataclass(frozen=True)
class GeneratedMolecularHamiltonian:
    """Immutable result of one PySCF and Jordan--Wigner calculation."""

    num_qubits: int
    terms: tuple[tuple[float, str], ...]
    hartree_fock_energy: float

    @property
    def identity_pauli(self) -> str:
        return "I" * self.num_qubits

    @property
    def identity_coefficient(self) -> float:
        return next(
            (
                coefficient
                for coefficient, pauli in self.terms
                if pauli == self.identity_pauli
            ),
            0.0,
        )

    def to_lcp(
        self,
        *,
        include_identity: bool = False,
    ) -> LCP:
        """Convert generated Pauli terms to a fresh LCP."""
        return LCP(
            {
                pauli: coefficient
                for coefficient, pauli in self.terms
                if include_identity or pauli != self.identity_pauli
            },
            num_qubits=self.num_qubits,
        )


@dataclass(frozen=True)
class MolecularHamiltonianPreset:
    """A named molecular specification from which an LCP is generated."""

    name: str
    description: str
    geometry: Geometry
    basis: str = "sto-3g"
    multiplicity: int = 1
    charge: int = 0
    occupied_indices: tuple[int, ...] | None = None
    active_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "geometry", _normalized_geometry(self.geometry))
        if self.occupied_indices is not None:
            object.__setattr__(
                self,
                "occupied_indices",
                tuple(self.occupied_indices),
            )
        if self.active_indices is not None:
            object.__setattr__(self, "active_indices", tuple(self.active_indices))

    def generate(
        self,
        *,
        coefficient_tolerance: float = 1e-12,
    ) -> GeneratedMolecularHamiltonian:
        """Run PySCF and return the cached Jordan--Wigner Hamiltonian."""
        tolerance = _validated_nonnegative_real(
            coefficient_tolerance,
            name="coefficient_tolerance",
        )
        return _generate_molecular_hamiltonian(
            self.geometry,
            self.basis,
            self.multiplicity,
            self.charge,
            self.occupied_indices,
            self.active_indices,
            tolerance,
        )

    @property
    def num_qubits(self) -> int:
        return self.generate().num_qubits

    @property
    def terms(self) -> tuple[tuple[float, str], ...]:
        return self.generate().terms

    @property
    def identity_pauli(self) -> str:
        return self.generate().identity_pauli

    @property
    def identity_coefficient(self) -> float:
        return self.generate().identity_coefficient

    def to_lcp(
        self,
        *,
        include_identity: bool = False,
        coefficient_tolerance: float = 1e-12,
    ) -> LCP:
        """Generate an LCP, optionally retaining the identity term."""
        return self.generate(
            coefficient_tolerance=coefficient_tolerance,
        ).to_lcp(
            include_identity=include_identity,
        )


H2_STO3G_JW = MolecularHamiltonianPreset(
    name="h2_sto3g_jw",
    description=(
        "H2, bond length 0.735 angstrom, STO-3G, Jordan-Wigner, "
        "four qubits"
    ),
    geometry=(
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 0.735)),
    ),
)

LIH_STO3G_ACTIVE_JW = MolecularHamiltonianPreset(
    name="lih_sto3g_active_jw",
    description=(
        "LiH, bond length 1.45 angstrom, STO-3G, frozen spatial orbital 0, "
        "active spatial orbitals 1-2, Jordan-Wigner, four qubits"
    ),
    geometry=(
        ("Li", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 1.45)),
    ),
    occupied_indices=(0,),
    active_indices=(1, 2),
)

H2O_STO3G_CAS_4E_4O_JW = MolecularHamiltonianPreset(
    name="h2o_sto3g_cas_4e_4o_jw",
    description=(
        "H2O, O-H distance 0.9576 angstrom, angle 104.5 degrees, STO-3G, "
        "CAS(4e,4o), Jordan-Wigner, eight qubits"
    ),
    geometry=(
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.75716, 0.0, 0.58626)),
        ("H", (-0.75716, 0.0, 0.58626)),
    ),
    occupied_indices=(0, 1, 2),
    active_indices=(3, 4, 5, 6),
)


def hydrogen_chain_preset(
    num_atoms: int,
    *,
    spacing: float = 1.0,
) -> MolecularHamiltonianPreset:
    """Construct an equally spaced hydrogen chain in the full STO-3G space."""
    if not isinstance(num_atoms, int) or isinstance(num_atoms, bool):
        raise TypeError("num_atoms must be an integer")
    if num_atoms <= 0:
        raise ValueError("num_atoms must be positive")
    spacing = float(spacing)
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")

    return MolecularHamiltonianPreset(
        name=f"h{num_atoms}_chain_sto3g_full_jw",
        description=(
            f"H{num_atoms} linear chain, spacing {spacing} angstrom, "
            "full STO-3G orbital space, Jordan-Wigner"
        ),
        geometry=tuple(
            ("H", (0.0, 0.0, atom_index * spacing))
            for atom_index in range(num_atoms)
        ),
        multiplicity=1 if num_atoms % 2 == 0 else 2,
    )


MOLECULAR_HAMILTONIANS: dict[str, MolecularHamiltonianPreset] = {
    H2_STO3G_JW.name: H2_STO3G_JW,
    LIH_STO3G_ACTIVE_JW.name: LIH_STO3G_ACTIVE_JW,
    H2O_STO3G_CAS_4E_4O_JW.name: H2O_STO3G_CAS_4E_4O_JW,
}


def _pauli_terms_from_qubit_operator(
    qubit_hamiltonian: object,
    *,
    num_qubits: int,
    coefficient_tolerance: float,
) -> tuple[tuple[float, str], ...]:
    """Convert OpenFermion's indexed terms to q_(n-1)...q_0 strings."""
    converted: list[tuple[float, str]] = []
    for indexed_paulis, raw_coefficient in qubit_hamiltonian.terms.items():
        coefficient = complex(raw_coefficient)
        if abs(coefficient) <= coefficient_tolerance:
            continue
        if abs(coefficient.imag) > coefficient_tolerance:
            raise ValueError("Jordan-Wigner Hamiltonian has a complex coefficient")

        symbols = ["I"] * num_qubits
        for qubit, symbol in indexed_paulis:
            symbols[num_qubits - 1 - qubit] = symbol
        converted.append((float(coefficient.real), "".join(symbols)))
    return tuple(sorted(converted, key=lambda term: term[1]))


@lru_cache(maxsize=8)
def _generate_molecular_hamiltonian(
    geometry: Geometry,
    basis: str,
    multiplicity: int,
    charge: int,
    occupied_indices: tuple[int, ...] | None,
    active_indices: tuple[int, ...] | None,
    coefficient_tolerance: float,
) -> GeneratedMolecularHamiltonian:
    """Run the electronic-structure calculation once per molecular preset."""
    from openfermion import MolecularData, get_fermion_operator, jordan_wigner
    from openfermionpyscf import run_pyscf

    with TemporaryDirectory(prefix="qdrift_pyscf_") as directory:
        molecule = MolecularData(
            list(geometry),
            basis,
            multiplicity,
            charge,
            filename=str(Path(directory) / "molecule"),
        )
        molecule = run_pyscf(molecule, run_scf=True)
        interaction_operator = molecule.get_molecular_hamiltonian(
            occupied_indices=(
                None if occupied_indices is None else list(occupied_indices)
            ),
            active_indices=(
                None if active_indices is None else list(active_indices)
            ),
        )
        qubit_hamiltonian = jordan_wigner(
            get_fermion_operator(interaction_operator)
        )

    num_qubits = int(interaction_operator.n_qubits)
    return GeneratedMolecularHamiltonian(
        num_qubits=num_qubits,
        terms=_pauli_terms_from_qubit_operator(
            qubit_hamiltonian,
            num_qubits=num_qubits,
            coefficient_tolerance=coefficient_tolerance,
        ),
        hartree_fock_energy=float(molecule.hf_energy),
    )


def generate_molecular_lcp(
    geometry: Sequence[tuple[str, Sequence[float]]],
    *,
    basis: str = "sto-3g",
    multiplicity: int = 1,
    charge: int = 0,
    occupied_indices: Sequence[int] | None = None,
    active_indices: Sequence[int] | None = None,
    include_identity: bool = False,
    coefficient_tolerance: float = 1e-12,
) -> LCP:
    """Generate an LCP for an arbitrary molecule from its geometry."""
    preset = MolecularHamiltonianPreset(
        name="custom",
        description="dynamically generated molecular Hamiltonian",
        geometry=_normalized_geometry(geometry),
        basis=basis,
        multiplicity=multiplicity,
        charge=charge,
        occupied_indices=(
            None if occupied_indices is None else tuple(occupied_indices)
        ),
        active_indices=None if active_indices is None else tuple(active_indices),
    )
    return preset.to_lcp(
        include_identity=include_identity,
        coefficient_tolerance=coefficient_tolerance,
    )
