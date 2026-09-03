r"""Generate molecular Hamiltonians with PySCF and OpenFermion.

PySCF computes molecular integrals from a geometry, OpenFermion applies the
Jordan--Wigner transformation, and the resulting Pauli terms are converted to
the project's :class:`operators.LCP` representation.

The identity term is excluded from generated LCP objects by default because it
contributes only a global phase.
"""

from __future__ import annotations

import math
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


# OpenFermion's own ``SymbolicOperator.compress`` uses 1e-8 as its default
# roundoff tolerance.  Keep that tolerance only for the forbidden imaginary
# part: real coefficients continue to use the user-selected coefficient
# cutoff, which is 1e-12 by default.
_HERMITICITY_ROUNDOFF_TOLERANCE = 1e-8


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


def _linear_symmetric_geometry(
    center: str,
    outer: str,
    bond_length: float,
) -> Geometry:
    """Return ``outer-center-outer`` on the z axis."""
    return (
        (outer, (0.0, 0.0, -bond_length)),
        (center, (0.0, 0.0, 0.0)),
        (outer, (0.0, 0.0, bond_length)),
    )


def _bent_symmetric_geometry(
    center: str,
    outer: str,
    bond_length: float,
    angle_degrees: float,
) -> Geometry:
    """Return a planar two-bond geometry symmetric about the z axis."""
    half_angle = math.radians(angle_degrees / 2.0)
    transverse = bond_length * math.sin(half_angle)
    longitudinal = bond_length * math.cos(half_angle)
    return (
        (center, (0.0, 0.0, 0.0)),
        (outer, (transverse, 0.0, longitudinal)),
        (outer, (-transverse, 0.0, longitudinal)),
    )


def _tetrahedral_geometry(bond_length: float) -> Geometry:
    """Return methane with a carbon at the origin."""
    coordinate = bond_length / math.sqrt(3.0)
    return (
        ("C", (0.0, 0.0, 0.0)),
        ("H", (coordinate, coordinate, coordinate)),
        ("H", (-coordinate, -coordinate, coordinate)),
        ("H", (-coordinate, coordinate, -coordinate)),
        ("H", (coordinate, -coordinate, -coordinate)),
    )


def _trigonal_pyramidal_geometry(
    bond_length: float,
    angle_degrees: float,
) -> Geometry:
    """Return a C3v ammonia geometry with nitrogen at the origin."""
    bond_angle = math.radians(angle_degrees)
    polar_cosine_squared = (2.0 * math.cos(bond_angle) + 1.0) / 3.0
    polar_cosine = math.sqrt(polar_cosine_squared)
    radius = bond_length * math.sqrt(1.0 - polar_cosine_squared)
    height = bond_length * polar_cosine
    hydrogens = tuple(
        (
            "H",
            (
                radius * math.cos(2.0 * math.pi * index / 3.0),
                radius * math.sin(2.0 * math.pi * index / 3.0),
                height,
            ),
        )
        for index in range(3)
    )
    return (("N", (0.0, 0.0, 0.0)), *hydrogens)


def _benzene_geometry(
    carbon_carbon_distance: float,
    carbon_hydrogen_distance: float,
) -> Geometry:
    """Return a planar D6h benzene geometry centered at the origin."""
    hydrogen_radius = carbon_carbon_distance + carbon_hydrogen_distance
    carbons = tuple(
        (
            "C",
            (
                carbon_carbon_distance
                * math.cos(math.pi * index / 3.0),
                carbon_carbon_distance
                * math.sin(math.pi * index / 3.0),
                0.0,
            ),
        )
        for index in range(6)
    )
    hydrogens = tuple(
        (
            "H",
            (
                hydrogen_radius * math.cos(math.pi * index / 3.0),
                hydrogen_radius * math.sin(math.pi * index / 3.0),
                0.0,
            ),
        )
        for index in range(6)
    )
    return carbons + hydrogens


NH3_STO3G_FULL_JW = MolecularHamiltonianPreset(
    name="nh3_sto3g_full_jw",
    description=(
        "NH3, N-H distance 1.012 angstrom, angle 106.7 degrees, full "
        "STO-3G orbital space, Jordan-Wigner, 16 qubits"
    ),
    geometry=_trigonal_pyramidal_geometry(1.012, 106.7),
)

CH4_STO3G_FULL_JW = MolecularHamiltonianPreset(
    name="ch4_sto3g_full_jw",
    description=(
        "CH4, tetrahedral C-H distance 1.087 angstrom, full STO-3G "
        "orbital space, Jordan-Wigner, 18 qubits"
    ),
    geometry=_tetrahedral_geometry(1.087),
)

CO_STO3G_FULL_JW = MolecularHamiltonianPreset(
    name="co_sto3g_full_jw",
    description=(
        "CO, bond length 1.128 angstrom, full STO-3G orbital space, "
        "Jordan-Wigner, 20 qubits"
    ),
    geometry=(
        ("C", (0.0, 0.0, 0.0)),
        ("O", (0.0, 0.0, 1.128)),
    ),
)

H2S_STO3G_FULL_JW = MolecularHamiltonianPreset(
    name="h2s_sto3g_full_jw",
    description=(
        "H2S, S-H distance 1.336 angstrom, angle 92.1 degrees, full "
        "STO-3G orbital space, Jordan-Wigner, 22 qubits"
    ),
    geometry=_bent_symmetric_geometry("S", "H", 1.336, 92.1),
)

C2H2_STO3G_FULL_JW = MolecularHamiltonianPreset(
    name="c2h2_sto3g_full_jw",
    description=(
        "C2H2, C-C distance 1.203 angstrom, C-H distance 1.060 angstrom, "
        "full STO-3G orbital space, Jordan-Wigner, 24 qubits"
    ),
    geometry=(
        ("H", (0.0, 0.0, -1.6615)),
        ("C", (0.0, 0.0, -0.6015)),
        ("C", (0.0, 0.0, 0.6015)),
        ("H", (0.0, 0.0, 1.6615)),
    ),
)

CO2_STO3G_FULL_JW = MolecularHamiltonianPreset(
    name="co2_sto3g_full_jw",
    description=(
        "CO2, linear C-O distance 1.160 angstrom, full STO-3G orbital "
        "space, Jordan-Wigner, 30 qubits"
    ),
    geometry=_linear_symmetric_geometry("C", "O", 1.160),
)

C6H6_STO3G_FULL_JW = MolecularHamiltonianPreset(
    name="c6h6_sto3g_full_jw",
    description=(
        "C6H6, planar D6h geometry with C-C distance 1.397 angstrom and "
        "C-H distance 1.090 angstrom, full STO-3G orbital space, "
        "Jordan-Wigner, 72 qubits"
    ),
    geometry=_benzene_geometry(1.397, 1.090),
)


MOLECULAR_HAMILTONIANS: dict[str, MolecularHamiltonianPreset] = {
    H2_STO3G_JW.name: H2_STO3G_JW,
    LIH_STO3G_ACTIVE_JW.name: LIH_STO3G_ACTIVE_JW,
    H2O_STO3G_CAS_4E_4O_JW.name: H2O_STO3G_CAS_4E_4O_JW,
    NH3_STO3G_FULL_JW.name: NH3_STO3G_FULL_JW,
    CH4_STO3G_FULL_JW.name: CH4_STO3G_FULL_JW,
    CO_STO3G_FULL_JW.name: CO_STO3G_FULL_JW,
    H2S_STO3G_FULL_JW.name: H2S_STO3G_FULL_JW,
    C2H2_STO3G_FULL_JW.name: C2H2_STO3G_FULL_JW,
    CO2_STO3G_FULL_JW.name: CO2_STO3G_FULL_JW,
    C6H6_STO3G_FULL_JW.name: C6H6_STO3G_FULL_JW,
}


def _pauli_terms_from_qubit_operator(
    qubit_hamiltonian: object,
    *,
    num_qubits: int,
    coefficient_tolerance: float,
    project_hermitian: bool = False,
) -> tuple[tuple[float, str], ...]:
    """Convert OpenFermion's indexed terms to q_(n-1)...q_0 strings.

    ``project_hermitian=True`` takes the Hermitian part by retaining the real
    coefficient of each Hermitian Pauli basis element.  This is used only for
    molecular Hamiltonians constructed from real PySCF integrals.  Generic
    callers retain the explicit imaginary-coefficient validation.
    """
    converted: list[tuple[float, str]] = []
    for indexed_paulis, raw_coefficient in qubit_hamiltonian.terms.items():
        coefficient = complex(raw_coefficient)
        imaginary_tolerance = max(
            coefficient_tolerance,
            _HERMITICITY_ROUNDOFF_TOLERANCE,
            1e-12 * abs(coefficient.real),
        )
        if (
            not project_hermitian
            and abs(coefficient.imag) > imaginary_tolerance
        ):
            raise ValueError("Jordan-Wigner Hamiltonian has a complex coefficient")
        real_coefficient = float(coefficient.real)
        if abs(real_coefficient) <= coefficient_tolerance:
            continue

        symbols = ["I"] * num_qubits
        for qubit, symbol in indexed_paulis:
            symbols[num_qubits - 1 - qubit] = symbol
        converted.append((real_coefficient, "".join(symbols)))
    return tuple(sorted(converted, key=lambda term: term[1]))


@lru_cache(maxsize=16)
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
            project_hermitian=True,
        ),
        hartree_fock_energy=float(molecule.hf_energy),
    )
