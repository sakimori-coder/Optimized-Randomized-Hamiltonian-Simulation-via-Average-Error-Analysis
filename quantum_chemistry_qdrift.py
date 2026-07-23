r"""Generate molecular Hamiltonians and compare two qDRIFT decompositions.

PySCF computes molecular integrals from the geometry stored in each preset.
OpenFermion then applies the Jordan--Wigner transformation and the resulting
Pauli terms are converted to :class:`lch.LCH`.  No Pauli coefficients are
stored in this file.

The identity term is excluded by default because it contributes only a global
phase.  Use ``--include-identity`` to sample it as part of qDRIFT instead.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeAlias

import numpy as np

from commuting_operator_norm_estimator import BuiltInMethod as GroupNormMethod
from lch import LCH
from main import QDriftComparison, compare_pauli_and_grouped_qdrift
from qdrift_variance_estimator import BuiltInMethod as VarianceMethod


VARIANCE_METHODS: tuple[VarianceMethod, ...] = (
    "contraction_bound",
    "pauli_l1_bound",
    "anticommuting_bound",
    "sdp_bound",
    "exact",
)
GROUP_NORM_METHODS: tuple[GroupNormMethod, ...] = (
    "coefficient_l1",
    "lp",
    "exact",
)

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

    def to_lch(
        self,
        *,
        include_identity: bool = False,
        pauli_rotation_depth: float = 1.0,
    ) -> LCH:
        """Convert generated Pauli terms to a fresh LCH."""
        depth = _validated_nonnegative_real(
            pauli_rotation_depth,
            name="pauli_rotation_depth",
        )
        return LCH(
            [
                (
                    coefficient,
                    pauli,
                    0.0 if pauli == self.identity_pauli else depth,
                )
                for coefficient, pauli in self.terms
                if include_identity or pauli != self.identity_pauli
            ],
            num_qubits=self.num_qubits,
        )


@dataclass(frozen=True)
class MolecularHamiltonianPreset:
    """A named molecular specification from which an LCH is generated."""

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

    def to_lch(
        self,
        *,
        include_identity: bool = False,
        pauli_rotation_depth: float = 1.0,
        coefficient_tolerance: float = 1e-12,
    ) -> LCH:
        """Generate an LCH, assigning zero cost to the identity rotation."""
        return self.generate(
            coefficient_tolerance=coefficient_tolerance,
        ).to_lch(
            include_identity=include_identity,
            pauli_rotation_depth=pauli_rotation_depth,
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


def generate_molecular_lch(
    geometry: Sequence[tuple[str, Sequence[float]]],
    *,
    basis: str = "sto-3g",
    multiplicity: int = 1,
    charge: int = 0,
    occupied_indices: Sequence[int] | None = None,
    active_indices: Sequence[int] | None = None,
    include_identity: bool = False,
    pauli_rotation_depth: float = 1.0,
    coefficient_tolerance: float = 1e-12,
) -> LCH:
    """Generate an LCH for an arbitrary molecule from its geometry."""
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
    return preset.to_lch(
        include_identity=include_identity,
        pauli_rotation_depth=pauli_rotation_depth,
        coefficient_tolerance=coefficient_tolerance,
    )


def run_molecular_qdrift_experiment(
    preset: MolecularHamiltonianPreset,
    *,
    time: float = 1.0,
    epsilon: float = 0.01,
    variance_methods: Sequence[VarianceMethod] = ("sdp_bound",),
    group_norm_method: GroupNormMethod = "lp",
    max_group_size: int = 12,
    include_identity: bool = False,
    pauli_rotation_depth: float = 1.0,
) -> dict[str, QDriftComparison]:
    """Compare Pauli and grouped qDRIFT for one molecular preset."""
    hamiltonian = preset.to_lch(
        include_identity=include_identity,
        pauli_rotation_depth=pauli_rotation_depth,
    )
    comparisons: dict[str, QDriftComparison] = {}
    for method in variance_methods:
        comparisons[method] = compare_pauli_and_grouped_qdrift(
            hamiltonian,
            time=time,
            epsilon=epsilon,
            variance_method=method,
            group_norm_method=group_norm_method,
            max_group_size=max_group_size,
        )
    return comparisons


def _maximum_probability(probabilities: np.ndarray) -> float:
    return float(np.max(probabilities)) if probabilities.size else 0.0


def _print_summary(comparisons: dict[str, QDriftComparison]) -> None:
    print(
        f"{'variance':<22} {'pauli N':>9} {'group N':>9} "
        f"{'pauli depth':>13} {'group depth':>13} {'ratio':>10}"
    )
    for method, comparison in comparisons.items():
        print(
            f"{method:<22} "
            f"{comparison.pauli.steps:>9d} "
            f"{comparison.grouped.steps:>9d} "
            f"{comparison.pauli.expected_total_depth:>13.6f} "
            f"{comparison.grouped.expected_total_depth:>13.6f} "
            f"{comparison.grouped_to_pauli_depth_ratio:>10.6f}"
        )


def _print_details(comparison: QDriftComparison) -> None:
    print("\ndetailed result for", comparison.variance_method)
    print(
        f"{'decomposition':<14} {'terms':>7} {'lambda':>12} {'B':>12} "
        f"{'lambda^2 B':>12} {'depth/step':>12} {'p_max':>12}"
    )
    for name, cost in (("pauli", comparison.pauli), ("grouped", comparison.grouped)):
        print(
            f"{name:<14} "
            f"{cost.num_sampling_terms:>7d} "
            f"{cost.lambda_sum:>12.8f} "
            f"{cost.normalized_variance_bound:>12.8f} "
            f"{cost.variance_constant:>12.8f} "
            f"{cost.expected_depth_per_step:>12.8f} "
            f"{_maximum_probability(cost.sampling_probabilities):>12.8f}"
        )


def main() -> None:
    parser = ArgumentParser(
        description="Compare Pauli and grouped qDRIFT for molecular Hamiltonians"
    )
    parser.add_argument(
        "--hamiltonian",
        choices=sorted(MOLECULAR_HAMILTONIANS),
        default=H2_STO3G_JW.name,
    )
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument(
        "--variance-method",
        choices=("all", *VARIANCE_METHODS),
        default="all",
    )
    parser.add_argument(
        "--group-norm-method",
        choices=GROUP_NORM_METHODS,
        default="lp",
    )
    parser.add_argument("--max-group-size", type=int, default=12)
    parser.add_argument("--pauli-rotation-depth", type=float, default=1.0)
    parser.add_argument("--include-identity", action="store_true")
    parser.add_argument("--print-groups", action="store_true")
    parser.add_argument("--print-probabilities", action="store_true")
    args = parser.parse_args()

    preset = MOLECULAR_HAMILTONIANS[args.hamiltonian]
    print(f"Hamiltonian: {preset.name}")
    print(preset.description)
    print("Generating the Hamiltonian with PySCF and OpenFermion...", flush=True)
    generated = preset.generate()
    variance_methods = (
        VARIANCE_METHODS
        if args.variance_method == "all"
        else (args.variance_method,)
    )
    comparisons = run_molecular_qdrift_experiment(
        preset,
        time=args.time,
        epsilon=args.epsilon,
        variance_methods=variance_methods,
        group_norm_method=args.group_norm_method,
        max_group_size=args.max_group_size,
        include_identity=args.include_identity,
        pauli_rotation_depth=args.pauli_rotation_depth,
    )

    print(
        f"generated_qubits={generated.num_qubits}, "
        f"generated_pauli_terms={len(generated.terms)}, "
        f"hartree_fock_energy={generated.hartree_fock_energy:.10f}"
    )
    print(f"include_identity={args.include_identity}")
    if not args.include_identity and generated.identity_coefficient != 0.0:
        print(
            "removed identity coefficient="
            f"{generated.identity_coefficient:.8f} (global phase)"
        )
    print(f"time={args.time}, epsilon={args.epsilon}")
    print(f"group_norm_method={args.group_norm_method}")
    _print_summary(comparisons)

    selected = comparisons["sdp_bound"] if "sdp_bound" in comparisons else next(
        iter(comparisons.values())
    )
    _print_details(selected)

    if args.print_groups:
        print("\ncommuting groups")
        for index, group in enumerate(selected.groups):
            print(
                f"group[{index}]: weight={group.normalization_weight:.10f}, "
                f"depth={group.rotation_depth}, "
                f"paulis=[{', '.join(group.pauli_strings)}]"
            )
    if args.print_probabilities:
        print("\npauli probabilities=", selected.pauli.sampling_probabilities)
        print("grouped probabilities=", selected.grouped.sampling_probabilities)


if __name__ == "__main__":
    main()
