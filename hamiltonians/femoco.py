r"""Load the Reiher FeMoco CAS(54e,54o) Hamiltonian from FCIDUMP."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np

from operators import LCP

from .chemistry import _validated_nonnegative_real


FEMOCO_NAME = "femoco_reiher_54e_54o_jw"
FEMOCO_SPATIAL_ORBITALS = 54
FEMOCO_ACTIVE_ELECTRONS = 54

_FOUR_UNIQUE_REAL_PATTERNS: tuple[tuple[str, float], ...] = (
    ("XXXX", -0.125),
    ("XXYY", 0.125),
    ("XYXY", -0.125),
    ("XYYX", -0.125),
    ("YXXY", -0.125),
    ("YXYX", -0.125),
    ("YYXX", 0.125),
    ("YYYY", -0.125),
)


def _triangular_index(row: int, column: int) -> int:
    if row < column:
        row, column = column, row
    return row * (row + 1) // 2 + column


def _packed_eri(
    packed: np.ndarray,
    first: int,
    second: int,
    third: int,
    fourth: int,
) -> float:
    first_pair = _triangular_index(first, second)
    second_pair = _triangular_index(third, fourth)
    return float(packed[_triangular_index(first_pair, second_pair)])


def _add(
    coefficients: dict[tuple[int, int], float],
    x_mask: int,
    z_mask: int,
    coefficient: float,
) -> None:
    if coefficient != 0.0:
        key = (x_mask, z_mask)
        coefficients[key] = coefficients.get(key, 0.0) + coefficient


def _range_mask(start: int, stop: int) -> int:
    return ((1 << stop) - 1) ^ ((1 << start) - 1)


def _add_one_body_hermitian(
    coefficients: dict[tuple[int, int], float],
    first: int,
    second: int,
    coefficient: float,
) -> None:
    if coefficient == 0.0:
        return
    if first == second:
        _add(coefficients, 0, 0, 0.5 * coefficient)
        _add(coefficients, 0, 1 << first, -0.5 * coefficient)
        return
    if first > second:
        first, second = second, first
    x_mask = (1 << first) | (1 << second)
    parity_mask = _range_mask(first + 1, second)
    _add(coefficients, x_mask, parity_mask, 0.5 * coefficient)
    _add(coefficients, x_mask, parity_mask | x_mask, 0.5 * coefficient)


def _masks_from_symbols(
    indexed_symbols: tuple[tuple[int, str], ...],
    parity_mask: int,
) -> tuple[int, int]:
    x_mask = 0
    z_mask = parity_mask
    for index, symbol in indexed_symbols:
        x_mask |= 1 << index
        if symbol == "Y":
            z_mask |= 1 << index
    return x_mask, z_mask


def _add_two_body_hermitian(
    coefficients: dict[tuple[int, int], float],
    first: int,
    second: int,
    third: int,
    fourth: int,
    coefficient: float,
) -> None:
    """Real-coefficient specialization of OpenFermion's JW pair map."""
    if coefficient == 0.0 or first == second or third == fourth:
        return
    unique_count = len({first, second, third, fourth})
    if unique_count == 4:
        sign = -1.0 if (first > second) ^ (third > fourth) else 1.0
        indices = (first, second, third, fourth)
        for symbols, pattern_coefficient in _FOUR_UNIQUE_REAL_PATTERNS:
            indexed = tuple(sorted(zip(indices, symbols)))
            a, b, c, d = (index for index, _ in indexed)
            parity = _range_mask(a + 1, b) | _range_mask(c + 1, d)
            x_mask, z_mask = _masks_from_symbols(indexed, parity)
            _add(
                coefficients,
                x_mask,
                z_mask,
                sign * pattern_coefficient * coefficient,
            )
        return
    if unique_count == 3:
        if first == third:
            a, b = (fourth, second) if second > fourth else (second, fourth)
            coefficient = -coefficient
            repeated = first
        elif first == fourth:
            a, b = (third, second) if second > third else (second, third)
            repeated = first
        elif second == third:
            a, b = (fourth, first) if first > fourth else (first, fourth)
            repeated = second
        else:
            a, b = (third, first) if first > third else (first, third)
            coefficient = -coefficient
            repeated = second
        x_mask = (1 << a) | (1 << b)
        parity = _range_mask(a + 1, b)
        for endpoint_z_mask in (0, x_mask):
            z_mask = parity | endpoint_z_mask
            hopping = 0.25 * coefficient
            _add(coefficients, x_mask, z_mask, hopping)
            _add(coefficients, x_mask, z_mask ^ (1 << repeated), -hopping)
        return
    if unique_count == 2:
        diagonal = -0.25 * coefficient if first == fourth else 0.25 * coefficient
        first_z = 1 << first
        second_z = 1 << second
        _add(coefficients, 0, 0, -diagonal)
        _add(coefficients, 0, first_z, diagonal)
        _add(coefficients, 0, second_z, diagonal)
        _add(coefficients, 0, first_z | second_z, -diagonal)


def _spin_orbital_two_body_coefficient(
    packed: np.ndarray,
    first: int,
    second: int,
    third: int,
    fourth: int,
) -> float:
    first_orbital, first_spin = divmod(first, 2)
    second_orbital, second_spin = divmod(second, 2)
    third_orbital, third_spin = divmod(third, 2)
    fourth_orbital, fourth_spin = divmod(fourth, 2)
    coefficient = 0.0
    if first_spin == fourth_spin and second_spin == third_spin:
        coefficient += _packed_eri(
            packed,
            first_orbital,
            fourth_orbital,
            second_orbital,
            third_orbital,
        )
    if first_spin == third_spin and second_spin == fourth_spin:
        coefficient -= _packed_eri(
            packed,
            first_orbital,
            third_orbital,
            second_orbital,
            fourth_orbital,
        )
    return coefficient


def _pauli_string(x_mask: int, z_mask: int, num_qubits: int) -> str:
    symbols = "IXZY"
    return "".join(
        symbols[((x_mask >> index) & 1) + 2 * ((z_mask >> index) & 1)]
        for index in range(num_qubits - 1, -1, -1)
    )


def _packed_integrals_to_pauli_terms(
    one_body: np.ndarray,
    packed_two_body: np.ndarray,
    core_energy: float,
    coefficient_tolerance: float,
) -> tuple[tuple[float, str], ...]:
    num_orbitals = one_body.shape[0]
    num_qubits = 2 * num_orbitals
    coefficients: dict[tuple[int, int], float] = {(0, 0): core_energy}
    for orbital in range(num_orbitals):
        for spin in range(2):
            mode = 2 * orbital + spin
            _add_one_body_hermitian(
                coefficients, mode, mode, float(one_body[orbital, orbital])
            )
    for first_orbital, second_orbital in combinations(range(num_orbitals), 2):
        coefficient = 0.5 * float(
            one_body[first_orbital, second_orbital]
            + one_body[second_orbital, first_orbital]
        )
        for spin in range(2):
            _add_one_body_hermitian(
                coefficients,
                2 * first_orbital + spin,
                2 * second_orbital + spin,
                coefficient,
            )

    spin_pair_classes: tuple[list[tuple[int, int]], ...] = ([], [], [])
    for first, second in combinations(range(num_qubits), 2):
        spin_pair_classes[(first & 1) + (second & 1)].append((first, second))
        coefficient = _spin_orbital_two_body_coefficient(
            packed_two_body, first, second, first, second
        )
        _add_two_body_hermitian(
            coefficients, first, second, first, second, coefficient
        )
    for pair_class in spin_pair_classes:
        for pair_index in range(len(pair_class) - 1):
            first, second = pair_class[pair_index]
            for other_pair_index in range(pair_index + 1, len(pair_class)):
                third, fourth = pair_class[other_pair_index]
                coefficient = _spin_orbital_two_body_coefficient(
                    packed_two_body, first, second, third, fourth
                )
                if coefficient != 0.0:
                    _add_two_body_hermitian(
                        coefficients,
                        first,
                        second,
                        third,
                        fourth,
                        coefficient,
                    )
    return tuple(
        (float(coefficient), _pauli_string(x_mask, z_mask, num_qubits))
        for (x_mask, z_mask), coefficient in coefficients.items()
        if abs(coefficient) > coefficient_tolerance
    )


@dataclass(frozen=True)
class GeneratedFcidumpHamiltonian:
    """Immutable Jordan--Wigner Hamiltonian generated from FCIDUMP."""

    num_qubits: int
    terms: tuple[tuple[float, str], ...]
    hartree_fock_energy: float
    num_spatial_orbitals: int
    num_electrons: int
    spin_projection_twice: int
    fcidump_path: str

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

    def to_lcp(self, *, include_identity: bool = False) -> LCP:
        return LCP(
            {
                pauli: coefficient
                for coefficient, pauli in self.terms
                if include_identity or pauli != self.identity_pauli
            },
            num_qubits=self.num_qubits,
        )


@dataclass(frozen=True)
class ReiherFeMocoHamiltonianPreset:
    """Reiher CAS(54e,54o) FeMoco Hamiltonian stored as FCIDUMP."""

    fcidump_path: str | Path
    name: str = FEMOCO_NAME
    description: str = (
        "FeMoco Reiher active-space Hamiltonian, CAS(54e,54o), FCIDUMP, "
        "Jordan-Wigner, 108 qubits"
    )

    def __post_init__(self) -> None:
        path = Path(self.fcidump_path).expanduser().resolve()
        object.__setattr__(self, "fcidump_path", str(path))

    def generate(
        self,
        *,
        coefficient_tolerance: float = 1e-12,
    ) -> GeneratedFcidumpHamiltonian:
        tolerance = _validated_nonnegative_real(
            coefficient_tolerance,
            name="coefficient_tolerance",
        )
        path = Path(self.fcidump_path)
        try:
            file_stat = path.stat()
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"FeMoco FCIDUMP was not found: {path}"
            ) from error
        return _generate_reiher_femoco(
            self.fcidump_path,
            file_stat.st_size,
            file_stat.st_mtime_ns,
            tolerance,
        )


@lru_cache(maxsize=2)
def _generate_reiher_femoco(
    fcidump_path: str,
    file_size: int,
    modification_time_ns: int,
    coefficient_tolerance: float,
) -> GeneratedFcidumpHamiltonian:
    del file_size, modification_time_ns
    path = Path(fcidump_path)
    from pyscf.tools import fcidump

    data = fcidump.read(str(path), verbose=False)
    num_orbitals = int(data["NORB"])
    num_electrons = int(data["NELEC"])
    if num_orbitals != FEMOCO_SPATIAL_ORBITALS:
        raise ValueError(
            "Reiher FeMoco FCIDUMP must have NORB=54, got "
            f"{num_orbitals}"
        )
    if num_electrons != FEMOCO_ACTIVE_ELECTRONS:
        raise ValueError(
            "Reiher FeMoco FCIDUMP must have NELEC=54, got "
            f"{num_electrons}"
        )

    one_body = np.asarray(data["H1"], dtype=np.float64)
    packed_two_body = np.asarray(data["H2"], dtype=np.float64)
    expected_packed_size = math.comb(
        math.comb(num_orbitals + 1, 2) + 1,
        2,
    )
    if one_body.shape != (num_orbitals, num_orbitals):
        raise ValueError("FCIDUMP one-electron integral shape is inconsistent")
    if packed_two_body.shape != (expected_packed_size,):
        raise ValueError("FCIDUMP two-electron integral shape is inconsistent")
    if not np.allclose(one_body, one_body.T, atol=1e-10, rtol=1e-10):
        raise ValueError("FCIDUMP one-electron integrals must be real symmetric")

    core_energy = float(data.get("ECORE", 0.0))
    num_qubits = 2 * num_orbitals
    terms = _packed_integrals_to_pauli_terms(
        one_body,
        packed_two_body,
        core_energy,
        coefficient_tolerance,
    )

    hartree_fock_energy = _closed_shell_hartree_fock_energy(
        one_body,
        packed_two_body,
        num_electrons=num_electrons,
        core_energy=core_energy,
    )
    return GeneratedFcidumpHamiltonian(
        num_qubits=num_qubits,
        terms=terms,
        hartree_fock_energy=hartree_fock_energy,
        num_spatial_orbitals=num_orbitals,
        num_electrons=num_electrons,
        spin_projection_twice=int(data.get("MS2", data.get("MS", 0))),
        fcidump_path=str(path),
    )


def _closed_shell_hartree_fock_energy(
    one_body: np.ndarray,
    packed_two_body: np.ndarray,
    *,
    num_electrons: int,
    core_energy: float,
) -> float:
    """Energy of the determinant occupying the first ``N/2`` orbitals."""
    if num_electrons % 2:
        return math.nan
    occupied = range(num_electrons // 2)
    energy = float(core_energy)
    energy += 2.0 * math.fsum(float(one_body[i, i]) for i in occupied)
    energy += math.fsum(
        2.0 * _packed_eri(packed_two_body, i, i, j, j)
        - _packed_eri(packed_two_body, i, j, j, i)
        for i in occupied
        for j in occupied
    )
    return energy


__all__ = [
    "FEMOCO_ACTIVE_ELECTRONS",
    "FEMOCO_NAME",
    "FEMOCO_SPATIAL_ORBITALS",
    "GeneratedFcidumpHamiltonian",
    "ReiherFeMocoHamiltonianPreset",
]
