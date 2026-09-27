import math
from itertools import combinations

import numpy as np
import pytest
from pyscf import gto

from hamiltonians import chemistry, syk
from operators import LCP


# Geometry and qubit counts from the molecular table in the paper.
TETRAHEDRAL_ANGLE = math.degrees(math.acos(-1 / 3))
MOLECULAR_CASES = [
    ("h2_sto3g_jw", 4, "sto-3g", ("H", "H"), 0.735, None),
    ("lih_sto3g_full_jw", 12, "sto-3g", ("Li", "H"), 1.45, None),
    ("beh2_sto3g_full_jw", 14, "sto-3g", ("Be", "H", "H"), 1.3264, 180.0),
    ("h2o_sto3g_full_jw", 14, "sto-3g", ("O", "H", "H"), 0.9576, 104.5),
    ("nh3_sto3g_full_jw", 16, "sto-3g", ("N", "H", "H", "H"), 1.012, 106.7),
    ("ch4_sto3g_full_jw", 18, "sto-3g", ("C", "H", "H", "H", "H"), 1.087, TETRAHEDRAL_ANGLE),
    ("n2_sto3g_full_jw", 20, "sto-3g", ("N", "N"), 1.0977, None),
    ("h2o_ccpvdz_full_jw", 48, "cc-pvdz", ("O", "H", "H"), 0.9576, 104.5),
    ("ch4_ccpvdz_full_jw", 68, "cc-pvdz", ("C", "H", "H", "H", "H"), 1.087, TETRAHEDRAL_ANGLE),
]


def test_molecular_presets_match_paper_table():
    assert set(chemistry.MOLECULAR_HAMILTONIANS) == {case[0] for case in MOLECULAR_CASES}


@pytest.mark.parametrize("name,num_qubits,basis,atoms,bond_length,angle", MOLECULAR_CASES)
def test_molecular_geometry_and_basis(name, num_qubits, basis, atoms, bond_length, angle):
    geometry, actual_basis = chemistry.MOLECULAR_HAMILTONIANS[name]
    assert actual_basis == basis
    assert sorted(atom for atom, _ in geometry) == sorted(atoms)

    # Count spin orbitals directly from PySCF's basis, without running SCF.
    molecule = gto.M(atom=geometry, basis=basis, unit="Angstrom", verbose=0)
    assert 2 * molecule.nao_nr() == num_qubits

    if len(atoms) == 2:
        distance = math.dist(geometry[0][1], geometry[1][1])
        assert distance == pytest.approx(bond_length, rel=0, abs=1e-12)
        return

    center = next(np.array(position) for atom, position in geometry if atom != "H")
    bonds = [np.array(position) - center for atom, position in geometry if atom == "H"]
    lengths = [np.linalg.norm(bond) for bond in bonds]
    assert lengths == pytest.approx([bond_length] * len(bonds), rel=0, abs=1e-12)
    for first, second in combinations(bonds, 2):
        cosine = np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
        assert cosine == pytest.approx(math.cos(math.radians(angle)), rel=0, abs=1e-12)


@pytest.mark.parametrize("molecule", ["h2o", "ch4"])
def test_basis_change_preserves_geometry(molecule):
    sto3g = chemistry.MOLECULAR_HAMILTONIANS[f"{molecule}_sto3g_full_jw"]
    ccpvdz = chemistry.MOLECULAR_HAMILTONIANS[f"{molecule}_ccpvdz_full_jw"]
    assert sto3g[0] == ccpvdz[0]


def test_h2_identity_free_lcp_reproduces_hartree_fock_energy(tmp_path):
    from openfermion import MolecularData, get_sparse_operator
    from openfermionpyscf import run_pyscf

    target = chemistry.generate("h2_sto3g_jw")
    assert isinstance(target, LCP)
    assert target.num_qubits == 4
    assert "IIII" not in target.terms

    # Independent reference including nuclear repulsion and the identity term.
    geometry, basis = chemistry.MOLECULAR_HAMILTONIANS["h2_sto3g_jw"]
    molecule = run_pyscf(MolecularData(
        geometry, basis, multiplicity=1, filename=str(tmp_path / "h2_reference"),
    ))
    full_matrix = get_sparse_operator(molecule.get_molecular_hamiltonian()).toarray()
    identity_coefficient = np.trace(full_matrix).real / 16

    # The Hartree-Fock determinant is |0011>: spin orbitals q0 and q1 occupied.
    hf_expectation = sum(
        coefficient * (-1) ** pauli[-2:].count("Z")
        for pauli, coefficient in target.terms.items()
        if all(symbol in "IZ" for symbol in pauli)
    )
    assert hf_expectation == pytest.approx(
        molecule.hf_energy - identity_coefficient, rel=0, abs=1e-10,
    )


def test_syk_is_reproducible_and_has_one_term_per_quartet():
    first = syk.generate(4, seed=13)
    second = syk.generate(4, seed=13)

    assert isinstance(first, LCP)
    assert first.terms == second.terms
    assert first.terms != syk.generate(4, seed=14).terms
    assert len(first.terms) == math.comb(8, 4)
    assert first.num_qubits == 4
    assert "IIII" not in first.terms
    assert syk.generate(4).terms == syk.generate(4, seed=42).terms


@pytest.mark.parametrize("num_qubits,coupling_scale,seed", [
    (2, 1.0, 42),
    (3, 2.5, 13),
    (4, 1e-10, 7),  # No intermediate library cutoff may erase weak couplings.
    (3, 0.0, 42),
])
def test_syk_matches_dense_majorana_hamiltonian(num_qubits, coupling_scale, seed):
    dimension = 2 ** num_qubits
    majoranas = []
    for site in range(num_qubits):
        # Construct annihilation operators directly in the occupation basis.
        # Site 0 is the leftmost tensor factor; earlier occupied sites set parity.
        annihilation = np.zeros((dimension, dimension), dtype=complex)
        bit = 1 << (num_qubits - 1 - site)
        for state in range(dimension):
            if state & bit:
                parity = (state >> (num_qubits - site)).bit_count()
                annihilation[state ^ bit, state] = (-1) ** parity
        creation = annihilation.conj().T
        majoranas.extend([annihilation + creation, -1j * (annihilation - creation)])

    # The paper specifies Var(J_abcd) = 3! J**2 / (2n)**3 and no extra
    # prefactor in H = sum_{a<b<c<d} J_abcd gamma_a gamma_b gamma_c gamma_d.
    variance = math.factorial(3) * coupling_scale ** 2 / (2 * num_qubits) ** 3
    rng = np.random.default_rng(seed)
    couplings = rng.normal(0, math.sqrt(variance), math.comb(2 * num_qubits, 4))
    expected = np.zeros((dimension, dimension), dtype=complex)
    for (a, b, c, d), coefficient in zip(combinations(range(2 * num_qubits), 4), couplings):
        expected += coefficient * (majoranas[a] @ majoranas[b] @ majoranas[c] @ majoranas[d])

    pauli_matrices = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1]),
    }
    target = syk.generate(num_qubits, coupling_scale=coupling_scale, seed=seed)
    actual = np.zeros_like(expected)
    for pauli, coefficient in target.terms.items():
        matrix = np.array([[1]])
        for symbol in pauli:
            matrix = np.kron(matrix, pauli_matrices[symbol])
        actual += coefficient * matrix

    assert target.num_qubits == num_qubits
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=coupling_scale * 1e-14)
    assert target.hs_norm() == pytest.approx(np.linalg.norm(couplings), rel=1e-13, abs=0)



@pytest.mark.parametrize("name", ["h2_sto3g_jw", "lih_sto3g_full_jw"])
def test_molecular_pauli_coefficients_match_openfermion(name, tmp_path):
    from openfermion import MolecularData, get_fermion_operator, jordan_wigner
    from openfermionpyscf import run_pyscf

    # Independent library transformation checks the molecular integral convention
    # and q0-rightmost conversion, including all off-diagonal Pauli terms.
    geometry, basis = chemistry.MOLECULAR_HAMILTONIANS[name]
    molecule = run_pyscf(MolecularData(
        geometry, basis, multiplicity=1, filename=str(tmp_path / name),
    ))
    reference = jordan_wigner(get_fermion_operator(molecule.get_molecular_hamiltonian()))
    num_qubits = molecule.n_qubits
    expected = {}
    for paulis, coefficient in reference.terms.items():
        if not paulis or abs(coefficient) <= 1e-12:
            continue
        symbols = ["I"] * num_qubits
        for site, symbol in paulis:
            symbols[num_qubits - 1 - site] = symbol
        expected["".join(symbols)] = float(coefficient.real)

    target = chemistry.generate(name)
    assert target.num_qubits == num_qubits
    assert target.terms == pytest.approx(expected, rel=0, abs=1e-11)
