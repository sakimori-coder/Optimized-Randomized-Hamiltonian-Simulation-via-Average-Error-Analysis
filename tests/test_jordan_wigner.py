import json
from functools import reduce
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from pyscf import ao2mo

from hamiltonians import jordan_wigner


# Saved before the earlier independently checked FeMoco implementation changed.
_REFERENCES = json.loads(
    (Path(__file__).parent / "fixtures" / "femoco_jw_coefficients.json").read_text()
)
_PAULIS = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]),
}


def _pauli_matrix(word):
    return reduce(np.kron, (_PAULIS[symbol] for symbol in word))


def _annihilation(site, num_qubits):
    # Occupation bit q0 is the rightmost tensor factor.
    operator = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)
    bit = 1 << site
    for state in range(2 ** num_qubits):
        if state & bit:
            sign = (-1) ** (state & (bit - 1)).bit_count()
            operator[state ^ bit, state] = sign
    return operator


def _spatial_integrals(reference_name):
    reference = _REFERENCES[reference_name]
    one_body = np.array(reference["one_body"])
    two_body = ao2mo.restore(1, np.array(reference["packed_two_body"]), len(one_body))
    return one_body, two_body


def _occupation_hamiltonian(one_body, two_body):
    """Apply creation/annihilation operators directly to occupation bitstrings."""
    num_orbitals = len(one_body)
    dimension = 2 ** (2 * num_orbitals)
    matrix = np.zeros((dimension, dimension))

    def add(coefficient, actions):
        # Actions are ordered from the rightmost operator to the leftmost.
        for column in range(dimension):
            state, amplitude = column, coefficient
            for site, create in actions:
                bit = 1 << site
                if bool(state & bit) == create:
                    break
                amplitude *= (-1) ** (state & (bit - 1)).bit_count()
                state ^= bit
            else:
                matrix[state, column] += amplitude

    for p, q in product(range(num_orbitals), repeat=2):
        for spin in range(2):
            add(one_body[p, q], [(2 * q + spin, False), (2 * p + spin, True)])
    for p, q, r, s in product(range(num_orbitals), repeat=4):
        for spin, other_spin in product(range(2), repeat=2):
            # H2 = 1/2 sum (pq|rs) a†_{pσ} a†_{rτ} a_{sτ} a_{qσ}.
            add(0.5 * two_body[p, q, r, s], [
                (2 * q + spin, False), (2 * s + other_spin, False),
                (2 * r + other_spin, True), (2 * p + spin, True),
            ])
    return matrix


@pytest.mark.parametrize("indices", [
    (), (0,), (1,), (4,), (5,), (0, 0), (0, 1), (1, 0),
    (1, 4, 1), (0, 2, 4), (0, 1, 2, 3), (5, 1, 4, 0), (5, 4, 5, 4),
])
def test_majorana_products_match_occupation_matrices(indices):
    num_qubits = 3
    annihilation = [_annihilation(site, num_qubits) for site in range(num_qubits)]
    majoranas = [
        operator
        for a in annihilation
        for operator in (a + a.conj().T, -1j * (a - a.conj().T))
    ]
    expected = np.eye(2 ** num_qubits, dtype=complex)
    for index in indices:
        expected = expected @ majoranas[index]

    phase, x_mask, z_mask = jordan_wigner.majorana_product(indices)
    word = jordan_wigner.pauli_string(x_mask, z_mask, num_qubits)
    np.testing.assert_array_equal(phase * _pauli_matrix(word), expected)


@pytest.mark.parametrize("reference_name", ["small_fcidump", "random_3", "random_4"])
def test_spatial_integrals_preserve_all_saved_pauli_coefficients(reference_name):
    one_body, two_body = _spatial_integrals(reference_name)
    target = jordan_wigner.from_spatial_integrals(one_body, two_body)
    assert target.num_qubits == 2 * len(one_body)
    assert target.terms == pytest.approx(_REFERENCES[reference_name]["terms"], rel=0, abs=1e-12)


@pytest.mark.parametrize("reference_name", ["small_fcidump", "random_3"])
def test_spatial_integrals_match_direct_occupation_hamiltonian(reference_name):
    one_body, two_body = _spatial_integrals(reference_name)
    target = jordan_wigner.from_spatial_integrals(one_body, two_body)
    expected = _occupation_hamiltonian(one_body, two_body)
    expected -= np.trace(expected) / len(expected) * np.eye(len(expected))
    actual = sum(coefficient * _pauli_matrix(word) for word, coefficient in target.terms.items())
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_small_multiorbital_integrals_preserve_final_pauli_coefficients():
    one_body, two_body = _spatial_integrals("random_4")
    scale = 1e-10
    target = jordan_wigner.from_spatial_integrals(scale * one_body, scale * two_body)
    expected = {
        word: scale * coefficient for word, coefficient in _REFERENCES["random_4"]["terms"].items()
        if abs(scale * coefficient) > 1e-12
    }
    assert target.terms == pytest.approx(expected, rel=1e-12, abs=1e-24)


def test_one_orbital_interaction_matches_occupation_basis_matrix():
    # H = h (n_up + n_down) + U n_up n_down, in |00>, |01>, |10>, |11>.
    h, interaction = 0.7, -0.2
    target = jordan_wigner.from_spatial_integrals(
        np.array([[h]]), np.array([[[[interaction]]]]),
    )
    actual = sum(coefficient * _pauli_matrix(word) for word, coefficient in target.terms.items())
    expected = np.diag([0.0, h, h, 2 * h + interaction])
    expected -= np.trace(expected) / 4 * np.eye(4)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("h,interaction,expected", [
    (4e-10, 0.0, {"IZ": -2e-10, "ZI": -2e-10}),
    (0.0, 4e-10, {"IZ": -1e-10, "ZI": -1e-10, "ZZ": 1e-10}),
    # Each Z contribution is below 1e-12, but their sum must survive.
    (1.5e-12, 3e-12, {"IZ": -1.5e-12, "ZI": -1.5e-12}),
    (2e-12, 0.0, {}),  # A Pauli coefficient exactly at the cutoff is omitted.
    (1.0, -2.0, {"ZZ": -0.5}),  # The one- and two-body Z terms cancel.
    (1.0, -2.0 + 2e-12, {"ZZ": (-2.0 + 2e-12) / 4}),
])
def test_cutoff_is_applied_after_adding_one_and_two_body_terms(h, interaction, expected):
    target = jordan_wigner.from_spatial_integrals(
        np.array([[h]]), np.array([[[[interaction]]]]),
    )
    assert target.terms == pytest.approx(expected, rel=1e-12, abs=1e-24)


def test_explicit_cutoff_and_zero_hamiltonian():
    one_body, two_body = np.array([[0.5]]), np.zeros((1, 1, 1, 1))
    target = jordan_wigner.from_spatial_integrals(one_body, two_body, coefficient_tolerance=0.25)
    assert target.terms == {}
    assert target.num_qubits == 2

    zero = jordan_wigner.from_spatial_integrals(0 * one_body, two_body, coefficient_tolerance=0)
    assert zero.terms == {}
    assert zero.num_qubits == 2


def test_pauli_products_match_all_two_qubit_matrix_products():
    words = ["".join(symbols) for symbols in product("IXYZ", repeat=2)]
    for first, second in product(words, repeat=2):
        masks = []
        for word in (first, second):
            x = sum(1 << q for q, symbol in enumerate(reversed(word)) if symbol in "XY")
            z = sum(1 << q for q, symbol in enumerate(reversed(word)) if symbol in "YZ")
            masks.extend((x, z))
        phase, x, z = jordan_wigner.pauli_product(*masks)
        actual = phase * _pauli_matrix(jordan_wigner.pauli_string(x, z, 2))
        expected = _pauli_matrix(first) @ _pauli_matrix(second)
        np.testing.assert_array_equal(actual, expected, err_msg=f"{first} times {second}")


@pytest.mark.parametrize("num_qubits", [7, 8, 9, 63, 64, 65])
def test_pauli_strings_at_rendering_block_boundaries(num_qubits):
    expected = ("XYZI" * ((num_qubits + 3) // 4))[:num_qubits]
    x = sum(1 << q for q, symbol in enumerate(reversed(expected)) if symbol in "XY")
    z = sum(1 << q for q, symbol in enumerate(reversed(expected)) if symbol in "YZ")
    assert jordan_wigner.pauli_string(x, z, num_qubits) == expected


def test_pauli_string_keeps_high_bits_in_108_qubit_hamiltonian():
    x = (1 << 107) | (1 << 64) | (1 << 7) | 1
    z = (1 << 107) | (1 << 65) | (1 << 8) | 1
    expected = "Y" + "I" * 41 + "ZX" + "I" * 55 + "ZX" + "I" * 6 + "Y"
    assert jordan_wigner.pauli_string(x, z, 108) == expected
