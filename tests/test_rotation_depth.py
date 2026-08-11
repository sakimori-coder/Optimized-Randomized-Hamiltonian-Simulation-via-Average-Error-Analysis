import pytest

from operators import LCP
from grouping.rotation_depth import minimum_pauli_rotation_depth


def test_empty_zero_and_identity_terms_have_zero_depth():
    assert minimum_pauli_rotation_depth(LCP({}, num_qubits=2)) == 0
    assert minimum_pauli_rotation_depth(LCP({"XX": 0})) == 0
    assert minimum_pauli_rotation_depth(LCP({"II": 3})) == 0


def test_independent_commuting_paulis_fit_in_one_layer():
    hamiltonian = LCP({"ZI": 1, "IZ": 2})

    assert minimum_pauli_rotation_depth(hamiltonian) == 1


def test_dependent_commuting_paulis_require_two_layers():
    # ZZ = (ZI)(IZ), so the three binary vectors have rank two.
    hamiltonian = LCP({"ZI": 1, "IZ": 2, "ZZ": 3})

    assert minimum_pauli_rotation_depth(hamiltonian) == 2


def test_dense_low_rank_subset_controls_exact_minimum():
    # The first seven terms are all nonzero vectors of a rank-three subspace,
    # so they alone need ceil(7 / 3) = 3 layers.  The three extra independent
    # terms make the global bound only ceil(10 / 6) = 2, demonstrating why all
    # subsets (equivalently, all flats) must be considered.
    hamiltonian = LCP(
        {
            "ZIIIII": 1,
            "IZIIII": 1,
            "IIZIII": 1,
            "ZZIIII": 1,
            "ZIZIII": 1,
            "IZZIII": 1,
            "ZZZIII": 1,
            "IIIZII": 1,
            "IIIIZI": 1,
            "IIIIIZ": 1,
        }
    )

    assert minimum_pauli_rotation_depth(hamiltonian) == 3


def test_general_commuting_paulis_use_symplectic_binary_vectors():
    hamiltonian = LCP({"XX": 1, "YY": 2, "ZZ": 3})

    # XX * YY is proportional to ZZ, so their F_2 rank is two.
    assert minimum_pauli_rotation_depth(hamiltonian) == 2


def test_rejects_non_commuting_terms():
    with pytest.raises(ValueError, match="commute pairwise"):
        minimum_pauli_rotation_depth(LCP({"X": 1, "Z": 1}))


def test_rejects_non_hermitian_coefficients():
    with pytest.raises(ValueError, match="coefficients must be real"):
        minimum_pauli_rotation_depth(LCP({"X": 1j}))


def test_rejects_non_lcp_input():
    with pytest.raises(TypeError, match="must be an LCP"):
        minimum_pauli_rotation_depth({"X": 1})
