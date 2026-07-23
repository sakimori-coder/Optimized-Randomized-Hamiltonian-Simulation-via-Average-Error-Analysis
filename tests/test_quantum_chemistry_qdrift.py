import pytest
from openfermion import QubitOperator

from quantum_chemistry_qdrift import (
    GeneratedMolecularHamiltonian,
    H2_STO3G_JW,
    H2O_STO3G_CAS_4E_4O_JW,
    LIH_STO3G_ACTIVE_JW,
    MOLECULAR_HAMILTONIANS,
    _pauli_terms_from_qubit_operator,
    run_molecular_qdrift_experiment,
)


def test_openfermion_terms_are_converted_to_project_qubit_order() -> None:
    operator = QubitOperator("", -3.0) + QubitOperator("X0 Y2", 0.25)
    terms = _pauli_terms_from_qubit_operator(
        operator,
        num_qubits=3,
        coefficient_tolerance=1e-12,
    )
    generated = GeneratedMolecularHamiltonian(
        num_qubits=3,
        terms=terms,
        hartree_fock_energy=-1.0,
    )

    assert terms == ((-3.0, "III"), (0.25, "YIX"))
    assert generated.to_lch().lcp_terms == [(0.25 + 0.0j, "YIX", 1.0)]
    assert generated.to_lch(
        include_identity=True,
        pauli_rotation_depth=2.5,
    ).lcp_terms == [
        (-3.0 + 0.0j, "III", 0.0),
        (0.25 + 0.0j, "YIX", 2.5),
    ]


def test_h2_preset_excludes_the_global_phase_by_default() -> None:
    traceless = H2_STO3G_JW.to_lch()
    full = H2_STO3G_JW.to_lch(include_identity=True)

    assert len(traceless.lcp_terms) == 14
    assert all(pauli != "IIII" for _, pauli, _ in traceless.lcp_terms)
    assert all(cost == 1.0 for _, _, cost in traceless.lcp_terms)

    assert len(full.lcp_terms) == 15
    identity_terms = [term for term in full.lcp_terms if term[1] == "IIII"]
    assert identity_terms == [
        (pytest.approx(-0.0905789861, abs=1e-5), "IIII", 0.0)
    ]


def test_h2_sdp_experiment_reproduces_the_grouped_depth_improvement() -> None:
    comparison = run_molecular_qdrift_experiment(
        H2_STO3G_JW,
        time=1.0,
        epsilon=0.01,
        variance_methods=("sdp_bound",),
        group_norm_method="lp",
    )["sdp_bound"]

    assert comparison.pauli.lambda_sum == pytest.approx(1.89449315, abs=1e-5)
    assert comparison.pauli.normalized_variance_bound == pytest.approx(
        0.998548632055,
        abs=1e-5,
    )
    assert comparison.pauli.steps == 717
    assert comparison.pauli.expected_total_depth == pytest.approx(717.0)

    assert len(comparison.groups) == 2
    assert comparison.grouped.lambda_sum == pytest.approx(1.20735121, abs=1e-5)
    assert comparison.grouped.normalized_variance_bound == pytest.approx(
        0.282844789893,
        abs=1e-5,
    )
    assert comparison.grouped.steps == 83
    assert comparison.grouped.expected_total_depth == pytest.approx(
        236.561789,
        abs=1e-4,
    )
    assert comparison.grouped_to_pauli_depth_ratio == pytest.approx(
        0.32993276,
        abs=1e-6,
    )


def test_h2_lp_group_norm_matches_the_known_group_structure() -> None:
    comparison = run_molecular_qdrift_experiment(
        H2_STO3G_JW,
        variance_methods=("contraction_bound",),
        group_norm_method="lp",
    )["contraction_bound"]

    assert comparison.groups[0].pauli_strings == (
        "IIIZ",
        "IIZI",
        "IIZZ",
        "IZII",
        "IZIZ",
        "IZZI",
        "ZIII",
        "ZIIZ",
        "ZIZI",
        "ZZII",
    )
    assert comparison.groups[0].normalization_weight == pytest.approx(1.02641999)
    assert comparison.groups[0].rotation_depth == 3

    assert comparison.groups[1].pauli_strings == (
        "XXYY",
        "XYYX",
        "YXXY",
        "YYXX",
    )
    assert comparison.groups[1].normalization_weight == pytest.approx(
        0.1809312,
        abs=1e-5,
    )
    assert comparison.groups[1].rotation_depth == 2


def test_lih_active_space_preset_excludes_the_global_phase_by_default() -> None:
    traceless = LIH_STO3G_ACTIVE_JW.to_lch()
    full = LIH_STO3G_ACTIVE_JW.to_lch(include_identity=True)
    coefficients = {
        pauli: coefficient for coefficient, pauli, _ in full.lcp_terms
    }

    assert MOLECULAR_HAMILTONIANS["lih_sto3g_active_jw"] is LIH_STO3G_ACTIVE_JW
    assert len(traceless.lcp_terms) == 26
    assert all(pauli != "IIII" for _, pauli, _ in traceless.lcp_terms)
    assert all(cost == 1.0 for _, _, cost in traceless.lcp_terms)

    assert len(full.lcp_terms) == 27
    identity_terms = [term for term in full.lcp_terms if term[1] == "IIII"]
    assert identity_terms == [
        (pytest.approx(-7.4989469, abs=1e-5), "IIII", 0.0)
    ]
    # Catch accidental reversal of the OpenFermion qubit-index convention.
    assert coefficients["IIIZ"] == pytest.approx(0.16199475, abs=1e-5)
    assert coefficients["ZIII"] == pytest.approx(-0.01324370, abs=1e-5)


def test_lih_sdp_experiment_reproduces_the_grouped_depth_improvement() -> None:
    comparison = run_molecular_qdrift_experiment(
        LIH_STO3G_ACTIVE_JW,
        time=1.0,
        epsilon=0.01,
        variance_methods=("sdp_bound",),
        group_norm_method="lp",
    )["sdp_bound"]

    assert comparison.pauli.lambda_sum == pytest.approx(0.8971267, abs=1e-5)
    assert comparison.pauli.normalized_variance_bound == pytest.approx(
        0.999788310587,
        abs=1e-5,
    )
    assert comparison.pauli.steps == 161
    assert comparison.pauli.expected_total_depth == pytest.approx(161.0)

    assert len(comparison.groups) == 4
    assert comparison.grouped.lambda_sum == pytest.approx(0.8441519, abs=1e-5)
    assert comparison.grouped.normalized_variance_bound == pytest.approx(
        0.139959889358,
        abs=1e-5,
    )
    assert comparison.grouped.steps == 20
    assert comparison.grouped.expected_total_depth == pytest.approx(
        57.27496,
        abs=1e-4,
    )
    assert comparison.grouped_to_pauli_depth_ratio == pytest.approx(
        0.355745,
        abs=1e-6,
    )


def test_h2o_cas_4e_4o_preset_generates_eight_qubit_lch() -> None:
    generated = H2O_STO3G_CAS_4E_4O_JW.generate()
    traceless = generated.to_lch()

    assert (
        MOLECULAR_HAMILTONIANS["h2o_sto3g_cas_4e_4o_jw"]
        is H2O_STO3G_CAS_4E_4O_JW
    )
    assert generated.num_qubits == 8
    assert len(generated.terms) == 105
    assert len(traceless.lcp_terms) == 104
    assert generated.identity_coefficient == pytest.approx(
        -73.11302673,
        abs=1e-5,
    )
