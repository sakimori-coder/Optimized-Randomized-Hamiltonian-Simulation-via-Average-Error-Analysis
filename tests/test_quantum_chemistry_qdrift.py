import pytest
from openfermion import QubitOperator

from hamiltonians.chemistry import (
    GeneratedMolecularHamiltonian,
    H2_STO3G_JW,
    H2O_STO3G_CAS_4E_4O_JW,
    LIH_STO3G_ACTIVE_JW,
    MOLECULAR_HAMILTONIANS,
    _pauli_terms_from_qubit_operator,
    hydrogen_chain_preset,
)
from operators import LCP
from quantum_chemistry_qdrift import (
    _print_summary,
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
    assert generated.to_lcp().terms == {"YIX": 0.25}
    assert generated.to_lcp(include_identity=True).terms == {
        "III": -3.0,
        "YIX": 0.25,
    }


def test_h2_preset_excludes_the_global_phase_by_default() -> None:
    traceless = H2_STO3G_JW.to_lcp()
    full = H2_STO3G_JW.to_lcp(include_identity=True)

    assert isinstance(traceless, LCP)
    assert len(traceless.terms) == 14
    assert "IIII" not in traceless.terms
    assert len(full.terms) == 15
    assert full.terms["IIII"] == pytest.approx(-0.0905789861, abs=1e-5)


def test_h2_sdp_experiment_reproduces_the_grouped_depth_improvement() -> None:
    comparison = run_molecular_qdrift_experiment(
        H2_STO3G_JW,
        time=1.0,
        epsilon=0.01,
        variance_methods=("sdp_bound",),
        group_norm_method="lp",
        grouping_method="greedy",
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
        grouping_method="greedy",
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
    traceless = LIH_STO3G_ACTIVE_JW.to_lcp()
    full = LIH_STO3G_ACTIVE_JW.to_lcp(include_identity=True)
    coefficients = full.terms

    assert MOLECULAR_HAMILTONIANS["lih_sto3g_active_jw"] is LIH_STO3G_ACTIVE_JW
    assert len(traceless.terms) == 26
    assert "IIII" not in traceless.terms
    assert len(full.terms) == 27
    assert coefficients["IIII"] == pytest.approx(-7.4989469, abs=1e-5)
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
        grouping_method="greedy",
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


def test_molecular_experiment_uses_chemistry_grouping_by_default() -> None:
    comparison = run_molecular_qdrift_experiment(
        H2_STO3G_JW,
        variance_methods=("contraction_bound",),
        group_norm_method="coefficient_l1",
    )["contraction_bound"]

    assert comparison.grouping_method == "chemistry"
    assert len(comparison.groups) == 2
    assert all(
        set(pauli) <= {"I", "Z"}
        for pauli in comparison.groups[0].pauli_strings
    )
    assert {
        tuple(
            index
            for index, symbol in enumerate(pauli)
            if symbol in "XY"
        )
        for pauli in comparison.groups[1].pauli_strings
    } == {(0, 1, 2, 3)}


def test_summary_prints_variance_values_for_every_method(capsys) -> None:
    comparisons = run_molecular_qdrift_experiment(
        H2_STO3G_JW,
        variance_methods=("contraction_bound", "pauli_l1_bound"),
        group_norm_method="coefficient_l1",
    )

    _print_summary(comparisons)
    output = capsys.readouterr().out

    assert "Decompositions" in output
    assert "normalized B" in output
    assert "lambda^2 B" in output
    assert "one-step time-t channel error" in output
    assert "normalized diamond distance" in output
    assert "second order" in output
    assert "Taylor remainder" in output
    assert "finite-time bound" in output
    assert "Variance method: contraction_bound" in output
    assert "Variance method: pauli_l1_bound" in output
    assert "grouped / pauli variance-constant ratio" in output
    assert "grouped / pauli total-depth ratio" in output
    for comparison in comparisons.values():
        assert f"{comparison.pauli.normalized_variance_bound:.8e}" in output
        assert f"{comparison.grouped.normalized_variance_bound:.8e}" in output
        assert f"{comparison.pauli.variance_constant:.8e}" in output
        assert f"{comparison.grouped.variance_constant:.8e}" in output
        assert f"{comparison.pauli.one_step_second_order_bound:.8e}" in output
        assert f"{comparison.grouped.one_step_second_order_bound:.8e}" in output
        assert f"{comparison.pauli.one_step_certified_upper_bound:.8e}" in output
        assert f"{comparison.grouped.one_step_certified_upper_bound:.8e}" in output


def test_h2o_cas_4e_4o_preset_generates_eight_qubit_lcp() -> None:
    generated = H2O_STO3G_CAS_4E_4O_JW.generate()
    traceless = generated.to_lcp()

    assert (
        MOLECULAR_HAMILTONIANS["h2o_sto3g_cas_4e_4o_jw"]
        is H2O_STO3G_CAS_4E_4O_JW
    )
    assert generated.num_qubits == 8
    assert len(generated.terms) == 105
    assert len(traceless.terms) == 104
    assert generated.identity_coefficient == pytest.approx(
        -73.11302673,
        abs=1e-5,
    )


def test_h20_chain_uses_the_full_sto3g_orbital_space() -> None:
    preset = hydrogen_chain_preset(
        20,
        spacing=1.0,
    )

    assert preset.name == "h20_chain_sto3g_full_jw"
    assert len(preset.geometry) == 20
    assert preset.geometry[0] == ("H", (0.0, 0.0, 0.0))
    assert preset.geometry[-1] == ("H", (0.0, 0.0, 19.0))
    assert preset.occupied_indices is None
    assert preset.active_indices is None
    assert preset.multiplicity == 1
