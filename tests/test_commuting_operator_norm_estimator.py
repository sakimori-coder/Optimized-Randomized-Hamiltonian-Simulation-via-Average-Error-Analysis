import math
from fractions import Fraction

import pytest

from grouping import (
    estimate_commuting_operator_norm_upper_bound,
)
from operators import LCP


def test_returns_coefficient_one_norm_bound() -> None:
    hamiltonian = LCP({"ZII": 1, "IZI": 1, "IIZ": 1, "ZZZ": -1})

    assert estimate_commuting_operator_norm_upper_bound(hamiltonian) == pytest.approx(
        4.0
    )


def test_exact_method_matches_the_known_operator_norm() -> None:
    hamiltonian = LCP({"ZII": 1, "IZI": 1, "IIZ": 1, "ZZZ": -1})

    sparse_norm = estimate_commuting_operator_norm_upper_bound(
        hamiltonian,
        method="exact",
    )

    assert sparse_norm == pytest.approx(2.0)


def test_frobenius_method_uses_pauli_orthogonality() -> None:
    hamiltonian = LCP({"ZI": 3.0, "IZ": 4.0})

    frobenius_norm = estimate_commuting_operator_norm_upper_bound(
        hamiltonian,
        method="frobenius",
    )

    assert frobenius_norm == pytest.approx(10.0)


def test_frobenius_method_does_not_materialize_a_matrix(monkeypatch) -> None:
    hamiltonian = LCP({"ZI": 3.0, "IZ": 4.0})

    def fail_if_materialized(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Frobenius norm must not materialize a matrix")

    monkeypatch.setattr(LCP, "to_csr", fail_if_materialized)
    monkeypatch.setattr(LCP, "to_matrix", fail_if_materialized)

    assert estimate_commuting_operator_norm_upper_bound(
        hamiltonian,
        method="frobenius",
    ) == pytest.approx(10.0)


def test_lp_method_uses_even_pauli_dependency() -> None:
    hamiltonian = LCP(
        {"ZII": 1, "IZI": 1, "IIZ": 1, "ZZZ": -0.25}
    )

    lp_bound = estimate_commuting_operator_norm_upper_bound(
        hamiltonian,
        method="lp",
    )

    assert lp_bound == pytest.approx(2.75)
    assert lp_bound < hamiltonian.coefficient_one_norm()


def test_lp_method_tracks_minus_identity_phase() -> None:
    # XX * YY * ZZ = -II, so the corresponding q parity must be odd.
    hamiltonian = LCP({"II": 2, "XX": 1, "YY": 1, "ZZ": 1})

    lp_bound = estimate_commuting_operator_norm_upper_bound(
        hamiltonian,
        method="lp",
    )

    assert lp_bound == pytest.approx(3.0)


def test_lp_method_maximizes_both_energy_signs() -> None:
    # The eigenvalues are 1 and -3, so maximizing E alone would return 1.
    hamiltonian = LCP({"XX": 1, "YY": 1, "ZZ": 1})

    assert estimate_commuting_operator_norm_upper_bound(
        hamiltonian,
        method="lp",
    ) == pytest.approx(3.0)


def test_lp_method_equals_l1_for_independent_paulis() -> None:
    hamiltonian = LCP({"II": 0.5, "ZI": 2, "IZ": -1})

    assert estimate_commuting_operator_norm_upper_bound(
        hamiltonian,
        method="lp",
    ) == pytest.approx(3.5)


def test_lp_method_is_a_relaxation_not_an_exact_solver() -> None:
    hamiltonian = LCP(
        {
            "ZIII": 0.25,
            "IZII": 0.25,
            "IIZI": 1,
            "ZZZI": 1,
            "IIIZ": 1,
            "ZZIZ": -1,
        }
    )

    lp_bound = estimate_commuting_operator_norm_upper_bound(
        hamiltonian,
        method="lp",
    )

    assert hamiltonian.operator_norm() == pytest.approx(2.5)
    assert lp_bound == pytest.approx(4.0)
    assert lp_bound < hamiltonian.coefficient_one_norm()


def test_handles_zero_hamiltonian() -> None:
    zero = LCP({}, num_qubits=3)

    assert estimate_commuting_operator_norm_upper_bound(zero) == 0.0
    assert estimate_commuting_operator_norm_upper_bound(
        zero,
        method="lp",
    ) == 0.0
    assert estimate_commuting_operator_norm_upper_bound(
        zero,
        method="exact",
    ) == 0.0


def test_coefficient_bound_does_not_materialize_a_matrix(monkeypatch) -> None:
    num_qubits = 1000
    hamiltonian = LCP(
        {
            "Z" + "I" * (num_qubits - 1): 1.0,
            "I" + "Z" + "I" * (num_qubits - 2): -0.5,
        }
    )

    def fail_if_materialized(*_args: object, **_kwargs: object) -> None:
        pytest.fail("coefficient l1 bound must not materialize a matrix")

    monkeypatch.setattr(LCP, "to_csr", fail_if_materialized)
    monkeypatch.setattr(LCP, "to_matrix", fail_if_materialized)

    assert estimate_commuting_operator_norm_upper_bound(hamiltonian) == pytest.approx(
        1.5
    )


def test_lp_bound_does_not_materialize_a_matrix(monkeypatch) -> None:
    num_qubits = 1000
    z1 = "Z" + "I" * (num_qubits - 1)
    z2 = "I" + "Z" + "I" * (num_qubits - 2)
    z3 = "II" + "Z" + "I" * (num_qubits - 3)
    z1z2z3 = "ZZZ" + "I" * (num_qubits - 3)
    hamiltonian = LCP({z1: 1, z2: 1, z3: 1, z1z2z3: -1})

    def fail_if_materialized(*_args: object, **_kwargs: object) -> None:
        pytest.fail("LP bound must not materialize a matrix")

    monkeypatch.setattr(LCP, "to_csr", fail_if_materialized)
    monkeypatch.setattr(LCP, "to_matrix", fail_if_materialized)

    assert estimate_commuting_operator_norm_upper_bound(
        hamiltonian,
        method="lp",
    ) == pytest.approx(2.0)


def test_lp_method_scales_large_coefficients_before_solving() -> None:
    scale = 8e307
    hamiltonian = LCP(
        {"ZII": scale, "IZI": scale, "IIZ": scale, "ZZZ": -scale}
    )

    lp_bound = estimate_commuting_operator_norm_upper_bound(
        hamiltonian,
        method="lp",
    )

    assert math.isfinite(lp_bound)
    assert lp_bound == pytest.approx(1.6e308)


def test_coefficient_l1_sum_is_rounded_up() -> None:
    coefficients = [9560342718.892494, 9478274870.593494, 565513677.2680869]
    hamiltonian = LCP(
        dict(zip(("ZII", "IZI", "IIZ"), coefficients))
    )

    bound = estimate_commuting_operator_norm_upper_bound(hamiltonian)
    exact_float_sum = sum(
        (Fraction.from_float(value) for value in coefficients),
        start=Fraction(0),
    )

    assert Fraction.from_float(bound) >= exact_float_sum


def test_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        estimate_commuting_operator_norm_upper_bound(
            LCP({"Z": 1.0}),
            method="not_a_method",  # type: ignore[arg-type]
        )
