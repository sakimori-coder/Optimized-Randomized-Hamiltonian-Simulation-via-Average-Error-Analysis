import pytest

from operators import LCP

from hamiltonians.pauli import build_lch_from_lcp_unit_cost
from hamiltonians.physics import (
    heisenberg_chain,
    schwinger_model,
    transverse_field_ising_chain,
)
from hamiltonians.random import build_random_lcp


def test_random_lcp_is_reproducible_and_has_requested_number_of_terms() -> None:
    first = build_random_lcp(num_terms=7, num_qubits=3, seed=1234)
    second = build_random_lcp(num_terms=7, num_qubits=3, seed=1234)

    assert first.terms == second.terms
    assert len(first.terms) == 7
    assert first.num_qubits == 3


def test_lcp_to_single_pauli_lch_preserves_coefficients_and_identity_cost() -> None:
    source = LCP({"II": 2.5, "XI": -1.25, "YZ": 0.75})

    converted = build_lch_from_lcp_unit_cost(source)

    assert [
        (coefficient, operator.terms, cost)
        for coefficient, operator, cost in converted.lcp_terms
    ] == [
        (2.5, {"II": 1.0}, 0.0),
        (-1.25, {"XI": 1.0}, 1.0),
        (0.75, {"YZ": 1.0}, 1.0),
    ]
    assert converted.lcp.terms == source.terms


def test_transverse_field_ising_chain_open_and_periodic_terms() -> None:
    open_chain = transverse_field_ising_chain(
        3,
        coupling=2.0,
        transverse_field=0.5,
    )
    periodic_chain = transverse_field_ising_chain(
        3,
        coupling=2.0,
        transverse_field=0.5,
        periodic=True,
    )

    open_terms = {
        "ZZI": -2.0,
        "IZZ": -2.0,
        "XII": -0.5,
        "IXI": -0.5,
        "IIX": -0.5,
    }
    assert open_chain.terms == open_terms
    assert periodic_chain.terms == {**open_terms, "ZIZ": -2.0}


def test_heisenberg_chain_has_exchange_and_longitudinal_field_terms() -> None:
    hamiltonian = heisenberg_chain(
        2,
        coupling=1.5,
        longitudinal_field=-0.25,
    )

    assert hamiltonian.terms == {
        "XX": 1.5,
        "YY": 1.5,
        "ZZ": 1.5,
        "ZI": -0.25,
        "IZ": -0.25,
    }


def test_schwinger_model_expands_hopping_mass_and_electric_terms() -> None:
    hamiltonian = schwinger_model(
        3,
        x=2.0,
        mu=4.0,
    )

    assert hamiltonian.terms == {
        "XXI": 1.0,
        "YYI": 1.0,
        "IXX": 1.0,
        "IYY": 1.0,
        "ZII": 2.5,
        "IZI": -2.0,
        "IIZ": 2.0,
        "ZZI": 0.5,
    }


def test_schwinger_model_can_retain_identity_shift() -> None:
    without_identity = schwinger_model(
        2,
        x=0.0,
        mu=0.0,
        background_field=0.25,
    )
    with_identity = schwinger_model(
        2,
        x=0.0,
        mu=0.0,
        background_field=0.25,
        include_identity=True,
    )

    assert without_identity.terms == {"ZI": 0.75}
    assert with_identity.terms == {
        "ZI": 0.75,
        "II": 0.8125,
    }

    mass_only = schwinger_model(
        1,
        x=0.0,
        mu=2.0,
        include_identity=True,
    )
    assert mass_only.terms == {"Z": 1.0, "I": 1.0}


def test_schwinger_model_requires_at_least_one_qubit() -> None:
    with pytest.raises(ValueError, match="positive"):
        schwinger_model(0)
