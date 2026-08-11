import pytest

from hamiltonians.chemistry import H2_STO3G_JW
from molecular_average_trace_distance import (
    _print_result,
    run_molecular_average_trace_distance_experiment,
)


@pytest.fixture(scope="module")
def h2_experiment():
    return run_molecular_average_trace_distance_experiment(
        H2_STO3G_JW,
        total_time=0.2,
        number_of_steps=1,
        num_initial_states=3,
        group_norm_method="lp",
        grouping_method="chemistry",
        max_group_size=12,
        trace_distance_method="low_rank",
        seed=42,
    )


def test_molecular_comparison_uses_the_chemistry_decomposition(
    h2_experiment,
) -> None:
    decompositions = h2_experiment.decompositions
    assert h2_experiment.estimates is not None
    assert h2_experiment.bounds is not None
    pauli, grouped = h2_experiment.estimates
    pauli_bound, grouped_bound = h2_experiment.bounds
    assert decompositions.generated_hamiltonian.num_qubits == 4
    assert len(decompositions.target_hamiltonian.terms) == 14
    assert [len(group) for group in decompositions.groups] == [10, 4]
    assert decompositions.grouping_method == "chemistry"
    assert pauli.mean == pytest.approx(0.1180949972)
    assert grouped.mean == pytest.approx(0.0052158812)
    assert grouped.mean / pauli.mean == pytest.approx(
        0.0441668262
    )
    assert pauli_bound.step_second_order_bound > 0.0
    assert grouped_bound.step_second_order_bound >= 0.0
    assert pauli_bound.step_average_trace_distance_upper_bound <= 1.0
    assert (
        grouped_bound.step_average_trace_distance_upper_bound
        <= 1.0
    )


def test_molecular_comparison_output_contains_samples_and_bounds(
    h2_experiment,
    capsys,
) -> None:
    assert h2_experiment.estimates is not None
    pauli, grouped = h2_experiment.estimates
    _print_result(h2_experiment)
    output = capsys.readouterr().out

    assert "Monte Carlo Haar-average trace distance" in output
    assert "grouped / pauli mean ratio" in output
    assert "grouped - pauli paired mean" in output
    assert "Haar-average trace-distance bounds" in output
    assert "tau(V)" in output
    assert "second-order term" in output
    assert f"{pauli.mean:.8e}" in output
    assert f"{grouped.mean:.8e}" in output


def test_bound_mode_skips_monte_carlo() -> None:
    experiment = run_molecular_average_trace_distance_experiment(
        H2_STO3G_JW,
        total_time=0.2,
        number_of_steps=1,
        group_norm_method="lp",
        grouping_method="chemistry",
        max_group_size=12,
        mode="bound",
    )

    assert experiment.estimates is None
    assert experiment.bounds is not None
