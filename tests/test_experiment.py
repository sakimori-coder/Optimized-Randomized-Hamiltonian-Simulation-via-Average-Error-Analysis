import math

from hamiltonians.chemistry import H2_STO3G_JW
from hamiltonians.syk import SykHamiltonianPreset
from qdrift_metric_comparison import (
    merge_partial_experiments,
    print_experiment,
    run_experiment,
    save_partial_experiment,
)


def test_molecular_experiment_outputs_exactly_the_three_ratios(capsys) -> None:
    experiment = run_experiment(
        H2_STO3G_JW,
        total_time=0.1,
        number_of_steps=2,
        num_initial_states=2,
        num_trajectories=4,
        max_group_size=4,
        num_workers=2,
        seed=7,
    )

    assert experiment.tau_m_ratio > 1.0
    assert math.isfinite(experiment.infidelity_ratio)
    assert math.isfinite(experiment.qpe_signal_error_ratio)
    print_experiment(experiment)
    output = capsys.readouterr().out
    assert "tau(M_p)" in output
    assert "average infidelity" in output
    assert "average QPE signal error" in output
    assert "ratio (pauli/grouped)" in output


def test_syk_uses_the_same_fixed_experiment_path() -> None:
    experiment = run_experiment(
        SykHamiltonianPreset(num_qubits=3, seed=4),
        total_time=0.1,
        number_of_steps=2,
        num_initial_states=2,
        num_trajectories=4,
        max_group_size=3,
        num_workers=1,
        seed=6,
    )

    assert experiment.decompositions.generated.num_qubits == 3
    assert math.isfinite(experiment.tau_m_ratio)
    assert math.isfinite(experiment.infidelity_ratio)
    assert math.isfinite(experiment.qpe_signal_error_ratio)


def test_pbs_partials_merge_complete_global_input_state_set(tmp_path) -> None:
    common = dict(
        preset=H2_STO3G_JW,
        total_time=0.1,
        number_of_steps=2,
        num_initial_states=4,
        num_trajectories=4,
        max_group_size=4,
        num_workers=1,
        seed=11,
    )
    for shard_index, state_indices in enumerate(((0, 2), (1, 3))):
        experiment = run_experiment(
            **common,
            initial_state_indices=state_indices,
        )
        save_partial_experiment(
            experiment,
            tmp_path,
            shard_index=shard_index,
            num_shards=2,
            seed=11,
        )

    merged = merge_partial_experiments(tmp_path)
    full = run_experiment(**common)

    assert merged.tau_m[0] == full.tau_m[0].tau_m
    assert merged.tau_m[1] == full.tau_m[1].tau_m
    assert merged.infidelity[0].mean == full.trajectory_metrics[0].infidelity.mean
    assert merged.infidelity[1].mean == full.trajectory_metrics[1].infidelity.mean
    assert (
        merged.qpe_signal_error[0].mean
        == full.trajectory_metrics[0].absolute_qpe_signal_error.mean
    )
    assert (
        merged.qpe_signal_error[1].mean
        == full.trajectory_metrics[1].absolute_qpe_signal_error.mean
    )
