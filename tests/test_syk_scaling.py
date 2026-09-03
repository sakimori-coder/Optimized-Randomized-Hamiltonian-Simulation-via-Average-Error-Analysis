import csv

import pytest

from hamiltonians.syk import SykHamiltonianPreset
from qdrift_metric_comparison import run_experiment
from syk_scaling_experiment import (
    SykSweepConfig,
    _calculate_tau_only,
    aggregate_sweep,
    run_sweep_shard,
)


def test_tau_only_path_matches_full_decomposition_tau() -> None:
    preset = SykHamiltonianPreset(num_qubits=4, seed=17)
    pauli, grouped, num_terms, num_groups = _calculate_tau_only(
        preset,
        max_group_size=4,
    )
    full = run_experiment(
        preset,
        total_time=0.05,
        number_of_steps=2,
        num_initial_states=2,
        num_trajectories=2,
        max_group_size=4,
        seed=3,
        num_workers=1,
        trajectory_chunks_per_state=1,
    )

    assert pauli.tau_m == pytest.approx(full.tau_m[0].tau_m)
    assert grouped.tau_m == pytest.approx(full.tau_m[1].tau_m)
    assert num_terms == len(full.decompositions.target)
    assert num_groups == len(full.decompositions.grouped.terms)


def test_sweep_shards_resume_and_aggregate(tmp_path) -> None:
    config = SykSweepConfig(
        min_qubits=2,
        max_qubits=3,
        num_realizations=2,
        statevector_max_qubits=2,
        number_of_steps=2,
        num_initial_states=2,
        num_trajectories=2,
        max_group_size=0,
        base_seed=9,
        num_workers=1,
        trajectory_chunks_per_state=1,
    )
    first = run_sweep_shard(
        config,
        tmp_path,
        task_index=0,
        num_tasks=2,
        tau_workers=2,
    )
    second = run_sweep_shard(
        config,
        tmp_path,
        task_index=1,
        num_tasks=2,
        tau_workers=2,
    )
    resumed = run_sweep_shard(
        config,
        tmp_path,
        task_index=0,
        num_tasks=2,
        tau_workers=1,
    )

    assert first == (2, 0)
    assert second == (2, 0)
    assert resumed == (0, 2)
    raw_path, summary_path, plot_path = aggregate_sweep(
        tmp_path,
        make_plot=False,
    )
    assert plot_path is None
    with raw_path.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    with summary_path.open(newline="", encoding="utf-8") as stream:
        summary = list(csv.DictReader(stream))

    assert len(raw) == 4
    assert len(summary) == 6
    assert all(
        float(row["pauli_lambda_time"]) == pytest.approx(1.0)
        for row in raw
    )
    q2 = [row for row in raw if row["num_qubits"] == "2"]
    q3 = [row for row in raw if row["num_qubits"] == "3"]
    assert all(row["infidelity_factor"] for row in q2)
    assert all(not row["infidelity_factor"] for row in q3)
