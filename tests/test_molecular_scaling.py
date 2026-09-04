import csv

import pytest

from hamiltonians.chemistry import H2_STO3G_JW
from molecular_scaling_experiment import (
    DEFAULT_SYSTEMS,
    FEMOCO_NAME,
    MolecularSweepConfig,
    aggregate_sweep,
    run_sweep_shard,
)


def test_default_molecular_benchmark_has_requested_ten_systems() -> None:
    assert len(DEFAULT_SYSTEMS) == 10
    assert DEFAULT_SYSTEMS[-1] == FEMOCO_NAME


def test_molecular_sweep_tau_only_resume_and_aggregate(tmp_path) -> None:
    config = MolecularSweepConfig(
        systems=(H2_STO3G_JW.name,),
        statevector_max_qubits=3,
        number_of_steps=2,
        num_initial_states=2,
        num_trajectories=2,
        max_group_size=0,
        base_seed=7,
        num_workers=1,
        trajectory_chunks_per_state=1,
    )

    assert run_sweep_shard(config, tmp_path) == (1, 0)
    assert run_sweep_shard(config, tmp_path) == (0, 1)
    csv_path, plot_path = aggregate_sweep(tmp_path, make_plot=False)

    assert plot_path is None
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["num_qubits"] == "4"
    assert rows[0]["actual_metrics_computed"] == "False"
    assert rows[0]["infidelity_factor"] == ""
    assert float(rows[0]["pauli_lambda_time"]) == pytest.approx(1.0)


def test_femoco_path_is_required_when_selected() -> None:
    with pytest.raises(ValueError, match="femoco_fcidump is required"):
        MolecularSweepConfig(systems=(FEMOCO_NAME,))
