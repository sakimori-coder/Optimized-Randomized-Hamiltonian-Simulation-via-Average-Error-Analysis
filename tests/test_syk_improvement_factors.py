import csv
import math
import os
from multiprocessing import get_context

import pytest

import improvement_factors
import syk_improvement_factors as experiment
from hamiltonians import syk
from operators import LCP


def test_sizes_and_realizations_run_together_without_nested_pools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "NUM_REALIZATIONS", 2)
    monkeypatch.setattr(experiment, "NUMBER_OF_STEPS", 3)
    monkeypatch.setattr(experiment, "NUM_INITIAL_STATES", 2)
    monkeypatch.setattr(experiment, "NUM_TRAJECTORIES", 5)
    monkeypatch.setattr(experiment, "NUM_WORKERS", 4)
    barrier = get_context("fork").Barrier(4)
    generate = syk.generate

    def generate_together(num_qubits, **kwargs):
        # Two realizations of each of two sizes must run simultaneously.
        # Waiting for one size to finish before starting the next would time out.
        barrier.wait(timeout=15)
        path = tmp_path / f"instance-{num_qubits}-{kwargs['seed']}.txt"
        path.write_text(str(os.getpid()))
        return generate(num_qubits, **kwargs)

    def forbid_trajectory_pool(*args, **kwargs):
        raise AssertionError("Each SYK task must run trajectories in its own process")

    monkeypatch.setattr(experiment.syk, "generate", generate_together)
    monkeypatch.setattr(improvement_factors, "ProcessPoolExecutor", forbid_trajectory_pool)
    experiment.main([5, 6])

    instance_files = list(tmp_path.glob("instance-*.txt"))
    worker_pids = {int(path.read_text()) for path in instance_files}
    assert len(instance_files) == 4
    assert len(worker_pids) == 4
    assert os.getpid() not in worker_pids

    with experiment.OUTPUT_PATH.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [(int(row["num_qubits"]), int(row["realization"])) for row in rows] == [
        (5, 0), (5, 1), (6, 0), (6, 1),
    ]
    hamiltonian_seeds = {row["hamiltonian_seed"] for row in rows}
    sampling_seeds = {row["sampling_seed"] for row in rows}
    assert len(hamiltonian_seeds) == len(sampling_seeds) == 4
    assert hamiltonian_seeds.isdisjoint(sampling_seeds)
    for row in rows:
        assert row["number_of_steps"] == "3"
        assert row["num_initial_states"] == "2"
        assert row["num_trajectories"] == "5"
        assert "num_workers" not in row
        hamiltonian = generate(
            int(row["num_qubits"]), coupling_scale=float(row["coupling_scale"]),
            seed=int(row["hamiltonian_seed"]),
        )
        assert int(row["num_single_terms"]) == len(hamiltonian.terms)
        assert float(row["total_time"]) * sum(abs(a) for a in hamiltonian.terms.values()) == pytest.approx(1)
        assert all(math.isfinite(float(row[key])) for key in ("I_M", "I_r", "I_sig"))


@pytest.mark.parametrize("num_workers", [1, 4])
def test_csv_order_and_large_system_cutoff(tmp_path, monkeypatch, num_workers):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "NUM_REALIZATIONS", 2)
    monkeypatch.setattr(experiment, "NUM_WORKERS", num_workers)

    def generate(num_qubits, **kwargs):
        return LCP({"X" + "I" * (num_qubits - 1): 1, "Z" + "I" * (num_qubits - 1): 1})

    def estimate(hamiltonian, grouped, **kwargs):
        assert hamiltonian.num_qubits == 15
        assert kwargs["num_workers"] == 1
        return 2.0, 3.0

    monkeypatch.setattr(experiment.syk, "generate", generate)
    monkeypatch.setattr(experiment, "estimate_i_r_and_i_sig", estimate)
    experiment.main([16, 15])

    with experiment.OUTPUT_PATH.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [(int(row["num_qubits"]), int(row["realization"])) for row in rows] == [
        (16, 0), (16, 1), (15, 0), (15, 1),
    ]
    assert [float(row["I_M"]) for row in rows] == [1.0] * 4
    assert [(row["I_r"], row["I_sig"]) for row in rows] == [
        ("", ""), ("", ""), ("2.0", "3.0"), ("2.0", "3.0"),
    ]


def test_worker_count_and_size_order_preserve_seeds_and_samples(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "NUM_REALIZATIONS", 3)
    monkeypatch.setattr(experiment, "NUMBER_OF_STEPS", 3)
    monkeypatch.setattr(experiment, "NUM_INITIAL_STATES", 2)
    monkeypatch.setattr(experiment, "NUM_TRAJECTORIES", 5)
    results = []
    for num_workers, sizes in ((1, [5, 6]), (4, [6, 5])):
        monkeypatch.setattr(experiment, "NUM_WORKERS", num_workers)
        experiment.main(sizes)
        with experiment.OUTPUT_PATH.open() as stream:
            rows = list(csv.DictReader(stream))
        rows.sort(key=lambda row: (int(row["num_qubits"]), int(row["realization"])))
        results.append(rows)
    assert results[0] == results[1]
