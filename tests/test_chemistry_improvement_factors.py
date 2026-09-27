import csv
import math

import pytest

import chemistry_improvement_factors as experiment
import improvement_factors
from hamiltonians import chemistry
from operators import LCP


def test_h2_uses_local_processes_and_saves_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "NUMBER_OF_STEPS", 3)
    monkeypatch.setattr(experiment, "NUM_INITIAL_STATES", 3)
    monkeypatch.setattr(experiment, "NUM_TRAJECTORIES", 11)
    monkeypatch.setattr(experiment, "NUM_WORKERS", 2)
    pool_sizes = []
    process_pool = improvement_factors.ProcessPoolExecutor

    def create_pool(*args, **kwargs):
        pool_sizes.append(kwargs["max_workers"])
        return process_pool(*args, **kwargs)

    monkeypatch.setattr(improvement_factors, "ProcessPoolExecutor", create_pool)
    experiment.main(["h2_sto3g_jw"])
    assert pool_sizes == [2]

    with experiment.OUTPUT_PATH.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    row = rows[0]
    assert row["molecule"] == "h2_sto3g_jw"
    assert row["num_qubits"] == "4"
    assert row["number_of_steps"] == "3"
    assert row["num_initial_states"] == "3"
    assert row["num_trajectories"] == "11"
    assert "num_workers" not in row
    assert row["seed"] == "42"
    hamiltonian = chemistry.generate("h2_sto3g_jw")
    assert float(row["total_time"]) * sum(abs(a) for a in hamiltonian.terms.values()) == pytest.approx(1)
    assert all(math.isfinite(float(row[key])) for key in ("I_M", "I_r", "I_sig"))


def test_molecules_run_in_order_and_large_systems_only_compute_i_m(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "NUM_WORKERS", 2)
    molecules = ["nh3_sto3g_full_jw", "ch4_sto3g_full_jw"]
    sizes = dict(zip(molecules, (16, 18)))
    events = []

    def generate(molecule):
        events.append(("generate", molecule))
        n = sizes[molecule]
        return LCP({"X" + "I" * (n - 1): 1, "Z" + "I" * (n - 1): 1})

    def estimate(hamiltonian, grouped, **kwargs):
        events.append(("estimate", hamiltonian.num_qubits))
        assert kwargs["num_workers"] == 2
        return 2.0, 3.0

    monkeypatch.setattr(experiment.chemistry, "generate", generate)
    monkeypatch.setattr(experiment, "estimate_i_r_and_i_sig", estimate)
    experiment.main(molecules)

    assert events == [("generate", molecules[0]), ("estimate", 16), ("generate", molecules[1])]
    with experiment.OUTPUT_PATH.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["molecule"] for row in rows] == molecules
    assert [float(row["I_M"]) for row in rows] == [1.0, 1.0]
    assert (rows[0]["I_r"], rows[0]["I_sig"]) == ("2.0", "3.0")
    assert (rows[1]["I_r"], rows[1]["I_sig"]) == ("", "")
