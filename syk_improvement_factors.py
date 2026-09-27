"""Evaluate (qubit count, realization) pairs in parallel within one node."""

import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from multiprocessing import get_context
from pathlib import Path

# qDRIFT workers use one Qulacs thread each.
if __name__ == "__main__":
    os.environ["QULACS_NUM_THREADS"] = "1"

from grouping import build_fermionic_lch
from hamiltonians import syk
from improvement_factors import calculate_i_m, estimate_i_r_and_i_sig


QUBIT_COUNTS = range(5, 51)
NUM_REALIZATIONS = 50
COUPLING_SCALE = 1.0
NUMBER_OF_STEPS = 100
NUM_INITIAL_STATES = 100
NUM_TRAJECTORIES = 1000
NUM_WORKERS = len(os.sched_getaffinity(0))
SEED = 42
OUTPUT_PATH = Path("results/syk_improvement_factors.csv")


def _calculate_realization(task):
    num_qubits, realization = task
    print(f"Computing SYK n={num_qubits}, realization={realization} ...", flush=True)
    # Seeds depend on the instance, not on which worker executes it.
    hamiltonian_seed = SEED + 2 * (num_qubits * NUM_REALIZATIONS + realization)
    sampling_seed = hamiltonian_seed + 1
    hamiltonian = syk.generate(
        num_qubits, coupling_scale=COUPLING_SCALE, seed=hamiltonian_seed,
    )
    grouped = build_fermionic_lch(hamiltonian)
    # Paper setting: t * lambda_single = 1.
    total_time = 1 / math.fsum(abs(a) for a in hamiltonian.terms.values())
    I_M = calculate_i_m(hamiltonian, grouped)

    # The paper estimates state-vector errors only through 15 qubits.
    I_r = I_sig = None
    if num_qubits <= 15:
        I_r, I_sig = estimate_i_r_and_i_sig(
            hamiltonian, grouped, total_time=total_time,
            number_of_steps=NUMBER_OF_STEPS,
            num_initial_states=NUM_INITIAL_STATES,
            num_trajectories=NUM_TRAJECTORIES,
            seed=sampling_seed, num_workers=1,
        )

    return [
        num_qubits, realization, hamiltonian_seed, sampling_seed,
        COUPLING_SCALE, len(hamiltonian.terms), len(grouped.terms), total_time,
        NUMBER_OF_STEPS, NUM_INITIAL_STATES, NUM_TRAJECTORIES,
        I_M, I_r, I_sig,
    ]


def main(qubit_counts):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as stream, ProcessPoolExecutor(
        max_workers=NUM_WORKERS, mp_context=get_context("fork"),
    ) as executor:
        writer = csv.writer(stream)
        writer.writerow([
            "num_qubits", "realization", "hamiltonian_seed", "sampling_seed",
            "coupling_scale", "num_single_terms", "num_groups", "total_time",
            "number_of_steps", "num_initial_states", "num_trajectories",
            "I_M", "I_r", "I_sig",
        ])
        tasks = product(qubit_counts, range(NUM_REALIZATIONS))
        # All sizes share one pool; only the parent writes rows, in task order.
        for row in executor.map(_calculate_realization, tasks):
            writer.writerow(row)
            stream.flush()
            print(f"  SYK n={row[0]}, realization={row[1]}: "
                  f"I_M={row[-3]}, I_r={row[-2]}, I_sig={row[-1]}", flush=True)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main([int(value) for value in sys.argv[1:]] or QUBIT_COUNTS)
