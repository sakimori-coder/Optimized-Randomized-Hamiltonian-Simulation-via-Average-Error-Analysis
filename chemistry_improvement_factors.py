"""Process molecules in order, parallelizing trajectories within each node."""

import csv
import math
import os
import sys
from pathlib import Path

# qDRIFT workers use one Qulacs thread each.
if __name__ == "__main__":
    os.environ["QULACS_NUM_THREADS"] = "1"

from grouping import build_fermionic_lch
from hamiltonians import chemistry
from improvement_factors import calculate_i_m, estimate_i_r_and_i_sig


MOLECULES = [*chemistry.MOLECULAR_HAMILTONIANS, chemistry.FEMOCO_NAME]
NUMBER_OF_STEPS = 100
NUM_INITIAL_STATES = 100
NUM_TRAJECTORIES = 1000
NUM_WORKERS = len(os.sched_getaffinity(0))
SEED = 42
OUTPUT_PATH = Path("results/chemistry_improvement_factors.csv")


def main(molecules):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "molecule", "num_qubits", "num_single_terms", "num_groups", "total_time",
            "number_of_steps", "num_initial_states", "num_trajectories", "seed",
            "I_M", "I_r", "I_sig",
        ])
        for molecule in molecules:
            print(f"Computing {molecule} ...", flush=True)
            hamiltonian = chemistry.generate(molecule)
            grouped = build_fermionic_lch(hamiltonian)
            # Paper setting: t * lambda_single = 1.
            total_time = 1 / math.fsum(abs(a) for a in hamiltonian.terms.values())
            I_M = calculate_i_m(hamiltonian, grouped)

            # The paper estimates state-vector errors only through 16 qubits.
            I_r = I_sig = None
            if hamiltonian.num_qubits <= 16:
                I_r, I_sig = estimate_i_r_and_i_sig(
                    hamiltonian, grouped, total_time=total_time,
                    number_of_steps=NUMBER_OF_STEPS,
                    num_initial_states=NUM_INITIAL_STATES,
                    num_trajectories=NUM_TRAJECTORIES,
                    seed=SEED, num_workers=NUM_WORKERS,
                )

            writer.writerow([
                molecule, hamiltonian.num_qubits, len(hamiltonian.terms),
                len(grouped.terms), total_time, NUMBER_OF_STEPS,
                NUM_INITIAL_STATES, NUM_TRAJECTORIES, SEED, I_M, I_r, I_sig,
            ])
            stream.flush()
            print(f"  I_M={I_M}, I_r={I_r}, I_sig={I_sig}", flush=True)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main(sys.argv[1:] or MOLECULES)
