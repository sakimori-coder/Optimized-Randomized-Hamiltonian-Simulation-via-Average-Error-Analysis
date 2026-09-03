#!/usr/bin/env bash

set -euo pipefail

# OpenPBS 20.0 has no -J 1-N%limit syntax.  Use exactly one long-running
# array subjob per node the user wants this sweep to occupy.
NUM_NODES=${NUM_NODES:-4}
NCPUS=${NCPUS:-96}
MEMORY=${MEMORY:-350gb}
WALLTIME=${WALLTIME:-48:00:00}

# State-vector realizations use NUM_WORKERS processes.  Tau-only realizations
# are independently parallelized with the more memory-conservative value.
NUM_WORKERS=${NUM_WORKERS:-$NCPUS}
TRAJECTORY_CHUNKS_PER_STATE=${TRAJECTORY_CHUNKS_PER_STATE:-$NUM_WORKERS}
TAU_WORKERS=${TAU_WORKERS:-8}

# Requested experiment: q=5,...,50; 50 seeded disorder realizations per q.
MIN_QUBITS=${MIN_QUBITS:-5}
MAX_QUBITS=${MAX_QUBITS:-50}
NUM_REALIZATIONS=${NUM_REALIZATIONS:-50}
STATEVECTOR_MAX_QUBITS=${STATEVECTOR_MAX_QUBITS:-15}
COUPLING_SCALE=${COUPLING_SCALE:-1.0}
TOTAL_TIME=${TOTAL_TIME:-0.3}
NUMBER_OF_STEPS=${NUMBER_OF_STEPS:-20}
NUM_INITIAL_STATES=${NUM_INITIAL_STATES:-20}
NUM_TRAJECTORIES=${NUM_TRAJECTORIES:-200}
# Zero means max_group_size = current number of qubits.
MAX_GROUP_SIZE=${MAX_GROUP_SIZE:-0}
BASE_SEED=${BASE_SEED:-42}

PROJECT_DIR=${PROJECT_DIR:-$(pwd -P)}
RESULT_DIR=${RESULT_DIR:-$PROJECT_DIR/results/syk_scaling_$(date +%Y%m%d_%H%M%S)_$$}
UV_EXECUTABLE=${UV_EXECUTABLE:-uv}
PBS_ENV_SCRIPT=${PBS_ENV_SCRIPT:-}
PBS_QUEUE=${PBS_QUEUE:-}
PBS_ACCOUNT=${PBS_ACCOUNT:-}

require_positive_integer() {
    local value=$1
    local name=$2
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer, got: $value" >&2
        exit 2
    fi
}

for setting in \
    NUM_NODES NCPUS NUM_WORKERS TRAJECTORY_CHUNKS_PER_STATE TAU_WORKERS \
    MIN_QUBITS MAX_QUBITS NUM_REALIZATIONS NUMBER_OF_STEPS \
    NUM_INITIAL_STATES NUM_TRAJECTORIES
do
    require_positive_integer "${!setting}" "$setting"
done
if (( MIN_QUBITS < 2 || MAX_QUBITS < MIN_QUBITS )); then
    echo "Require 2 <= MIN_QUBITS <= MAX_QUBITS" >&2
    exit 2
fi
if (( NUM_WORKERS > NCPUS )); then
    echo "NUM_WORKERS cannot exceed NCPUS" >&2
    exit 2
fi
if (( TAU_WORKERS > NCPUS )); then
    echo "TAU_WORKERS cannot exceed NCPUS" >&2
    exit 2
fi
if ! command -v qsub >/dev/null 2>&1; then
    echo "qsub was not found in PATH" >&2
    exit 127
fi

mkdir -p "$RESULT_DIR"
RESULT_DIR=$(cd "$RESULT_DIR" && pwd -P)

export PROJECT_DIR RESULT_DIR NUM_NODES NCPUS NUM_WORKERS
export TRAJECTORY_CHUNKS_PER_STATE TAU_WORKERS MIN_QUBITS MAX_QUBITS
export NUM_REALIZATIONS STATEVECTOR_MAX_QUBITS COUPLING_SCALE TOTAL_TIME
export NUMBER_OF_STEPS NUM_INITIAL_STATES NUM_TRAJECTORIES MAX_GROUP_SIZE
export BASE_SEED UV_EXECUTABLE PBS_ENV_SCRIPT

exported_variables=(
    PROJECT_DIR RESULT_DIR NUM_NODES NCPUS NUM_WORKERS
    TRAJECTORY_CHUNKS_PER_STATE TAU_WORKERS MIN_QUBITS MAX_QUBITS
    NUM_REALIZATIONS STATEVECTOR_MAX_QUBITS COUPLING_SCALE TOTAL_TIME
    NUMBER_OF_STEPS NUM_INITIAL_STATES NUM_TRAJECTORIES MAX_GROUP_SIZE
    BASE_SEED UV_EXECUTABLE PBS_ENV_SCRIPT
)
variable_list=$(IFS=,; echo "${exported_variables[*]}")

qsub_arguments=(
    -J "1-${NUM_NODES}"
    -l "select=1:ncpus=${NCPUS}:mem=${MEMORY}"
    -l "walltime=${WALLTIME}"
    -v "$variable_list"
)
if [[ -n "$PBS_QUEUE" ]]; then
    qsub_arguments+=(-q "$PBS_QUEUE")
fi
if [[ -n "$PBS_ACCOUNT" ]]; then
    qsub_arguments+=(-A "$PBS_ACCOUNT")
fi

job_id=$(qsub "${qsub_arguments[@]}" "$PROJECT_DIR/pbs/syk_scaling_array.pbs")

echo "Submitted SYK scaling array: $job_id"
echo "Result directory:            $RESULT_DIR"
echo "Array shards/nodes:          $NUM_NODES"
echo "Total realization tasks:     $(((MAX_QUBITS - MIN_QUBITS + 1) * NUM_REALIZATIONS))"
echo
echo "Monitor with:"
echo "  qstat -t $job_id"
echo
echo "If walltime expires, submit again with the same RESULT_DIR; completed"
echo "realizations will be validated and skipped."
echo
echo "After all array subjobs finish, aggregate with:"
echo "  cd $PROJECT_DIR"
echo "  uv run --frozen python syk_scaling_experiment.py aggregate --output-dir $RESULT_DIR"
echo
echo "Or submit the lightweight aggregate job:"
echo "  qsub -v PROJECT_DIR=$PROJECT_DIR,RESULT_DIR=$RESULT_DIR pbs/aggregate_syk_scaling.pbs"
