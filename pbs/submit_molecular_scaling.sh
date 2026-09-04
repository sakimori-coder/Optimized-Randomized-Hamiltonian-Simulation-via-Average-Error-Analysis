#!/usr/bin/env bash

set -euo pipefail

# One long-running array shard per requested node.  Change NUM_NODES without
# changing the experiment itself; checkpoints make resubmission safe.
NUM_NODES=${NUM_NODES:-4}
NCPUS=${NCPUS:-96}
MEMORY=${MEMORY:-350gb}
WALLTIME=${WALLTIME:-100:00:00}

NUM_WORKERS=${NUM_WORKERS:-$NCPUS}
TRAJECTORY_CHUNKS_PER_STATE=${TRAJECTORY_CHUNKS_PER_STATE:-$NUM_WORKERS}
STATEVECTOR_MAX_QUBITS=${STATEVECTOR_MAX_QUBITS:-16}
NUMBER_OF_STEPS=${NUMBER_OF_STEPS:-100}
NUM_INITIAL_STATES=${NUM_INITIAL_STATES:-20}
NUM_TRAJECTORIES=${NUM_TRAJECTORIES:-200}
MAX_GROUP_SIZE=${MAX_GROUP_SIZE:-0}
COEFFICIENT_TOLERANCE=${COEFFICIENT_TOLERANCE:-1e-12}
BASE_SEED=${BASE_SEED:-42}

PROJECT_DIR=${PROJECT_DIR:-$(pwd -P)}
RESULT_DIR=${RESULT_DIR:-$PROJECT_DIR/results/molecular_scaling_$(date +%Y%m%d_%H%M%S)_$$}
FEMOCO_FCIDUMP=${FEMOCO_FCIDUMP:-}
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
    NUM_NODES NCPUS NUM_WORKERS TRAJECTORY_CHUNKS_PER_STATE \
    NUMBER_OF_STEPS NUM_INITIAL_STATES NUM_TRAJECTORIES
do
    require_positive_integer "${!setting}" "$setting"
done
if (( NUM_NODES > 10 )); then
    echo "NUM_NODES cannot exceed the 10 molecular systems" >&2
    exit 2
fi
if (( NUM_WORKERS > NCPUS )); then
    echo "NUM_WORKERS cannot exceed NCPUS" >&2
    exit 2
fi
if [[ -z "$FEMOCO_FCIDUMP" ]]; then
    echo "Set FEMOCO_FCIDUMP to the Reiher 54e/54o FCIDUMP path" >&2
    exit 2
fi
if [[ ! -f "$FEMOCO_FCIDUMP" ]]; then
    echo "FeMoco FCIDUMP was not found: $FEMOCO_FCIDUMP" >&2
    exit 2
fi
if ! command -v qsub >/dev/null 2>&1; then
    echo "qsub was not found in PATH" >&2
    exit 127
fi

mkdir -p "$RESULT_DIR"
RESULT_DIR=$(cd "$RESULT_DIR" && pwd -P)
FEMOCO_FCIDUMP=$(cd "$(dirname "$FEMOCO_FCIDUMP")" && pwd -P)/$(basename "$FEMOCO_FCIDUMP")

export PROJECT_DIR RESULT_DIR FEMOCO_FCIDUMP NUM_NODES NCPUS NUM_WORKERS
export TRAJECTORY_CHUNKS_PER_STATE STATEVECTOR_MAX_QUBITS NUMBER_OF_STEPS
export NUM_INITIAL_STATES NUM_TRAJECTORIES MAX_GROUP_SIZE
export COEFFICIENT_TOLERANCE BASE_SEED UV_EXECUTABLE PBS_ENV_SCRIPT

exported_variables=(
    PROJECT_DIR RESULT_DIR FEMOCO_FCIDUMP NUM_NODES NCPUS NUM_WORKERS
    TRAJECTORY_CHUNKS_PER_STATE STATEVECTOR_MAX_QUBITS NUMBER_OF_STEPS
    NUM_INITIAL_STATES NUM_TRAJECTORIES MAX_GROUP_SIZE
    COEFFICIENT_TOLERANCE BASE_SEED UV_EXECUTABLE PBS_ENV_SCRIPT
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

job_id=$(qsub "${qsub_arguments[@]}" "$PROJECT_DIR/pbs/molecular_scaling_array.pbs")

echo "Submitted molecular scaling array: $job_id"
echo "Result directory:                  $RESULT_DIR"
echo "Array shards/nodes:                $NUM_NODES"
echo "FeMoco FCIDUMP:                    $FEMOCO_FCIDUMP"
echo
echo "Monitor with:"
echo "  qstat -t -n -1 $job_id"
echo
echo "After all subjobs finish, aggregate with:"
echo "  cd $PROJECT_DIR"
echo "  uv run --frozen python molecular_scaling_experiment.py aggregate --output-dir $RESULT_DIR"
