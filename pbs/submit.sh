#!/usr/bin/env bash

set -euo pipefail

# OpenPBS 20.0 does not yet support the later "-J 1-N%limit" syntax.  Use one
# array subjob per requested node, so changing NUM_NODES changes both the
# partition count and maximum number of nodes this experiment can occupy.
NUM_NODES=${NUM_NODES:-4}
NUM_SHARDS=$NUM_NODES

# One array subjob occupies one of the user's 96-core, ~370-GB nodes.  Request
# 350 GB by default to leave headroom for site/system use.
NCPUS=${NCPUS:-96}
MEMORY=${MEMORY:-350gb}
WALLTIME=${WALLTIME:-12:00:00}
NUM_WORKERS=${NUM_WORKERS:-$NCPUS}
TRAJECTORY_CHUNKS_PER_STATE=${TRAJECTORY_CHUNKS_PER_STATE:-$NUM_WORKERS}

# Experiment settings.
SOURCE_KIND=${SOURCE_KIND:-molecule}
HAMILTONIAN=${HAMILTONIAN:-h2o_sto3g_cas_4e_4o_jw}
SYK_QUBITS=${SYK_QUBITS:-10}
SYK_COUPLING_SCALE=${SYK_COUPLING_SCALE:-1.0}
SYK_SEED=${SYK_SEED:-42}
TOTAL_TIME=${TOTAL_TIME:-0.3}
NUMBER_OF_STEPS=${NUMBER_OF_STEPS:-20}
NUM_INITIAL_STATES=${NUM_INITIAL_STATES:-100}
NUM_TRAJECTORIES=${NUM_TRAJECTORIES:-500}
MAX_GROUP_SIZE=${MAX_GROUP_SIZE:-12}
SEED=${SEED:-42}

PROJECT_DIR=${PROJECT_DIR:-$(pwd -P)}
RESULT_DIR=${RESULT_DIR:-$PROJECT_DIR/results/qdrift_$(date +%Y%m%d_%H%M%S)}
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

require_positive_integer "$NUM_NODES" NUM_NODES
require_positive_integer "$NCPUS" NCPUS
require_positive_integer "$NUM_WORKERS" NUM_WORKERS
require_positive_integer "$TRAJECTORY_CHUNKS_PER_STATE" \
    TRAJECTORY_CHUNKS_PER_STATE
require_positive_integer "$NUM_INITIAL_STATES" NUM_INITIAL_STATES
require_positive_integer "$NUM_TRAJECTORIES" NUM_TRAJECTORIES

if (( NUM_SHARDS > NUM_INITIAL_STATES )); then
    echo "NUM_SHARDS cannot exceed NUM_INITIAL_STATES" >&2
    exit 2
fi
if (( NUM_WORKERS > NCPUS )); then
    echo "NUM_WORKERS cannot exceed requested NCPUS" >&2
    exit 2
fi
if [[ "$SOURCE_KIND" != molecule && "$SOURCE_KIND" != syk ]]; then
    echo "SOURCE_KIND must be molecule or syk" >&2
    exit 2
fi
if ! command -v qsub >/dev/null 2>&1; then
    echo "qsub was not found in PATH" >&2
    exit 127
fi

mkdir -p "$RESULT_DIR"
RESULT_DIR=$(cd "$RESULT_DIR" && pwd -P)

export PROJECT_DIR RESULT_DIR NUM_SHARDS SOURCE_KIND HAMILTONIAN
export SYK_QUBITS SYK_COUPLING_SCALE SYK_SEED TOTAL_TIME NUMBER_OF_STEPS
export NUM_INITIAL_STATES NUM_TRAJECTORIES MAX_GROUP_SIZE SEED NUM_WORKERS
export TRAJECTORY_CHUNKS_PER_STATE UV_EXECUTABLE PBS_ENV_SCRIPT

exported_variables=(
    PROJECT_DIR RESULT_DIR NUM_SHARDS SOURCE_KIND HAMILTONIAN
    SYK_QUBITS SYK_COUPLING_SCALE SYK_SEED TOTAL_TIME NUMBER_OF_STEPS
    NUM_INITIAL_STATES NUM_TRAJECTORIES MAX_GROUP_SIZE SEED NUM_WORKERS
    TRAJECTORY_CHUNKS_PER_STATE UV_EXECUTABLE PBS_ENV_SCRIPT
)
variable_list=$(IFS=,; echo "${exported_variables[*]}")

qsub_arguments=(
    -J "1-${NUM_SHARDS}"
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

job_id=$(qsub "${qsub_arguments[@]}" "$PROJECT_DIR/pbs/qdrift_array.pbs")

echo "Submitted PBS array: $job_id"
echo "Result directory:     $RESULT_DIR"
echo "Array shards/nodes:   $NUM_NODES"
echo
echo "Monitor all subjobs with:"
echo "  qstat -t $job_id"
echo
echo "After every subjob finishes successfully, merge with:"
echo "  cd $PROJECT_DIR"
echo "  uv run --frozen python qdrift_metric_comparison.py --merge-partials $RESULT_DIR"
echo
echo "Or submit the lightweight merge job from this project directory:"
echo "  qsub -v PROJECT_DIR=$PROJECT_DIR,RESULT_DIR=$RESULT_DIR pbs/merge_results.pbs"
