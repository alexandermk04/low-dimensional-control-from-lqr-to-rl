#!/bin/bash
# Submit one experiment as a SLURM job array. Run ON the cluster.
#
#   cluster/submit.sh <experiment> <shards> <time> [cpus] [mem] [extra sbatch args...]
#   cluster/submit.sh ch4_true_model 4 00:30:00
#   cluster/submit.sh ch6_perception 50 03:00:00 4 16G --dependency=afterok:123456
#
# Every array task runs the same experiment script and trains only every
# <shards>-th grid entry (see experiments.store.in_shard). Runs already stored
# are skipped.
set -euo pipefail

VENV=${VENV:-$WORK/venvs/thesis}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # Inside the job: pin every thread pool to the reserved cores
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
    source "$VENV/bin/activate"
    exec python -u -m "experiments.$EXPERIMENT"
fi

EXPERIMENT=$1 SHARDS=$2 TIME=$3 CPUS=${4:-1} MEM=${5:-4G}
shift $(( $# < 5 ? $# : 5 ))

cd "$(dirname "$0")/.."
mkdir -p cluster/logs

sbatch \
    --job-name="$EXPERIMENT" \
    --array="0-$((SHARDS - 1))" \
    --time="$TIME" \
    --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --output="cluster/logs/${EXPERIMENT}_%A_%a.out" \
    --export=ALL,EXPERIMENT="$EXPERIMENT",VENV="$VENV" \
    "$@" \
    cluster/submit.sh
