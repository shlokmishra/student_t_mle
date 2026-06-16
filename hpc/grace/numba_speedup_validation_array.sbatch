#!/usr/bin/env bash
#SBATCH --job-name=numba_speedup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/numba_speedup_validation/%A_%a.out
#SBATCH --error=logs/numba_speedup_validation/%A_%a.err

set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-GCCcore/14.3.0 Python/3.13.5}"
CASE_TSV="${CASE_TSV:-hpc/grace/numba_speedup_validation_cases.tsv}"
OUT_DIR="${OUT_DIR:-results/numba_speedup_validation}"

mkdir -p logs/numba_speedup_validation "${OUT_DIR}/task_results"

if command -v module >/dev/null 2>&1 && [ -n "${PYTHON_MODULE}" ]; then
  module purge
  # shellcheck disable=SC2086
  module load ${PYTHON_MODULE}
fi

if [ ! -f .venv/bin/activate ]; then
  echo "Missing .venv. Run: bash scripts/grace_setup_env.sh"
  exit 2
fi

source .venv/bin/activate

if [ ! -f "${CASE_TSV}" ]; then
  echo "Missing ${CASE_TSV}. Generate the case table before submitting."
  exit 2
fi

row="$(awk -v task_id="${SLURM_ARRAY_TASK_ID}" 'BEGIN {FS="\t"} NR > 1 && $1 == task_id {print $0}' "${CASE_TSV}")"
if [ -z "${row}" ]; then
  echo "No numba speedup row for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 2
fi

backend="$(printf '%s\n' "${row}" | awk -F '\t' '{print $2}')"
k_value="$(printf '%s\n' "${row}" | awk -F '\t' '{print $3}')"
n_value="$(printf '%s\n' "${row}" | awk -F '\t' '{print $4}')"
seed="$(printf '%s\n' "${row}" | awk -F '\t' '{print $5}')"
repeat="$(printf '%s\n' "${row}" | awk -F '\t' '{print $6}')"
iterations="$(printf '%s\n' "${row}" | awk -F '\t' '{print $7}')"
warmup_iterations="$(printf '%s\n' "${row}" | awk -F '\t' '{print $8}')"

task_out="${OUT_DIR}/task_results/task_${SLURM_ARRAY_TASK_ID}_${backend}_k${k_value}_n${n_value}_seed${seed}_repeat${repeat}.csv"

echo "START task=${SLURM_ARRAY_TASK_ID} backend=${backend} k=${k_value} n=${n_value} seed=${seed} repeat=${repeat} iterations=${iterations} warmup_iterations=${warmup_iterations} out=${task_out}"
heartbeat() {
  start_epoch="$(date +%s)"
  while true; do
    sleep 60
    now_epoch="$(date +%s)"
    elapsed="$((now_epoch - start_epoch))"
    echo "PROGRESS task=${SLURM_ARRAY_TASK_ID} backend=${backend} k=${k_value} n=${n_value} seed=${seed} repeat=${repeat} elapsed_sec=${elapsed}"
  done
}
heartbeat &
heartbeat_pid="$!"
trap 'kill "${heartbeat_pid}" >/dev/null 2>&1 || true' EXIT

python scripts/benchmark_student_gibbs_backends.py \
  --backends "${backend}" \
  --k-values "${k_value}" \
  --n-values "${n_value}" \
  --iterations "${iterations}" \
  --warmup-iterations "${warmup_iterations}" \
  --seeds "${seed}" \
  --out "${task_out}"

kill "${heartbeat_pid}" >/dev/null 2>&1 || true
trap - EXIT
echo "DONE task=${SLURM_ARRAY_TASK_ID} backend=${backend} k=${k_value} n=${n_value} seed=${seed} repeat=${repeat} out=${task_out}"
