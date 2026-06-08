#!/usr/bin/env bash
#SBATCH --job-name=timing_fairness
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/timing_fairness/%A_%a.out
#SBATCH --error=logs/timing_fairness/%A_%a.err

set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-}"
CASE_TSV="${CASE_TSV:-hpc/grace/timing_fairness_cases.tsv}"
OUT_DIR="${OUT_DIR:-results/timing_fairness_audit}"

mkdir -p logs/timing_fairness "${OUT_DIR}"

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
  echo "Missing ${CASE_TSV}. Generate and review timing-fairness cases before submitting."
  exit 2
fi

row="$(awk -v task_id="${SLURM_ARRAY_TASK_ID}" 'NR > 1 && $1 == task_id {print $0}' "${CASE_TSV}")"
if [ -z "${row}" ]; then
  echo "No timing fairness row for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 2
fi

case_id="$(printf '%s\n' "${row}" | awk -F '\t' '{print $2}')"
method="$(printf '%s\n' "${row}" | awk -F '\t' '{print $3}')"
seed="$(printf '%s\n' "${row}" | awk -F '\t' '{print $4}')"
repeat="$(printf '%s\n' "${row}" | awk -F '\t' '{print $5}')"

python scripts/run_timing_fairness_audit.py \
  --case-id "${case_id}" \
  --method "${method}" \
  --seed "${seed}" \
  --repeat "${repeat}" \
  --out "${OUT_DIR}"
