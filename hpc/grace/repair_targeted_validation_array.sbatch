#!/usr/bin/env bash
#SBATCH --job-name=repair_targeted_val
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/targeted_validation_repair/%A_%a.out
#SBATCH --error=logs/targeted_validation_repair/%A_%a.err

set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-}"
CASE_CONFIG="${CASE_CONFIG:-configs/targeted_validation_cases.yaml}"
REPAIR_TSV="${REPAIR_TSV:-hpc/grace/repair_targeted_validation_cases.tsv}"

mkdir -p logs/targeted_validation_repair

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

if [ ! -f "${REPAIR_TSV}" ]; then
  echo "Missing repair list: ${REPAIR_TSV}"
  exit 2
fi

row="$(awk -v task_id="${SLURM_ARRAY_TASK_ID}" 'BEGIN {FS="\t"} NR == task_id + 1 {print $0}' "${REPAIR_TSV}")"
if [ -z "${row}" ]; then
  echo "No repair row for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 2
fi

IFS=$'\t' read -r case_id model k n method seed initialization num_iterations burn_in diagnostic_thin force_overwrite output_dir <<< "${row}"

if [ "${force_overwrite}" != "true" ]; then
  echo "Repair row ${case_id} does not set force_overwrite=true"
  exit 2
fi

case_leaf="case_${case_id}"
if [ "$(basename "${output_dir}")" != "${case_leaf}" ]; then
  echo "Repair output_dir ${output_dir} does not match ${case_leaf}"
  exit 2
fi

out_parent="$(dirname "${output_dir}")"
mkdir -p "${out_parent}"

echo "Repairing ${case_id}: model=${model} k=${k} n=${n} method=${method} seed=${seed} init=${initialization}"
echo "Target settings: num_iterations=${num_iterations} burn_in=${burn_in} diagnostic_thin=${diagnostic_thin}"

python scripts/run_targeted_validation.py \
  --case-config "${CASE_CONFIG}" \
  --case-id "${case_id}" \
  --out "${out_parent}" \
  --num-iterations "${num_iterations}" \
  --burn-in "${burn_in}" \
  --seed "${seed}" \
  --diagnostic-thin "${diagnostic_thin}" \
  --save-transition-diagnostics \
  --save-latent-diagnostics \
  --save-rattle-energy-diagnostics \
  --save-branch-diagnostics \
  --save-initialization-diagnostics \
  --force
