#!/usr/bin/env bash
#SBATCH --job-name=final_prod_v1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/final_production_v1/%A_%a.out
#SBATCH --error=logs/final_production_v1/%A_%a.err

set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-}"
CASE_CONFIG="${CASE_CONFIG:-configs/final_production_v1_cases.yaml}"
CASE_TSV="${CASE_TSV:-results/final_production_v1/final_production_cases.tsv}"
OUT_DIR="${OUT_DIR:-results/final_production_v1}"
FORCE_FLAG="${FORCE_FLAG:-}"

mkdir -p logs/final_production_v1 "${OUT_DIR}"

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
  python scripts/make_grace_final_production_jobs.py \
    --config "${CASE_CONFIG}" \
    --out-dir "${OUT_DIR}" \
    --case-tsv "${CASE_TSV}"
fi

case_id="$(awk -v task_id="${SLURM_ARRAY_TASK_ID}" 'BEGIN {FS="\t"} NR > 1 && $1 == task_id {print $2}' "${CASE_TSV}")"
if [ -z "${case_id}" ]; then
  echo "No case id for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 2
fi

case_dir="${OUT_DIR}/case_${case_id}"
if [ -f "${case_dir}/run_metadata.json" ] && [ "${FORCE_FLAG}" != "--force" ]; then
  echo "Case ${case_id} already complete; set FORCE_FLAG=--force to rerun."
  exit 0
fi

echo "Running final production v1 case ${case_id}"
python scripts/run_final_production_case.py \
  --case-config "${CASE_CONFIG}" \
  --case-id "${case_id}" \
  --out "${OUT_DIR}" \
  ${FORCE_FLAG}
