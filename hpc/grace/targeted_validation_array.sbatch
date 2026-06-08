#!/usr/bin/env bash
#SBATCH --job-name=targeted_val
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/targeted_validation/%A_%a.out
#SBATCH --error=logs/targeted_validation/%A_%a.err

set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-}"
CASE_CONFIG="${CASE_CONFIG:-configs/targeted_validation_cases.yaml}"
CASE_TSV="${CASE_TSV:-hpc/grace/targeted_validation_cases.tsv}"
OUT_DIR="${OUT_DIR:-results/targeted_validation_runs}"
DIAGNOSTIC_THIN="${DIAGNOSTIC_THIN:-20}"
FORCE_FLAG="${FORCE_FLAG:-}"

mkdir -p logs/targeted_validation "${OUT_DIR}"

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
  echo "Missing ${CASE_TSV}. Run scripts/make_grace_targeted_validation_jobs.py first."
  exit 2
fi

CASE_ID="$(awk -v task_id="${SLURM_ARRAY_TASK_ID}" 'NR > 1 && $1 == task_id {print $2}' "${CASE_TSV}")"
if [ -z "${CASE_ID}" ]; then
  echo "No case id for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 2
fi

case_dir="${OUT_DIR}/case_${CASE_ID}"
if [ -f "${case_dir}/run_metadata.json" ] && [ "${FORCE_FLAG}" != "--force" ]; then
  echo "Case ${CASE_ID} already complete; set FORCE_FLAG=--force to rerun."
  exit 0
fi

python scripts/run_targeted_validation.py \
  --case-config "${CASE_CONFIG}" \
  --case-id "${CASE_ID}" \
  --out "${OUT_DIR}" \
  --diagnostic-thin "${DIAGNOSTIC_THIN}" \
  --save-transition-diagnostics \
  --save-latent-diagnostics \
  --save-rattle-energy-diagnostics \
  --save-branch-diagnostics \
  --save-initialization-diagnostics \
  ${FORCE_FLAG}
