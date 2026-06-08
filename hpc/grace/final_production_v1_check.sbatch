#!/usr/bin/env bash
#SBATCH --job-name=final_prod_v1_check
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/final_production_v1/check_%j.out
#SBATCH --error=logs/final_production_v1/check_%j.err

set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-}"
CASE_CONFIG="${CASE_CONFIG:-configs/final_production_v1_cases.yaml}"
OUT_DIR="${OUT_DIR:-results/final_production_v1}"
CASE_TSV="${CASE_TSV:-${OUT_DIR}/final_production_cases.tsv}"

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

python scripts/make_grace_final_production_jobs.py \
  --config "${CASE_CONFIG}" \
  --out-dir "${OUT_DIR}" \
  --case-tsv "${CASE_TSV}" \
  --check
