#!/usr/bin/env bash
#SBATCH --job-name=check_targeted_val
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/targeted_validation/check_%j.out
#SBATCH --error=logs/targeted_validation/check_%j.err

set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-}"
RUN_DIR="${RUN_DIR:-results/targeted_validation_runs}"
CASE_TSV="${CASE_TSV:-hpc/grace/targeted_validation_cases.tsv}"

mkdir -p logs/targeted_validation "${RUN_DIR}"

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

python scripts/check_targeted_validation_outputs.py \
  --run-dir "${RUN_DIR}" \
  --case-table "${CASE_TSV}"
