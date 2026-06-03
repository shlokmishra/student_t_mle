#!/usr/bin/env bash
#SBATCH --job-name=ref_all
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/ref_all_%j.out

set -euo pipefail

# Full all-model reference audit for the dashboard.
# Defaults avoid exact adaptive t_abram because it can be much slower at B=100000.
# To request all three KDE forms:
#   BANDWIDTHS=scott,SJ_transform,t_abram sbatch scripts/grace_reference_audit.sbatch

PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11}"
B_VALUES="${B_VALUES:-100000}"
SEEDS="${SEEDS:-123,456,789}"
N_VALUES="${N_VALUES:-10,20,50}"
K_VALUES="${K_VALUES:-1,2,3}"
BANDWIDTHS="${BANDWIDTHS:-scott,SJ_transform}"
GRID_SIZE="${GRID_SIZE:-2500}"
OUT_CSV="${OUT_CSV:-reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv}"

if command -v module >/dev/null 2>&1; then
  module purge
  module load "${PYTHON_MODULE}"
fi

if [ ! -f .venv/bin/activate ]; then
  echo "Missing .venv. Run: bash scripts/grace_setup_env.sh"
  exit 2
fi

source .venv/bin/activate
mkdir -p logs "$(dirname "${OUT_CSV}")"

python reporting/diagnostics/audit_reference_all_models.py \
  --models student_t logistic laplace \
  --k-values "${K_VALUES}" \
  --n-values "${N_VALUES}" \
  --B-values "${B_VALUES}" \
  --seeds "${SEEDS}" \
  --bandwidths "${BANDWIDTHS}" \
  --grid-size "${GRID_SIZE}" \
  --out-csv "${OUT_CSV}" \
  --overwrite

