#!/usr/bin/env bash
#SBATCH --job-name=cost_all
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/cost_all_%j.out

set -euo pipefail

# Long Gibbs/RATTLE cost audit for Student-t, logistic, and Laplace.
# Laplace RATTLE is intentionally not applicable and is reported as such.

PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11}"
NUM_ITERATIONS="${NUM_ITERATIONS:-50000}"
BURN_IN="${BURN_IN:-10000}"
SEED="${SEED:-0}"
N_VALUES="${N_VALUES:-10,20,50}"
K_VALUES="${K_VALUES:-1,2,3}"
OUT_DIR="${OUT_DIR:-results/cost_audit}"
RUN_STATUS="${RUN_STATUS:-full}"
LATENT_DIAGNOSTIC_THIN="${LATENT_DIAGNOSTIC_THIN:-25}"
LATENT_DIAGNOSTIC_MAX_ROWS="${LATENT_DIAGNOSTIC_MAX_ROWS:-5000}"

if command -v module >/dev/null 2>&1; then
  module purge
  module load "${PYTHON_MODULE}"
fi

if [ ! -f .venv/bin/activate ]; then
  echo "Missing .venv. Run: bash scripts/grace_setup_env.sh"
  exit 2
fi

source .venv/bin/activate
mkdir -p logs "${OUT_DIR}"

python scripts/run_cost_audit.py \
  --models student_t logistic laplace \
  --methods gibbs rattle \
  --k-values "${K_VALUES}" \
  --n-values "${N_VALUES}" \
  --mu-star 0 \
  --num-iterations "${NUM_ITERATIONS}" \
  --burn-in "${BURN_IN}" \
  --seed "${SEED}" \
  --out "${OUT_DIR}" \
  --run-status "${RUN_STATUS}" \
  --save-latent-diagnostics \
  --latent-diagnostic-thin "${LATENT_DIAGNOSTIC_THIN}" \
  --latent-diagnostic-max-rows "${LATENT_DIAGNOSTIC_MAX_ROWS}"

