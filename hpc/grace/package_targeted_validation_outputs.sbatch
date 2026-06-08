#!/usr/bin/env bash
#SBATCH --job-name=package_targeted_val
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --output=logs/targeted_validation/package_%j.out
#SBATCH --error=logs/targeted_validation/package_%j.err

set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-}"
RUN_DIR="${RUN_DIR:-results/targeted_validation_runs}"
LOGS_DIR="${LOGS_DIR:-logs/targeted_validation}"
INCLUDE_FULL_CHAINS="${INCLUDE_FULL_CHAINS:-0}"

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

chain_flag=()
if [ "${INCLUDE_FULL_CHAINS}" = "1" ]; then
  chain_flag=(--include-full-chains)
fi

python scripts/package_targeted_validation_for_local.py \
  --run-dir "${RUN_DIR}" \
  --logs-dir "${LOGS_DIR}" \
  --out-dir "${RUN_DIR}" \
  "${chain_flag[@]}"
