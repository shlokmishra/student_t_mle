#!/usr/bin/env bash
#SBATCH --job-name=postproc
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/postproc_%j.out

set -euo pipefail

# Build dashboard-ready summaries after the reference and cost jobs finish.

PYTHON_MODULE="${PYTHON_MODULE:-}"
REFERENCE_CSV="${REFERENCE_CSV:-reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv}"
SUMMARY_DIR="${SUMMARY_DIR:-reporting/diagnostic_outputs/model_reference_audit/full_summary}"

if command -v module >/dev/null 2>&1 && [ -n "${PYTHON_MODULE}" ]; then
  module purge
  # PYTHON_MODULE may contain a prerequisite toolchain plus Python.
  # shellcheck disable=SC2086
  module load ${PYTHON_MODULE}
fi

if [ ! -f .venv/bin/activate ]; then
  echo "Missing .venv. Run: bash scripts/grace_setup_env.sh"
  exit 2
fi

source .venv/bin/activate
mkdir -p logs "${SUMMARY_DIR}"

python reporting/diagnostics/summarize_reference_all_models.py \
  --in-csv "${REFERENCE_CSV}" \
  --out-dir "${SUMMARY_DIR}"

python scripts/prepare_dashboard_cache.py
python scripts/check_dashboard_cache.py
python scripts/check_dashboard_data.py
