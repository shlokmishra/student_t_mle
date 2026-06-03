#!/usr/bin/env bash
set -euo pipefail

# Run this once on Grace from the repository root before submitting jobs.
# Override PYTHON_MODULE if Grace exposes a different Python module name.

PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11}"

if command -v module >/dev/null 2>&1; then
  module purge
  module load "${PYTHON_MODULE}"
else
  echo "Environment modules are unavailable in this shell; using current python."
fi

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p logs

echo "Grace environment is ready. Submit jobs with:"
echo "  bash scripts/grace_submit_all.sh"

