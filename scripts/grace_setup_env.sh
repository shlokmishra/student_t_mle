#!/usr/bin/env bash
set -euo pipefail

# Run this once on Grace from the repository root before submitting jobs.
# Grace module names are version/toolchain specific. Find the exact module with:
#   module spider Python
# or:
#   mla Python
#
# Then, if needed, run for example:
#   PYTHON_MODULE="GCCcore/10.2.0 Python/3.8.6" bash scripts/grace_setup_env.sh

PYTHON_MODULE="${PYTHON_MODULE:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if command -v module >/dev/null 2>&1 && [ -n "${PYTHON_MODULE}" ]; then
  module purge
  # PYTHON_MODULE may contain a prerequisite toolchain plus Python.
  # shellcheck disable=SC2086
  module load ${PYTHON_MODULE}
else
  echo "No PYTHON_MODULE provided; using the current ${PYTHON_BIN} on PATH."
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Could not find ${PYTHON_BIN}."
  echo "Run: module spider Python"
  echo "Then rerun with something like:"
  echo "  PYTHON_MODULE=\"GCCcore/10.2.0 Python/3.8.6\" bash scripts/grace_setup_env.sh"
  exit 2
fi

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p logs

echo "Grace environment is ready. Submit jobs with:"
echo "  bash scripts/grace_submit_all.sh"
