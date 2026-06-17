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
PYTHON_BIN="${PYTHON_BIN:-}"

if ! command -v module >/dev/null 2>&1; then
  # Grace commonly exposes Environment Modules via shell init rather than PATH.
  for module_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
    if [ -f "${module_init}" ]; then
      # shellcheck disable=SC1090
      source "${module_init}"
      break
    fi
  done
fi

if command -v module >/dev/null 2>&1 && [ -n "${PYTHON_MODULE}" ]; then
  module purge
  # PYTHON_MODULE may contain a prerequisite toolchain plus Python.
  # shellcheck disable=SC2086
  module load ${PYTHON_MODULE}
else
  echo "No usable module command with PYTHON_MODULE provided; using the current Python on PATH."
fi

if [ -z "${PYTHON_BIN}" ]; then
  for candidate in python python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Could not find ${PYTHON_BIN}."
  echo "Run: module spider Python"
  echo "Then rerun with something like:"
  echo "  PYTHON_MODULE=\"GCCcore/10.2.0 Python/3.8.6\" bash scripts/grace_setup_env.sh"
  exit 2
fi

python_version="$("${PYTHON_BIN}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
python_path="$(command -v "${PYTHON_BIN}")"
echo "Using ${PYTHON_BIN} at ${python_path} (version ${python_version})"

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        "Grace environment setup requires Python >= 3.10. "
        f"Detected {sys.version.split()[0]}. "
        "Load the requested Grace Python module or set PYTHON_BIN explicitly."
    )
PY

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p logs

echo "Grace environment is ready. Submit jobs with:"
echo "  bash scripts/grace_submit_all.sh"
