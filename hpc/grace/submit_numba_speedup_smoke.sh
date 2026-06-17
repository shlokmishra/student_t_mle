#!/usr/bin/env bash
set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-GCCcore/14.3.0 Python/3.13.5}"
CONFIG="${CONFIG:-configs/numba_speedup_validation_cases.yaml}"
CASE_TSV="${CASE_TSV:-hpc/grace/numba_speedup_validation_smoke_cases.tsv}"
OUT_DIR="${OUT_DIR:-results/numba_speedup_validation_smoke}"
MANIFEST="${MANIFEST:-${OUT_DIR}/run_manifest.json}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

"${PYTHON_BIN}" scripts/make_grace_numba_speedup_jobs.py \
  --config "${CONFIG}" \
  --case-tsv "${CASE_TSV}" \
  --manifest "${MANIFEST}" \
  --mode smoke

NUM_CASES="$(awk 'NR > 1 {count += 1} END {print count + 0}' "${CASE_TSV}")"
mkdir -p logs/numba_speedup_validation "${OUT_DIR}/task_results"

export PYTHON_MODULE
export CASE_TSV
export OUT_DIR
sbatch --array=1-${NUM_CASES}%${MAX_PARALLEL} hpc/grace/numba_speedup_validation_array.sbatch
