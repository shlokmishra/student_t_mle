#!/usr/bin/env bash
set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-GCCcore/14.3.0 Python/3.13.5}"
CASE_CONFIG="${CASE_CONFIG:-configs/final_production_v1_cases.yaml}"
OUT_DIR="${OUT_DIR:-results/final_production_v1}"
CASE_TSV="${CASE_TSV:-${OUT_DIR}/final_production_cases.tsv}"
MAX_PARALLEL="${MAX_PARALLEL:-12}"

export PYTHON_MODULE
export CASE_CONFIG
export CASE_TSV
export OUT_DIR

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

"${PYTHON_BIN}" scripts/make_grace_final_production_jobs.py \
  --config "${CASE_CONFIG}" \
  --out-dir "${OUT_DIR}" \
  --case-tsv "${CASE_TSV}"

NUM_CASES="$(awk 'NR > 1 {count += 1} END {print count + 0}' "${CASE_TSV}")"
if [ "${NUM_CASES}" -le 0 ]; then
  echo "No final production cases generated in ${CASE_TSV}" >&2
  exit 2
fi

sbatch --array=1-${NUM_CASES}%${MAX_PARALLEL} hpc/grace/final_production_v1_array.sbatch
