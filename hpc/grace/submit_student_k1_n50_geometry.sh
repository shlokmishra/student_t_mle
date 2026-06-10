#!/usr/bin/env bash
set -euo pipefail
PYTHON_MODULE="${PYTHON_MODULE:-GCCcore/14.3.0 Python/3.13.5}"
CASE_CONFIG="${CASE_CONFIG:-configs/student_k1_n50_geometry_cases.yaml}"
CASE_TSV="${CASE_TSV:-hpc/grace/student_k1_n50_geometry_cases.tsv}"
OUT_DIR="${OUT_DIR:-results/student_k1_n50_geometry_runs}"
SAVE_FULL_LATENT_DIAGNOSTICS="${SAVE_FULL_LATENT_DIAGNOSTICS:-}"
if [ -z "${SAVE_FULL_LATENT_DIAGNOSTICS}" ]; then
  SAVE_FULL_LATENT_DIAGNOSTICS="--save-full-latent-diagnostics"
fi
FULL_LATENT_DIAGNOSTIC_MAX_ROWS="${FULL_LATENT_DIAGNOSTIC_MAX_ROWS:-0}"
NUM_CASES=15
MAX_PARALLEL="${MAX_PARALLEL:-8}"
sbatch --export=ALL,CASE_CONFIG="${CASE_CONFIG}",CASE_TSV="${CASE_TSV}",OUT_DIR="${OUT_DIR}",SAVE_FULL_LATENT_DIAGNOSTICS="${SAVE_FULL_LATENT_DIAGNOSTICS}",FULL_LATENT_DIAGNOSTIC_MAX_ROWS="${FULL_LATENT_DIAGNOSTIC_MAX_ROWS}" --array=1-${NUM_CASES}%${MAX_PARALLEL} hpc/grace/targeted_validation_array.sbatch
