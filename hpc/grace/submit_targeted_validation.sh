#!/usr/bin/env bash
set -euo pipefail
PYTHON_MODULE="${PYTHON_MODULE:-GCCcore/14.3.0 Python/3.13.5}"
NUM_CASES=114
MAX_PARALLEL="${MAX_PARALLEL:-8}"
sbatch --array=1-${NUM_CASES}%${MAX_PARALLEL} hpc/grace/targeted_validation_array.sbatch
