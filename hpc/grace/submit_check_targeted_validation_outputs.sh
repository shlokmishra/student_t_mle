#!/usr/bin/env bash
set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-GCCcore/14.3.0 Python/3.13.5}"
TARGETED_VALIDATION_JOBID="${TARGETED_VALIDATION_JOBID:-18774750}"

export PYTHON_MODULE
sbatch --dependency=afterany:${TARGETED_VALIDATION_JOBID} hpc/grace/check_targeted_validation_outputs.sbatch
