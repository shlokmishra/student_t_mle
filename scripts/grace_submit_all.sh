#!/usr/bin/env bash
set -euo pipefail

# Submit the Grace experiment suite from the repository root.
# First run:
#   bash scripts/grace_setup_env.sh
#
# Then submit:
#   bash scripts/grace_submit_all.sh
#
# Optional overrides:
#   PYTHON_MODULE="GCCcore/10.2.0 Python/3.8.6" bash scripts/grace_submit_all.sh
#
# By default this keeps total requested cores at 8 by running the two big jobs
# sequentially. If your allocation allows 16 cores, set:
#   CONCURRENT_BIG_JOBS=1 bash scripts/grace_submit_all.sh

mkdir -p logs

echo "Using up to ${MAX_PARALLEL:-8} parallel workers per job."
echo "Each worker pins BLAS/OpenMP thread counts to 1 to avoid oversubscription."
echo "Reference defaults: B=${B_VALUES:-100000}, bandwidths=${BANDWIDTHS:-scott,SJ_transform,t_abram}."
echo "Cost defaults: NUM_ITERATIONS=${NUM_ITERATIONS:-100000}, BURN_IN=${BURN_IN:-20000}."

ref_job="$(sbatch --parsable scripts/grace_reference_audit.sbatch)"
if [ "${CONCURRENT_BIG_JOBS:-0}" = "1" ]; then
  cost_job="$(sbatch --parsable scripts/grace_cost_audit.sbatch)"
  post_job="$(sbatch --parsable --dependency=afterok:${ref_job}:${cost_job} scripts/grace_postprocess_dashboard.sbatch)"
else
  cost_job="$(sbatch --parsable --dependency=afterok:${ref_job} scripts/grace_cost_audit.sbatch)"
  post_job="$(sbatch --parsable --dependency=afterok:${cost_job} scripts/grace_postprocess_dashboard.sbatch)"
fi

echo "Submitted reference audit job: ${ref_job}"
echo "Submitted cost audit job: ${cost_job}"
echo "Submitted postprocess job: ${post_job}"
echo
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/ref_all_${ref_job}.out"
echo "  tail -f logs/cost_all_${cost_job}.out"
echo "  tail -f logs/postproc_${post_job}.out"
