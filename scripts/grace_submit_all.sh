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
#   BANDWIDTHS=scott,SJ_transform,t_abram NUM_ITERATIONS=100000 bash scripts/grace_submit_all.sh

mkdir -p logs

ref_job="$(sbatch --parsable scripts/grace_reference_audit.sbatch)"
cost_job="$(sbatch --parsable scripts/grace_cost_audit.sbatch)"
post_job="$(sbatch --parsable --dependency=afterok:${ref_job}:${cost_job} scripts/grace_postprocess_dashboard.sbatch)"

echo "Submitted reference audit job: ${ref_job}"
echo "Submitted cost audit job: ${cost_job}"
echo "Submitted postprocess job: ${post_job}"
echo
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/ref_all_${ref_job}.out"
echo "  tail -f logs/cost_all_${cost_job}.out"
echo "  tail -f logs/postproc_${post_job}.out"

