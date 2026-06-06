#!/usr/bin/env bash
#SBATCH --job-name=cost_laplace
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/cost_laplace_%j.out

set -euo pipefail

# Rerun only Laplace known-scale sampler/cost rows and replace the Laplace rows
# in the existing all-model cost-audit CSVs. Student-t and Logistic rows are
# kept exactly as they are.

PYTHON_MODULE="${PYTHON_MODULE:-}"
NUM_ITERATIONS="${NUM_ITERATIONS:-100000}"
BURN_IN="${BURN_IN:-20000}"
SEEDS="${SEEDS:-123,456,789}"
LAPLACE_N_VALUES="${LAPLACE_N_VALUES:-11,21,51}"
METHODS="${METHODS:-gibbs rattle}"
OUT_DIR="${OUT_DIR:-results/cost_audit}"
RUN_STATUS="${RUN_STATUS:-full}"
LATENT_DIAGNOSTIC_THIN="${LATENT_DIAGNOSTIC_THIN:-25}"
LATENT_DIAGNOSTIC_MAX_ROWS="${LATENT_DIAGNOSTIC_MAX_ROWS:-5000}"
MAX_PARALLEL="${MAX_PARALLEL:-${SLURM_CPUS_PER_TASK:-8}}"
TASK_DIR="${TASK_DIR:-results/cost_audit_laplace_rerun_parts}"
REFERENCE_CSV="${REFERENCE_CSV:-reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv}"
ANALYSIS_OUT_DIR="${ANALYSIS_OUT_DIR:-results/analysis_report}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if command -v module >/dev/null 2>&1 && [ -n "${PYTHON_MODULE}" ]; then
  module purge
  # PYTHON_MODULE may contain a prerequisite toolchain plus Python.
  # shellcheck disable=SC2086
  module load ${PYTHON_MODULE}
fi

if [ ! -f .venv/bin/activate ]; then
  echo "Missing .venv. Run: bash scripts/grace_setup_env.sh"
  exit 2
fi

source .venv/bin/activate
mkdir -p logs "${OUT_DIR}" "${ANALYSIS_OUT_DIR}"
rm -rf "${TASK_DIR}"
mkdir -p "${TASK_DIR}"

IFS=',' read -r -a laplace_n_array <<< "${LAPLACE_N_VALUES}"
IFS=',' read -r -a seed_array <<< "${SEEDS}"
# METHODS is intentionally space-separated because run_cost_audit takes methods
# as argparse nargs.
# shellcheck disable=SC2206
method_array=(${METHODS})

run_limited() {
  while [ "$(jobs -pr | wc -l | tr -d ' ')" -ge "${MAX_PARALLEL}" ]; do
    wait -n
  done
  "$@" &
}

task_index=0
for method in "${method_array[@]}"; do
  for n in "${laplace_n_array[@]}"; do
    for seed in "${seed_array[@]}"; do
      task_index=$((task_index + 1))
      part_dir="${TASK_DIR}/part_$(printf '%04d' "${task_index}")_laplace_${method}_n${n}_seed${seed}"
      echo "submit Laplace cost task ${task_index}: method=${method} n=${n} seed=${seed}"
      run_limited python scripts/run_cost_audit.py \
        --models laplace \
        --methods "${method}" \
        --n-values "${n}" \
        --mu-star 0 \
        --num-iterations "${NUM_ITERATIONS}" \
        --burn-in "${BURN_IN}" \
        --seed "${seed}" \
        --out "${part_dir}" \
        --run-status "${RUN_STATUS}" \
        --save-latent-diagnostics \
        --latent-diagnostic-thin "${LATENT_DIAGNOSTIC_THIN}" \
        --latent-diagnostic-max-rows "${LATENT_DIAGNOSTIC_MAX_ROWS}"
    done
  done
done

wait

replace_cost_file() {
  local filename="$1"
  local main_csv="${OUT_DIR}/${filename}"
  local pattern="${TASK_DIR}/*/${filename}"
  if ! compgen -G "${pattern}" > /dev/null; then
    echo "No replacement parts found for ${filename}; skipping."
    return
  fi
  python scripts/replace_laplace_reference_rows.py \
    --main-csv "${main_csv}" \
    --new-parts-glob "${pattern}" \
    --out-csv "${main_csv}" \
    --skip-if-empty \
    --allow-missing-main
}

replace_cost_file "chain_samples.csv"
replace_cost_file "cost_ledger.csv"
replace_cost_file "posterior_summaries.csv"
replace_cost_file "diagnostic_summary.csv"
replace_cost_file "latent_x_diagnostics.csv"

if [ "${RUN_ANALYSIS}" = "1" ]; then
  python reporting/diagnostics/analyze_full_comparison.py \
    --reference-csv "${REFERENCE_CSV}" \
    --cost-dir "${OUT_DIR}" \
    --out-dir "${ANALYSIS_OUT_DIR}"
fi

echo "Replaced Laplace rows in ${OUT_DIR}"
echo "Refreshed analysis report: ${RUN_ANALYSIS}"
