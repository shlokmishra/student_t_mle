#!/usr/bin/env bash
#SBATCH --job-name=ref_laplace
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=logs/ref_laplace_%j.out

set -euo pipefail

# Rerun only Laplace known-scale reference rows and replace the Laplace rows in
# the existing all-model reference CSVs. Student-t and Logistic rows are kept
# exactly as they are.

PYTHON_MODULE="${PYTHON_MODULE:-}"
B_VALUES="${B_VALUES:-100000}"
SEEDS="${SEEDS:-123,456,789}"
LAPLACE_N_VALUES="${LAPLACE_N_VALUES:-11,21,51}"
BANDWIDTHS="${BANDWIDTHS:-scott,SJ_transform,t_abram}"
GRID_SIZE="${GRID_SIZE:-2500}"
OUT_CSV="${OUT_CSV:-reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv}"
DENSITY_OUT_CSV="${DENSITY_OUT_CSV:-reporting/diagnostic_outputs/model_reference_audit/reference_all_models_density_grid.csv}"
SUMMARY_DIR="${SUMMARY_DIR:-reporting/diagnostic_outputs/model_reference_audit/full_summary}"
MAX_PARALLEL="${MAX_PARALLEL:-${SLURM_CPUS_PER_TASK:-8}}"
TASK_DIR="${TASK_DIR:-reporting/diagnostic_outputs/model_reference_audit/laplace_rerun_parts}"

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
mkdir -p logs "$(dirname "${OUT_CSV}")" "$(dirname "${DENSITY_OUT_CSV}")" "${SUMMARY_DIR}"
rm -rf "${TASK_DIR}"
mkdir -p "${TASK_DIR}"

IFS=',' read -r -a laplace_n_array <<< "${LAPLACE_N_VALUES}"
IFS=',' read -r -a seed_array <<< "${SEEDS}"

run_limited() {
  while [ "$(jobs -pr | wc -l | tr -d ' ')" -ge "${MAX_PARALLEL}" ]; do
    wait -n
  done
  "$@" &
}

task_index=0
for n in "${laplace_n_array[@]}"; do
  for seed in "${seed_array[@]}"; do
    task_index=$((task_index + 1))
    part_csv="${TASK_DIR}/part_$(printf '%04d' "${task_index}")_laplace_n${n}_seed${seed}.csv"
    part_density_csv="${TASK_DIR}/density_$(printf '%04d' "${task_index}")_laplace_n${n}_seed${seed}.csv"
    echo "submit Laplace reference task ${task_index}: n=${n} seed=${seed}"
    run_limited python reporting/diagnostics/audit_reference_all_models.py \
      --models laplace \
      --n-values "${n}" \
      --B-values "${B_VALUES}" \
      --seeds "${seed}" \
      --bandwidths "${BANDWIDTHS}" \
      --grid-size "${GRID_SIZE}" \
      --out-csv "${part_csv}" \
      --density-out-csv "${part_density_csv}" \
      --overwrite
  done
done

wait

python scripts/replace_laplace_reference_rows.py \
  --main-csv "${OUT_CSV}" \
  --new-parts-glob "${TASK_DIR}/part_*.csv" \
  --out-csv "${OUT_CSV}"

python scripts/replace_laplace_reference_rows.py \
  --main-csv "${DENSITY_OUT_CSV}" \
  --new-parts-glob "${TASK_DIR}/density_*.csv" \
  --out-csv "${DENSITY_OUT_CSV}"

python reporting/diagnostics/summarize_reference_all_models.py \
  --in-csv "${OUT_CSV}" \
  --out-dir "${SUMMARY_DIR}"

echo "Replaced Laplace rows in:"
echo "  ${OUT_CSV}"
echo "  ${DENSITY_OUT_CSV}"
echo "Refreshed summaries in:"
echo "  ${SUMMARY_DIR}"
