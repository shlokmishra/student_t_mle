#!/usr/bin/env bash
#SBATCH --job-name=lap_ref_patch
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/lap_ref_patch_%j.out

set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-}"
PATCH_DIR="${PATCH_DIR:-reporting/diagnostic_outputs/model_reference_audit/laplace_interval_patch}"
B_VALUES="${B_VALUES:-100000}"
SEEDS="${SEEDS:-123,456,789}"
N_VALUES="${N_VALUES:-10,20,50}"
GRID_SIZE="${GRID_SIZE:-2500}"
BANDWIDTHS="${BANDWIDTHS:-scott,SJ_transform,t_abram}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if command -v module >/dev/null 2>&1 && [ -n "${PYTHON_MODULE}" ]; then
  module purge
  # shellcheck disable=SC2086
  module load ${PYTHON_MODULE}
fi

if [ ! -f .venv/bin/activate ]; then
  echo "Missing .venv. Run: bash scripts/grace_setup_env.sh"
  exit 2
fi

source .venv/bin/activate
rm -rf "${PATCH_DIR}"
mkdir -p logs "${PATCH_DIR}"

IFS=',' read -r -a n_array <<< "${N_VALUES}"
IFS=',' read -r -a seed_array <<< "${SEEDS}"

for n in "${n_array[@]}"; do
  for seed in "${seed_array[@]}"; do
    echo "patch Laplace interval reference: n=${n} seed=${seed}"
    python reporting/diagnostics/audit_reference_all_models.py \
      --models laplace \
      --n-values "${n}" \
      --B-values "${B_VALUES}" \
      --seeds "${seed}" \
      --bandwidths "${BANDWIDTHS}" \
      --grid-size "${GRID_SIZE}" \
      --out-csv "${PATCH_DIR}/part_laplace_n${n}_seed${seed}.csv" \
      --density-out-csv "${PATCH_DIR}/density_laplace_n${n}_seed${seed}.csv" \
      --overwrite
  done
done

python scripts/patch_laplace_reference_outputs.py \
  --patch-dir "${PATCH_DIR}" \
  --summary-csv reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv \
  --density-csv reporting/diagnostic_outputs/model_reference_audit/reference_all_models_density_grid.csv
