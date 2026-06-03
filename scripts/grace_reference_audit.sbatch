#!/usr/bin/env bash
#SBATCH --job-name=ref_all
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=logs/ref_all_%j.out

set -euo pipefail

# Full all-model reference audit for the dashboard.
# Defaults run all three KDE forms. Exact adaptive t_abram can be much slower
# than scott/SJ_transform at B=100000.

PYTHON_MODULE="${PYTHON_MODULE:-}"
B_VALUES="${B_VALUES:-100000}"
SEEDS="${SEEDS:-123,456,789}"
N_VALUES="${N_VALUES:-10,20,50}"
K_VALUES="${K_VALUES:-1,2,3}"
BANDWIDTHS="${BANDWIDTHS:-scott,SJ_transform,t_abram}"
GRID_SIZE="${GRID_SIZE:-2500}"
OUT_CSV="${OUT_CSV:-reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv}"
DENSITY_OUT_CSV="${DENSITY_OUT_CSV:-reporting/diagnostic_outputs/model_reference_audit/reference_all_models_density_grid.csv}"
MAX_PARALLEL="${MAX_PARALLEL:-${SLURM_CPUS_PER_TASK:-8}}"
TASK_DIR="${TASK_DIR:-reporting/diagnostic_outputs/model_reference_audit/reference_all_models_parts}"

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
rm -rf "${TASK_DIR}"
mkdir -p logs "$(dirname "${OUT_CSV}")" "$(dirname "${DENSITY_OUT_CSV}")" "${TASK_DIR}"

IFS=',' read -r -a n_array <<< "${N_VALUES}"
IFS=',' read -r -a k_array <<< "${K_VALUES}"
IFS=',' read -r -a seed_array <<< "${SEEDS}"

run_limited() {
  while [ "$(jobs -pr | wc -l | tr -d ' ')" -ge "${MAX_PARALLEL}" ]; do
    wait -n
  done
  "$@" &
}

task_index=0
for model in student_t logistic laplace; do
  for n in "${n_array[@]}"; do
    for seed in "${seed_array[@]}"; do
      if [ "${model}" = "student_t" ]; then
        for k in "${k_array[@]}"; do
          task_index=$((task_index + 1))
          part_csv="${TASK_DIR}/part_$(printf '%04d' "${task_index}")_${model}_k${k}_n${n}_seed${seed}.csv"
          part_density_csv="${TASK_DIR}/density_$(printf '%04d' "${task_index}")_${model}_k${k}_n${n}_seed${seed}.csv"
          echo "submit reference task ${task_index}: model=${model} k=${k} n=${n} seed=${seed}"
          run_limited python reporting/diagnostics/audit_reference_all_models.py \
            --models "${model}" \
            --k-values "${k}" \
            --n-values "${n}" \
            --B-values "${B_VALUES}" \
            --seeds "${seed}" \
            --bandwidths "${BANDWIDTHS}" \
            --grid-size "${GRID_SIZE}" \
            --out-csv "${part_csv}" \
            --density-out-csv "${part_density_csv}" \
            --overwrite
        done
      else
        task_index=$((task_index + 1))
        part_csv="${TASK_DIR}/part_$(printf '%04d' "${task_index}")_${model}_n${n}_seed${seed}.csv"
        part_density_csv="${TASK_DIR}/density_$(printf '%04d' "${task_index}")_${model}_n${n}_seed${seed}.csv"
        echo "submit reference task ${task_index}: model=${model} n=${n} seed=${seed}"
        run_limited python reporting/diagnostics/audit_reference_all_models.py \
          --models "${model}" \
          --n-values "${n}" \
          --B-values "${B_VALUES}" \
          --seeds "${seed}" \
          --bandwidths "${BANDWIDTHS}" \
          --grid-size "${GRID_SIZE}" \
          --out-csv "${part_csv}" \
          --density-out-csv "${part_density_csv}" \
          --overwrite
      fi
    done
  done
done

wait

python -c 'import sys; from pathlib import Path; import pandas as pd; out=Path(sys.argv[1]); parts=sorted(Path(sys.argv[2]).glob("part_*.csv")); out.parent.mkdir(parents=True, exist_ok=True); pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(out, index=False); print(f"merged {len(parts)} summary files into {out}")' "${OUT_CSV}" "${TASK_DIR}"
python -c 'import sys; from pathlib import Path; import pandas as pd; out=Path(sys.argv[1]); parts=sorted(Path(sys.argv[2]).glob("density_*.csv")); out.parent.mkdir(parents=True, exist_ok=True); pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(out, index=False); print(f"merged {len(parts)} density files into {out}")' "${DENSITY_OUT_CSV}" "${TASK_DIR}"
