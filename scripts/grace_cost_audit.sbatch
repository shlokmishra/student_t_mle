#!/usr/bin/env bash
#SBATCH --job-name=cost_all
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/cost_all_%j.out

set -euo pipefail

# Long Gibbs/RATTLE cost audit for Student-t, logistic, and Laplace.
# Laplace RATTLE is intentionally not applicable and is reported as such.

PYTHON_MODULE="${PYTHON_MODULE:-}"
NUM_ITERATIONS="${NUM_ITERATIONS:-100000}"
BURN_IN="${BURN_IN:-20000}"
SEED="${SEED:-0}"
N_VALUES="${N_VALUES:-10,20,50}"
K_VALUES="${K_VALUES:-1,2,3}"
OUT_DIR="${OUT_DIR:-results/cost_audit}"
RUN_STATUS="${RUN_STATUS:-full}"
LATENT_DIAGNOSTIC_THIN="${LATENT_DIAGNOSTIC_THIN:-25}"
LATENT_DIAGNOSTIC_MAX_ROWS="${LATENT_DIAGNOSTIC_MAX_ROWS:-5000}"
MAX_PARALLEL="${MAX_PARALLEL:-${SLURM_CPUS_PER_TASK:-8}}"
TASK_DIR="${TASK_DIR:-results/cost_audit_parts}"

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
mkdir -p logs "${OUT_DIR}" "${TASK_DIR}"

IFS=',' read -r -a n_array <<< "${N_VALUES}"
IFS=',' read -r -a k_array <<< "${K_VALUES}"

run_limited() {
  while [ "$(jobs -pr | wc -l | tr -d ' ')" -ge "${MAX_PARALLEL}" ]; do
    wait -n
  done
  "$@" &
}

task_index=0
for model in student_t logistic laplace; do
  for method in gibbs rattle; do
    for n in "${n_array[@]}"; do
      if [ "${model}" = "student_t" ]; then
        for k in "${k_array[@]}"; do
          task_index=$((task_index + 1))
          part_dir="${TASK_DIR}/part_$(printf '%04d' "${task_index}")_${model}_${method}_k${k}_n${n}"
          echo "submit cost task ${task_index}: model=${model} method=${method} k=${k} n=${n}"
          run_limited python scripts/run_cost_audit.py \
            --models "${model}" \
            --methods "${method}" \
            --k-values "${k}" \
            --n-values "${n}" \
            --mu-star 0 \
            --num-iterations "${NUM_ITERATIONS}" \
            --burn-in "${BURN_IN}" \
            --seed "${SEED}" \
            --out "${part_dir}" \
            --run-status "${RUN_STATUS}" \
            --save-latent-diagnostics \
            --latent-diagnostic-thin "${LATENT_DIAGNOSTIC_THIN}" \
            --latent-diagnostic-max-rows "${LATENT_DIAGNOSTIC_MAX_ROWS}"
        done
      else
        task_index=$((task_index + 1))
        part_dir="${TASK_DIR}/part_$(printf '%04d' "${task_index}")_${model}_${method}_n${n}"
        echo "submit cost task ${task_index}: model=${model} method=${method} n=${n}"
        run_limited python scripts/run_cost_audit.py \
          --models "${model}" \
          --methods "${method}" \
          --n-values "${n}" \
          --mu-star 0 \
          --num-iterations "${NUM_ITERATIONS}" \
          --burn-in "${BURN_IN}" \
          --seed "${SEED}" \
          --out "${part_dir}" \
          --run-status "${RUN_STATUS}" \
          --save-latent-diagnostics \
          --latent-diagnostic-thin "${LATENT_DIAGNOSTIC_THIN}" \
          --latent-diagnostic-max-rows "${LATENT_DIAGNOSTIC_MAX_ROWS}"
      fi
    done
  done
done

wait

python -c 'import sys; from pathlib import Path; import pandas as pd; out=Path(sys.argv[1]); root=Path(sys.argv[2]); names=["chain_samples.csv","cost_ledger.csv","posterior_summaries.csv","diagnostic_summary.csv","latent_x_diagnostics.csv"]; out.mkdir(parents=True, exist_ok=True); [pd.concat([pd.read_csv(p) for p in sorted(root.glob("*/"+name))], ignore_index=True).to_csv(out/name, index=False) for name in names if list(root.glob("*/"+name))]; print(f"merged cost audit parts into {out}")' "${OUT_DIR}" "${TASK_DIR}"
