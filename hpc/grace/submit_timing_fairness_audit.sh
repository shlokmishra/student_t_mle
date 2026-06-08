#!/usr/bin/env bash
set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-GCCcore/14.3.0 Python/3.13.5}"
CONFIG="${CONFIG:-configs/timing_fairness_cases.yaml}"
CASE_TSV="${CASE_TSV:-hpc/grace/timing_fairness_cases.tsv}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

python - "${CONFIG}" "${CASE_TSV}" <<'PY'
from pathlib import Path
import sys

from scripts.targeted_validation_config import load_case_config

config = load_case_config(Path(sys.argv[1]))
out = Path(sys.argv[2])
rows = []
defaults = config["defaults"]
for case in config["cases"]:
    for method in case["methods"]:
        for seed in case.get("seeds", defaults.get("seeds", [0])):
            for repeat in range(int(case.get("repeats", defaults.get("repeats", 1)))):
                rows.append((len(rows) + 1, case["case_id"], method, int(seed), repeat))
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as handle:
    handle.write("task_index\tcase_id\tmethod\tseed\trepeat\n")
    for row in rows:
        handle.write("\t".join(str(value) for value in row) + "\n")
print(len(rows))
PY

NUM_CASES="$(awk 'NR > 1 {count += 1} END {print count + 0}' "${CASE_TSV}")"

export PYTHON_MODULE
export CASE_TSV
sbatch --array=1-${NUM_CASES}%${MAX_PARALLEL} hpc/grace/timing_fairness_array.sbatch
