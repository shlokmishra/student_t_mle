"""Generate Grace array inputs for targeted validation cases."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.targeted_validation_config import expanded_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-config", type=Path, default=Path("configs/targeted_validation_cases.yaml"))
    parser.add_argument("--out-tsv", type=Path, default=Path("hpc/grace/targeted_validation_cases.tsv"))
    parser.add_argument("--submit-script", type=Path, default=Path("hpc/grace/submit_targeted_validation.sh"))
    parser.add_argument("--array-script", type=Path, default=Path("hpc/grace/targeted_validation_array.sbatch"))
    parser.add_argument("--max-parallel", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = expanded_cases(args.case_config)
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", encoding="utf-8") as handle:
        handle.write("task_index\tcase_id\n")
        for index, case in enumerate(cases, start=1):
            handle.write(f"{index}\t{case['case_id']}\n")
    args.submit_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "PYTHON_MODULE=\"${PYTHON_MODULE:-GCCcore/14.3.0 Python/3.13.5}\"\n"
        f"NUM_CASES={len(cases)}\n"
        f"MAX_PARALLEL=\"${{MAX_PARALLEL:-{args.max_parallel}}}\"\n"
        "sbatch --array=1-${NUM_CASES}%${MAX_PARALLEL} hpc/grace/targeted_validation_array.sbatch\n",
        encoding="utf-8",
    )
    args.submit_script.chmod(0o755)
    print(f"wrote {len(cases)} cases to {args.out_tsv}")
    print(f"submit with: PYTHON_MODULE=\"GCCcore/14.3.0 Python/3.13.5\" bash {args.submit_script}")


if __name__ == "__main__":
    main()
