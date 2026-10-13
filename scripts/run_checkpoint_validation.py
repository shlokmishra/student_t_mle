"""Bound existing tests and the smoke example; save commands, logs and exit codes."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "validation.json").exists():
        parser.error("use a fresh output directory; existing results are preserved")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    env = os.environ.copy()
    limits = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  NUMBA_NUM_THREADS="1", JAX_PLATFORMS="cpu",
                  MPLCONFIGDIR="/tmp/student_t_mle-baseline-matplotlib")
    env.update(limits)
    cases = [
        ("pytest", 900, ["-m", "pytest", "tests", "-q", "--durations=15", f"--junitxml={out / 'pytest.xml'}"]),
        ("example", 120, ["examples/score_conditioned_student.py"]),
        ("selected-example", 120, ["examples/selected_mle_student.py"]),
        ("reference-help", 60, ["scripts/validate_selected_mle_reference.py", "--help"]),
        ("selected-gibbs-help", 60, ["scripts/validate_student_selected_mle_gibbs.py", "--help"]),
        ("components-help", 60, ["reporting/diagnostics/verify_student_n3_components.py", "--help"]),
    ]
    results = []
    for name, ceiling, command in cases:
        command = [sys.executable, *command]
        start = time.monotonic()
        with (out / f"{name}.txt").open("w") as stream:
            try:
                run = subprocess.run(command, cwd=root, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, timeout=ceiling)
                status = run.returncode
            except subprocess.TimeoutExpired:
                status = "timeout"
        results.append(dict(name=name, command=command, exit_code=status,
                            wall_seconds=time.monotonic() - start, ceiling_seconds=ceiling))
        print(json.dumps(results[-1]), flush=True)
    record = dict(python=sys.version, platform=platform.platform(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), environment=limits,
                  results=results)
    (out / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
    raise SystemExit(0 if all(r["exit_code"] == 0 for r in results) else 1)


if __name__ == "__main__":
    main()
