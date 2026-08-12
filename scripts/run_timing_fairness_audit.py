"""Run one timing-fairness audit case.

This reuses the targeted validation runner so sampler transition logic stays
untouched. The Grace submit wrapper is prepared separately and is not submitted
by the completion-check workflow.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.targeted_validation_config import load_case_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/timing_fairness_cases.yaml"))
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--method", required=True, choices=["gibbs", "rattle"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/timing_fairness_audit"))
    parser.add_argument("--gibbs-backend", choices=["jax_loop", "jax_scan", "jax_scan_block_z", "numba"], default="jax_loop")
    return parser.parse_args()


def expanded_timing_cases(config_path: Path) -> list[dict]:
    config = load_case_config(config_path)
    defaults = config["defaults"]
    rows = []
    for case in config["cases"]:
        for method in case["methods"]:
            for seed in case.get("seeds", defaults.get("seeds", [0])):
                for repeat in range(int(case.get("repeats", defaults.get("repeats", 1)))):
                    rows.append({**defaults, **case, "method": method, "seed": int(seed), "repeat": repeat})
    return rows


def main() -> None:
    args = parse_args()
    matches = [
        row
        for row in expanded_timing_cases(args.config)
        if row["case_id"] == args.case_id and row["method"] == args.method and row["seed"] == args.seed and row["repeat"] == args.repeat
    ]
    if not matches:
        raise SystemExit(f"No timing fairness case found for {args}")
    case = matches[0]
    synthetic_case_id = f"{case['case_id']}_{case['method']}_seed{case['seed']}_repeat{case['repeat']}"
    args.out.mkdir(parents=True, exist_ok=True)
    generated_config = args.out / f"{synthetic_case_id}.yaml"
    base_case_id = synthetic_case_id
    generated_config.write_text(
        "\n".join(
            [
                "defaults:",
                f"  num_iterations: {int(case['num_iterations'])}",
                f"  burn_in: {int(case['burn_in'])}",
                "  diagnostic_thin: 20",
                f"  seeds: [{int(case['seed'])}]",
                "  initializations: [central]",
                "",
                "cases:",
                f"  - case_id: {base_case_id}",
                f"    model: {case['model']}",
                f"    k: {'' if case.get('k') is None else case.get('k')}",
                f"    n: {int(case['n'])}",
                f"    method: {case['method']}",
                "    diagnostic_only: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    metadata_path = args.out / f"{synthetic_case_id}.timing_metadata.json"
    metadata_path.write_text(json.dumps({"case": case, "started_at_unix": time.time()}, indent=2, sort_keys=True), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "scripts/run_targeted_validation.py",
            "--case-config",
            str(generated_config),
            "--case-id",
            f"{base_case_id}_seed{case['seed']}_init_central",
            "--out",
            str(args.out),
            "--gibbs-backend",
            str(args.gibbs_backend),
            "--diagnostic-thin",
            "20",
            "--save-transition-diagnostics",
            "--save-latent-diagnostics",
            "--save-rattle-energy-diagnostics",
            "--save-branch-diagnostics",
            "--save-initialization-diagnostics",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
