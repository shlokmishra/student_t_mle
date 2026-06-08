"""Run the staged model-comparison analysis pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "analysis_pipeline"
LOG_PATH = OUT_DIR / "pipeline_log.txt"
BLOCKERS_PATH = OUT_DIR / "blockers.md"


@dataclass
class CommandResult:
    label: str
    command: list[str]
    returncode: int
    elapsed_sec: float
    status: str
    stdout_tail: str
    stderr_tail: str
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "medium", "full", "all"], default="all")
    parser.add_argument("--skip-full-if-medium-sec", type=float, default=900.0)
    return parser.parse_args()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def append_log(text: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def add_blocker(text: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = "# Analysis Pipeline Blockers\n\n" if not BLOCKERS_PATH.exists() else ""
    with BLOCKERS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(prefix + f"- {text}\n")


def run_command(label: str, command: list[str], timeout_sec: float | None = None) -> CommandResult:
    append_log(f"\n## {label}\n$ {' '.join(command)}")
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        elapsed = time.perf_counter() - start
        status = "passed" if completed.returncode == 0 else "failed"
        result = CommandResult(
            label=label,
            command=command,
            returncode=int(completed.returncode),
            elapsed_sec=elapsed,
            status=status,
            stdout_tail=completed.stdout[-4000:],
            stderr_tail=completed.stderr[-4000:],
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        result = CommandResult(
            label=label,
            command=command,
            returncode=124,
            elapsed_sec=elapsed,
            status="timeout",
            stdout_tail=stdout[-4000:],
            stderr_tail=stderr[-4000:],
            error=f"timed out after {timeout_sec} seconds",
        )
    append_log(f"status={result.status} returncode={result.returncode} elapsed_sec={result.elapsed_sec:.2f}")
    if result.stdout_tail:
        append_log("stdout_tail:\n" + result.stdout_tail)
    if result.stderr_tail:
        append_log("stderr_tail:\n" + result.stderr_tail)
    if result.status != "passed":
        add_blocker(f"{label} {result.status}: {result.error or result.stderr_tail.strip() or result.stdout_tail.strip()}")
    return result


def result_dict(result: CommandResult) -> dict:
    return {
        "label": result.label,
        "command": result.command,
        "returncode": result.returncode,
        "elapsed_sec": result.elapsed_sec,
        "status": result.status,
        "stdout_tail": result.stdout_tail,
        "stderr_tail": result.stderr_tail,
        "error": result.error,
    }


def preflight() -> list[CommandResult]:
    required = [
        ROOT / "dashboard" / "app.py",
        ROOT / "dashboard" / "pages" / "1_Posterior_Comparison.py",
        ROOT / "dashboard" / "pages" / "2_Cost_Audit.py",
        ROOT / "dashboard" / "pages" / "3_Model_Validity_Audit.py",
        ROOT / "dashboard" / "pages" / "4_Analysis_Report.py",
        ROOT / "dashboard" / "pages" / "5_KDE_Correctness.py",
        ROOT / "dashboard" / "pages" / "6_Sampler_Correctness.py",
        ROOT / "dashboard" / "pages" / "7_Efficiency.py",
        ROOT / "dashboard" / "pages" / "8_Geometry.py",
        ROOT / "models" / "model_registry.py",
        ROOT / "reporting" / "diagnostics" / "audit_reference_all_models.py",
        ROOT / "scripts" / "run_cost_audit.py",
    ]
    for path in required:
        if not path.exists():
            add_blocker(f"missing required file: {path.relative_to(ROOT)}")
    tests = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_model_registry_and_location_conventions.py",
        "tests/test_student_rattle_diagnostics.py",
        "tests/test_kde_reference_moments.py",
        "tests/test_logistic_rattle_diagnostics.py",
        "tests/test_laplace_reference_targets.py",
    ]
    return [
        run_command("preflight tests", tests, timeout_sec=300),
        run_command("preflight dashboard check", [sys.executable, "scripts/check_dashboard_data.py"], timeout_sec=120),
    ]


def smoke_reference() -> list[CommandResult]:
    out_csv = "reporting/diagnostic_outputs/model_reference_audit/reference_all_models_smoke.csv"
    summary_dir = "reporting/diagnostic_outputs/model_reference_audit/smoke_summary"
    return [
        run_command(
            "smoke reference audit",
            [
                sys.executable,
                "reporting/diagnostics/audit_reference_all_models.py",
                "--models",
                "student_t",
                "logistic",
                "laplace",
                "--k-values",
                "1,2,3",
                "--n-values",
                "10,20,50",
                "--laplace-n-values",
                "11",
                "--B-values",
                "5000",
                "--seeds",
                "123",
                "--bandwidths",
                "scott,SJ_transform",
                "--out-csv",
                out_csv,
                "--overwrite",
            ],
            timeout_sec=900,
        ),
        run_command(
            "smoke reference summary",
            [sys.executable, "reporting/diagnostics/summarize_reference_all_models.py", "--in-csv", out_csv, "--out-dir", summary_dir],
            timeout_sec=120,
        ),
    ]


def smoke_sampler() -> list[CommandResult]:
    return [
        run_command(
            "smoke sampler audit",
            [
                sys.executable,
                "scripts/run_cost_audit.py",
                "--models",
                "student_t",
                "logistic",
                "laplace",
                "--methods",
                "gibbs",
                "rattle",
                "--k-values",
                "1,2,3",
                "--n-values",
                "10",
                "--laplace-n-values",
                "11",
                "--num-iterations",
                "1000",
                "--burn-in",
                "200",
                "--seed",
                "0",
                "--run-status",
                "smoke",
                "--out",
                "results/cost_audit_smoke/",
            ],
            timeout_sec=900,
        )
    ]


def tuning() -> list[CommandResult]:
    return [
        run_command(
            "rattle tuning",
            [
                sys.executable,
                "scripts/tune_rattle_grid.py",
                "--models",
                "student_t",
                "logistic",
                "--k-values",
                "1,2,3",
                "--n-values",
                "10",
                "--step-sizes",
                "0.005,0.01,0.02,0.05",
                "--leapfrog-steps",
                "5,10,20",
                "--num-iterations",
                "3000",
                "--burn-in",
                "500",
                "--seed",
                "0",
                "--out",
                "results/rattle_tuning/",
            ],
            timeout_sec=1800,
        )
    ]


def medium_sampler() -> list[CommandResult]:
    settings_path = ROOT / "results" / "rattle_tuning" / "recommended_rattle_settings.json"
    cmd = [
        sys.executable,
        "scripts/run_cost_audit.py",
        "--models",
        "student_t",
        "logistic",
        "laplace",
        "--methods",
        "gibbs",
        "rattle",
        "--k-values",
        "1,2,3",
        "--n-values",
        "10",
        "--num-iterations",
        "10000",
        "--burn-in",
        "2000",
        "--seed",
        "0",
        "--run-status",
        "medium",
        "--out",
        "results/cost_audit_medium/",
    ]
    if settings_path.exists():
        cmd.extend(["--rattle-settings-json", str(settings_path)])
    return [run_command("medium sampler audit", cmd, timeout_sec=1800)]


def full_reference() -> list[CommandResult]:
    out_csv = "reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv"
    summary_dir = "reporting/diagnostic_outputs/model_reference_audit/full_summary"
    return [
        run_command(
            "full reference audit",
            [
                sys.executable,
                "reporting/diagnostics/audit_reference_all_models.py",
                "--models",
                "student_t",
                "logistic",
                "laplace",
                "--k-values",
                "1,2,3",
                "--n-values",
                "10,20,50",
                "--laplace-n-values",
                "11,21,51",
                "--B-values",
                "100000",
                "--seeds",
                "123,456,789",
                "--bandwidths",
                "scott,SJ_transform",
                "--out-csv",
                out_csv,
                "--overwrite",
            ],
            timeout_sec=3600,
        ),
        run_command(
            "full reference summary",
            [sys.executable, "reporting/diagnostics/summarize_reference_all_models.py", "--in-csv", out_csv, "--out-dir", summary_dir],
            timeout_sec=180,
        ),
    ]


def full_sampler() -> list[CommandResult]:
    settings_path = ROOT / "results" / "rattle_tuning" / "recommended_rattle_settings.json"
    cmd = [
        sys.executable,
        "scripts/run_cost_audit.py",
        "--models",
        "student_t",
        "logistic",
        "laplace",
        "--methods",
        "gibbs",
        "rattle",
        "--k-values",
        "1,2,3",
        "--n-values",
        "10,20,50",
        "--laplace-n-values",
        "11,21,51",
        "--num-iterations",
        "10000",
        "--burn-in",
        "2000",
        "--seed",
        "0",
        "--run-status",
        "full",
        "--out",
        "results/cost_audit/",
    ]
    if settings_path.exists():
        cmd.extend(["--rattle-settings-json", str(settings_path)])
    return [run_command("full sampler audit", cmd, timeout_sec=3600)]


def persist_stage(name: str, results: list[CommandResult]) -> dict:
    status = "passed" if all(result.status == "passed" for result in results) else "partial"
    payload = {"stage": name, "status": status, "results": [result_dict(result) for result in results]}
    write_json(OUT_DIR / f"{name}_status.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    BLOCKERS_PATH.write_text("# Analysis Pipeline Blockers\n\n", encoding="utf-8")

    pipeline: dict[str, dict] = {}
    preflight_results = preflight()
    pipeline["preflight"] = {"status": "passed" if all(r.status == "passed" for r in preflight_results) else "partial", "results": [result_dict(r) for r in preflight_results]}

    if args.stage in {"smoke", "all"}:
        smoke_results = smoke_reference() + smoke_sampler()
        pipeline["smoke"] = persist_stage("smoke", smoke_results)

    if args.stage in {"medium", "all"}:
        medium_results = tuning() + medium_sampler()
        pipeline["medium"] = persist_stage("medium", medium_results)

    if args.stage in {"full", "all"}:
        medium_elapsed = 0.0
        if "medium" in pipeline:
            medium_elapsed = sum(result["elapsed_sec"] for result in pipeline["medium"]["results"])
        if args.stage == "all" and medium_elapsed > args.skip_full_if_medium_sec:
            add_blocker(f"full stage skipped because medium elapsed_sec={medium_elapsed:.2f} exceeded threshold={args.skip_full_if_medium_sec:.2f}")
            pipeline["full"] = {"stage": "full", "status": "skipped", "results": []}
            write_json(OUT_DIR / "full_status.json", pipeline["full"])
        else:
            full_results = full_reference() + full_sampler()
            pipeline["full"] = persist_stage("full", full_results)

    pipeline["blockers_file"] = str(BLOCKERS_PATH)
    pipeline["log_file"] = str(LOG_PATH)
    for stage_name in ["smoke", "medium", "full"]:
        stage_path = OUT_DIR / f"{stage_name}_status.json"
        if stage_name not in pipeline and stage_path.exists():
            try:
                pipeline[stage_name] = json.loads(stage_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                add_blocker(f"could not parse existing stage status: {stage_path}")
    write_json(OUT_DIR / "pipeline_status.json", pipeline)
    print(f"wrote pipeline status to {OUT_DIR}")


if __name__ == "__main__":
    main()
