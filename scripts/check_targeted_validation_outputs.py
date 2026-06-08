"""Check Grace targeted-validation outputs without rerunning samplers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.targeted_validation_config import find_case


REQUIRED_ALTERNATIVES = {
    "chain_samples": ["chain_samples.csv", "chain_samples.parquet"],
}
REQUIRED_FILES = [
    "posterior_summaries.csv",
    "cost_ledger.csv",
    "transition_diagnostics.csv",
    "latent_diagnostics.csv",
    "initialization_diagnostics.csv",
    "run_metadata.json",
]
OPTIONAL_FILES = [
    "rattle_energy_diagnostics.csv",
    "branch_diagnostics.csv",
]
ROW_COUNT_FILES = [
    "chain_samples.csv",
    "chain_samples.parquet",
    "posterior_summaries.csv",
    "cost_ledger.csv",
    "transition_diagnostics.csv",
    "latent_diagnostics.csv",
    "initialization_diagnostics.csv",
    "rattle_energy_diagnostics.csv",
    "branch_diagnostics.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/targeted_validation_runs"))
    parser.add_argument("--case-table", type=Path, default=Path("hpc/grace/targeted_validation_cases.tsv"))
    parser.add_argument("--case-config", type=Path, default=Path("configs/targeted_validation_cases.yaml"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/targeted_validation"))
    return parser.parse_args()


def read_case_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def row_count(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        if path.suffix == ".parquet":
            return int(len(pd.read_parquet(path)))
        return int(len(pd.read_csv(path)))
    except Exception:
        return None


def find_error_log(logs_dir: Path, task_index: str) -> str:
    matches = sorted(logs_dir.glob(f"*_{task_index}.err"))
    if not matches:
        return ""
    nonempty = [path for path in matches if path.exists() and path.stat().st_size > 0]
    return str(nonempty[-1] if nonempty else matches[-1])


def read_runtime(metadata_path: Path) -> float | None:
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    ledger_path = metadata.get("files", {}).get("ledger")
    if not ledger_path:
        return None
    ledger = Path(ledger_path)
    if not ledger.exists():
        return None
    try:
        frame = pd.read_csv(ledger)
    except Exception:
        return None
    for column in ["runtime_sec", "elapsed_wall_time_sec", "wall_time_sec"]:
        if column in frame.columns and len(frame) > 0:
            value = pd.to_numeric(frame[column], errors="coerce").iloc[0]
            return None if pd.isna(value) else float(value)
    runtime_columns = [column for column in frame.columns if "runtime" in column.lower() or "wall" in column.lower()]
    for column in runtime_columns:
        value = pd.to_numeric(frame[column], errors="coerce").iloc[0]
        if not pd.isna(value):
            return float(value)
    return None


def classify_case(row: dict[str, str], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_id = row["case_id"]
    case = find_case(args.case_config, case_id)
    case_dir = args.run_dir / f"case_{case_id}"
    missing: list[str] = []
    required_present: dict[str, bool] = {}

    for logical_name, alternatives in REQUIRED_ALTERNATIVES.items():
        present = any((case_dir / name).exists() and (case_dir / name).stat().st_size > 0 for name in alternatives)
        required_present[logical_name] = present
        if not present:
            missing.append(" or ".join(alternatives))
    for filename in REQUIRED_FILES:
        present = (case_dir / filename).exists() and (case_dir / filename).stat().st_size > 0
        required_present[filename] = present
        if not present:
            missing.append(filename)

    counts = {filename: row_count(case_dir / filename) for filename in ROW_COUNT_FILES}
    optional_present = {filename: (case_dir / filename).exists() for filename in OPTIONAL_FILES}
    metadata_path = case_dir / "run_metadata.json"
    metadata_status = ""
    if metadata_path.exists():
        try:
            metadata_status = str(json.loads(metadata_path.read_text(encoding="utf-8")).get("status", ""))
        except json.JSONDecodeError:
            missing.append("valid run_metadata.json")

    if not case_dir.exists():
        status = "missing"
    elif not missing and metadata_status == "completed":
        status = "complete"
    elif not missing:
        status = "partial"
    else:
        status = "failed" if any(path.exists() for path in case_dir.iterdir()) else "missing"

    error_log = find_error_log(args.logs_dir, str(row["task_index"]))
    report = {
        "case_id": case_id,
        "model": case["model"],
        "k": "" if case.get("k") is None else case.get("k"),
        "n": int(case["n"]),
        "method": case["method"],
        "seed": int(case["seed"]),
        "initialization": case["initialization"],
        "diagnostic_only": bool(case.get("diagnostic_only", False)),
        "status": status,
        "missing_files": ";".join(missing),
        "runtime_sec": read_runtime(metadata_path),
        "error_log_path": error_log,
    }
    for filename, count in counts.items():
        safe = filename.replace(".", "_")
        report[f"{safe}_rows"] = count
    for filename, present in optional_present.items():
        safe = filename.replace(".", "_")
        report[f"{safe}_present"] = present

    missing_rows = [
        {
            "case_id": case_id,
            "model": case["model"],
            "k": "" if case.get("k") is None else case.get("k"),
            "n": int(case["n"]),
            "method": case["method"],
            "seed": int(case["seed"]),
            "initialization": case["initialization"],
            "missing_file": item,
            "case_dir": str(case_dir),
            "error_log_path": error_log,
        }
        for item in missing
    ]
    return report, missing_rows


def write_failed_cases(reports: list[dict[str, Any]], path: Path) -> None:
    failed = [row for row in reports if row["status"] != "complete"]
    fields = ["case_id", "model", "k", "n", "method", "seed", "initialization", "diagnostic_only", "status", "missing_files", "error_log_path"]
    pd.DataFrame(failed, columns=fields).to_csv(path, sep="\t", index=False)


def write_manifest(args: argparse.Namespace, reports: list[dict[str, Any]], path: Path) -> None:
    status_counts = pd.Series([row["status"] for row in reports]).value_counts().to_dict()
    cases = []
    for row in reports:
        case_dir = args.run_dir / f"case_{row['case_id']}"
        files = {}
        if case_dir.exists():
            for file_path in sorted(path for path in case_dir.iterdir() if path.is_file()):
                files[file_path.name] = {"path": str(file_path), "bytes": file_path.stat().st_size}
        cases.append({"case_id": row["case_id"], "status": row["status"], "files": files})
    manifest = {
        "run_dir": str(args.run_dir),
        "case_table": str(args.case_table),
        "total_cases": len(reports),
        "status_counts": status_counts,
        "cases": cases,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def write_summary(reports: list[dict[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(reports)
    counts = frame["status"].value_counts().to_dict()
    lines = [
        "# Grace Targeted Validation Completion Summary",
        "",
        f"- Total cases: {len(frame)}",
        f"- Complete: {counts.get('complete', 0)}",
        f"- Partial: {counts.get('partial', 0)}",
        f"- Failed: {counts.get('failed', 0)}",
        f"- Missing: {counts.get('missing', 0)}",
        "",
    ]
    incomplete = frame[frame["status"] != "complete"]
    if incomplete.empty:
        lines.append("All targeted validation cases are complete.")
    else:
        lines.extend(["## Incomplete Cases", ""])
        for _, row in incomplete.iterrows():
            missing = row["missing_files"] or "no required files missing; metadata/status needs review"
            lines.append(f"- {row['case_id']}: {row['status']} ({missing})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_write_resubmit_script(reports: list[dict[str, Any]], case_table: Path, path: Path) -> None:
    failed_ids = [row["case_id"] for row in reports if row["status"] != "complete"]
    if not failed_ids:
        if path.exists():
            path.unlink()
        return
    quoted = " ".join(failed_ids)
    content = f"""#!/usr/bin/env bash
set -euo pipefail

# Prepared only. Review failed_cases.tsv before running.
PYTHON_MODULE="${{PYTHON_MODULE:-GCCcore/14.3.0 Python/3.13.5}}"
CASE_TSV="${{CASE_TSV:-{case_table}}}"
OUT_DIR="${{OUT_DIR:-results/targeted_validation_runs}}"
FAILED_CASES=({quoted})

for case_id in "${{FAILED_CASES[@]}}"; do
  task_id="$(awk -v case_id="${{case_id}}" 'NR > 1 && $2 == case_id {{print $1}}' "${{CASE_TSV}}")"
  if [ -z "${{task_id}}" ]; then
    echo "No task id found for ${{case_id}}" >&2
    continue
  fi
  echo "Would resubmit ${{case_id}} as array task ${{task_id}}"
done

echo "To actually rerun failed cases, submit the needed array indexes manually after reviewing resources."
"""
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    rows = read_case_table(args.case_table)
    reports: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for row in rows:
        report, missing = classify_case(row, args)
        reports.append(report)
        missing_rows.extend(missing)

    pd.DataFrame(reports).to_csv(args.run_dir / "job_completion_report.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(args.run_dir / "missing_outputs.csv", index=False)
    write_failed_cases(reports, args.run_dir / "failed_cases.tsv")
    write_manifest(args, reports, args.run_dir / "grace_output_manifest.json")
    write_summary(reports, args.run_dir / "grace_completion_summary.md")
    maybe_write_resubmit_script(reports, args.case_table, Path("hpc/grace/resubmit_failed_targeted_validation.sh"))

    counts = pd.Series([row["status"] for row in reports]).value_counts().to_dict()
    print(json.dumps({"total_cases": len(reports), "status_counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
