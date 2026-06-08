"""Generate and validate Grace final-production v1 runset metadata."""

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

from scripts.targeted_validation_config import expanded_cases


REQUIRED_FILES = [
    "run_metadata.json",
    "posterior_summaries.csv",
    "cost_ledger.csv",
    "latent_diagnostics.csv",
    "transition_diagnostics.csv",
    "branch_diagnostics.csv",
    "rattle_energy_diagnostics.csv",
    "geometry_diagnostics.csv",
    "initialization_diagnostics.csv",
]
CHAIN_ALTERNATIVES = ["chain_samples.csv", "chain_samples.parquet"]
ROW_COUNT_FILES = REQUIRED_FILES[1:] + CHAIN_ALTERNATIVES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/final_production_v1_cases.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/final_production_v1"))
    parser.add_argument("--case-tsv", type=Path, default=Path("results/final_production_v1/final_production_cases.tsv"))
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def row_count(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        if path.suffix == ".parquet":
            return int(len(pd.read_parquet(path)))
        return int(len(pd.read_csv(path)))
    except Exception:
        return None


def write_cases(config: Path, out_dir: Path, case_tsv: Path) -> list[dict[str, Any]]:
    cases = expanded_cases(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_tsv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_index",
        "case_id",
        "model",
        "k",
        "n",
        "method",
        "seed",
        "initialization",
        "diagnostic_only",
        "num_iterations",
        "burn_in",
        "diagnostic_thin",
        "output_dir",
    ]
    with case_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index, case in enumerate(cases, start=1):
            writer.writerow(
                {
                    "task_index": index,
                    "case_id": case["case_id"],
                    "model": case["model"],
                    "k": "" if case.get("k") is None else case.get("k"),
                    "n": int(case["n"]),
                    "method": case["method"],
                    "seed": int(case["seed"]),
                    "initialization": case["initialization"],
                    "diagnostic_only": bool(case.get("diagnostic_only", False)),
                    "num_iterations": int(case["num_iterations"]),
                    "burn_in": int(case["burn_in"]),
                    "diagnostic_thin": int(case["diagnostic_thin"]),
                    "output_dir": str(out_dir / f"case_{case['case_id']}"),
                }
            )
    return cases


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def classify_case(row: dict[str, str], out_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_dir = out_dir / f"case_{row['case_id']}"
    missing = []
    if not any((case_dir / name).exists() and (case_dir / name).stat().st_size > 0 for name in CHAIN_ALTERNATIVES):
        missing.append("chain_samples.csv or chain_samples.parquet")
    for filename in REQUIRED_FILES:
        path = case_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            missing.append(filename)
    metadata_status = ""
    metadata = {}
    metadata_path = case_dir / "run_metadata.json"
    if metadata_path.exists() and metadata_path.stat().st_size > 0:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_status = str(metadata.get("status", ""))
        except json.JSONDecodeError:
            missing.append("valid run_metadata.json")
    if not case_dir.exists():
        status = "missing"
    elif missing:
        status = "failed" if any(case_dir.iterdir()) else "missing"
    elif metadata_status == "completed":
        status = "complete"
    else:
        status = "partial"
    report = {
        "case_id": row["case_id"],
        "model": row["model"],
        "k": row["k"],
        "n": int(row["n"]),
        "method": row["method"],
        "seed": int(row["seed"]),
        "initialization": row["initialization"],
        "diagnostic_only": row["diagnostic_only"],
        "status": status,
        "missing_files": ";".join(missing),
        "runtime_sec": None,
        "output_dir": str(case_dir),
    }
    ledger_path = case_dir / "cost_ledger.csv"
    if ledger_path.exists():
        try:
            ledger = pd.read_csv(ledger_path)
            if "wall_time_sec" in ledger and not ledger.empty:
                report["runtime_sec"] = float(pd.to_numeric(ledger["wall_time_sec"], errors="coerce").iloc[0])
        except Exception:
            pass
    for filename in ROW_COUNT_FILES:
        report[f"{filename.replace('.', '_')}_rows"] = row_count(case_dir / filename)
    missing_rows = [{**{key: report[key] for key in ["case_id", "model", "k", "n", "method", "seed", "initialization", "status"]}, "missing_file": item, "output_dir": str(case_dir)} for item in missing]
    return report, missing_rows


def write_manifest(out_dir: Path, case_tsv: Path, reports: list[dict[str, Any]]) -> None:
    status_counts = pd.Series([row["status"] for row in reports]).value_counts().to_dict() if reports else {}
    manifest = {
        "runset": "final_production_v1",
        "out_dir": str(out_dir),
        "case_tsv": str(case_tsv),
        "total_cases": len(reports),
        "status_counts": status_counts,
        "cases": reports,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def write_reports(out_dir: Path, case_tsv: Path) -> None:
    rows = read_cases(case_tsv)
    reports = []
    missing = []
    for row in rows:
        report, missing_rows = classify_case(row, out_dir)
        reports.append(report)
        missing.extend(missing_rows)
    pd.DataFrame(reports).to_csv(out_dir / "job_completion_report.csv", index=False)
    pd.DataFrame(missing).to_csv(out_dir / "missing_outputs.csv", index=False)
    failed = [row for row in reports if row["status"] != "complete"]
    failed_fields = ["case_id", "model", "k", "n", "method", "seed", "initialization", "diagnostic_only", "status", "missing_files", "output_dir"]
    pd.DataFrame(failed, columns=failed_fields).to_csv(out_dir / "failed_cases.tsv", sep="\t", index=False)
    counts = pd.Series([row["status"] for row in reports]).value_counts().to_dict() if reports else {}
    lines = [
        "# Grace Final Production v1 Completion Summary",
        "",
        f"- Total cases: {len(reports)}",
        f"- Complete: {counts.get('complete', 0)}",
        f"- Partial: {counts.get('partial', 0)}",
        f"- Failed: {counts.get('failed', 0)}",
        f"- Missing: {counts.get('missing', 0)}",
        "",
    ]
    if failed:
        lines.extend(["## Incomplete Cases", ""])
        lines.extend(f"- {row['case_id']}: {row['status']} ({row['missing_files']})" for row in failed)
    else:
        lines.append("All final production v1 cases are complete.")
    (out_dir / "grace_completion_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_manifest(out_dir, case_tsv, reports)
    print(json.dumps({"total_cases": len(reports), "status_counts": counts}, sort_keys=True))


def main() -> None:
    args = parse_args()
    cases = write_cases(args.config, args.out_dir, args.case_tsv)
    if args.check:
        write_reports(args.out_dir, args.case_tsv)
    else:
        write_manifest(args.out_dir, args.case_tsv, [])
        print(json.dumps({"case_tsv": str(args.case_tsv), "total_cases": len(cases)}, sort_keys=True))


if __name__ == "__main__":
    main()
