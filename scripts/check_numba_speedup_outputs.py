"""Validate Grace Student Gibbs backend timing outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/numba_speedup_validation"))
    parser.add_argument("--case-table", type=Path, default=Path("hpc/grace/numba_speedup_validation_cases.tsv"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/numba_speedup_validation"))
    return parser.parse_args()


def read_case_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def result_path(run_dir: Path, row: dict[str, str]) -> Path:
    return (
        run_dir
        / "task_results"
        / f"task_{row['task_index']}_{row['backend']}_k{row['k']}_n{row['n']}_seed{row['seed']}_repeat{row['repeat']}.csv"
    )


def find_error_log(logs_dir: Path, task_index: str) -> str:
    matches = sorted(logs_dir.glob(f"*_{task_index}.err"))
    nonempty = [path for path in matches if path.exists() and path.stat().st_size > 0]
    if nonempty:
        return str(nonempty[-1])
    return str(matches[-1]) if matches else ""


def classify(row: dict[str, str], args: argparse.Namespace) -> tuple[dict, dict | None, pd.DataFrame | None]:
    path = result_path(args.run_dir, row)
    missing = []
    status = "complete"
    frame = None
    result_row = None
    if not path.exists() or path.stat().st_size == 0:
        missing.append(str(path))
        status = "missing"
    else:
        try:
            frame = pd.read_csv(path)
        except Exception:
            frame = None
            status = "failed"
        if frame is None or len(frame) != 1:
            status = "failed"
        else:
            result_row = frame.iloc[0].to_dict()
            checks = [
                str(result_row.get("backend")) == str(row["backend"]),
                int(result_row.get("n", -1)) == int(row["n"]),
                float(result_row.get("k", np.nan)) == float(row["k"]),
                int(result_row.get("iterations", -1)) == int(row["num_iterations"]),
                np.isfinite(float(result_row.get("elapsed_sec", np.nan))),
                float(result_row.get("elapsed_sec", np.nan)) > 0.0,
            ]
            if not all(checks):
                status = "failed"
    report = {
        **row,
        "status": status,
        "result_file": str(path),
        "missing_files": ";".join(missing),
        "error_log_path": find_error_log(args.logs_dir, row["task_index"]),
    }
    if result_row is not None:
        for key in ["elapsed_sec", "iterations_per_sec", "pair_updates_per_sec", "mu_acceptance_rate", "pair_acceptance_rate", "z_acceptance_rate", "block_z_acceptance_rate", "posterior_mu_mean", "posterior_mu_sd"]:
            report[key] = result_row.get(key, np.nan)
    return report, result_row, frame


def write_summary(reports: list[dict], path: Path) -> None:
    frame = pd.DataFrame(reports)
    counts = frame["status"].value_counts().to_dict()
    lines = [
        "# Numba Speedup Validation Completion Summary",
        "",
        f"- Total cases: {len(frame)}",
        f"- Complete: {counts.get('complete', 0)}",
        f"- Failed: {counts.get('failed', 0)}",
        f"- Missing: {counts.get('missing', 0)}",
        "",
    ]
    complete = frame[frame["status"].eq("complete")].copy()
    if not complete.empty:
        complete["iterations_per_sec"] = pd.to_numeric(complete["iterations_per_sec"], errors="coerce")
        grouped = (
            complete.groupby(["backend", "k", "n"], dropna=False)["iterations_per_sec"]
            .agg(["count", "mean", "median"])
            .reset_index()
            .sort_values(["k", "n", "backend"])
        )
        lines.extend(["## Iterations/Sec By Backend", ""])
        lines.append("| backend | k | n | count | mean | median |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, row in grouped.iterrows():
            lines.append(
                f"| {row['backend']} | {row['k']} | {row['n']} | {int(row['count'])} | "
                f"{float(row['mean']):.6g} | {float(row['median']):.6g} |"
            )
        lines.append("")
    incomplete = frame[~frame["status"].eq("complete")]
    if not incomplete.empty:
        lines.extend(["## Incomplete Cases", ""])
        for _, row in incomplete.iterrows():
            lines.append(f"- task {row['task_index']} {row['backend']} k={row['k']} n={row['n']} seed={row['seed']} repeat={row['repeat']}: {row['status']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(args: argparse.Namespace, reports: list[dict], path: Path) -> None:
    files = {}
    if args.run_dir.exists():
        for file_path in sorted(p for p in args.run_dir.rglob("*") if p.is_file()):
            files[str(file_path)] = {"bytes": file_path.stat().st_size}
    manifest = {
        "run_dir": str(args.run_dir),
        "case_table": str(args.case_table),
        "logs_dir": str(args.logs_dir),
        "total_cases": len(reports),
        "status_counts": pd.Series([row["status"] for row in reports]).value_counts().to_dict(),
        "files": files,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    rows = read_case_table(args.case_table)
    reports = []
    frames = []
    for row in rows:
        report, _, frame = classify(row, args)
        reports.append(report)
        if frame is not None and report["status"] == "complete":
            frame = frame.copy()
            frame["task_index"] = int(row["task_index"])
            frame["repeat"] = int(row["repeat"])
            frames.append(frame)

    report_frame = pd.DataFrame(reports)
    report_frame.to_csv(args.run_dir / "job_completion_report.csv", index=False)
    report_frame[~report_frame["status"].eq("complete")].to_csv(args.run_dir / "failed_cases.tsv", sep="\t", index=False)
    missing = report_frame[report_frame["missing_files"].astype(str).ne("")]
    missing.to_csv(args.run_dir / "missing_outputs.csv", index=False)
    if frames:
        pd.concat(frames, ignore_index=True, sort=False).to_csv(args.run_dir / "benchmark_results.csv", index=False)
    write_manifest(args, reports, args.run_dir / "grace_output_manifest.json")
    write_summary(reports, args.run_dir / "grace_completion_summary.md")
    print(json.dumps({"run_dir": str(args.run_dir), "total_cases": len(reports), "complete": int(report_frame["status"].eq("complete").sum())}, sort_keys=True))


if __name__ == "__main__":
    main()
