"""Create a transfer package for Grace Student Gibbs backend timing outputs."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import pandas as pd


REPORT_FILES = [
    "run_manifest.json",
    "job_completion_report.csv",
    "missing_outputs.csv",
    "failed_cases.tsv",
    "grace_output_manifest.json",
    "grace_completion_summary.md",
    "benchmark_results.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/numba_speedup_validation"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/numba_speedup_validation"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/numba_speedup_validation"))
    parser.add_argument("--include-task-csvs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-tar", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def add_if_exists(files: list[Path], path: Path) -> None:
    if path.exists() and path.is_file() and path not in files:
        files.append(path)


def selected_logs(run_dir: Path, logs_dir: Path) -> list[Path]:
    report_path = run_dir / "job_completion_report.csv"
    if not report_path.exists():
        return sorted(logs_dir.glob("*.out")) + sorted(logs_dir.glob("*.err"))
    report = pd.read_csv(report_path)
    failed = report[~report["status"].eq("complete")]
    files: list[Path] = []
    for _, row in failed.iterrows():
        log_path = str(row.get("error_log_path", ""))
        if log_path:
            add_if_exists(files, Path(log_path))
    if files:
        return files
    return [path for path in sorted(logs_dir.glob("*.err")) if path.stat().st_size > 0]


def collect_files(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    for filename in REPORT_FILES:
        add_if_exists(files, args.run_dir / filename)
    if args.include_task_csvs:
        for path in sorted((args.run_dir / "task_results").glob("*.csv")):
            add_if_exists(files, path)
    for path in selected_logs(args.run_dir, args.logs_dir):
        add_if_exists(files, path)
    return files


def write_filelist(files: list[Path], path: Path) -> None:
    path.write_text("\n".join(str(file_path) for file_path in files) + "\n", encoding="utf-8")


def write_manifest(files: list[Path], args: argparse.Namespace, path: Path) -> None:
    manifest = {
        "run_dir": str(args.run_dir),
        "logs_dir": str(args.logs_dir),
        "include_task_csvs": bool(args.include_task_csvs),
        "file_count": len(files),
        "total_bytes": sum(file_path.stat().st_size for file_path in files),
        "files": [{"path": str(file_path), "bytes": file_path.stat().st_size} for file_path in files],
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def write_tar(files: list[Path], path: Path) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for file_path in files:
            archive.add(file_path, arcname=str(file_path))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = collect_files(args)
    manifest_path = args.out_dir / "numba_speedup_transfer_manifest.json"
    filelist_path = args.out_dir / "numba_speedup_transfer_filelist.txt"
    tar_path = args.out_dir / "numba_speedup_outputs_for_local.tar.gz"
    write_manifest(files, args, manifest_path)
    write_filelist(files, filelist_path)
    if args.make_tar:
        write_tar(files + [manifest_path, filelist_path], tar_path)
    print(json.dumps({"file_count": len(files), "manifest": str(manifest_path), "tar": str(tar_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
