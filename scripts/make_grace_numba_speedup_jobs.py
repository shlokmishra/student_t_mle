"""Generate Grace case tables and run manifests for Student Gibbs backend timing."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/numba_speedup_validation_cases.yaml"))
    parser.add_argument("--case-tsv", type=Path, default=Path("hpc/grace/numba_speedup_validation_cases.tsv"))
    parser.add_argument("--manifest", type=Path, default=Path("results/numba_speedup_validation/run_manifest.json"))
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def mode_settings(config: dict, mode: str) -> dict:
    settings = dict(config["defaults"])
    if mode == "smoke":
        settings.update(config.get("smoke", {}))
    return settings


def build_rows(settings: dict) -> list[dict]:
    rows = []
    for backend in settings["backends"]:
        for k in settings["k_values"]:
            for n in settings["n_values"]:
                for seed in settings["seeds"]:
                    for repeat in range(int(settings["repeats"])):
                        rows.append(
                            {
                                "task_index": len(rows) + 1,
                                "backend": str(backend),
                                "k": float(k),
                                "n": int(n),
                                "seed": int(seed),
                                "repeat": int(repeat),
                                "num_iterations": int(settings["num_iterations"]),
                                "warmup_iterations": int(settings["warmup_iterations"]),
                            }
                        )
    return rows


def write_tsv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(config: dict, settings: dict, rows: list[dict], args: argparse.Namespace) -> None:
    runset = config.get("runset", {})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "runset_name": runset.get("name", "numba_speedup_validation"),
        "mode": args.mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "case_tsv": str(args.case_tsv),
        "output_dir": runset.get("output_dir", "results/numba_speedup_validation"),
        "logs_dir": runset.get("logs_dir", "logs/numba_speedup_validation"),
        "total_cases": len(rows),
        "settings": settings,
        "cases": rows,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    settings = mode_settings(config, args.mode)
    rows = build_rows(settings)
    write_tsv(rows, args.case_tsv)
    write_manifest(config, settings, rows, args)
    print(json.dumps({"mode": args.mode, "case_tsv": str(args.case_tsv), "manifest": str(args.manifest), "num_cases": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
