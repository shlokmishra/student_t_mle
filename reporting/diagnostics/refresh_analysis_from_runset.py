"""Refresh downstream derived analyses from a named runset without overwriting baseline outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.run_registry import load_common_run_outputs, load_run_registry, resolve_runset_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runset", default="targeted_validation")
    parser.add_argument("--run-registry", type=Path, default=Path("configs/analysis_run_registry.yaml"))
    parser.add_argument("--refresh", nargs="+", default=["correctness", "efficiency", "geometry"])
    parser.add_argument("--out-root", type=Path, default=Path("results/refreshed_analysis/"))
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_empty(path: Path, columns: list[str]) -> None:
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def run_command(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    return proc.returncode, (proc.stdout + "\n" + proc.stderr).strip()


def compare_verdicts(new_dir: Path) -> pd.DataFrame:
    baseline = read_csv(ROOT / "results" / "sampler_correctness_audit" / "final_sampler_verdict_table.csv")
    new = read_csv(new_dir / "refreshed_correctness_summary.csv")
    if baseline.empty or new.empty:
        return pd.DataFrame(
            columns=["model", "k", "n", "method", "old_verdict", "new_verdict", "changed", "reason"]
        )
    keys = ["model", "k", "n", "method"]
    if "new_recommended_verdict" in new.columns:
        keep = keys + [col for col in ["new_recommended_verdict", "new_safe_to_present", "recommendation_reason", "recommended_action"] if col in new.columns]
        new = new[keep].rename(
            columns={
                "new_recommended_verdict": "verdict",
                "new_safe_to_present": "safe_to_present",
                "recommendation_reason": "main_warning",
            }
        )
    old_cols = keys + [col for col in ["verdict", "safe_to_present", "main_warning"] if col in baseline.columns]
    new_cols = keys + [col for col in ["verdict", "safe_to_present", "main_warning"] if col in new.columns]
    merged = baseline[old_cols].merge(new[new_cols], on=keys, how="inner", suffixes=("_old", "_new"))
    merged["changed"] = (
        merged["verdict_old"].astype(str).ne(merged["verdict_new"].astype(str))
        | merged["safe_to_present_old"].astype(str).ne(merged["safe_to_present_new"].astype(str))
    )
    merged["reason"] = "Derived comparison only; inspect refreshed reports for details."
    return merged.rename(columns={"verdict_old": "old_verdict", "verdict_new": "new_verdict"})


def main() -> None:
    args = parse_args()
    registry = load_run_registry(args.run_registry)
    runset = resolve_runset_paths(args.runset, registry)
    outputs = load_common_run_outputs(runset)
    out_dir = args.out_root / args.runset
    out_dir.mkdir(parents=True, exist_ok=True)

    actions: list[dict] = []
    tables = outputs["tables"]

    if "correctness" in args.refresh:
        target = out_dir / "refreshed_correctness_summary.csv"
        code, text = run_command(
            [
                sys.executable,
                "reporting/diagnostics/analyze_targeted_validation.py",
                "--runs-dir",
                str(runset.run_dir),
                "--reference-csv",
                str(runset.reference_csv or "reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv"),
                "--verdict-csv",
                "results/sampler_correctness_audit/final_sampler_verdict_table.csv",
            ]
        )
        recs = read_csv(runset.run_dir / "upgraded_verdict_recommendations.csv")
        if not recs.empty:
            recs.to_csv(target, index=False)
            actions.append({"refresh": "correctness", "status": "ok" if code == 0 else "failed", "detail": text[-1000:]})
        else:
            write_empty(target, ["model", "k", "n", "method", "recommended_action"])
            actions.append({"refresh": "correctness", "status": "no_recommendations", "detail": text[-1000:]})

    if "efficiency" in args.refresh:
        target = out_dir / "refreshed_efficiency_summary.csv"
        if not tables.get("chain_samples", pd.DataFrame()).empty and not tables.get("cost_ledger", pd.DataFrame()).empty:
            eff_dir = out_dir / "efficiency"
            code, text = run_command(
                [
                    sys.executable,
                    "reporting/diagnostics/analyze_efficiency.py",
                    "--cost-dir",
                    str(runset.run_dir),
                    "--correctness-dir",
                    "results/sampler_correctness_audit/",
                    "--reference-csv",
                    str(runset.reference_csv or "reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv"),
                    "--out-dir",
                    str(eff_dir),
                ]
            )
            summary = read_csv(eff_dir / "efficiency_summary.csv")
            summary.to_csv(target, index=False)
            actions.append({"refresh": "efficiency", "status": "ok" if code == 0 else "failed", "detail": text[-1000:]})
        else:
            write_empty(target, ["model", "k", "n", "method", "status"])
            actions.append({"refresh": "efficiency", "status": "skipped_missing_chain_or_cost", "detail": str(runset.run_dir)})

    if "geometry" in args.refresh:
        geom_dir = out_dir / "geometry"
        code, text = run_command(
            [
                sys.executable,
                "reporting/diagnostics/analyze_geometry.py",
                "--run-registry",
                str(args.run_registry),
                "--runsets",
                args.runset,
                "--reference-csv",
                str(runset.reference_csv or "reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv"),
                "--correctness-dir",
                "results/sampler_correctness_audit/",
                "--efficiency-dir",
                "results/efficiency_audit/",
                "--out-dir",
                str(geom_dir),
            ]
        )
        summary = read_csv(geom_dir / "geometry_summary.csv")
        summary.to_csv(out_dir / "refreshed_geometry_summary.csv", index=False)
        actions.append({"refresh": "geometry", "status": "ok" if code == 0 else "failed", "detail": text[-1000:]})

    changes = compare_verdicts(out_dir)
    changes.to_csv(out_dir / "verdict_changes.csv", index=False)
    report = [
        "# Refreshed Analysis Report",
        "",
        f"Runset: `{args.runset}`",
        f"Run directory: `{runset.run_dir}`",
        "",
        "This refresh does not overwrite baseline results.",
        "",
        "## Actions",
        "",
        pd.DataFrame(actions).to_markdown(index=False) if actions else "_No actions requested._",
        "",
        "## Verdict Changes",
        "",
        changes.to_markdown(index=False) if not changes.empty else "_No comparable verdict table available._",
        "",
    ]
    (out_dir / "refresh_report.md").write_text("\n".join(report), encoding="utf-8")
    manifest = {"runset": args.runset, "out_dir": str(out_dir), "actions": actions}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
