"""Tune conservative RATTLE settings on a small grid."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["student_t", "logistic"], choices=["student_t", "logistic"])
    parser.add_argument("--k-values", type=_floats, default=[1.0, 2.0, 3.0])
    parser.add_argument("--n-values", type=_ints, default=[10])
    parser.add_argument("--step-sizes", type=_floats, default=[0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--leapfrog-steps", type=_ints, default=[5, 10, 20])
    parser.add_argument("--num-iterations", type=int, default=3000)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/rattle_tuning/"))
    parser.add_argument("--projection-failure-threshold", type=float, default=0.05)
    parser.add_argument("--reverse-check-failure-threshold", type=float, default=0.05)
    return parser.parse_args()


def settings_key(model: str, k: float, n: int) -> str:
    if model == "student_t":
        return f"student_t:k={float(k):g}:n={int(n)}"
    return f"{model}:n={int(n)}"


def model_k_values(model: str, k_values: list[float]) -> list[float]:
    return k_values if model == "student_t" else [np.nan]


def run_config(args: argparse.Namespace, model: str, k: float, n: int, step_size: float, leapfrog_steps: int) -> dict:
    run_name = settings_key(model, k, n).replace(":", "_").replace("=", "") + f"_eps{step_size:g}_L{leapfrog_steps}"
    run_dir = args.out / "runs" / run_name
    cmd = [
        sys.executable,
        "scripts/run_cost_audit.py",
        "--models",
        model,
        "--methods",
        "rattle",
        "--n-values",
        str(int(n)),
        "--num-iterations",
        str(int(args.num_iterations)),
        "--burn-in",
        str(int(args.burn_in)),
        "--seed",
        str(int(args.seed)),
        "--rattle-step-size",
        str(float(step_size)),
        "--rattle-num-steps",
        str(int(leapfrog_steps)),
        "--run-status",
        "tuning",
        "--out",
        str(run_dir),
    ]
    if model == "student_t":
        cmd.extend(["--k-values", str(float(k))])
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    row = {
        "model": model,
        "k": float(k) if np.isfinite(k) else np.nan,
        "n": int(n),
        "rattle_step_size": float(step_size),
        "rattle_num_steps": int(leapfrog_steps),
        "returncode": int(completed.returncode),
        "run_dir": str(run_dir),
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    }
    diag_path = run_dir / "diagnostic_summary.csv"
    ledger_path = run_dir / "cost_ledger.csv"
    if completed.returncode == 0 and diag_path.exists():
        diag = pd.read_csv(diag_path)
        if not diag.empty:
            first = diag.iloc[0].to_dict()
            row.update(first)
    if ledger_path.exists():
        ledger = pd.read_csv(ledger_path)
        if not ledger.empty:
            row["projection_evals"] = float(ledger.iloc[0].get("projection_evals", np.nan))
            row["projection_failures"] = float(ledger.iloc[0].get("projection_failures", np.nan))
            row["reverse_check_failures"] = float(ledger.iloc[0].get("reverse_check_failures", np.nan))
    return row


def choose_recommendations(summary: pd.DataFrame, args: argparse.Namespace) -> dict:
    recommendations = {"settings": [], "by_key": {}}
    for (model, k, n), part in summary.groupby(["model", "k", "n"], dropna=False):
        scored = part.copy()
        scored["projection_failure_rate"] = pd.to_numeric(scored.get("projection_failure_rate"), errors="coerce")
        scored["reverse_check_failure_rate"] = pd.to_numeric(scored.get("reverse_check_failure_rate"), errors="coerce")
        scored["acceptance_rate"] = pd.to_numeric(scored.get("acceptance_rate"), errors="coerce")
        scored["ess_per_sec"] = pd.to_numeric(scored.get("ess_per_sec"), errors="coerce")
        ok = scored[
            (scored["returncode"].eq(0))
            & (scored["projection_failure_rate"].fillna(1.0) <= args.projection_failure_threshold)
            & (scored["reverse_check_failure_rate"].fillna(1.0) <= args.reverse_check_failure_threshold)
        ].copy()
        preferred = ok[(ok["acceptance_rate"] >= 0.4) & (ok["acceptance_rate"] <= 0.95)].copy()
        candidates = preferred if not preferred.empty else ok
        status = "ok" if not preferred.empty else ("warning" if not ok.empty else "warning")
        if candidates.empty:
            candidates = scored.copy()
            candidates["badness"] = (
                candidates["projection_failure_rate"].fillna(1.0)
                + candidates["reverse_check_failure_rate"].fillna(1.0)
                + (candidates["acceptance_rate"].fillna(0.0) - 0.65).abs()
            )
            best = candidates.sort_values(["badness", "ess_per_sec"], ascending=[True, False]).iloc[0]
        else:
            best = candidates.sort_values("ess_per_sec", ascending=False).iloc[0]
        record = {
            "model": str(model),
            "k": float(k) if pd.notna(k) else np.nan,
            "n": int(n),
            "rattle_step_size": float(best["rattle_step_size"]),
            "rattle_num_steps": int(best["rattle_num_steps"]),
            "acceptance_rate": float(best.get("acceptance_rate", np.nan)),
            "ess_per_sec": float(best.get("ess_per_sec", np.nan)),
            "projection_failure_rate": float(best.get("projection_failure_rate", np.nan)),
            "reverse_check_failure_rate": float(best.get("reverse_check_failure_rate", np.nan)),
            "status": status,
        }
        recommendations["settings"].append(record)
        recommendations["by_key"][settings_key(str(model), float(k), int(n))] = record
    return recommendations


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in args.models:
        for k in model_k_values(model, args.k_values):
            for n in args.n_values:
                for step_size in args.step_sizes:
                    for leapfrog_steps in args.leapfrog_steps:
                        row = run_config(args, model, k, int(n), float(step_size), int(leapfrog_steps))
                        rows.append(row)
                        print(
                            "completed tuning "
                            f"model={model} k={k} n={n} eps={step_size} L={leapfrog_steps} returncode={row['returncode']}"
                        )
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "tuning_summary.csv", index=False)
    recommendations = choose_recommendations(summary, args) if not summary.empty else {"settings": [], "by_key": {}}
    with (args.out / "recommended_rattle_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(recommendations), handle, indent=2, allow_nan=False)
    print(f"wrote tuning outputs to {args.out}")


if __name__ == "__main__":
    main()
