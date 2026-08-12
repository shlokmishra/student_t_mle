"""Small local RATTLE scaling benchmark for the Student-t location sampler.

This script is intentionally lighter than scripts/run_cost_audit.py: it runs a
short scaling grid, prints progress after each case, and writes compact summary
artifacts without saving full chains.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.random as random
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.cost_ledger import CostLedger
from models.loc_student_rattle import run_rattle


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def effective_sample_size(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n <= 1:
        return float(n)
    centered = values - np.mean(values)
    var = float(np.dot(centered, centered) / n)
    if var <= 0 or not np.isfinite(var):
        return float(n)
    autocorr_sum = 0.0
    for lag in range(1, n):
        acov = float(np.dot(centered[:-lag], centered[lag:]) / n)
        rho = acov / var
        if rho <= 0:
            break
        autocorr_sum += rho
    return float(max(n / (1.0 + 2.0 * autocorr_sum), 1.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-values", type=_floats, default=[2.0])
    parser.add_argument("--n-values", type=_ints, default=[10, 20, 50, 100, 200])
    parser.add_argument("--seeds", type=_ints, default=[0, 1, 2])
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--burn-in", type=int, default=50)
    parser.add_argument("--mu-star", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("results/local_rattle_scaling"))
    parser.add_argument("--proposal-std-mu", type=float, default=0.3)
    parser.add_argument("--prior-mean", type=float, default=0.0)
    parser.add_argument("--prior-std", type=float, default=10.0)
    parser.add_argument("--rattle-step-size", type=float, default=0.05)
    parser.add_argument("--rattle-num-steps", type=int, default=2)
    parser.add_argument("--rattle-proj-tol", type=float, default=1e-10)
    parser.add_argument("--rattle-proj-max-iters", type=int, default=25)
    parser.add_argument("--rattle-reverse-position-tol", type=float, default=5e-3)
    parser.add_argument("--rattle-reverse-momentum-tol", type=float, default=5e-3)
    parser.add_argument(
        "--rattle-projection-mode",
        choices=["paper_fixed_direction", "normal_newton_legacy"],
        default="paper_fixed_direction",
    )
    parser.add_argument("--rattle-include-gram-correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reverse-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initialization", choices=["central", "tail_heavy", "random"], default="central")
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def base_params(args: argparse.Namespace, k: float, n: int, seed: int) -> dict:
    return {
        "n": int(n),
        "k": float(k),
        "num_iterations_T": int(args.iterations),
        "proposal_std_mu": float(args.proposal_std_mu),
        "prior_mean": float(args.prior_mean),
        "prior_std": float(args.prior_std),
        "rattle_step_size": float(args.rattle_step_size),
        "rattle_num_steps": int(args.rattle_num_steps),
        "rattle_proj_tol": float(args.rattle_proj_tol),
        "rattle_proj_max_iters": int(args.rattle_proj_max_iters),
        "rattle_reverse_position_tol": float(args.rattle_reverse_position_tol),
        "rattle_reverse_momentum_tol": float(args.rattle_reverse_momentum_tol),
        "rattle_projection_mode": str(args.rattle_projection_mode),
        "rattle_include_gram_correction": bool(args.rattle_include_gram_correction),
        "reverse_check": bool(args.reverse_check),
        "initialization": str(args.initialization),
        "initialization_seed": int(seed),
    }


def _safe_rate(count: float, denom: float) -> float:
    return float(count) / max(float(denom), 1.0)


def run_case(args: argparse.Namespace, k: float, n: int, seed: int) -> dict:
    params = base_params(args, k, n, seed)
    ledger = CostLedger(
        method="rattle",
        model="student_t",
        n=int(n),
        k=float(k),
        mu_star=float(args.mu_star),
        seed=int(seed),
        iterations=int(args.iterations),
    )
    ledger.start()
    chain = run_rattle(random.PRNGKey(seed), args.mu_star, params, verbose=False, cost_ledger=ledger)
    ledger.stop()

    mus = np.asarray(chain["mu_chain"], dtype=float)
    burn_in = min(max(int(args.burn_in), 0), max(mus.size - 1, 0))
    post = mus[burn_in:]
    deltas = np.diff(post) if post.size > 1 else np.asarray([], dtype=float)
    abs_deltas = np.abs(deltas)
    sq_deltas = deltas * deltas

    ess_mu = effective_sample_size(post)
    elapsed_sec = float(ledger.counters.get("wall_time_sec", np.nan))
    iterations = max(int(args.iterations), 1)
    diag = dict(chain.get("projection_diagnostics", {}))
    counters = dict(ledger.counters)

    projection_evals = float(counters.get("projection_evals", np.nan))
    hmc_proposals = float(counters.get("hmc_proposals", iterations))
    projection_iterations_total = float(diag.get("projection_iterations_total", np.nan))

    row = {
        "model": "student_t",
        "method": "rattle",
        "k": float(k),
        "n": int(n),
        "seed": int(seed),
        "iterations": int(args.iterations),
        "burn_in": int(args.burn_in),
        "mu_star": float(args.mu_star),
        "elapsed_sec": elapsed_sec,
        "sec_per_iteration": elapsed_sec / iterations,
        "mu_acceptance_rate": float(chain.get("mu_acceptance_rate", np.nan)),
        "x_acceptance_rate": float(chain.get("x_acceptance_rate", np.nan)),
        "ess_mu": float(ess_mu),
        "ess_mu_per_sec": float(ess_mu) / max(elapsed_sec, 1e-12),
        "posterior_mean_mu": float(np.mean(post)) if post.size else np.nan,
        "posterior_sd_mu": float(np.std(post)) if post.size else np.nan,
        "mean_abs_delta_mu": float(np.mean(abs_deltas)) if abs_deltas.size else np.nan,
        "median_abs_delta_mu": float(np.median(abs_deltas)) if abs_deltas.size else np.nan,
        "esjd_mu": float(np.mean(sq_deltas)) if sq_deltas.size else np.nan,
        "esjd_mu_per_sec": (float(np.mean(sq_deltas)) if sq_deltas.size else np.nan) / max(elapsed_sec, 1e-12),
        "projection_failure_count": int(diag.get("projection_failure_count", counters.get("projection_failures", 0))),
        "projection_failure_rate": _safe_rate(
            float(diag.get("projection_failure_count", counters.get("projection_failures", 0))),
            hmc_proposals,
        ),
        "reverse_check_failure_count": int(
            diag.get("reverse_check_failure_count", counters.get("reverse_check_failures", 0))
        ),
        "reverse_check_failure_rate": _safe_rate(
            float(diag.get("reverse_check_failure_count", counters.get("reverse_check_failures", 0))),
            hmc_proposals,
        ),
        "projection_iterations_total": projection_iterations_total,
        "projection_iterations_per_iteration": projection_iterations_total / iterations,
        "max_projection_residual": float(diag.get("max_projection_residual", np.nan)),
        "mean_projection_residual": float(diag.get("mean_projection_residual", np.nan)),
        "max_manifold_residual": float(diag.get("max_manifold_residual", np.nan)),
        "max_reverse_position_error": float(diag.get("max_reverse_position_error", np.nan)),
        "max_reverse_momentum_error": float(diag.get("max_reverse_momentum_error", np.nan)),
        "student_grad_evals_per_iteration": float(counters.get("student_grad_evals", np.nan)) / iterations,
        "constraint_grad_evals_per_iteration": float(counters.get("constraint_grad_evals", np.nan)) / iterations,
        "student_logpdf_evals_per_iteration": float(counters.get("student_logpdf_evals", np.nan)) / iterations,
        "projection_evals_per_iteration": projection_evals / iterations,
        "projection_mode": str(args.rattle_projection_mode),
        "gram_correction_enabled": bool(args.rattle_include_gram_correction),
        "reverse_check": bool(args.reverse_check),
        "rattle_step_size": float(args.rattle_step_size),
        "rattle_num_steps": int(args.rattle_num_steps),
    }
    return row


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "elapsed_sec",
        "sec_per_iteration",
        "mu_acceptance_rate",
        "x_acceptance_rate",
        "ess_mu",
        "ess_mu_per_sec",
        "posterior_sd_mu",
        "mean_abs_delta_mu",
        "esjd_mu",
        "esjd_mu_per_sec",
        "projection_failure_rate",
        "reverse_check_failure_rate",
        "projection_iterations_per_iteration",
        "student_grad_evals_per_iteration",
        "constraint_grad_evals_per_iteration",
        "student_logpdf_evals_per_iteration",
        "projection_evals_per_iteration",
        "max_projection_residual",
        "max_reverse_position_error",
    ]
    grouped = rows.groupby(["k", "n"], as_index=False)
    summary = grouped[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if part).rstrip("_") if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary["num_seeds"] = grouped.size()["size"].to_numpy()
    return summary


def write_report(args: argparse.Namespace, rows: pd.DataFrame, summary: pd.DataFrame) -> None:
    report_path = args.out / "local_rattle_scaling_report.md"
    key_cols = [
        "k",
        "n",
        "sec_per_iteration_mean",
        "sec_per_iteration_std",
        "ess_mu_per_sec_mean",
        "x_acceptance_rate_mean",
        "projection_failure_rate_mean",
        "reverse_check_failure_rate_mean",
        "projection_iterations_per_iteration_mean",
    ]
    table = summary[key_cols].to_markdown(index=False, floatfmt=".6g")
    text = f"""# Local RATTLE Scaling Probe

Run settings:

- iterations: {int(args.iterations)}
- burn-in: {int(args.burn_in)}
- seeds: {','.join(str(seed) for seed in args.seeds)}
- k values: {','.join(str(k) for k in args.k_values)}
- n values: {','.join(str(n) for n in args.n_values)}
- reverse_check: {bool(args.reverse_check)}
- projection_mode: {args.rattle_projection_mode}
- gram_correction_enabled: {bool(args.rattle_include_gram_correction)}

## Summary

{table}

## Output Files

- rows: `{args.out / "local_rattle_scaling_rows.csv"}`
- summary: `{args.out / "local_rattle_scaling_summary.csv"}`
"""
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.warmup:
        print("[warmup] start untimed RATTLE/JAX warm-up for all k,n shapes", flush=True)
        warmup_args = argparse.Namespace(**vars(args))
        warmup_args.iterations = 2
        warmup_args.burn_in = 0
        for k in args.k_values:
            for n in args.n_values:
                _ = run_case(warmup_args, float(k), int(n), int(args.seeds[0]))
        print("[warmup] done", flush=True)
    total = len(args.k_values) * len(args.n_values) * len(args.seeds)
    rows = []
    case_idx = 0
    for k in args.k_values:
        for n in args.n_values:
            for seed in args.seeds:
                case_idx += 1
                print(
                    f"[{case_idx}/{total}] start student_t RATTLE k={k:g} n={n} seed={seed} "
                    f"iterations={args.iterations}",
                    flush=True,
                )
                row = run_case(args, float(k), int(n), int(seed))
                rows.append(row)
                print(
                    f"[{case_idx}/{total}] done k={k:g} n={n} seed={seed} "
                    f"elapsed={row['elapsed_sec']:.3f}s sec/iter={row['sec_per_iteration']:.6g} "
                    f"ess/sec={row['ess_mu_per_sec']:.6g} x_acc={row['x_acceptance_rate']:.3f} "
                    f"proj_fail={row['projection_failure_rate']:.3g} rev_fail={row['reverse_check_failure_rate']:.3g}",
                    flush=True,
                )

    rows_df = pd.DataFrame(rows)
    summary_df = summarize(rows_df)
    rows_path = args.out / "local_rattle_scaling_rows.csv"
    summary_path = args.out / "local_rattle_scaling_summary.csv"
    rows_df.to_csv(rows_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    write_report(args, rows_df, summary_df)
    print(f"wrote rows to {rows_path}", flush=True)
    print(f"wrote summary to {summary_path}", flush=True)
    print(f"wrote report to {args.out / 'local_rattle_scaling_report.md'}", flush=True)


if __name__ == "__main__":
    main()
